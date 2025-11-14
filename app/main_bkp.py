from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from datetime import datetime
from urllib.parse import urlparse
import os
import json

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from app.db.engine import Base, engine, SessionLocal
from app.models.models import Website, Product
from app.models.llmtest import LLMTest

from app.services.llm_checker import run_llm_visibility_check
from app.services.prompt_packs import list_prompt_packs, load_prompt_pack
from app.services.prompt_generator import (
    generate_prompt_pack_for_product,
    save_prompt_pack_to_file,
)
from app.services.visibility_report import (
    fetch_page_html_via_scraperapi,
    build_page_snapshot,
    generate_visibility_report_markdown,
)

# Load environment variables once at startup
load_dotenv()

app = FastAPI(title="LLM Visibility API", version="0.3.0")


# --- DB Session Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Create tables on startup ---
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


# --- Pydantic Schemas ---

class AnalyzeRequest(BaseModel):
    url: HttpUrl


class AnalyzeResult(BaseModel):
    website_domain: str
    product_url: str
    page_title: Optional[str] = None
    has_product_jsonld: bool


class ProductOut(BaseModel):
    id: int
    website_domain: str
    url: str
    title: Optional[str]
    has_product_jsonld: bool
    last_checked: datetime

    class Config:
        from_attributes = True


class LLMCheckRequest(BaseModel):
    product_id: int
    prompt: str
    model: Optional[str] = "gpt-4.1-mini"


class LLMCheckResult(BaseModel):
    product_id: int
    model_used: str
    appeared: bool
    matched_domain_or_url: Optional[str] = None
    snippet: Optional[str] = None


class PromptPackSummary(BaseModel):
    id: str
    name: str
    category: Optional[str] = None
    language: Optional[str] = None
    num_prompts: int


class LLMRunBatchRequest(BaseModel):
    product_id: int
    pack_id: str
    model: Optional[str] = "gpt-4.1-mini"


class LLMRunBatchResult(BaseModel):
    product_id: int
    pack_id: str
    model_used: str
    total_prompts: int
    appeared_count: int
    visibility_score: float  # 0.0 to 1.0


class GeneratePromptPackRequest(BaseModel):
    product_id: int
    num_prompts: int = 50
    pack_id: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None  # optional manual override


class GeneratePromptPackResponse(BaseModel):
    pack_id: str
    num_prompts: int
    file_path: str


class VisibilityReportRequest(BaseModel):
    product_id: int
    model: Optional[str] = "gpt-4.1"  # can use gpt-4.1-mini if needed


class VisibilityReportResponse(BaseModel):
    product_id: int
    model_used: str
    report_markdown: str


# --- Basic Health Check ---

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


# --- Analyze Website / Product Page ---

@app.post("/analyze-website", response_model=AnalyzeResult)
async def analyze_website(payload: AnalyzeRequest, db: Session = Depends(get_db)):
    target_url = str(payload.url)
    parsed = urlparse(target_url)
    domain = parsed.netloc.lower()

    # 1) Fetch HTML via ScraperAPI (external crawler)
    SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")
    if not SCRAPER_API_KEY:
        raise HTTPException(status_code=500, detail="SCRAPER_API_KEY not set in .env")

    api_url = f"http://api.scraperapi.com/?api_key={SCRAPER_API_KEY}&url={target_url}&render=true"

    try:
        async with httpx.AsyncClient(timeout=40) as client:
            resp = await client.get(api_url)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fetch error: {e}")

    # 2) Parse HTML
    soup = BeautifulSoup(html, "lxml")

    # title preference: <meta property="og:title">, then <title>, then first <h1>
    page_title = None
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        page_title = og_title["content"].strip()
    if not page_title and soup.title and soup.title.string:
        page_title = soup.title.string.strip()
    if not page_title:
        h1 = soup.find("h1")
        if h1 and h1.get_text():
            page_title = h1.get_text().strip()

    # 3) Look for any Product JSON-LD
    has_product_jsonld = False
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue

        def contains_product(d: Any) -> bool:
            # handle dict or list
            if isinstance(d, dict):
                t = d.get("@type")
                if t == "Product":
                    return True
                if isinstance(t, list) and "Product" in t:
                    return True
                # dive into nested graph or itemListElement
                for k in ("@graph", "itemListElement", "mainEntity"):
                    if k in d:
                        if contains_product(d[k]):
                            return True
            elif isinstance(d, list):
                return any(contains_product(x) for x in d)
            return False

        if contains_product(data):
            has_product_jsonld = True
            break

    # 4) Upsert Website & Product
    website = db.query(Website).filter(Website.domain == domain).one_or_none()
    if not website:
        website = Website(domain=domain)
        db.add(website)
        db.flush()  # assign id

    product = db.query(Product).filter(Product.url == target_url).one_or_none()
    if not product:
        product = Product(
            website_id=website.id,
            url=target_url,
            title=page_title,
            has_product_jsonld=has_product_jsonld,
            last_checked=datetime.utcnow(),
        )
        db.add(product)
    else:
        product.title = page_title
        product.has_product_jsonld = has_product_jsonld
        product.last_checked = datetime.utcnow()

    db.commit()

    return AnalyzeResult(
        website_domain=website.domain,
        product_url=product.url,
        page_title=page_title,
        has_product_jsonld=has_product_jsonld,
    )


# --- List Products ---

@app.get("/products", response_model=List[ProductOut])
def list_products(db: Session = Depends(get_db)):
    rows = (
        db.query(Product)
        .join(Website, Product.website_id == Website.id)
        .all()
    )

    result: List[ProductOut] = []
    for p in rows:
        result.append(
            ProductOut(
                id=p.id,
                website_domain=p.website.domain if p.website else "",
                url=p.url,
                title=p.title,
                has_product_jsonld=p.has_product_jsonld,
                last_checked=p.last_checked,
            )
        )
    return result


# --- Single LLM Visibility Check ---

@app.post("/run-llm-check", response_model=LLMCheckResult)
def run_llm_check(payload: LLMCheckRequest, db: Session = Depends(get_db)):
    # 1) Get product and domain
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    parsed = urlparse(product.url)
    domain = parsed.netloc

    # 2) Run the model check
    result = run_llm_visibility_check(
        prompt=payload.prompt,
        domain=domain,
        model=payload.model,
    )

    # 3) Persist test result
    row = LLMTest(
        product_id=product.id,
        model_used=result["model"],
        prompt=payload.prompt,
        appeared=result["appeared"],
        matched_domain=result["matched"] or None,
        snippet=result["snippet"],
    )
    db.add(row)
    db.commit()

    return LLMCheckResult(
        product_id=product.id,
        model_used=result["model"],
        appeared=result["appeared"],
        matched_domain_or_url=result["matched"] or None,
        snippet=result["snippet"],
    )


# --- Prompt Packs Listing ---

@app.get("/prompt-packs", response_model=List[PromptPackSummary])
def get_prompt_packs():
    packs_raw = list_prompt_packs()
    return [
        PromptPackSummary(
            id=p["id"],
            name=p["name"],
            category=p.get("category"),
            language=p.get("language"),
            num_prompts=p.get("num_prompts", 0),
        )
        for p in packs_raw
    ]


# --- Batch LLM Visibility Check using a Prompt Pack ---

@app.post("/run-llm-batch", response_model=LLMRunBatchResult)
def run_llm_batch(payload: LLMRunBatchRequest, db: Session = Depends(get_db)):
    # 1) Get product
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # 2) Get domain from URL
    parsed = urlparse(product.url)
    domain = parsed.netloc
    if not domain:
        raise HTTPException(status_code=400, detail="Invalid product URL domain")

    # 3) Load prompt pack
    try:
        pack = load_prompt_pack(payload.pack_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    prompts = pack.get("prompts", [])
    if not prompts:
        raise HTTPException(status_code=400, detail="Prompt pack has no prompts")

    total = len(prompts)
    appeared_count = 0
    rows: List[LLMTest] = []

    # 4) Run tests for each prompt
    for prompt in prompts:
        result = run_llm_visibility_check(
            prompt=prompt,
            domain=domain,
            model=payload.model,
        )
        if result["appeared"]:
            appeared_count += 1

        row = LLMTest(
            product_id=product.id,
            model_used=result["model"],
            prompt=prompt,
            appeared=result["appeared"],
            matched_domain=result["matched"] or None,
            snippet=result["snippet"],
        )
        rows.append(row)

    # 5) Save all rows in a single commit
    if rows:
        db.add_all(rows)
        db.commit()

    visibility_score = appeared_count / total if total > 0 else 0.0

    return LLMRunBatchResult(
        product_id=product.id,
        pack_id=payload.pack_id,
        model_used=payload.model,
        total_prompts=total,
        appeared_count=appeared_count,
        visibility_score=visibility_score,
    )


# --- Auto Prompt Pack Generation ---

@app.post("/generate-prompt-pack", response_model=GeneratePromptPackResponse)
def generate_prompt_pack(payload: GeneratePromptPackRequest, db: Session = Depends(get_db)):
    # 1) Get product from DB
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # For now we don't have a category column; you can extend Product later.
    inferred_category = payload.category or None

    # 2) Generate pack dict via LLM
    pack = generate_prompt_pack_for_product(
        product_id=product.id,
        product_title=product.title or product.url,
        product_url=product.url,
        category=inferred_category,
        num_prompts=payload.num_prompts,
        pack_id=payload.pack_id,
        name=payload.name,
    )

    # 3) Save to prompt_packs/{id}.json
    file_path = save_prompt_pack_to_file(pack)

    return GeneratePromptPackResponse(
        pack_id=pack["id"],
        num_prompts=len(pack.get("prompts", [])),
        file_path=file_path,
    )


# --- LLM Visibility Technical Report ---

@app.post("/visibility-report", response_model=VisibilityReportResponse)
async def visibility_report(payload: VisibilityReportRequest, db: Session = Depends(get_db)):
    # 1) Fetch product
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    parsed = urlparse(product.url)
    domain = parsed.netloc

    # 2) Compute basic visibility metrics from LLMTest
    tests = db.query(LLMTest).filter(LLMTest.product_id == product.id).all()

    total_tests = len(tests)
    appeared_count = sum(1 for t in tests if t.appeared)

    # Per-model breakdown
    per_model: Dict[str, Dict[str, Any]] = {}
    for t in tests:
        m = t.model_used
        pm = per_model.setdefault(m, {"total": 0, "appeared": 0})
        pm["total"] += 1
        if t.appeared:
            pm["appeared"] += 1

    overall_visibility = (appeared_count / total_tests) if total_tests > 0 else 0.0

    visibility_metrics = {
        "product_id": product.id,
        "url": product.url,
        "total_tests": total_tests,
        "appeared_count": appeared_count,
        "overall_visibility_score": overall_visibility,
        "per_model": {
            model: {
                "total": data["total"],
                "appeared": data["appeared"],
                "visibility_score": (data["appeared"] / data["total"])
                if data["total"] > 0
                else 0.0,
            }
            for model, data in per_model.items()
        },
    }

    # 3) Fetch HTML & build snapshot
    try:
        html = await fetch_page_html_via_scraperapi(product.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching page HTML: {e}")

    snapshot = build_page_snapshot(html)

    # 4) Generate report via LLM
    try:
        report_md = generate_visibility_report_markdown(
            product_title=product.title or product.url,
            product_url=product.url,
            domain=domain,
            has_product_jsonld=product.has_product_jsonld,
            visibility_metrics=visibility_metrics,
            page_snapshot=snapshot,
            model=payload.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {e}")

    return VisibilityReportResponse(
        product_id=product.id,
        model_used=payload.model,
        report_markdown=report_md,
    )