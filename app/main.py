from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, HttpUrl, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict, Any
from datetime import datetime
from urllib.parse import urlparse
import os
import json
import re

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from app.db.engine import Base, engine, SessionLocal
from app.models.models import Website, Product
from app.models.llmtest import LLMTest
from app.models.prompt_models import PromptPack, Prompt
from app.models.user_models import User
from app.config import DEFAULT_LLM_MODEL, DEFAULT_REPORT_MODEL

from app.services.llm_checker import (
    run_llm_visibility_check,
    run_llm_visibility_batch_check,  # NEW
)
from app.services.prompt_generator import (
    generate_prompt_pack_for_product,
    save_prompt_pack_to_file,
)
from app.services.google_prompt_generator import (
    generate_google_prompt_pack_for_product,
)
from app.services.prompt_packs import (
    load_prompt_pack,
    PROMPT_PACKS_DIR,
)
from app.services.visibility_report import (
    fetch_page_html_via_scraperapi,
    build_page_snapshot,
    generate_visibility_report_markdown,
    save_report_markdown_to_file,
    REPORTS_DIR,
)
from app.services.auth_utils import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)

load_dotenv()

app = FastAPI(title="LLM Visibility API", version="0.7.0")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://llm-visibility-frontend-production.up.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========= Multi-tenant helpers (schema-per-tenant) =========

TENANT_HEADER = "X-Tenant"  # header that identifies the client/tenant

# For registration-time validation of client codes
TENANT_CODE_REGEX = re.compile(r"^[a-z0-9_]{3,32}$")


def normalize_and_validate_tenant_code(raw_code: str) -> str:
    """
    Normalise and validate a tenant code that the client enters at registration.

    Rules:
    - lowercase
    - 3–32 chars
    - only a–z, 0–9, underscore
    - must NOT start with 'tenant_'
    - cannot be 'public'
    """
    if not raw_code:
        raise HTTPException(status_code=400, detail="Tenant code is required")

    code = raw_code.strip().lower()

    if code.startswith("tenant_"):
        raise HTTPException(
            status_code=400,
            detail="Tenant code should not start with 'tenant_'. Just use the short code, e.g. 'acme_client'.",
        )

    if not TENANT_CODE_REGEX.fullmatch(code):
        raise HTTPException(
            status_code=400,
            detail="Invalid tenant code. Use 3–32 characters: lowercase letters, numbers and underscore only.",
        )

    if code == "public":
        raise HTTPException(
            status_code=400,
            detail="This tenant code is reserved. Please choose a different one.",
        )

    return code


def schema_name_for_tenant(code: str) -> str:
    """
    Our convention: tenant code 'test_client' -> schema 'tenant_test_client'
    """
    return f"tenant_{code}"


def provision_new_tenant(
    tenant_code: str,
    admin_email: str,
    admin_password: str,
) -> str:
    """
    Creates:
    - schema: tenant_<code>
    - all tables in that schema (using Base.metadata)
    - an admin user in that schema

    Returns the schema name.
    """
    # Validate & normalise code
    code = normalize_and_validate_tenant_code(tenant_code)
    schema_name = schema_name_for_tenant(code)

    # 1) Check if schema already exists
    with engine.connect() as conn:
        existing = conn.execute(
            text(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = :name"
            ),
            {"name": schema_name},
        ).scalar()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="That tenant code is already in use. Please choose another.",
            )

    # 2) Create schema + tables + admin user in a single transaction
    hashed = hash_password(admin_password)

    # Use a transactional connection
    with engine.begin() as conn:
        # Create schema
        conn.execute(text(f'CREATE SCHEMA "{schema_name}"'))

        # Set search_path so all DDL/DML go into this schema
        conn.execute(text(f'SET search_path TO "{schema_name}"'))

        # Create all tables defined on Base in this schema
        Base.metadata.create_all(bind=conn)

        # Insert the initial admin user into this tenant's users table
        conn.execute(
            text(
                """
                INSERT INTO users (email, hashed_password, is_active, is_admin, created_at)
                VALUES (:email, :hashed_password, TRUE, TRUE, NOW())
                """
            ),
            {"email": admin_email, "hashed_password": hashed},
        )

    return schema_name


def _normalize_schema_name(raw: str) -> str:
    """
    Turn a client identifier into a safe Postgres schema name.
    Examples:
      "demo_client"     -> "tenant_demo_client"
      "tenant_client_b" -> "tenant_client_b" (kept as is)

    This is used for request-time routing via the X-Tenant header.
    """
    raw = (raw or "").strip().lower()
    if not raw:
        return "public"

    if raw.startswith("tenant_"):
        schema = raw
    else:
        schema = f"tenant_{raw}"

    # basic safety check – only allow letters, digits, underscore
    if not re.fullmatch(r"[a-zA-Z0-9_]+", schema):
        raise HTTPException(status_code=400, detail="Invalid tenant name")

    return schema


def _ensure_tenant_schema(schema: str):
    """
    Ensure the tenant schema exists.

    IMPORTANT:
    - This NO LONGER creates schemas or tables.
    - New schemas are provisioned ONLY via `provision_new_tenant`
      (used by /auth/register-client).
    """
    if schema == "public":
        # public gets its tables created at startup
        return

    with engine.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT 1 FROM information_schema.schemata "
                "WHERE schema_name = :name"
            ),
            {"name": schema},
        ).scalar()

    if not exists:
        # Unknown / unprovisioned tenant
        raise HTTPException(
            status_code=404,
            detail="Unknown workspace / tenant. Please check your tenant code.",
        )


def get_tenant_schema(request: Request) -> str:
    """
    Read tenant from header and normalise to schema name.
    If header missing, fall back to 'public'.
    """
    raw_tenant = request.headers.get(TENANT_HEADER)
    return _normalize_schema_name(raw_tenant or "")


# --- DB Session Dependency (now tenant-aware) ---
def get_db(request: Request):
    """
    Open a DB session scoped to the tenant's schema using search_path.

    NOTE:
    - This will now ONLY work for existing, provisioned schemas.
    - New schemas are created via /auth/register-client (provision_new_tenant).
    """
    db = SessionLocal()
    try:
        schema = get_tenant_schema(request)

        # Ensure the schema already exists (no auto-creation here)
        _ensure_tenant_schema(schema)

        # Set search_path on this session so all queries are schema-scoped
        db.execute(text(f'SET search_path TO "{schema}"'))

        yield db
    finally:
        db.close()


# --- Create tables on startup (public schema only) ---
@app.on_event("startup")
def on_startup():
    # Base public schema – useful for default/demo tenant
    Base.metadata.create_all(bind=engine)


# --- Schemas (Pydantic models) ---

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
    model: Optional[str] = DEFAULT_LLM_MODEL


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
    model: Optional[str] = DEFAULT_LLM_MODEL


class LLMRunBatchResult(BaseModel):
    product_id: int
    pack_id: str
    model_used: str
    total_prompts: int
    appeared_count: int
    visibility_score: float


class GeneratePromptPackRequest(BaseModel):
    product_id: int
    num_prompts: int = 50
    pack_id: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None


class GeneratePromptPackResponse(BaseModel):
    pack_id: str
    num_prompts: int
    file_path: str
    download_url: str


class GenerateGooglePromptPackRequest(BaseModel):
    product_id: int
    num_prompts: int = 50
    pack_id: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None  # logical product/category tag


class GenerateGooglePromptPackResponse(BaseModel):
    pack_id: str
    num_prompts: int
    file_path: str
    download_url: str


class VisibilityReportRequest(BaseModel):
    product_id: int
    model: Optional[str] = DEFAULT_REPORT_MODEL


class VisibilityReportResponse(BaseModel):
    product_id: int
    model_used: str
    report_markdown: str
    file_path: str
    download_url: str


class PromptPerformance(BaseModel):
    prompt_id: int
    index: int
    text: str
    total_runs: int
    appeared_count: int
    visibility_score: float


# ----- Auth Pydantic models -----

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: EmailStr
    is_active: bool
    is_admin: bool

    class Config:
        from_attributes = True


class RegisterAdminRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ----- Client registration models -----

class ClientRegistrationRequest(BaseModel):
    tenant_code: str   # what the client will type, e.g. "test_client"
    admin_email: EmailStr
    admin_password: str


class ClientRegistrationResponse(BaseModel):
    tenant_code: str
    schema_name: str
    admin_email: EmailStr
    message: str


# ----- Auth helpers (dependencies) -----

def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Decodes JWT, loads user from the CURRENT tenant schema.
    """
    from jose import JWTError

    try:
        payload = decode_access_token(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    subject = payload.get("sub")
    if subject is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.query(User).filter(User.email == subject).one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is inactive")

    return user


# --- Endpoints ---

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


# ----- Client registration endpoint -----

@app.post("/auth/register-client", response_model=ClientRegistrationResponse)
def register_client(payload: ClientRegistrationRequest):
    """
    Public endpoint: a new client can self-register.

    Behaviour:
    - Validates tenant_code format
    - Checks if tenant_code/schema already exists
    - Creates schema + tables for this tenant
    - Creates an admin user in that schema

    The client will then use:
      - tenant_code (e.g. 'test_client')
      - admin_email
      - admin_password
    on the normal /auth/login endpoint, with X-Tenant set to tenant_code.
    """
    # We validate and normalise inside provision_new_tenant
    schema_name = provision_new_tenant(
        tenant_code=payload.tenant_code,
        admin_email=payload.admin_email,
        admin_password=payload.admin_password,
    )

    # Return the normalised tenant code as well
    code = normalize_and_validate_tenant_code(payload.tenant_code)

    return ClientRegistrationResponse(
        tenant_code=code,
        schema_name=schema_name,
        admin_email=payload.admin_email,
        message=(
            "Workspace created successfully. "
            "You can now log in using this tenant code, email and password."
        ),
    )


# ----- Auth endpoints -----

@app.post("/auth/register-admin", response_model=UserOut)
def register_admin(
    payload: RegisterAdminRequest,
    db: Session = Depends(get_db),
    request: Request = None,
):
    """
    Creates an admin user for the current tenant (schema defined by X-Tenant header).
    You can later restrict or remove this endpoint in production.
    """
    existing = db.query(User).filter(User.email == payload.email).one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="User with this email already exists")

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        is_admin=True,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=Token)
def login(
    payload: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Login within the current tenant (from X-Tenant header).
    Returns a JWT access token.
    """
    user = db.query(User).filter(User.email == payload.email).one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is inactive")

    # Get tenant code from header to embed in the token (optional)
    tenant_raw = request.headers.get(TENANT_HEADER) or ""
    tenant_code = tenant_raw.strip() or "public"

    access_token = create_access_token(
        subject=user.email,
        tenant=tenant_code,
    )

    return Token(access_token=access_token)


# ----- Core product / visibility endpoints -----

@app.post("/analyze-website", response_model=AnalyzeResult)
async def analyze_website(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    target_url = str(payload.url)
    parsed = urlparse(target_url)
    domain = parsed.netloc.lower()

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

    soup = BeautifulSoup(html, "lxml")

    # Derive title
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

    # Detect Product JSON-LD
    has_product_jsonld = False
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue

        def contains_product(d: Any) -> bool:
            if isinstance(d, dict):
                t = d.get("@type")
                if t == "Product":
                    return True
                if isinstance(t, list) and "Product" in t:
                    return True
                for k in ("@graph", "itemListElement", "mainEntity"):
                    if k in d and contains_product(d[k]):
                        return True
            elif isinstance(d, list):
                return any(contains_product(x) for x in d)
            return False

        if contains_product(data):
            has_product_jsonld = True
            break

    # Upsert Website & Product
    website = db.query(Website).filter(Website.domain == domain).one_or_none()
    if not website:
        website = Website(domain=domain)
        db.add(website)
        db.flush()

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

    # Cache values BEFORE commit to avoid ObjectDeletedError / lazy reload issues
    website_domain_value = website.domain
    product_url_value = product.url
    has_product_jsonld_value = has_product_jsonld
    page_title_value = page_title

    db.commit()

    return AnalyzeResult(
        website_domain=website_domain_value,
        product_url=product_url_value,
        page_title=page_title_value,
        has_product_jsonld=has_product_jsonld_value,
    )


@app.get("/products", response_model=List[ProductOut])
def list_products(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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


@app.post("/run-llm-check", response_model=LLMCheckResult)
def run_llm_check(
    payload: LLMCheckRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    parsed = urlparse(product.url)
    domain = parsed.netloc

    result = run_llm_visibility_check(
        prompt=payload.prompt,
        domain=domain,
        model=payload.model,
    )

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


@app.get("/prompt-packs", response_model=List[PromptPackSummary])
def get_prompt_packs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List prompt packs for the CURRENT TENANT ONLY.
    Uses the tenant-scoped DB (schema set by get_db).
    """
    db_packs = db.query(PromptPack).all()

    results: List[PromptPackSummary] = []
    for pack in db_packs:
        num_prompts = len(pack.prompts) if pack.prompts is not None else 0
        results.append(
            PromptPackSummary(
                id=pack.pack_key,
                name=pack.name or pack.pack_key,
                category=pack.category,
                language=pack.language,
                num_prompts=num_prompts,
            )
        )
    return results


@app.post("/run-llm-batch", response_model=LLMRunBatchResult)
def run_llm_batch(
    payload: LLMRunBatchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    parsed = urlparse(product.url)
    domain = parsed.netloc
    if not domain:
        raise HTTPException(status_code=400, detail="Invalid product URL domain")

    # Load pack from JSON
    try:
        pack_data = load_prompt_pack(payload.pack_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    prompts_list = pack_data.get("prompts", [])
    if not prompts_list:
        raise HTTPException(status_code=400, detail="Prompt pack has no prompts")

    total = len(prompts_list)

    # Ensure PromptPack exists in DB
    db_pack = (
        db.query(PromptPack)
        .filter(PromptPack.pack_key == payload.pack_id)
        .one_or_none()
    )
    if not db_pack:
        db_pack = PromptPack(
            pack_key=payload.pack_id,
            name=pack_data.get("name"),
            category=pack_data.get("category"),
            source=pack_data.get("source") or "unknown",
            language=pack_data.get("language", "en"),
            product_id=product.id,
        )
        db.add(db_pack)
        db.flush()

    # Load existing Prompt rows for this pack
    existing_prompts = {
        p.index: p for p in db.query(Prompt).filter(Prompt.pack_id == db_pack.id).all()
    }

    # Ensure we have Prompt rows for each index
    for idx, prompt_text in enumerate(prompts_list):
        if idx not in existing_prompts:
            prompt_row = Prompt(
                pack_id=db_pack.id,
                index=idx,
                text=prompt_text,
            )
            db.add(prompt_row)
            db.flush()
            existing_prompts[idx] = prompt_row

    # --- NEW: Single LLM batch call for all prompts ---
    try:
        batch_results = run_llm_visibility_batch_check(
            prompts=prompts_list,
            domain=domain,
            model=payload.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running LLM batch: {e}")
    # --------------------------------------------------

    appeared_count = 0
    rows: List[LLMTest] = []

    # batch_results[i] corresponds to prompts_list[i] (best-effort)
    for idx, prompt_text in enumerate(prompts_list):
        if idx < len(batch_results):
            res = batch_results[idx]
            appeared = res.get("appeared", False)
            matched = res.get("matched") or None
            snippet = res.get("snippet") or ""
            model_used = res.get("model") or payload.model
        else:
            # Fallback if model omitted some indexes
            appeared = False
            matched = None
            snippet = ""
            model_used = payload.model

        if appeared:
            appeared_count += 1

        prompt_row = existing_prompts[idx]

        row = LLMTest(
            product_id=product.id,
            pack_id=db_pack.id,
            prompt_id=prompt_row.id,
            model_used=model_used,
            prompt=prompt_text,
            appeared=appeared,
            matched_domain=matched,
            snippet=snippet,
        )
        rows.append(row)

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


@app.post("/generate-prompt-pack", response_model=GeneratePromptPackResponse)
def generate_prompt_pack(
    payload: GeneratePromptPackRequest,
    db: Session = Depends(get_db),
    request: Request = None,
    current_user: User = Depends(get_current_user),
):
    """
    Source A: LLM-generated high-intent prompts (behavioural categories).
    """
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    inferred_category = payload.category or None

    pack = generate_prompt_pack_for_product(
        product_id=product.id,
        product_title=product.title or product.url,
        product_url=product.url,
        category=inferred_category,
        num_prompts=payload.num_prompts,
        pack_id=payload.pack_id,
        name=payload.name,
    )

    # Save JSON file
    file_path = save_prompt_pack_to_file(pack)
    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/download/prompt-pack/{pack['id']}"

    # --- Sync DB PromptPack + Prompt records ---
    pack_key = pack["id"]
    db_pack = (
        db.query(PromptPack)
        .filter(PromptPack.pack_key == pack_key)
        .one_or_none()
    )
    if not db_pack:
        db_pack = PromptPack(
            pack_key=pack_key,
            name=pack.get("name"),
            category=pack.get("category"),
            source=pack.get("source") or "auto_generated_high_intent",
            language=pack.get("language", "en"),
            product_id=product.id,
        )
        db.add(db_pack)
        db.flush()

    # Clear existing prompts for this pack (if re-generating)
    db.query(Prompt).filter(Prompt.pack_id == db_pack.id).delete()

    prompts = pack.get("prompts", [])
    for idx, text in enumerate(prompts):
        db.add(
            Prompt(
                pack_id=db_pack.id,
                index=idx,
                text=text,
            )
        )

    db.commit()
    # --- End sync ---

    return GeneratePromptPackResponse(
        pack_id=pack["id"],
        num_prompts=len(pack.get("prompts", [])),
        file_path=file_path,
        download_url=download_url,
    )


@app.post("/generate-google-prompt-pack", response_model=GenerateGooglePromptPackResponse)
def generate_google_prompt_pack(
    payload: GenerateGooglePromptPackRequest,
    db: Session = Depends(get_db),
    request: Request = None,
    current_user: User = Depends(get_current_user),
):
    """
    Source B: Google-seeded high-intent prompts (Scrapingdog 'people also ask').
    """
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    inferred_category = payload.category or None

    try:
        pack = generate_google_prompt_pack_for_product(
            product_id=product.id,
            product_title=product.title or product.url,
            product_url=product.url,
            category=inferred_category,
            num_prompts=payload.num_prompts,
            pack_id=payload.pack_id,
            name=payload.name,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Google prompt pack: {e}")

    # Save JSON file
    file_path = save_prompt_pack_to_file(pack)
    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/download/prompt-pack/{pack['id']}"

    # --- Sync DB PromptPack + Prompt records ---
    pack_key = pack["id"]
    db_pack = (
        db.query(PromptPack)
        .filter(PromptPack.pack_key == pack_key)
        .one_or_none()
    )
    if not db_pack:
        db_pack = PromptPack(
            pack_key=pack_key,
            name=pack.get("name"),
            category=pack.get("category"),
            source=pack.get("source") or "google_people_also_ask",
            language=pack.get("language", "en"),
            product_id=product.id,
        )
        db.add(db_pack)
        db.flush()

    # Clear existing prompts for this pack (if re-generating)
    db.query(Prompt).filter(Prompt.pack_id == db_pack.id).delete()

    prompts = pack.get("prompts", [])
    for idx, text in enumerate(prompts):
        db.add(
            Prompt(
                pack_id=db_pack.id,
                index=idx,
                text=text,
            )
        )

    db.commit()
    # --- End sync ---

    return GenerateGooglePromptPackResponse(
        pack_id=pack["id"],
        num_prompts=len(pack.get("prompts", [])),
        file_path=file_path,
        download_url=download_url,
    )


@app.post("/visibility-report", response_model=VisibilityReportResponse)
async def visibility_report(
    payload: VisibilityReportRequest,
    db: Session = Depends(get_db),
    request: Request = None,
    current_user: User = Depends(get_current_user),
):
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    parsed = urlparse(product.url)
    domain = parsed.netloc

    tests = db.query(LLMTest).filter(LLMTest.product_id == product.id).all()

    total_tests = len(tests)
    appeared_count = sum(1 for t in tests if t.appeared)

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
                "visibility_score": (data["appeared"] / data["total"]) if data["total"] > 0 else 0.0,
            }
            for model, data in per_model.items()
        },
    }

    try:
        html = await fetch_page_html_via_scraperapi(product.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching page HTML: {e}")

    snapshot = build_page_snapshot(html)

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

    try:
        file_path = save_report_markdown_to_file(product.id, report_md)
        base_url = str(request.base_url).rstrip("/")
        filename = os.path.basename(file_path)
        download_url = f"{base_url}/download/report/{filename}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving report: {e}")

    return VisibilityReportResponse(
        product_id=product.id,
        model_used=payload.model,
        report_markdown=report_md,
        file_path=file_path,
        download_url=download_url,
    )


@app.get("/download/prompt-pack/{pack_id}")
def download_prompt_pack(
    pack_id: str,
):
    """
    Download a prompt pack JSON file by pack_id.

    Note:
    - We do not enforce a tenant DB check here because the browser
      cannot send the X-Tenant header on a normal link click.
    - We still validate the pack_id to avoid path traversal.
    """
    if ".." in pack_id or "/" in pack_id or "\\" in pack_id:
        raise HTTPException(status_code=400, detail="Invalid pack_id")

    fname = f"{pack_id}.json"
    fpath = os.path.join(PROMPT_PACKS_DIR, fname)

    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail="Prompt pack file not found")

    return FileResponse(
        path=fpath,
        media_type="application/json",
        filename=fname,
    )


@app.get("/download/report/{filename}")
def download_report(
    filename: str,
):
    """
    Download a Markdown report file by filename.
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    fpath = os.path.join(REPORTS_DIR, filename)

    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        path=fpath,
        media_type="text/markdown",
        filename=filename,
    )


@app.get("/prompt-stats/{pack_id}", response_model=List[PromptPerformance])
def prompt_stats(
    pack_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns per-prompt performance for a given pack_id (pack_key).
    """
    db_pack = (
        db.query(PromptPack)
        .filter(PromptPack.pack_key == pack_id)
        .one_or_none()
    )
    if not db_pack:
        raise HTTPException(status_code=404, detail="Prompt pack not found in DB")

    prompts = (
        db.query(Prompt)
        .filter(Prompt.pack_id == db_pack.id)
        .order_by(Prompt.index)
        .all()
    )

    results: List[PromptPerformance] = []

    for prompt in prompts:
        tests = db.query(LLMTest).filter(LLMTest.prompt_id == prompt.id).all()
        total_runs = len(tests)
        appeared_count = sum(1 for t in tests if t.appeared)
        visibility_score = (appeared_count / total_runs) if total_runs > 0 else 0.0

        results.append(
            PromptPerformance(
                prompt_id=prompt.id,
                index=prompt.index,
                text=prompt.text,
                total_runs=total_runs,
                appeared_count=appeared_count,
                visibility_score=visibility_score,
            )
        )

    return results