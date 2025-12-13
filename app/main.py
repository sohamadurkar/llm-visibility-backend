from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
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
import uuid  # NEW

import threading

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from app.db.engine import Base, engine, SessionLocal
from app.models.models import Website, Product
from app.models.llmtest import LLMTest
from app.models.prompt_models import PromptPack, Prompt
from app.models.user_models import User
from app.models.content_models import ContentArticle
from app.config import DEFAULT_LLM_MODEL, DEFAULT_REPORT_MODEL

from app.services.llm_checker import (
    run_llm_visibility_check,
    run_llm_visibility_batch_check,
)
from app.services.prompt_generator import (
    generate_prompt_pack_for_product,
    save_prompt_pack_to_file,
    generate_persona_prompt_pack_for_product,  # NEW
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
# NEW: competitor report service
from app.services.batch_competitor_report import (
    generate_competitor_report_markdown_for_batch,
)
from app.services.content_generator import (
    ARTICLE_ANGLES,
    generate_article_html_for_angle,
    generate_dynamic_angle_labels_for_product,  # NEW
)

load_dotenv()

app = FastAPI(title="LLM Visibility API", version="0.7.0")

# ------------------------------------------------------------------
# Tenant schema ensure cache (process-wide)
# ------------------------------------------------------------------
_ENSURED_SCHEMAS: set[str] = set()
_ENSURE_LOCK = threading.Lock()

# ✅ NEW: safe identifier quoting helper (schema name)
def _quote_ident(name: str) -> str:
    # basic safety check – only allow letters, digits, underscore
    if not re.fullmatch(r"[a-zA-Z0-9_]+", name or ""):
        raise HTTPException(status_code=400, detail="Invalid tenant name")
    return f'"{name}"'

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

#  CORS setup
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

    if raw == "public":
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

    ✅ UPDATED:
    - Uses process-wide cache to avoid hitting information_schema on every request.
    """
    if schema == "public":
        # public gets its tables created at startup
        return

    # Fast path: already verified in this process
    if schema in _ENSURED_SCHEMAS:
        return

    with _ENSURE_LOCK:
        if schema in _ENSURED_SCHEMAS:
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

        # Cache as verified
        _ENSURED_SCHEMAS.add(schema)


def get_tenant_schema(request: Request) -> str:
    """
    Read tenant from header and normalise to schema name.
    If header missing, fall back to 'public'.
    """
    raw_tenant = request.headers.get(TENANT_HEADER)
    return _normalize_schema_name(raw_tenant or "")


# ✅ NEW: helper for URL paths (tenant code, not schema)
def tenant_code_for_path_from_request(request: Request) -> str:
    """
    Returns a tenant code suitable for putting in URL paths.
    - If header missing -> "public"
    - If header is "tenant_xxx" -> "xxx"
    - Else -> header value (lowercased)
    """
    tenant_raw = request.headers.get(TENANT_HEADER) or ""
    tenant_code = tenant_raw.strip().lower() or "public"
    if tenant_code.startswith("tenant_"):
        return tenant_code[len("tenant_"):]
    return tenant_code


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

        # Ensure the schema already exists (cached; no auto-creation here)
        _ensure_tenant_schema(schema)

        # ✅ UPDATED: safe schema quoting + include public fallback
        qschema = _quote_ident(schema)
        db.execute(text(f"SET search_path TO {qschema}, public"))

        # 🔹 Lightweight migration: make sure angle_label exists in this schema too
        try:
            db.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS content_articles
                    ADD COLUMN IF NOT EXISTS angle_label VARCHAR
                    """
                )
            )
        except Exception:
            # Don't break requests if migration fails for some reason
            pass

        yield db
    finally:
        db.close()


# --- Create tables on startup (public schema only) ---
@app.on_event("startup")
def on_startup():
    # Base public schema – useful for default/demo tenant
    Base.metadata.create_all(bind=engine)

    # 🔹 Ensure new angle_label column exists on public.content_articles
    from sqlalchemy import text as _text
    with engine.begin() as conn:
        conn.execute(
            _text(
                """
                ALTER TABLE IF EXISTS content_articles
                ADD COLUMN IF NOT EXISTS angle_label VARCHAR
                """
            )
        )


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
    batch_id: str  # NEW


class LLMRunBatchStartResponse(BaseModel):
    status: str = "started"
    batch_id: str  # NEW


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


# NEW: Persona prompt pack models
class GeneratePersonaPromptPackRequest(BaseModel):
    product_id: int
    persona: str
    num_prompts: int = 50
    pack_id: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None


class GeneratePersonaPromptPackResponse(BaseModel):
    pack_id: str
    num_prompts: int
    file_path: str
    download_url: str


class VisibilityReportRequest(BaseModel):
    product_id: int
    model: Optional[str] = DEFAULT_REPORT_MODEL
    pack_id: Optional[str] = None    # NEW
    batch_id: Optional[str] = None   # NEW


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


class ArticleOut(BaseModel):
    id: int
    product_id: int
    angle_key: str
    angle_label: str
    title: str
    slug: str
    meta_description: Optional[str] = None
    is_published: bool
    public_url: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class GenerateArticlesRequest(BaseModel):
    """
    Optional subset of angles to regenerate.
    If omitted, we generate all 10 canonical angles.
    """
    angles: Optional[List[str]] = None
    model: Optional[str] = DEFAULT_REPORT_MODEL


# ----- NEW: Batch competitor report models -----

class BatchCompetitorReportRequest(BaseModel):
    product_id: int
    batch_id: str
    model: Optional[str] = DEFAULT_REPORT_MODEL


class BatchCompetitorReportResponse(BaseModel):
    product_id: int
    batch_id: str
    pack_id: Optional[str] = None
    model_used: str
    report_markdown: str
    file_path: str
    download_url: str


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


# ----- Behaviour classifier for prompts -----


def classify_prompt_behaviour(text: str) -> str:
    """
    Heuristic classifier that assigns a behaviour label to a prompt
    based on its wording. Matches the buyer modes used in generation:
      - discovery
      - close_alternative
      - budget
      - occasion
      - fit_practical
      - trend_popularity
      - channel
    """
    t = text.lower()

    # Budget / price-led
    if any(kw in t for kw in [
        "under £", "under$", "cheap", "cheapest", "budget",
        "affordable", "on sale", "discount", "deal", "best price",
        "good value", "value for money",
    ]):
        return "budget"

    # Occasion / use-case
    if any(kw in t for kw in [
        "wedding", "bridal", "office", "work", "interview", "party",
        "evening", "casual", "formal", "school", "holiday", "vacation",
        "gift", "present", "birthday", "anniversary", "christmas",
        "new year", "weekend", "summer", "winter",
    ]):
        return "occasion"

    # Fit / comfort / practicality
    if any(kw in t for kw in [
        "comfortable", "comfort", "all-day", "all day",
        "wide fit", "narrow fit", "arch support", "support",
        "durable", "waterproof", "breathable", "lightweight",
        "for walking", "for running", "for standing", "for travel",
    ]):
        return "fit_practical"

    # Trend / popularity / best-rated
    if any(kw in t for kw in [
        "trending", "on trend", "fashionable", "stylish",
        "most popular", "best rated", "top rated", "top-rated",
        "best sellers", "best-selling", "viral", "popular right now",
    ]):
        return "trend_popularity"

    # Channel / where to buy
    if any(kw in t for kw in [
        "online in the uk", "online uk", "uk websites", "uk shops",
        "high street", "department stores", "stores near me",
        "near me", "local shops",
    ]):
        return "channel"

    # Close-alternative search (similar / like this)
    if any(kw in t for kw in [
        "similar", "alternatives", "like this", "like these",
        "instead of", "something like", "similar style", "similar design",
    ]):
        return "close_alternative"

    # Generic discovery (fallback)
    if any(kw in t for kw in [
        "best", "good", "recommend", "show me", "which", "what are",
        "ideas for", "options for", "options to",
    ]):
        return "discovery"

    # Default bucket
    return "discovery"


def _article_slug(product_id: int, angle_key: str) -> str:
    """
    Generate a UNIQUE slug per article, even if the same angle_key is reused.
    Example: product 12, angle 'fit_comfort' -> 'product-12-fit-comfort-1a2b3c4d'
    """
    import re as _re
    safe_angle = _re.sub(r"[^a-z0-9\-]+", "-", angle_key.lower().replace("_", "-")).strip("-")
    short = uuid.uuid4().hex[:8]
    return f"product-{product_id}-{safe_angle}-{short}"


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


# --- Background batch helper ---

def _run_llm_batch_background(
    tenant_schema: str,
    product_id: int,
    pack_id: str,
    model: Optional[str],
    batch_id: str,  # NEW
):
    """
    Background task: runs the full LLM batch for a given tenant/product/pack
    without blocking the original HTTP request.
    """
    db = SessionLocal()
    try:
        # Ensure we are operating inside the correct tenant schema
        qschema = _quote_ident(tenant_schema)
        db.execute(text(f"SET search_path TO {qschema}, public"))

        product = (
            db.query(Product)
            .filter(Product.id == product_id)
            .one_or_none()
        )
        if not product:
            return

        parsed = urlparse(product.url)
        domain = parsed.netloc
        if not domain:
            return

        # Load pack from DB first, then from file if needed
        db_pack = (
            db.query(PromptPack)
            .filter(PromptPack.pack_key == pack_id)
            .one_or_none()
        )

        pack_data = None

        if db_pack and db_pack.pack_json:
            pack_data = db_pack.pack_json
        else:
            try:
                pack_data = load_prompt_pack(pack_id)
            except FileNotFoundError:
                return  # nothing to do

            if db_pack:
                db_pack.pack_json = pack_data
            else:
                db_pack = PromptPack(
                    pack_key=pack_id,
                    name=pack_data.get("name"),
                    category=pack_data.get("category"),
                    source=pack_data.get("source") or "unknown",
                    language=pack_data.get("language", "en"),
                    product_id=product.id,
                    pack_json=pack_data,
                )
                db.add(db_pack)
                db.flush()

        prompts_list = pack_data.get("prompts", []) if isinstance(pack_data, dict) else []
        if not prompts_list:
            return

        # Ensure Prompt rows for each index
        existing_prompts = {
            p.index: p
            for p in db.query(Prompt).filter(Prompt.pack_id == db_pack.id).all()
        }

        for idx, prompt_text in enumerate(prompts_list):
            if idx not in existing_prompts:
                behaviour = classify_prompt_behaviour(prompt_text)
                prompt_row = Prompt(
                    pack_id=db_pack.id,
                    index=idx,
                    text=prompt_text,
                    behaviour=behaviour,
                )
                db.add(prompt_row)
                db.flush()
                existing_prompts[idx] = prompt_row

        # Run batch LLM check
        batch_results = run_llm_visibility_batch_check(
            prompts=prompts_list,
            domain=domain,
            model=model,
        )

        rows: List[LLMTest] = []
        for idx, prompt_text in enumerate(prompts_list):
            res = batch_results[idx]
            prompt_row = existing_prompts[idx]

            row = LLMTest(
                batch_id=batch_id,  # NEW
                product_id=product.id,
                pack_id=db_pack.id,
                prompt_id=prompt_row.id,
                model_used=res["model"],
                prompt=prompt_text,
                appeared=res["appeared"],
                matched_domain=res["matched"] or None,
                snippet=res["snippet"],
                llm_answer=res.get("answer"),  # NEW – full answer if available
            )
            rows.append(row)

        if rows:
            db.add_all(rows)
            db.commit()

    finally:
        db.close()


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

    # Detect relevant JSON-LD (Product, Service, Offer, etc.)
    RELEVANT_SCHEMA_TYPES = {
        "Product",
        "Service",
        "Offer",
        "AggregateOffer",
        "Review",
        "AggregateRating",
        "Organization",
        "LocalBusiness",
        "Place",
        "Article",
        "BlogPosting",
        "WebPage",
        "FAQPage",
        "HowTo",
        "ItemList",
        "Event",
        "Course",
        "SoftwareApplication",
    }

    def contains_relevant_schema(d: Any) -> bool:
        """
        Recursively check if the JSON-LD structure contains any of the
        relevant schema.org types (Product, Service, Offer, etc.).
        """
        if isinstance(d, dict):
            t = d.get("@type")
            if isinstance(t, str):
                if t in RELEVANT_SCHEMA_TYPES:
                    return True
            elif isinstance(t, list):
                if any(isinstance(x, str) and x in RELEVANT_SCHEMA_TYPES for x in t):
                    return True

            # Look into common nested containers
            for key in (
                "@graph",
                "itemListElement",
                "mainEntity",
                "about",
                "hasOfferCatalog",
            ):
                if key in d and contains_relevant_schema(d[key]):
                    return True

        elif isinstance(d, list):
            return any(contains_relevant_schema(x) for x in d)

        return False

    has_product_jsonld = False  # name kept for DB compatibility
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue

        if contains_relevant_schema(data):
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

    product_id = payload.product_id

    row = LLMTest(
        product_id=product_id,
        model_used=result["model"],
        prompt=payload.prompt,
        appeared=result["appeared"],
        matched_domain=result["matched"] or None,
        snippet=result["snippet"],
        llm_answer=result.get("answer"),  # NEW – full answer if available
    )
    db.add(row)
    db.commit()

    return LLMCheckResult(
        product_id=product_id,
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
    """
    Synchronous version of batch run. For large packs, prefer /run-llm-batch-async
    from the frontend to avoid browser timeouts.
    """
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    parsed = urlparse(product.url)
    domain = parsed.netloc
    if not domain:
        raise HTTPException(status_code=400, detail="Invalid product URL domain")

    # --- Load pack from DB first, fall back to JSON file for legacy packs ---
    db_pack = (
        db.query(PromptPack)
        .filter(PromptPack.pack_key == payload.pack_id)
        .one_or_none()
    )

    pack_data = None

    if db_pack and db_pack.pack_json:
        pack_data = db_pack.pack_json
    else:
        # Legacy behaviour – try file
        try:
            pack_data = load_prompt_pack(payload.pack_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Prompt pack not found. If this is an old pack, please regenerate it.",
            )

        # If we had a db_pack but no JSON, backfill it
        if db_pack:
            db_pack.pack_json = pack_data
        else:
            db_pack = PromptPack(
                pack_key=payload.pack_id,
                name=pack_data.get("name"),
                category=pack_data.get("category"),
                source=pack_data.get("source") or "unknown",
                language=pack_data.get("language", "en"),
                product_id=product.id,
                pack_json=pack_data,
            )
            db.add(db_pack)
            db.flush()

    prompts_list = pack_data.get("prompts", []) if isinstance(pack_data, dict) else []
    if not prompts_list:
        raise HTTPException(status_code=400, detail="Prompt pack has no prompts")

    total = len(prompts_list)

    # Load existing Prompt rows for this pack
    existing_prompts = {
        p.index: p for p in db.query(Prompt).filter(Prompt.pack_id == db_pack.id).all()
    }

    # Ensure Prompt row exists for each index
    for idx, prompt_text in enumerate(prompts_list):
        if idx not in existing_prompts:
            behaviour = classify_prompt_behaviour(prompt_text)
            prompt_row = Prompt(
                pack_id=db_pack.id,
                index=idx,
                text=prompt_text,
                behaviour=behaviour,
            )
            db.add(prompt_row)
            db.flush()
            existing_prompts[idx] = prompt_row

    # NEW: create a batch_id for this run
    batch_id = uuid.uuid4().hex

    # Batch LLM call
    try:
        batch_results = run_llm_visibility_batch_check(
            prompts=prompts_list,
            domain=domain,
            model=payload.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running LLM batch: {e}")

    appeared_count = 0
    rows: List[LLMTest] = []

    # batch_results[i] corresponds to prompts_list[i]
    for idx, prompt_text in enumerate(prompts_list):
        res = batch_results[idx]
        if res["appeared"]:
            appeared_count += 1

        prompt_row = existing_prompts[idx]

        row = LLMTest(
            batch_id=batch_id,  # NEW
            product_id=product.id,
            pack_id=db_pack.id,
            prompt_id=prompt_row.id,
            model_used=res["model"],
            prompt=prompt_text,
            appeared=res["appeared"],
            matched_domain=res["matched"] or None,
            snippet=res["snippet"],
            llm_answer=res.get("answer"),  # NEW – full answer if available
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
        batch_id=batch_id,  # NEW
    )


@app.post("/run-llm-batch-async", response_model=LLMRunBatchStartResponse)
def run_llm_batch_async(
    payload: LLMRunBatchRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fire-and-forget variant: starts the batch run in the background and
    returns immediately. Frontend should poll /prompt-stats/{pack_id}
    to see results instead of waiting for this endpoint to finish work.
    """
    product = db.query(Product).filter(Product.id == payload.product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Validate pack exists and has prompts (DB first, then file)
    db_pack = (
        db.query(PromptPack)
        .filter(PromptPack.pack_key == payload.pack_id)
        .one_or_none()
    )

    pack_data = None

    if db_pack and db_pack.pack_json:
        pack_data = db_pack.pack_json
    else:
        try:
            pack_data = load_prompt_pack(payload.pack_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Prompt pack not found. If this is an old pack, please regenerate it.",
            )

        if db_pack:
            db_pack.pack_json = pack_data
        else:
            db_pack = PromptPack(
                pack_key=payload.pack_id,
                name=pack_data.get("name"),
                category=pack_data.get("category"),
                source=pack_data.get("source") or "unknown",
                language=pack_data.get("language", "en"),
                product_id=product.id,
                pack_json=pack_data,
            )
            db.add(db_pack)
            db.flush()

    prompts_list = pack_data.get("prompts", []) if isinstance(pack_data, dict) else []
    if not prompts_list:
        raise HTTPException(status_code=400, detail="Prompt pack has no prompts")

    # Determine the tenant schema for this request
    tenant_schema = get_tenant_schema(request)

    # NEW: create a batch_id here and pass to background task
    batch_id = uuid.uuid4().hex

    # Schedule background job
    background_tasks.add_task(
        _run_llm_batch_background,
        tenant_schema=tenant_schema,
        product_id=payload.product_id,
        pack_id=payload.pack_id,
        model=payload.model,
        batch_id=batch_id,  # NEW
    )

    return LLMRunBatchStartResponse(status="started", batch_id=batch_id)


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
    tenant_code_for_path = tenant_code_for_path_from_request(request)
    download_url = f"{base_url}/download/{tenant_code_for_path}/prompt-pack/{pack['id']}"

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
            pack_json=pack,
        )
        db.add(db_pack)
        db.flush()
    else:
        # If re-generating, update metadata + JSON
        db_pack.name = pack.get("name")
        db_pack.category = pack.get("category")
        db_pack.source = pack.get("source") or db_pack.source
        db_pack.language = pack.get("language", "en")
        db_pack.product_id = product.id
        db_pack.pack_json = pack

    # Clear existing prompts for this pack (if re-generating)
    db.query(Prompt).filter(Prompt.pack_id == db_pack.id).delete()

    prompts = pack.get("prompts", [])
    for idx, text in enumerate(prompts):
        behaviour = classify_prompt_behaviour(text)
        db.add(
            Prompt(
                pack_id=db_pack.id,
                index=idx,
                text=text,
                behaviour=behaviour,
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


# NEW: Persona prompt pack endpoint
@app.post("/generate-persona-prompt-pack", response_model=GeneratePersonaPromptPackResponse)
def generate_persona_prompt_pack(
    payload: GeneratePersonaPromptPackRequest,
    db: Session = Depends(get_db),
    request: Request = None,
    current_user: User = Depends(get_current_user),
):
    """
    Source C: Persona-based high-intent prompts.

    - Requires a free-text persona description (e.g. "busy parent in London on a tight budget").
    - Prompts are generated in the voice and constraints of this persona.
    """
    product = (
        db.query(Product)
        .filter(Product.id == payload.product_id)
        .one_or_none()
    )
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    persona = (payload.persona or "").strip()
    if not persona:
        raise HTTPException(status_code=400, detail="Persona description is required")

    inferred_category = payload.category or None

    pack = generate_persona_prompt_pack_for_product(
        product_id=product.id,
        product_title=product.title or product.url,
        product_url=product.url,
        persona=persona,
        category=inferred_category,
        num_prompts=payload.num_prompts,
        pack_id=payload.pack_id,
        name=payload.name,
    )

    # Save JSON file
    file_path = save_prompt_pack_to_file(pack)
    base_url = str(request.base_url).rstrip("/")
    tenant_code_for_path = tenant_code_for_path_from_request(request)
    download_url = f"{base_url}/download/{tenant_code_for_path}/prompt-pack/{pack['id']}"

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
            source=pack.get("source") or "persona_high_intent",
            language=pack.get("language", "en"),
            product_id=product.id,
            pack_json=pack,
        )
        db.add(db_pack)
        db.flush()
    else:
        # If re-generating, update metadata + JSON
        db_pack.name = pack.get("name")
        db_pack.category = pack.get("category")
        db_pack.source = pack.get("source") or db_pack.source
        db_pack.language = pack.get("language", "en")
        db_pack.product_id = product.id
        db_pack.pack_json = pack

    # Clear existing prompts for this pack (if re-generating)
    db.query(Prompt).filter(Prompt.pack_id == db_pack.id).delete()

    prompts = pack.get("prompts", [])
    for idx, text in enumerate(prompts):
        behaviour = classify_prompt_behaviour(text)
        db.add(
            Prompt(
                pack_id=db_pack.id,
                index=idx,
                text=text,
                behaviour=behaviour,
            )
        )

    db.commit()
    # --- End sync ---

    return GeneratePersonaPromptPackResponse(
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
    tenant_code_for_path = tenant_code_for_path_from_request(request)
    download_url = f"{base_url}/download/{tenant_code_for_path}/prompt-pack/{pack['id']}"

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
            pack_json=pack,  # NEW – store full JSON
        )
        db.add(db_pack)
        db.flush()
    else:
        # If re-generating, update metadata + JSON
        db_pack.name = pack.get("name")
        db_pack.category = pack.get("category")
        db_pack.source = pack.get("source") or db_pack.source
        db_pack.language = pack.get("language", "en")
        db_pack.product_id = product.id
        db_pack.pack_json = pack

    # Clear existing prompts for this pack (if re-generating)
    db.query(Prompt).filter(Prompt.pack_id == db_pack.id).delete()

    prompts = pack.get("prompts", [])
    for idx, text in enumerate(prompts):
        behaviour = classify_prompt_behaviour(text)
        db.add(
            Prompt(
                pack_id=db_pack.id,
                index=idx,
                text=text,
                behaviour=behaviour,
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

    # Build base query for tests
    tests_query = db.query(LLMTest).filter(LLMTest.product_id == product.id)

    # Optional: restrict to a specific pack (pack_key)
    if payload.pack_id:
        db_pack = (
            db.query(PromptPack)
            .filter(PromptPack.pack_key == payload.pack_id)
            .one_or_none()
        )
        if not db_pack:
            raise HTTPException(status_code=404, detail="Prompt pack not found for report")
        tests_query = tests_query.filter(LLMTest.pack_id == db_pack.id)

    # Optional: restrict to a specific batch
    if payload.batch_id:
        tests_query = tests_query.filter(LLMTest.batch_id == payload.batch_id)

    tests = tests_query.all()

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
        "pack_id": payload.pack_id,
        "batch_id": payload.batch_id,
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


# ----- NEW: Batch competitor analysis endpoint -----

@app.post("/batch-competitor-report", response_model=BatchCompetitorReportResponse)
async def batch_competitor_report(
    payload: BatchCompetitorReportRequest,
    db: Session = Depends(get_db),
    request: Request = None,
    current_user: User = Depends(get_current_user),
):
    """
    Generate a fully LLM-written competitor analysis report for a specific batch.

    Uses:
    - all LLMTest rows for (product_id, batch_id)
    - full answers (llm_answer) where available, falling back to snippet
    """
    product = (
        db.query(Product)
        .filter(Product.id == payload.product_id)
        .one_or_none()
    )
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    tests = (
        db.query(LLMTest)
        .filter(
            LLMTest.product_id == product.id,
            LLMTest.batch_id == payload.batch_id,
        )
        .all()
    )

    if not tests:
        raise HTTPException(
            status_code=404,
            detail="No LLM tests found for this batch and product.",
        )

    # Use pack from first row (all rows in a batch share same pack_id)
    pack = None
    if tests[0].pack_id is not None:
        pack = (
            db.query(PromptPack)
            .filter(PromptPack.id == tests[0].pack_id)
            .one_or_none()
        )

    model_to_use = payload.model or DEFAULT_REPORT_MODEL

    try:
        report_md = generate_competitor_report_markdown_for_batch(
            product=product,
            pack=pack,
            tests=tests,
            model=model_to_use,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating competitor report: {e}",
        )

    # Save markdown to file, reuse existing report mechanism
    try:
        file_path = save_report_markdown_to_file(product.id, report_md)
        base_url = str(request.base_url).rstrip("/")
        filename = os.path.basename(file_path)
        download_url = f"{base_url}/download/report/{filename}"
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving competitor report: {e}",
        )

    pack_key = None
    if pack:
        pack_key = pack.pack_key

    return BatchCompetitorReportResponse(
        product_id=product.id,
        batch_id=payload.batch_id,
        pack_id=pack_key,
        model_used=model_to_use,
        report_markdown=report_md,
        file_path=file_path,
        download_url=download_url,
    )


# ✅ NEW: Tenant-aware prompt pack download (no headers required)
@app.get("/download/{tenant_code}/prompt-pack/{pack_id}")
def download_prompt_pack_for_tenant(tenant_code: str, pack_id: str):
    """
    Tenant-aware download URL that works with plain <a href> clicks.
    """
    if ".." in pack_id or "/" in pack_id or "\\" in pack_id:
        raise HTTPException(status_code=400, detail="Invalid pack_id")

    # 1) Try file first (legacy)
    fname = f"{pack_id}.json"
    fpath = os.path.join(PROMPT_PACKS_DIR, fname)
    if os.path.isfile(fpath):
        return FileResponse(
            path=fpath,
            media_type="application/json",
            filename=fname,
        )

    # 2) DB fallback in correct tenant schema
    schema = _normalize_schema_name(tenant_code)  # "test_client" -> "tenant_test_client"
    _ensure_tenant_schema(schema)

    db = SessionLocal()
    try:
        qschema = _quote_ident(schema)
        db.execute(text(f"SET search_path TO {qschema}, public"))

        pack = (
            db.query(PromptPack)
            .filter(PromptPack.pack_key == pack_id)
            .one_or_none()
        )
        if not pack or not pack.pack_json:
            raise HTTPException(status_code=404, detail="Prompt pack not found")

        return JSONResponse(content=pack.pack_json)
    finally:
        db.close()


@app.get("/download/prompt-pack/{pack_id}")
def download_prompt_pack(pack_id: str, request: Request):
    """
    Download a prompt pack JSON file by pack_id.

    Behaviour:
    - Try to serve the JSON file from disk (legacy behaviour).
    - If it's missing, fall back to the DB-stored pack_json (tenant-aware).
    """
    if ".." in pack_id or "/" in pack_id or "\\" in pack_id:
        raise HTTPException(status_code=400, detail="Invalid pack_id")

    # 1) Try file first (legacy)
    fname = f"{pack_id}.json"
    fpath = os.path.join(PROMPT_PACKS_DIR, fname)
    if os.path.isfile(fpath):
        return FileResponse(
            path=fpath,
            media_type="application/json",
            filename=fname,
        )

    # 2) Tenant-aware DB fallback (requires header)
    tenant_schema = get_tenant_schema(request)
    _ensure_tenant_schema(tenant_schema)

    db = SessionLocal()
    try:
        qschema = _quote_ident(tenant_schema)
        db.execute(text(f"SET search_path TO {qschema}, public"))

        pack = (
            db.query(PromptPack)
            .filter(PromptPack.pack_key == pack_id)
            .one_or_none()
        )
        if not pack or not pack.pack_json:
            raise HTTPException(status_code=404, detail="Prompt pack not found")

        return JSONResponse(content=pack.pack_json)
    finally:
        db.close()


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


@app.get("/public/{tenant_code}/articles/{slug}", response_class=HTMLResponse)
def get_public_article(
    tenant_code: str,
    slug: str,
):
    """
    Public HTML endpoint for a content article.

    This is intended for:
    - human readers (can be linked from your frontend)
    - search engines and LLMs to crawl

    Tenant is encoded in the path, so we can route to the correct schema
    without relying on the X-Tenant header.
    """
    # Normalise + validate tenant -> schema
    schema = _normalize_schema_name(tenant_code)
    _ensure_tenant_schema(schema)

    db = SessionLocal()
    try:
        qschema = _quote_ident(schema)
        db.execute(text(f"SET search_path TO {qschema}, public"))

        article = (
            db.query(ContentArticle)
            .filter(
                ContentArticle.slug == slug,
                ContentArticle.is_published == True,
            )
            .one_or_none()
        )
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        # Basic HTML wrapper around the stored fragment
        # (content_html already contains <article>...</article>)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{article.title}</title>
  <meta name="description" content="{article.meta_description or ''}">
</head>
<body>
  {article.content_html}
</body>
</html>
"""
        return HTMLResponse(content=html, status_code=200)
    finally:
        db.close()


@app.get("/products/{product_id}/articles", response_model=List[ArticleOut])
def list_product_articles(
    product_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    product = db.query(Product).filter(Product.id == product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    articles = (
        db.query(ContentArticle)
        .filter(ContentArticle.product_id == product.id)
        .order_by(ContentArticle.angle_key)
        .all()
    )

    # Figure out which tenant code to use in the public URL path
    tenant_raw = request.headers.get(TENANT_HEADER) or ""
    tenant_code = (tenant_raw.strip().lower() or "public")

    # If someone ever sends "tenant_xxx" as header, strip the prefix for the URL path
    if tenant_code.startswith("tenant_"):
        tenant_code_for_path = tenant_code[len("tenant_") :]
    else:
        tenant_code_for_path = tenant_code

    base_url = str(request.base_url).rstrip("/")

    results: List[ArticleOut] = []
    for a in articles:
        # Prefer the stored dynamic angle_label, fallback to canonical / key
        angle_label = (
            a.angle_label
            or ARTICLE_ANGLES.get(a.angle_key, a.angle_key)
        )

        public_url = (
            f"{base_url}/public/{tenant_code_for_path}/articles/{a.slug}"
        )

        results.append(
            ArticleOut(
                id=a.id,
                product_id=a.product_id,
                angle_key=a.angle_key,
                angle_label=angle_label,
                title=a.title,
                slug=a.slug,
                meta_description=a.meta_description,
                is_published=a.is_published,
                public_url=public_url,
                created_at=a.created_at,
                updated_at=a.updated_at,
            )
        )
    return results


@app.post("/products/{product_id}/articles/generate", response_model=List[ArticleOut])
async def generate_product_articles(
    product_id: int,
    payload: GenerateArticlesRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    product = db.query(Product).filter(Product.id == product_id).one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    parsed = urlparse(product.url)
    domain = parsed.netloc

    # Determine which canonical angles to generate
    if payload.angles:
        angle_keys: List[str] = []
        for key in payload.angles:
            if key not in ARTICLE_ANGLES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown angle key: {key}",
                )
            angle_keys.append(key)
    else:
        angle_keys = list(ARTICLE_ANGLES.keys())

    # Try to get a page snapshot to ground the content
    page_snapshot = None
    try:
        html = await fetch_page_html_via_scraperapi(product.url)
        page_snapshot = build_page_snapshot(html)
    except Exception:
        # If snapshot fails, we still generate more generic content
        page_snapshot = None

    # 🔹 Ask LLM for product-specific labels for all 10 canonical angles
    # (or at least the ones we'll use; we still call once and then subset)
    dynamic_labels = generate_dynamic_angle_labels_for_product(
        product_title=product.title or product.url,
        product_url=product.url,
        domain=domain,
        page_snapshot=page_snapshot,
        model=payload.model,
    )

    generated_articles: List[ArticleOut] = []

    # 🔹 Figure out tenant code + base URL to build public_url
    tenant_raw = request.headers.get(TENANT_HEADER) or ""
    tenant_code = tenant_raw.strip().lower() or "public"

    # If header was "tenant_xxx", strip prefix for the URL path
    if tenant_code.startswith("tenant_"):
        tenant_code_for_path = tenant_code[len("tenant_") :]
    else:
        tenant_code_for_path = tenant_code

    base_url = str(request.base_url).rstrip("/")

    for angle_key in angle_keys:
        # Use dynamic per-product label if available; fall back to canonical
        angle_label = dynamic_labels.get(angle_key) or ARTICLE_ANGLES[angle_key]

        article_data = generate_article_html_for_angle(
            product_title=product.title or product.url,
            product_url=product.url,
            domain=domain,
            angle_key=angle_key,
            angle_label=angle_label,
            page_snapshot=page_snapshot,
            model=payload.model,
        )

        # 🔹 ALWAYS create a NEW article row – do NOT overwrite existing ones
        slug = _article_slug(product.id, angle_key)

        article = ContentArticle(
            product_id=product.id,
            angle_key=angle_key,
            angle_label=angle_label,
            slug=slug,
            title=article_data["title"],
            meta_description=article_data.get("meta_description") or None,
            content_html=article_data["content_html"],
            is_published=True,
        )
        db.add(article)
        db.flush()

        # 🔹 Build the public URL exactly like the GET endpoint
        public_url = (
            f"{base_url}/public/{tenant_code_for_path}/articles/{article.slug}"
        )

        generated_articles.append(
            ArticleOut(
                id=article.id,
                product_id=article.product_id,
                angle_key=article.angle_key,
                angle_label=angle_label,
                title=article.title,
                slug=article.slug,
                meta_description=article.meta_description,
                is_published=article.is_published,
                public_url=public_url,  # ✅ REQUIRED FIELD
                created_at=article.created_at,
                updated_at=article.updated_at,
            )
        )

    db.commit()
    return generated_articles


@app.get("/prompt-stats/{pack_id}", response_model=List[PromptPerformance])
def prompt_stats(
    pack_id: str,
    batch_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_pack = (
        db.query(PromptPack)
        .filter(PromptPack.pack_key == pack_id)
        .one_or_none()
    )
    if not db_pack:
        raise HTTPException(status_code=404, detail="Prompt pack not found in DB")

    # Build an OUTER JOIN so prompts with zero tests still return
    join_cond = (LLMTest.prompt_id == Prompt.id)
    if batch_id:
        join_cond = and_(join_cond, LLMTest.batch_id == batch_id)

    rows = (
        db.query(
            Prompt.id.label("prompt_id"),
            Prompt.index.label("index"),
            Prompt.text.label("text"),
            func.count(LLMTest.id).label("total_runs"),
            func.coalesce(
                func.sum(
                    case((LLMTest.appeared == True, 1), else_=0)
                ),
                0,
            ).label("appeared_count"),
        )
        .outerjoin(LLMTest, join_cond)
        .filter(Prompt.pack_id == db_pack.id)
        .group_by(Prompt.id, Prompt.index, Prompt.text)
        .order_by(Prompt.index)
        .all()
    )

    results: List[PromptPerformance] = []
    for r in rows:
        total_runs = int(r.total_runs or 0)
        appeared_count = int(r.appeared_count or 0)
        visibility_score = (appeared_count / total_runs) if total_runs > 0 else 0.0

        results.append(
            PromptPerformance(
                prompt_id=r.prompt_id,
                index=r.index,
                text=r.text,
                total_runs=total_runs,
                appeared_count=appeared_count,
                visibility_score=visibility_score,
            )
        )

    return results