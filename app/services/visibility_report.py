import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup
import httpx

from app.config import DEFAULT_REPORT_MODEL  # NEW

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")

# Root dir = project root (llm-visibility)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def _ensure_keys():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    if not SCRAPER_API_KEY:
        raise RuntimeError("SCRAPER_API_KEY not set in environment")


async def fetch_page_html_via_scraperapi(url: str) -> str:
    """
    Fetches fully rendered HTML for a page using ScraperAPI.
    """
    _ensure_keys()
    api_url = f"http://api.scraperapi.com/?api_key={SCRAPER_API_KEY}&url={url}&render=true"

    async with httpx.AsyncClient(timeout=40) as client:
        resp = await client.get(api_url)
    resp.raise_for_status()
    return resp.text


def build_page_snapshot(html: str, max_body_chars: int = 2000, max_schema_chars: int = 2000) -> Dict[str, Any]:
    """
    Extracts a compact snapshot of the page for the LLM:
    - title
    - meta description
    - headings
    - main text sample
    - product/Service/Offer JSON-LD snippet (if present)
    """
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = None
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Meta description
    meta_desc = None
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        meta_desc = md["content"].strip()

    # Headings
    headings: List[Dict[str, str]] = []
    for level in ["h1", "h2", "h3"]:
        for tag in soup.find_all(level):
            text = (tag.get_text() or "").strip()
            if text:
                headings.append({"level": level, "text": text})
            if len(headings) >= 15:
                break
        if len(headings) >= 15:
            break

    # Main body text sample
    paragraphs = []
    for p in soup.find_all("p"):
        t = (p.get_text() or "").strip()
        if t:
            paragraphs.append(t)
        if sum(len(x) for x in paragraphs) > max_body_chars:
            break
    main_text = "\n\n".join(paragraphs)
    if len(main_text) > max_body_chars:
        main_text = main_text[:max_body_chars]

    # Structured data (Product/Service/etc.) JSON-LD snippet
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

    def contains_relevant_schema(d):
        if isinstance(d, dict):
            t = d.get("@type")
            if isinstance(t, str):
                if t in RELEVANT_SCHEMA_TYPES:
                    return True
            elif isinstance(t, list):
                if any(isinstance(x, str) and x in RELEVANT_SCHEMA_TYPES for x in t):
                    return True

            for key in ("@graph", "itemListElement", "mainEntity", "about", "hasOfferCatalog"):
                if key in d and contains_relevant_schema(d[key]):
                    return True

        elif isinstance(d, list):
            return any(contains_relevant_schema(x) for x in d)

        return False

    product_jsonld_snippet = None
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue

        if contains_relevant_schema(data):
            raw = json.dumps(data, ensure_ascii=False, indent=2)
            product_jsonld_snippet = raw[:max_schema_chars]
            break

    snapshot: Dict[str, Any] = {
        "title": title,
        "meta_description": meta_desc,
        "headings": headings,
        "main_text_sample": main_text,
        "product_jsonld_snippet": product_jsonld_snippet,
    }
    return snapshot


def generate_visibility_report_markdown(
    product_title: str,
    product_url: str,
    domain: str,
    has_product_jsonld: bool,
    visibility_metrics: Dict[str, Any],
    page_snapshot: Dict[str, Any],
    model: str = DEFAULT_REPORT_MODEL,  # now comes from config
) -> str:
    """
    Calls OpenAI to generate a technical visibility report in Markdown.
    """
    _ensure_keys()
    client = OpenAI(api_key=OPENAI_API_KEY)

    snapshot_json = json.dumps(page_snapshot, ensure_ascii=False)
    metrics_json = json.dumps(visibility_metrics, ensure_ascii=False)

    system_msg = (
        "You are a senior technical SEO and LLM-visibility consultant. "
        "You specialise in making specific product pages more likely to be discovered and cited "
        "by large language models (LLMs) such as ChatGPT, Claude, Gemini, and Perplexity. "
        "You only give concrete, implementation-ready recommendations tied directly to the provided page."
    )

    user_msg = f"""
Analyse the following product page and its current LLM visibility performance.

Product:
- Title: {product_title}
- URL: {product_url}
- Domain: {domain}
- Has Product JSON-LD: {has_product_jsonld}

LLM Visibility Metrics (from automated tests):
{metrics_json}

Page snapshot (parsed from the live HTML):
{snapshot_json}

TASK:
1. DO NOT give generic SEO advice.
2. DO NOT talk in vague terms like "improve SEO" or "get more backlinks" without specifics.
3. Base every recommendation on what you see in the snapshot (title, meta, headings, main_text_sample, JSON-LD).
4. Focus on changes that will make this specific URL more likely to be:
   - parsed and understood by LLMs,
   - included in LLM answers,
   - cited as a source (URL/domain appearing in responses).

OUTPUT:
Write a technical report in MARKDOWN with the following sections:

## 1. Summary
- Briefly describe the current situation and the main weaknesses for LLM visibility.

## 2. Structured Data & Schema.org
- Identify any issues or gaps in Product JSON-LD (or its absence).
- Suggest exact fields and examples to add or correct.
- If JSON-LD is missing important attributes (e.g., brand, colour, material, offers, aggregateRating, review), specify them.

## 3. On-page Content for LLMs
- Based on the headings and main_text_sample, identify missing information that LLMs would find useful when recommending this product (e.g., use cases, comparisons, sizing, fit, unique value).
- Suggest concrete new paragraphs, FAQs, or Q&A blocks that could be added (write a few example FAQs in full).

## 4. Internal Linking & Context
- Suggest how this product page should be linked from category pages, editorial content, or guides to help LLMs see it as a good candidate answer. Be specific to the domain and product type.

## 5. Implementation Checklist (Prioritised)
- Provide a numbered checklist of concrete tasks, ordered by impact (High/Medium/Low).
- Each task should be 1–2 sentences and actionable, e.g. “Add a Product JSON-LD block with brand, colour 'claret', material 'velvet', heel height, and offer price.”

Keep the tone concise but precise. Assume the reader is a technical marketer or developer.
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.35,
    )

    report = completion.choices[0].message.content or ""
    return report


def save_report_markdown_to_file(product_id: int, report_markdown: str) -> str:
    """
    Saves the markdown report to reports/product_<id>_<timestamp>.md
    and returns the absolute file path.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"product_{product_id}_{timestamp}.md"
    fpath = os.path.join(REPORTS_DIR, fname)

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    return fpath