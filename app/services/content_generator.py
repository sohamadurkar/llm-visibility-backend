# app/services/content_generator.py

import os
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from app.config import DEFAULT_REPORT_MODEL  # reuse your report model

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _ensure_api_key():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")


# Canonical angle keys + human labels
ARTICLE_ANGLES: Dict[str, str] = {
    "fit_comfort": "Fit & Comfort",
    "occasion": "Occasion-specific",
    "budget_value": "Budget & Value",
    "trend_style": "Trend & Style",
    "alternatives": "Alternatives & Similar Products",
    "use_case": "Use-case & Scenario",
    "materials_durability": "Materials & Durability",
    "buying_guide": "Buying Guide",
    "faq": "FAQ",
    "care_maintenance": "Care & Maintenance",
}


def generate_article_html_for_angle(
    *,
    product_title: str,
    product_url: str,
    domain: str,
    angle_key: str,
    angle_label: str,
    page_snapshot: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate a structured HTML article for a specific product + angle.

    Returns:
      {
        "title": ...,
        "meta_description": ...,
        "content_html": "<article>...</article>"
      }
    """
    _ensure_api_key()
    client = OpenAI(api_key=OPENAI_API_KEY)

    model_to_use = model or DEFAULT_REPORT_MODEL

    snapshot_json = ""
    if page_snapshot:
        # Keep snapshot bounded so we don't blow token limits
        raw = json.dumps(page_snapshot, ensure_ascii=False)
        snapshot_json = raw[:7000]

    system_msg = (
        "You are an expert ecommerce SEO and content strategist. "
        "You write long-form, well-structured HTML articles that help search engines "
        "and large language models (LLMs) understand and recommend specific product pages. "
        "You always write in clear, plain English and keep the structure scannable."
    )

    user_msg = f"""
Write a detailed HTML article for a specific product page.

Product:
- Title: {product_title}
- URL: {product_url}
- Domain: {domain}

Angle to focus on: {angle_label}

If page snapshot data is provided, use it to stay grounded in the actual content:

Page snapshot (may be truncated JSON):
{snapshot_json or "(no snapshot available)"}

GOALS:
- Help human readers understand this product from the angle: "{angle_label}".
- Help search engines and LLMs understand when this product is a good recommendation.
- Naturally mention the product and link to its URL at least once.

CONTENT REQUIREMENTS:
1) Output MUST be valid HTML fragment, no <html> or <head> tags.
   Wrap the main content in a single <article> element.

2) Inside <article>, include:
   - One <h1> main heading that clearly reflects both the product and the angle.
   - Several <h2> and <h3> subsections with descriptive headings.
   - Short paragraphs and bullet lists where helpful.
   - At least one <a> link to the product URL: {product_url}.

3) Style:
   - Neutral, helpful, and factual tone.
   - Make it skimmable with headings and bullet points.
   - Do NOT invent very specific technical claims that are not implied by the snapshot.
   - If some details are unknown, speak in generic terms ("for many buyers", "often", etc.).

4) SEO & LLM considerations:
   - Explain who this product is best for, in the context of the angle.
   - Mention relevant scenarios, fit, use-cases, and decision criteria.
   - Use natural language, not keyword stuffing.

5) Also provide, at the very top as an HTML comment:
   - A suggested page title (<title> equivalent)
   - A one-sentence meta description

   Format exactly like this:

   <!--
   TITLE: ...
   META: ...
   -->

Return ONLY the HTML fragment (the comment + <article>...</article>), no markdown, no extra explanation.
"""

    completion = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.45,
    )

    html = completion.choices[0].message.content or ""

    # Very simple extraction of TITLE / META from the top comment, if present
    title = product_title
    meta = ""
    if html.startswith("<!--"):
        end_idx = html.find("-->")
        if end_idx != -1:
            header_block = html[4:end_idx].strip()
            # naive parsing
            for line in header_block.splitlines():
                line = line.strip()
                if line.upper().startswith("TITLE:"):
                    title = line.split(":", 1)[1].strip() or product_title
                elif line.upper().startswith("META:"):
                    meta = line.split(":", 1)[1].strip() or ""

            # strip the comment from the content_html
            html = html[end_idx + 3 :].lstrip()

    return {
        "title": title,
        "meta_description": meta,
        "content_html": html.strip(),
    }