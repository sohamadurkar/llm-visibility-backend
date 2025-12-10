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


# Canonical angle keys + default/fallback human labels.
# Keys are aligned with the frontend ContentAngleKey union.
ARTICLE_ANGLES: Dict[str, str] = {
    "fit_comfort": "Fit & Comfort",
    "occasion_specific": "Occasion-specific",
    "budget_value": "Budget & Value",
    "trend_style": "Trend & Style",
    "alternatives_similar": "Alternatives & Similar Products",
    "use_case": "Use-case & Scenario",
    "materials_durability": "Materials & Durability",
    "buying_guide": "Buying Guide",
    "faq": "FAQ",
    "care_maintenance": "Care & Maintenance",
}


def generate_dynamic_angle_labels_for_product(
    *,
    product_title: str,
    product_url: str,
    domain: str,
    page_snapshot: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, str]:
    """
    Ask the LLM to create product-specific angle labels for all 10 canonical
    angle keys.

    Returns a dict:
        {
          "fit_comfort": "Supportive heels for all-day weddings",
          "occasion_specific": "Wedding-guest and party outfits",
          ...
        }

    If anything goes wrong (API error / JSON parse), falls back to ARTICLE_ANGLES.
    """
    _ensure_api_key()
    client = OpenAI(api_key=OPENAI_API_KEY)

    model_to_use = model or DEFAULT_REPORT_MODEL

    snapshot_json = ""
    if page_snapshot:
        # Keep snapshot bounded so we don't blow token limits
        raw = json.dumps(page_snapshot, ensure_ascii=False)
        snapshot_json = raw[:7000]

    # Build a description of the canonical buckets so the model
    # knows what each key is meant to capture.
    canonical_description = """
We use 10 canonical angle keys to describe different buying and search intents:

- fit_comfort: comfort, fit, support, sizing, all-day wear.
- occasion_specific: specific events or contexts (weddings, office, travel, school, etc.).
- budget_value: price, affordability, value for money, deals.
- trend_style: fashion, style, trendiness, looks, aesthetic.
- alternatives_similar: alternatives, similar products, comparisons, "something like this".
- use_case: concrete scenarios and jobs-to-be-done (walking, running, standing all day, commuting, etc.).
- materials_durability: materials, build quality, durability, sustainability.
- buying_guide: how to choose, decision criteria, what to look for before buying.
- faq: frequently asked questions buyers have about this product.
- care_maintenance: cleaning, storage, care, how to keep it in good condition over time.
""".strip()

    system_msg = (
        "You are an expert ecommerce content strategist. "
        "For a given product page, you propose short, product-specific content angles "
        "that could each be the focus of a long-form article. "
        "You respond ONLY in strict JSON."
    )

    user_msg = f"""
We are working on long-form content for one specific product page.

Product:
- Title: {product_title}
- URL: {product_url}
- Domain: {domain}

{canonical_description}

TASK:
For EACH of the 10 angle keys, propose ONE short, product-specific label that:
- clearly describes how we will frame this product for that angle,
- is human-readable and specific (not generic like "Fit & Comfort"),
- is at most 80 characters,
- is suitable as a section heading in a content hub.

If page snapshot data is provided, use it to make the labels grounded in the real page:

Page snapshot (may be truncated JSON):
{snapshot_json or "(no snapshot available)"}

OUTPUT FORMAT (strict JSON, no extra text, no comments):

{{
  "fit_comfort": "…",
  "occasion_specific": "…",
  "budget_value": "…",
  "trend_style": "…",
  "alternatives_similar": "…",
  "use_case": "…",
  "materials_durability": "…",
  "buying_guide": "…",
  "faq": "…",
  "care_maintenance": "…"
}}
""".strip()

    try:
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
        )

        raw = completion.choices[0].message.content or ""
        data = json.loads(raw)

        result: Dict[str, str] = {}
        for key, default_label in ARTICLE_ANGLES.items():
            val = data.get(key, "").strip() if isinstance(data, dict) else ""
            if not isinstance(val, str) or not val:
                # Fallback to generic label
                result[key] = default_label
            else:
                # Ensure it isn't absurdly long
                if len(val) > 120:
                    val = val[:117].rstrip() + "..."
                result[key] = val

        return result

    except Exception:
        # On any error, just return the generic defaults
        return dict(ARTICLE_ANGLES)


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

    `angle_key` should be one of the canonical keys (fit_comfort, occasion_specific, ...).
    `angle_label` is the product-specific, human-facing label for this article
    (e.g. "Supportive heels for all-day weddings").

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

Angle key: {angle_key}
Angle to focus on (human label): {angle_label}

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
""".strip()

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