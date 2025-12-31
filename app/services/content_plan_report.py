# app/services/content_plan_report.py

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

from app.config import DEFAULT_REPORT_MODEL

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _ensure_api_key():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")


def generate_content_plan_markdown(
    product_title: str,
    product_url: str,
    domain: str,
    has_product_jsonld: bool,
    page_snapshot: Dict[str, Any],
    model: str = DEFAULT_REPORT_MODEL,
) -> str:
    """
    Generate a detailed CONTENT PLAN (not the article itself)
    for ONE very strong long-form article about the product.

    Output: Markdown with clear sections: goal, audience, outline,
    key talking points, FAQs, internal links, etc.
    """
    _ensure_api_key()
    client = OpenAI(api_key=OPENAI_API_KEY)

    snapshot_json = json.dumps(page_snapshot, ensure_ascii=False)

    system_msg = (
        "You are a senior content strategist and SEO/LLM visibility consultant. "
        "Your job is to design action-ready content plans for long-form articles "
        "that can rank in search AND be used as strong source pages for LLMs "
        "like ChatGPT, Gemini, Claude, and Perplexity."
    )

    user_msg = f"""
Create a detailed CONTENT PLAN (not the final article text) for ONE very strong long-form article about this product.

Product context:
- Title: {product_title}
- URL: {product_url}
- Domain: {domain}
- Has Product JSON-LD: {has_product_jsonld}

Page snapshot (from the live HTML):
{snapshot_json}

GOAL:
- The article should be the single best, most comprehensive evergreen page about this product on the site.
- It must be structured so that:
  - Search engines can understand and rank it.
  - LLMs can easily extract, summarise, and cite it when answering product-related questions.

CONSTRAINTS:
- Do NOT write the article itself.
- Focus on structure, angle, talking points, and requirements.
- Keep everything grounded in what you see in the snapshot (title, meta, headings, main_text_sample, JSON-LD, etc.).
- If something is missing in the snapshot but important, explicitly call it out as a gap.

OUTPUT:
Write the plan in MARKDOWN with these sections:

## 1. Article Goal & Positioning
- 3–5 bullet points describing what this article should achieve (for users + LLMs).

## 2. Target Audience & Core Use Cases
- Who this article is for (1–2 short persona descriptions).
- The top 3–6 use cases or scenarios this article must address.

## 3. Working Title & H1 Suggestions
- 3 alternative H1 ideas (short, clear, product-specific).

## 4. Detailed Outline (H2/H3 Structure)
- Provide a full outline for a ~1,800–2,500 word article.
- Use nested bullet points to show H2/H3 structure.
- For each H2 section, add 1–2 bullets explaining the objective of that section.

## 5. Key Talking Points & Data Requirements
- Bullet list of specific facts, data points, measurements, specifications, policies, or guarantees that MUST be added or clarified on the page.
- Clearly mark which items are missing from the current snapshot and need stakeholder input.

## 6. E-E-A-T & LLM Trust Signals
- Concrete suggestions for:
  - Author / reviewer info
  - References / evidence
  - Photos, diagrams, or visuals
  - Any specific trust signals (e.g. guarantees, returns, safety info, certifications).

## 7. Internal Linking & Context
- List 5–10 suggested internal links (by type, not by exact URL), such as:
  - Category pages
  - Related products
  - Buying guides
  - FAQs / support pages
- Explain how each internal link type helps LLMs treat this article as an authoritative source.

## 8. FAQ Block (Questions Only)
- Provide 6–10 FAQ questions ONLY (no answers).
- Focus on the highest-intent, most practical questions that potential buyers and LLMs would care about.

## 9. Implementation Checklist
- Numbered checklist of concrete tasks required to implement this content plan.
- Each item should be 1–2 sentences and actionable.

Keep the tone concise and practical. Assume the reader is a content lead working with a developer and SEO/LLM specialist.
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.35,
    )

    plan = completion.choices[0].message.content or ""
    return plan