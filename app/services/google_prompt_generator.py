import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from app.services.google_queries import (
    collect_google_queries_for_product,
)
from app.services.prompt_packs import PROMPT_PACKS_DIR
from app.config import DEFAULT_GOOGLE_PACK_MODEL  # centralised model

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _ensure_api_key():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")


def _slugify(text: str, max_length: int = 40) -> str:
    text = text.lower()
    slug = []
    for ch in text:
        if ch.isalnum():
            slug.append(ch)
        elif ch in " _-":
            slug.append("-")
    s = "".join(slug).strip("-")
    if len(s) > max_length:
        s = s[:max_length].rstrip("-")
    return s or "pack"


def _generate_prompts_from_google_queries(
    product_title: str,
    product_url: str,
    category: Optional[str],
    num_prompts: int,
    google_signals: List[str],
) -> List[str]:
    """
    Uses OpenAI to turn Google-based signals (PAA, Related Searches,
    Titles & Snippets) into high-intent shopping prompts.
    """
    _ensure_api_key()
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Limit how many raw signals we send in to keep token usage reasonable
    google_signals = google_signals[:80]

    system_msg = (
        "You are an expert e-commerce growth marketer and search strategist. "
        "You specialise in transforming real Google search behaviour into "
        "high-intent shopping prompts that a user would ask an AI assistant when "
        "they are actively looking to BUY a product, not just research it."
    )

    signals_json = json.dumps(google_signals, ensure_ascii=False, indent=2)

    user_msg = f"""
You are given REAL Google-derived signals related to a product, combined from:

- Source A – People Also Ask (PAA) questions
- Source B – Related searches
- Source C – Organic result titles & snippets

Product context:

- Product title: {product_title}
- Product page URL: {product_url}
- Product/category: {category or "unknown"}

Here are the combined Google signals as a JSON array of strings:

{signals_json}

Your tasks:

1) Filter out:
   - purely informational / educational content,
   - generic "what is" / "why" content that is not clearly about buying,
   - things that don't reflect commercial / transactional intent.

2) From the remaining items, infer and rewrite them into
   NATURAL-LANGUAGE SHOPPING PROMPTS that a user would ask an AI assistant
   (like ChatGPT) when they are actively looking to discover and BUY products
   like this.

   - Include verbs like "buy", "find", "show me options", "recommend",
     "best X to buy", "under £X", etc.
   - Aim to cover behaviours such as:
        • product discovery in buy-mode
        • specific product / close alternatives
        • price- and budget-sensitive purchasing
        • occasion / use-case driven buying
        • fit / comfort / practicality before buying
        • trend / popularity driven buying
        • brand / store preference
   - Use UK context where appropriate (e.g. currency £, UK retailers).

3) Generate up to {num_prompts} prompts.
   - They must ALL be high purchase intent.
   - They must ALL sound like real user queries.

4) Output:
   Return STRICTLY valid JSON in this exact format:

   {{
     "prompts": [
       "prompt 1",
       "prompt 2",
       "prompt 3"
     ]
   }}

   - No markdown.
   - No comments.
   - No extra keys.
"""

    completion = client.chat.completions.create(
        model=DEFAULT_GOOGLE_PACK_MODEL,  # now comes from config/env
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.5,
    )

    content = completion.choices[0].message.content or ""
    data = json.loads(content)
    prompts = data.get("prompts", [])

    cleaned: List[str] = []
    for p in prompts:
        if isinstance(p, str):
            s = p.strip()
            if s:
                cleaned.append(s)

    if not cleaned:
        raise RuntimeError("LLM returned no prompts from Google signals")

    return cleaned[:num_prompts]


def generate_google_prompt_pack_for_product(
    product_id: int,
    product_title: str,
    product_url: str,
    category: Optional[str],
    num_prompts: int = 50,
    pack_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a prompt pack based on real Google signals (Sources A, B, C).
    Uses versioned pack IDs so each generation is a new pack.
    """
    # 1) Collect Google signals via Scrapingdog (PAA + related + titles/snippets)
    google_signals = collect_google_queries_for_product(
        product_title=product_title,
        category=category,
        max_questions=80,
    )

    if not google_signals:
        raise RuntimeError("Could not obtain any Google-based signals for this product/category")

    # 2) Turn them into high-intent prompts
    prompts = _generate_prompts_from_google_queries(
        product_title=product_title,
        product_url=product_url,
        category=category,
        num_prompts=num_prompts,
        google_signals=google_signals,
    )

    # 3) Pack metadata (versioned ID)
    if pack_id:
        base_id = pack_id
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        base_id = f"google_auto_{product_id}_{_slugify(product_title)}_{timestamp}"

    pack_name = name or f"Google Seeded Pack for {product_title[:60]}"

    pack: Dict[str, Any] = {
        "id": base_id,
        "name": pack_name,
        "category": category or "google_seeded_high_intent",
        "language": "en",
        "source": "google_people_also_ask_related_titles_snippets",
        "prompts": prompts,
    }
    return pack