import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime  # ✅ NEW

from dotenv import load_dotenv
from openai import OpenAI

from app.services.google_queries import (
    collect_google_queries_for_product,
)
from app.services.prompt_packs import PROMPT_PACKS_DIR

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
    google_queries: List[str],
) -> List[str]:
    """
    Uses OpenAI to turn raw Google queries into high-intent shopping prompts.
    """
    _ensure_api_key()
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Limit how many raw queries we send in to keep token usage reasonable
    google_queries = google_queries[:80]

    system_msg = (
        "You are an expert e-commerce growth marketer and search strategist. "
        "You specialise in transforming raw Google search queries into "
        "high-intent shopping prompts that a user would ask an AI assistant when "
        "they are actively looking to BUY a product, not just research it."
    )

    queries_json = json.dumps(google_queries, ensure_ascii=False, indent=2)

    user_msg = f"""
You are given REAL Google user queries (from 'people also ask') related to a product:

- Product title: {product_title}
- Product page URL: {product_url}
- Product/category: {category or "unknown"}

Here are the raw Google queries as a JSON array of strings:

{queries_json}

Your tasks:

1) Filter out:
   - purely informational / educational questions,
   - generic "what is" / "why" questions that are not clearly about buying,
   - things that don't reflect commercial / transactional intent.

2) From the remaining queries, rewrite and expand them into
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
        model="gpt-4.1-mini",
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
        raise RuntimeError("LLM returned no prompts from Google queries")

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
    Build a prompt pack based on real Google queries (Source B).

    IMPORTANT:
    - If pack_id is not provided, we generate a UNIQUE pack ID each time
      using product_id + slug + timestamp. This means every generation is
      a new versioned pack linked to the same product.
    """
    # 1) Collect real queries from Google via Scrapingdog
    google_queries = collect_google_queries_for_product(
        product_title=product_title,
        category=category,
        max_questions=80,
    )

    if not google_queries:
        raise RuntimeError("Could not obtain any Google-based queries for this product/category")

    # 2) Turn them into high-intent prompts
    prompts = _generate_prompts_from_google_queries(
        product_title=product_title,
        product_url=product_url,
        category=category,
        num_prompts=num_prompts,
        google_queries=google_queries,
    )

    # 3) Pack metadata
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
        "source": "google_people_also_ask",
        "prompts": prompts,
    }
    return pack