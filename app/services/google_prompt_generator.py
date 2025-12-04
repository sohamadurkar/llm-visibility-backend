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
    Titles & Snippets) into high-intent, brand-neutral, CUSTOMER-LIKE
    shopping prompts.

    Principles:
    - Prompts sound like actual buyers searching.
    - No story / lore / “that famous X” references.
    - No brand names, store names, domains, or reuse of product_title text.
    - Short, direct, 1-sentence queries.
    """
    _ensure_api_key()
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Limit how many raw signals we send in to keep token usage reasonable
    google_signals = google_signals[:80]

    system_msg = (
        "You are an expert in real-world e-commerce search behaviour. "
        "You generate short, natural-language BUYING QUERIES that real shoppers would ask an AI assistant. "
        "You NEVER write marketing copy, themed descriptions, lore, story references, or fan-tribute style language. "
        "You ONLY produce what a buyer would actually type or say when trying to find products to purchase. "
        "All prompts must be brand-neutral and must NOT include specific brand names, store names, or website domains."
    )

    signals_json = json.dumps(google_signals, ensure_ascii=False, indent=2)

    user_msg = f"""
You are given REAL Google-derived signals related to a product, combined from:

- Source A – People Also Ask (PAA) questions
- Source B – Related searches
- Source C – Organic result titles & snippets

Product context (FOR YOUR UNDERSTANDING ONLY):

- Internal product title: {product_title}
- Product page URL: {product_url}
- Product/category: {category or "unknown"}

The shopper does NOT know this title, brand, or URL.
They are just a normal buyer searching.

Here are the combined Google signals as a JSON array of strings:

{signals_json}

========================
STEP 1 – FILTER
========================
From these Google signals, mentally filter out:

- purely informational / educational content,
- generic "what is" / "why" / "history of" content,
- anything that is not clearly about buying or choosing products,
- content that is mainly about news, reviews, or opinions.

You will NOT output the filtered items; you only use them as a basis.

========================
STEP 2 – TURN INTO REAL BUYER QUERIES
========================
Now, based on the remaining (more commercial) signals,
GENERATE NEW NATURAL-LANGUAGE SHOPPING PROMPTS that a user would ask an AI assistant
(like ChatGPT) when they are actively looking to discover and BUY products like this.

Very important behavioural rules:

1) REAL BUYING INTENT
   - Every prompt must show clear intent to buy or choose products.
   - Use patterns like:
     "buy", "where can I find", "best X to buy", "which X should I get",
     "good options under £X", "recommend", "show me", "with fast delivery", etc.
   - No research-only or informational questions.

2) PROMPT STYLE = HOW HUMANS SEARCH
   - Each prompt:
       • is 1 sentence (max 2 if absolutely needed),
       • is short and direct (roughly 8–25 words),
       • is written in simple, natural language.
   - Do NOT explain backstory, meaning, inspiration, themes, or lore.
   - Do NOT reference any “famous”, “iconic”, or “classic” songs, movies, memes, stories, art, or rivalries.
   - Do NOT say things like:
       "that famous X", "that iconic Y", "that classic duet", "that viral meme", etc.
   - Do NOT narrate what is printed on the product in story form.
   - The product_title and URL are ONLY for you to infer generic product type, features, use-cases, etc.
     You MUST NOT reuse phrases, names, or references from them.

3) COVER KEY BUYER MODES ACROSS THE WHOLE SET
   Spread the prompts so that, across the list, you naturally cover:

   (a) Product discovery (buy mode)
       - User knows the general type and wants options.
       - e.g. "best {{category or "products"}} to buy for everyday use"

   (b) Close-alternative search
       - User wants similar items (style, use-case, features) without knowing specific brands.
       - e.g. "similar {{category or "items"}} with [feature/fit/colour] I can buy online"

   (c) Price / budget-led buying
       - User filters by budget or value.
       - e.g. "{{category or "products"}} under £50 with good reviews"

   (d) Occasion / use-case driven buying
       - User has a specific event or scenario.
       - e.g. "{{category or "products"}} suitable for weddings / office / travel / gifts"

   (e) Fit / comfort / practicality
       - Comfort, fit, material, durability, practicality.
       - e.g. "comfortable {{category or "products"}} for all-day wear"

   (f) Trend / popularity / best-rated
       - User wants trending or best-rated options.
       - e.g. "most popular {{category or "products"}} this season"

   (g) Channel preference
       - User may mention generic channels:
         "online in the UK", "UK websites", "high street shops", "department stores".
       - Do NOT mention any specific retailer or marketplace by name.

4) STRICT BRAND / STORE / DOMAIN / TITLE NEUTRALITY
   - The shopper does NOT know the brand, store, or domain.
   - NEVER mention:
       • the product's brand,
       • any store / retailer by name,
       • any website/domain or URL,
       • any proper noun derived from the product_title or URL.
   - If any Google signals contain brand names, store names, or domains,
     you MUST NOT carry them into the prompts.
     Replace them with generic phrases like:
       "reputable brands", "UK department stores", "online retailers".
   - NEVER paraphrase or echo the internal product_title.

5) WHAT TO AVOID (CRITICAL)
   DO NOT generate:
   - Lore / story / rivalry / character / plot references of any kind.
   - Lyrics, quotes, or indirect references to songs, shows, movies, or memes.
   - Fan-tribute / collector language that encodes the identity of the item.
   - Long, descriptive, themed or poetic prompts.

   BAD patterns you MUST NOT use:
   - "that famous [song/duet/book/movie] where ..."
   - "two characters arguing about ..."
   - "that iconic scene where ..."
   - "T-shirt with artwork of [implied famous duet / story / character]"

   GOOD direction:
   - Focus on type, fit, material, price, colour, style, size, delivery, quality, use-case.

6) UK CONTEXT
   - Use £ for prices.
   - Assume the shopper is in the UK, unless the category clearly suggests otherwise.
   - Use neutral wording for channels: "online in the UK", "UK websites", "UK shops".

7) OUTPUT FORMAT (STRICT)
   - Generate up to {num_prompts} prompts.
   - They must ALL be high purchase intent and sound like real buyer queries.
   - Return STRICTLY valid JSON with exactly this structure:

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
        temperature=0.4,
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

    Prompts are:
    - Seeded from real Google behaviour,
    - Explicitly high-purchase-intent,
    - Brand / store / domain / title neutral,
    - Free of lore/theme/reference style cues.
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