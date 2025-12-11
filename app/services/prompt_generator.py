import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid  # NEW

from dotenv import load_dotenv
from openai import OpenAI

from app.services.prompt_packs import PROMPT_PACKS_DIR
from app.config import DEFAULT_PROMPT_PACK_MODEL  # NEW

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _ensure_api_key():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")


def _generate_prompts_with_llm(
    product_title: str,
    product_url: str,
    category: Optional[str],
    num_prompts: int = 50,
) -> List[str]:
    """
    Uses OpenAI to generate a list of high-intent, CUSTOMER-LIKE shopping prompts
    for a product/category.

    Core principles (enforced via the system + user prompt):
    - Prompts must sound like REAL buyers talking to an assistant.
    - No storylines / themes / references / “that famous X” type language.
    - No brand / store / domain names, and no reuse of product_title text.
    """
    _ensure_api_key()

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_msg = (
        "You are an expert in real-world e-commerce search behaviour. "
        "You generate short, natural-language BUYING QUERIES that real shoppers would ask an AI assistant. "
        "You NEVER write marketing copy, lore, story references, or themed descriptions. "
        "You ONLY write what a buyer would actually type or say when trying to find products to purchase."
    )

    user_msg = f"""
You must generate {num_prompts} DIVERSE, HIGH-INTENT shopping prompts for products like:

- Internal product title (for your understanding only): {product_title}
- Product page URL (for context only): {product_url}
- Product/category (for context only): {category or "unknown"}

The user does NOT know the internal title, brand, or URL. They are just a normal shopper.

========================
ABSOLUTE BEHAVIOURAL RULES
========================

1) REAL BUYER INTENT ONLY (not research)
   - Every prompt must reflect a shopper who is READY TO BUY or very close:
     - Use patterns like: "buy", "where can I find", "best X to buy", "which X should I get",
       "options under £X", "recommend", "show me", "good value", "with fast delivery", etc.
   - No purely informational or educational questions.
   - The user is clearly expecting concrete product suggestions they could actually purchase.

2) PROMPT STYLE = HOW HUMANS SEARCH
   - Each prompt must be:
     - 1 sentence (max 2 if absolutely needed),
     - Short and to the point (roughly 8–25 words),
     - Written in simple, natural language.
   - Do NOT explain backstory, meaning, inspiration, themes, or lore.
   - Do NOT reference any “famous”, “iconic”, or “classic” songs, movies, memes, stories, art, characters, or rivalries.
   - Do NOT say things like “that famous X”, “that iconic Y”, “that classic duet”, “that viral meme”, etc.
   - Do NOT describe what is printed on the product in a narrative way.
   - The product_title and URL are ONLY for you to infer generic product type, features, use-cases, etc.
     You MUST NOT reuse phrases, names, or references from them.

3) COVER KEY BUYER MODES ACROSS THE WHOLE SET
   Spread the prompts so that, across the list, you naturally cover:

   (a) Product discovery (buy mode)
       - User knows the general type and wants options.
       - e.g. "best {category or "product"} to buy for everyday use"

   (b) Close-alternative search
       - User wants similar items (style, use-case, features) without knowing specific brands.
       - e.g. "similar {category or "items"} with [feature/fit/colour] I can buy online"

   (c) Price / budget-led buying
       - User filters by budget or value.
       - e.g. "{category or "product"} under £50 with good reviews"

   (d) Occasion / use-case driven buying
       - User has a specific event or scenario.
       - e.g. "{category or "product"} suitable for weddings / office / travel / gifts"

   (e) Fit / comfort / practicality
       - Comfort, fit, material, durability, practicality.
       - e.g. "comfortable {category or "product"} for all-day wear"

   (f) Trend / popularity / best-rated
       - User wants trending or best-rated options.
       - e.g. "most popular {category or "product"} this season"

   (g) Channel preference
       - User may mention generic marketplaces (e.g. "online", "high street", "UK retailers").
       - Do NOT mention any specific retailer or marketplace by name.
       - Keep it generic like "online shops", "UK websites", "high street stores", etc.

4) STRICT BRAND, STORE, DOMAIN & TITLE NEUTRALITY
   - NEVER mention:
     - The product's brand,
     - The product's store name,
     - The website/domain or any URL,
     - Any proper noun derived from the product_title or URL.
   - NEVER paraphrase or echo the product title.
   - Assume the shopper has NEVER heard of this specific product/brand.

5) WHAT TO AVOID (VERY IMPORTANT)
   DO NOT generate:
   - Lore / story / rivalry / character / plot references of any kind.
   - Lyrics, quotes, or indirect references to songs, shows, movies, or memes.
   - Fan language ("obsessed with", "stan", etc.) that encodes the identity of the item.
   - “Collector” or “fan tribute” style prompts that sneak in identity clues.
   - Long, descriptive, themed or poetic language.

   Examples of BAD patterns you MUST NOT use:
   - "that famous [song/duet/book/movie] where ..."
   - "two characters arguing about ..."
   - "that iconic scene where ..."
   - "T-shirt with artwork of [named or clearly implied story/duet/etc.]"

   Instead, stay GENERIC and PRODUCT-CENTRIC:
   - Talk about type, fit, material, price, colour, style, use-case, size, delivery, quality, etc.

6) UK CONTEXT
   - Use £ for prices.
   - Assume the shopper is in the UK, unless the category clearly implies a different region.
   - Retail channel wording can be like "online in the UK", "UK websites", "UK shops".

7) OUTPUT FORMAT (MUST FOLLOW EXACTLY)
   - Return STRICTLY valid JSON with this exact structure:

   {{
     "prompts": [
       "prompt 1",
       "prompt 2",
       "prompt 3"
     ]
   }}

   - No markdown, no explanations, no comments, no extra keys.
   - Return exactly {num_prompts} prompts if possible, but at least 30 minimum.
"""

    completion = client.chat.completions.create(
        model=DEFAULT_PROMPT_PACK_MODEL,  # now from config
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
        raise RuntimeError("LLM returned no prompts")

    return cleaned[:num_prompts]


def _generate_persona_prompts_with_llm(
    product_title: str,
    product_url: str,
    persona: str,
    category: Optional[str],
    num_prompts: int = 50,
) -> List[str]:
    """
    Uses OpenAI to generate a list of high-intent, CUSTOMER-LIKE shopping prompts
    for a product/category, tailored to a specific buyer persona.
    """
    _ensure_api_key()

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_msg = (
        "You are an expert in real-world e-commerce search behaviour. "
        "You generate short, natural-language BUYING QUERIES that real shoppers would ask an AI assistant. "
        "You NEVER write marketing copy, lore, story references, or themed descriptions. "
        "You ONLY write what a buyer would actually type or say when trying to find products to purchase."
    )

    user_msg = f"""
You must generate {num_prompts} DIVERSE, HIGH-INTENT shopping prompts for this product and buyer persona.

PRODUCT CONTEXT (for your understanding only):
- Internal product title: {product_title}
- Product page URL: {product_url}
- Product/category (for context only): {category or "unknown"}

BUYER PERSONA (write prompts in the voice and intent of this persona):
\"\"\"{persona}\"\"\"

The user does NOT know the internal title, brand, or URL. They are just a normal shopper matching this persona.

========================
ABSOLUTE BEHAVIOURAL RULES
========================

1) REAL BUYER INTENT ONLY (not research)
   - Every prompt must reflect this persona being READY TO BUY or very close:
     - Use patterns like: "buy", "where can I find", "best X to buy", "which X should I get",
       "options under £X", "recommend", "show me", "good value", "with fast delivery", etc.
   - No purely informational or educational questions.
   - The persona clearly expects concrete product suggestions they could actually purchase.

2) PROMPT STYLE = HOW THIS PERSONA SEARCHES
   - Each prompt must be:
     - 1 sentence (max 2 if absolutely needed),
     - Short and to the point (roughly 8–25 words),
     - Written in simple, natural language.
   - You may reflect persona-specific constraints: budget, family situation, use-case, style, etc.
   - Do NOT explain backstory, meaning, inspiration, themes, or lore.
   - Do NOT reference any “famous”, “iconic”, or “classic” songs, movies, memes, stories, art, characters, or rivalries.
   - Do NOT say things like “that famous X”, “that iconic Y”, “that classic duet”, “that viral meme”, etc.
   - The product_title and URL are ONLY for you to infer generic product type, features, use-cases, etc.
     You MUST NOT reuse phrases, names, or references from them.

3) COVER KEY BUYER MODES ACROSS THE WHOLE SET
   Spread the prompts so that, across the list, you naturally cover:

   (a) Product discovery (buy mode)
   (b) Close-alternative search
   (c) Price / budget-led buying
   (d) Occasion / use-case driven buying
   (e) Fit / comfort / practicality
   (f) Trend / popularity / best-rated
   (g) Channel preference (generic "online", "UK websites", etc.)

4) STRICT BRAND, STORE, DOMAIN & TITLE NEUTRALITY
   - NEVER mention brand / store / domain / URL.
   - NEVER paraphrase the product title.
   - Assume the persona has NEVER heard of this specific product/brand.

5) WHAT TO AVOID (VERY IMPORTANT)
   - No lore, story, rivalry, character or plot references.
   - No lyrics, quotes, or indirect references to songs, shows, movies, or memes.
   - No fan / collector style language.
   - No long descriptive or themed language.

6) UK CONTEXT (default)
   - Use £ for prices.
   - Assume the persona is in the UK unless the category clearly implies otherwise.

7) OUTPUT FORMAT (MUST FOLLOW EXACTLY)
   - Return STRICTLY valid JSON with this exact structure:

   {{
     "prompts": [
       "prompt 1",
       "prompt 2",
       "prompt 3"
     ]
   }}

   - No markdown, no explanations, no comments, no extra keys.
   - Return exactly {num_prompts} prompts if possible, but at least 30 minimum.
"""

    completion = client.chat.completions.create(
        model=DEFAULT_PROMPT_PACK_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.5,
    )

    content = completion.choices[0].message.content or ""

    try:
        data = json.loads(content)
        prompts = data.get("prompts", [])
    except Exception:
        # fallback: split by lines if model didn't return JSON
        prompts = [
            line.strip()
            for line in content.split("\n")
            if line.strip()
        ]

    cleaned: List[str] = []
    for p in prompts:
        if isinstance(p, str):
            s = p.strip()
            if s:
                cleaned.append(s)

    if not cleaned:
        raise RuntimeError("LLM returned no prompts for persona pack")

    return cleaned[:num_prompts]


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


def generate_prompt_pack_for_product(
    product_id: int,
    product_title: str,
    product_url: str,
    category: Optional[str],
    num_prompts: int = 50,
    pack_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns a dict in the same structure as your JSON prompt pack files.
    Prompts are generated with explicit high-purchase-intent logic and are:

    - Short, natural buyer queries.
    - Brand / store / domain / title neutral.
    - Free of storylines, references, or theme-based descriptions.

    IMPORTANT:
    - If pack_id is not provided, we generate a UNIQUE pack ID each time
      using product_id + slug + timestamp.
    """
    prompts = _generate_prompts_with_llm(
        product_title=product_title,
        product_url=product_url,
        category=category,
        num_prompts=num_prompts,
    )

    if pack_id:
        base_id = pack_id
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        base_id = f"auto_{product_id}_{_slugify(product_title)}_{timestamp}"

    pack_name = name or f"Auto Pack for {product_title[:60]}"

    pack: Dict[str, Any] = {
        "id": base_id,
        "name": pack_name,
        "category": category or "auto_generated_high_intent",
        "language": "en",
        "source": "auto_generated_high_intent",
        "prompts": prompts,
    }
    return pack


def generate_persona_prompt_pack_for_product(
    product_id: int,
    product_title: str,
    product_url: str,
    persona: str,
    category: Optional[str] = None,
    num_prompts: int = 50,
    pack_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns a dict for a persona-based prompt pack.

    - Prompts are generated in the voice and intent of the given persona.
    - Same high-purchase-intent rules, but persona-specific constraints are allowed.
    """
    prompts = _generate_persona_prompts_with_llm(
      product_title=product_title,
      product_url=product_url,
      persona=persona,
      category=category,
      num_prompts=num_prompts,
    )

    if pack_id:
        base_id = pack_id
    else:
        short = uuid.uuid4().hex[:8]
        base_id = f"persona_{product_id}_{short}"

    default_name = f"Persona pack – {persona[:40]}".strip()
    pack_name = name or default_name

    pack: Dict[str, Any] = {
        "id": base_id,
        "name": pack_name,
        "category": category or "persona_high_intent",
        "language": "en",
        "source": "persona_high_intent",
        "product_id": product_id,
        "persona": persona,
        "prompts": prompts,
    }
    return pack


def save_prompt_pack_to_file(pack: Dict[str, Any]) -> str:
    os.makedirs(PROMPT_PACKS_DIR, exist_ok=True)
    pack_id = pack.get("id") or "pack"
    fname = f"{pack_id}.json"
    fpath = os.path.join(PROMPT_PACKS_DIR, fname)

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    return fpath