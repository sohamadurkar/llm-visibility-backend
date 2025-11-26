import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime  # ✅ NEW

from dotenv import load_dotenv
from openai import OpenAI

from app.services.prompt_packs import PROMPT_PACKS_DIR

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
    Uses OpenAI to generate a list of high-intent shopping prompts for a product/category.

    Behavioural logic applied:
    - Only generate prompts where the user is clearly looking to BUY (commercial / transactional intent).
    - Cover (as much as possible) these 7 purchase behaviours:
        1) Product discovery within a buy mindset
        2) Specific product / close alternatives
        3) Price- and budget-sensitive buying
        4) Occasion / use-case driven buying
        5) Fit / comfort / practicality concerns before buying
        6) Trend / popularity driven buying
        7) Brand / store preference driven buying

    - Avoid purely informational / research queries (e.g. “are velvet shoes good for winter?”).
    - Prompts should sound like real user queries to an AI assistant with the intent to find a product to purchase.
    """
    _ensure_api_key()

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_msg = (
        "You are an expert e-commerce growth marketer and search strategist. "
        "You design natural-language shopping queries that REAL users would ask an AI assistant "
        "when they are actively looking to BUY a product, not just research it."
    )

    user_msg = f"""
You must generate {num_prompts} DIVERSE, HIGH-INTENT shopping prompts for products like:

- Product title: {product_title}
- Product page URL: {product_url}
- Product/category: {category or "unknown"}

Very important behavioural rules:

1) ALL prompts must reflect a user who is READY TO BUY or very close to making a purchase.
   - Use verbs and patterns like "buy", "where can I find", "best X to buy", "options under £X",
     "recommend", "show me", "which X should I buy", etc.
   - The user is NOT asking for definitions or generic information; they want product suggestions and links.

2) You MUST cover the following high-intent behaviour types across the whole set
   (not necessarily labelled, but reflected in the prompts):

   (a) Product discovery (buy mode)
       - User knows the general type of item and wants suggestions to buy.
       - e.g. "best velvet Mary Jane pumps for women under £200 available online"

   (b) Specific product / close-alternative search
       - User is looking for this product or very similar ones to buy.
       - e.g. "velvet Mary Jane pumps similar to {product_title} available in the UK"

   (c) Price- / budget-sensitive purchasing
       - User is deciding what to buy based on price or deals.
       - e.g. "velvet Mary Jane shoes under £150 with good quality"

   (d) Occasion / use-case driven buying
       - User wants a product for a specific event or scenario.
       - e.g. "velvet pumps suitable for winter weddings" or "heels for cocktail parties"

   (e) Fit / comfort / practicality based buying
       - User wants to buy but cares about comfort, heel height, walkability, etc.
       - e.g. "comfortable velvet pumps with low heel for long events"

   (f) Trend / popularity driven buying
       - User wants to buy something that is trending or popular right now.
       - e.g. "trending velvet Mary Jane pumps this season"

   (g) Brand / store preference
       - User is ready to buy but prefers certain brands or stores (like John Lewis, Penelope Chilvers, etc.).
       - e.g. "velvet Mary Jane pumps available at John Lewis"

3) DO NOT generate:
   - informational-only queries (e.g. "are velvet shoes in style?")
   - purely educational questions (e.g. "what are Mary Jane shoes?")
   - generic fashion advice not clearly linked to finding a product to buy.

4) Tone & form:
   - Each prompt must be a standalone user query, in natural language.
   - Assume the user is talking to an AI assistant (like ChatGPT) to discover products and buy them.
   - Use UK context when relevant (currency £, retailers in the UK), unless the category suggests otherwise.

5) Output format:
   - Return STRICTLY valid JSON with this exact structure:

   {{
     "prompts": [
       "prompt 1",
       "prompt 2",
       "prompt 3"
     ]
   }}

   - No markdown, no explanations, no comments, no extra keys.
   - Exactly {num_prompts} prompts if possible; a minimum of 30 if you cannot reach {num_prompts}.
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

    # Parse JSON as instructed
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

    # Trim to desired size
    return cleaned[:num_prompts]


def _slugify(text: str, max_length: int = 40) -> str:
    """
    Simple slug generator for filenames / IDs.
    """
    text = text.lower()
    slug = []
    for ch in text:
        if ch.isalnum():
            slug.append(ch)
        elif ch in " _-":
            slug.append("-")
        # ignore everything else
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
    Prompts are generated with explicit high-purchase-intent logic.

    IMPORTANT:
    - If pack_id is not provided, we generate a UNIQUE pack ID each time
      using product_id + slug + timestamp. This means each generation is
      a new versioned pack and old packs remain intact.
    """
    prompts = _generate_prompts_with_llm(product_title, product_url, category, num_prompts)

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
        # You can optionally add a source flag here if you like:
        # "source": "auto_generated_high_intent",
        "prompts": prompts,
    }
    return pack


def save_prompt_pack_to_file(pack: Dict[str, Any]) -> str:
    """
    Saves the pack dict as JSON under prompt_packs/{id}.json and returns full path.
    """
    os.makedirs(PROMPT_PACKS_DIR, exist_ok=True)
    pack_id = pack.get("id") or "pack"
    fname = f"{pack_id}.json"
    fpath = os.path.join(PROMPT_PACKS_DIR, fname)

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    return fpath