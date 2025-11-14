import os
import json
from typing import Dict, Any, List, Optional

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
    Uses OpenAI to generate a list of shopping-style prompts for a product/category.
    """
    _ensure_api_key()

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_msg = (
        "You are an expert e-commerce growth marketer. "
        "You generate diverse, natural-language shopping queries that real users "
        "might ask an AI assistant when looking for products."
    )

    user_msg = f"""
Generate {num_prompts} diverse search-style prompts that a shopper could ask an AI assistant when looking for products like:

Title: {product_title}
URL: {product_url}
Category: {category or "unknown"}

The prompts should sound like real questions or shopping intents, not instructions to a developer.
Vary intents: price, quality, events (wedding, party), style, comfort, occasions, trends, etc.

Output STRICTLY in valid JSON format with this exact structure:

{{
  "prompts": [
    "prompt 1",
    "prompt 2",
    "prompt 3"
  ]
}}

No markdown, no explanations, no extra keys, no comments.
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
    )

    content = completion.choices[0].message.content or ""
    data = json.loads(content)  # will raise if not valid JSON
    prompts = data.get("prompts", [])

    # Clean and truncate
    cleaned: List[str] = []
    for p in prompts:
        if isinstance(p, str):
            s = p.strip()
            if s:
                cleaned.append(s)

    if not cleaned:
        raise RuntimeError("LLM returned no prompts")

    return cleaned[:num_prompts]


def _slugify(text: str, max_length: int = 40) -> str:
    """
    Simple slug generator for filenames / IDs.
    """
    text = text.lower()
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
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
    """
    prompts = _generate_prompts_with_llm(product_title, product_url, category, num_prompts)

    base_id = pack_id or f"auto_{product_id}_{_slugify(product_title)}"
    pack_name = name or f"Auto Pack for {product_title[:60]}"

    pack: Dict[str, Any] = {
        "id": base_id,
        "name": pack_name,
        "category": category or "auto_generated",
        "language": "en",
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