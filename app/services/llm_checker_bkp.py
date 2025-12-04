import os
import re
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI

from app.config import DEFAULT_LLM_MODEL

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Basic URL finder
URL_REGEX = re.compile(r"https?://[^\s\]\)\"'>]+", re.IGNORECASE)


def check_presence_in_text(text: str, domain: str) -> Tuple[bool, str]:
    """
    Returns (appeared, matched_domain_or_url)
    - True if domain appears or any URL with that domain appears.
    - Also returns first matching URL/domain for storage.
    """
    text_lower = text.lower()
    if domain.lower() in text_lower:
        return True, domain

    for url in URL_REGEX.findall(text):
        try:
            if domain.lower() in url.lower():
                return True, url
        except Exception:
            continue

    return False, ""


def run_llm_visibility_check(
    prompt: str,
    domain: str,
    model: str | None = None,
) -> dict:
    """
    Sends a single prompt to an LLM and checks if the brand/domain appears
    in the response. Returns a dict with appeared, matched, snippet, model.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Decide which model to use: explicit override or default from config
    model_to_use = model or DEFAULT_LLM_MODEL

    # Keep the prompt focused. Ask the model to list product recommendations with URLs.
    system = (
        "You are a helpful shopping assistant. "
        "When listing products, include the product names and direct product URLs. "
        "Prefer UK-available options if not specified."
    )

    completion = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = completion.choices[0].message.content or ""
    appeared, matched = check_presence_in_text(answer, domain)

    # Keep a short snippet (first 800 chars)
    snippet = answer[:800]

    return {
        "appeared": appeared,
        "matched": matched,
        "snippet": snippet,
        "model": model_to_use,
    }