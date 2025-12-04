# llm_checker.py

import os
import re
import json                      # NEW
from typing import Tuple, List   # UPDATED
from dotenv import load_dotenv
from openai import OpenAI

from app.config import DEFAULT_LLM_MODEL

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

URL_REGEX = re.compile(r"https?://[^\s\]\)\"'>]+", re.IGNORECASE)


def check_presence_in_text(text: str, domain: str) -> Tuple[bool, str]:
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
    Single-prompt version (kept for /run-llm-check and reuse).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=OPENAI_API_KEY)
    model_to_use = model or DEFAULT_LLM_MODEL

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
    snippet = answer[:800]

    return {
        "appeared": appeared,
        "matched": matched,
        "snippet": snippet,
        "model": model_to_use,
    }


def run_llm_visibility_batch_check(
    prompts: List[str],
    domain: str,
    model: str | None = None,
) -> List[dict]:
    """
    Batch version: sends MANY prompts in ONE OpenAI call and returns
    a list of results (same order as `prompts`).
    Each result has: appeared, matched, snippet, model.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=OPENAI_API_KEY)
    model_to_use = model or DEFAULT_LLM_MODEL

    system = (
        "You are a helpful shopping assistant. "
        "You will receive multiple user prompts. "
        "For EACH prompt, you must answer it as if the user asked it individually, "
        "listing product names and direct product URLs where appropriate. "
        "Prefer UK-available options if not specified."
    )

    # We send the prompts as JSON so the model can index them cleanly
    prompts_payload = [{"index": i, "text": p} for i, p in enumerate(prompts)]
    prompts_json = json.dumps(prompts_payload, ensure_ascii=False)

    user_msg = f"""
You are given a JSON array called "prompts". Each item has:
- "index": an integer index
- "text": the user's shopping query

prompts = {prompts_json}

For EACH item, you must produce an answer as if you were replying to that query alone.

Return STRICTLY valid JSON in this format:

{{
  "results": [
    {{
      "index": 0,
      "answer": "full natural language answer for prompt index 0, including product URLs"
    }},
    {{
      "index": 1,
      "answer": "full natural language answer for prompt index 1, including product URLs"
    }}
  ]
}}

- Do NOT omit any indexes.
- Do NOT add extra keys.
- The results must be in an array named "results".
"""

    completion = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    content = completion.choices[0].message.content or ""
    data = json.loads(content)
    raw_results = data.get("results", [])

    # Build a mapping from index -> answer string
    answers_by_index: dict[int, str] = {}
    for item in raw_results:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        ans = item.get("answer") or ""
        if isinstance(ans, str):
            answers_by_index[idx] = ans

    results: List[dict] = []
    for i, _prompt in enumerate(prompts):
        answer = answers_by_index.get(i, "")
        appeared, matched = check_presence_in_text(answer, domain)
        snippet = answer[:800]
        results.append(
            {
                "appeared": appeared,
                "matched": matched,
                "snippet": snippet,
                "model": model_to_use,
            }
        )

    return results