# app/services/batch_competitor_report.py

from typing import List, Dict, Any, Optional
from textwrap import shorten

from app.models.models import Product
from app.models.llmtest import LLMTest
from app.models.prompt_models import PromptPack
from app.config import DEFAULT_REPORT_MODEL

from openai import OpenAI


# Initialise OpenAI client (uses OPENAI_API_KEY from env)
client = OpenAI()


def _safe_text(x: Optional[str], max_chars: int = 4000) -> str:
    """
    Safely coerce text to a reasonably bounded string for prompts.
    """
    if not x:
        return ""
    return shorten(x, width=max_chars, placeholder=" [...]")


def generate_competitor_report_markdown_for_batch(
    product: Product,
    pack: Optional[PromptPack],
    tests: List[LLMTest],
    model: Optional[str] = None,
) -> str:
    """
    Fully LLM-generated competitor analysis for a single batch.

    Inputs:
      - product: the client's product row (with .website relationship if loaded)
      - pack: the PromptPack used for this batch, if available
      - tests: all LLMTest rows belonging to this batch (same batch_id)
      - model: optional override of DEFAULT_REPORT_MODEL

    Output:
      - Markdown string describing:
        * Where the client's brand appeared
        * Where competitors appeared instead
        * What winning brands did differently
        * Concrete recommendations for this client
    """
    if not tests:
        return "# Competitor Analysis\n\nNo LLM tests found for this batch."

    model = model or DEFAULT_REPORT_MODEL

    client_domain = product.website.domain if getattr(product, "website", None) else None
    client_url = product.url
    client_title = product.title or product.url
    has_schema = bool(getattr(product, "has_product_jsonld", False))

    # Split into wins (client appeared) vs losses (competitors)
    wins: List[Dict[str, Any]] = []
    losses: List[Dict[str, Any]] = []

    for t in tests:
        full_answer_raw = t.llm_answer or t.snippet or ""
        entry = {
            "prompt": t.prompt,
            "full_answer": _safe_text(full_answer_raw, max_chars=4000),
            "appeared": bool(t.appeared),
            "matched_domain": t.matched_domain,
        }
        if t.appeared:
            wins.append(entry)
        else:
            losses.append(entry)

    pack_label = pack.name if pack else None
    pack_key = pack.pack_key if pack else None

    # Compact JSON-like context for the LLM
    context: Dict[str, Any] = {
        "client": {
            "domain": client_domain,
            "url": client_url,
            "title": client_title,
            "has_product_or_service_schema": has_schema,
        },
        "prompt_pack": {
            "name": pack_label,
            "id": pack_key,
        },
        "summary": {
            "total_tests": len(tests),
            "wins": len(wins),
            "losses": len(losses),
        },
        "wins": wins,
        "losses": losses,
    }

    # System prompt: role + expectations
    system_prompt = (
        "You are a senior SEO & AI search strategist specialising in large language models.\n"
        "You receive data for a single brand's product or service page and a batch of LLM tests.\n\n"
        "You are given:\n"
        "- The client's domain, URL, title and whether Product/Service JSON-LD exists.\n"
        "- A set of prompts and full LLM answers.\n"
        "- For each answer, whether the client's domain appeared, and which domain matched.\n\n"
        "Your job is to produce a COMPETITOR ANALYSIS REPORT for THIS BRAND and THIS BATCH ONLY.\n"
        "Focus on:\n"
        "1) Where the client appears (wins): what in the answers suggests why the LLM is confident recommending this brand?\n"
        "2) Where the client does NOT appear (losses): which other brands win instead and what do their answers emphasise?\n"
        "3) The intent signals present in the answers: use-cases, occasions, price level, region, audience, style, features, authority signals, etc.\n"
        "4) Content & UX patterns: structure of the answer, use of bullets vs paragraphs, topical coverage, comparisons, brand authority cues.\n"
        "5) Structured data & on-page hints: where visible (e.g. ratings, stock, offers), infer how this could affect LLM training.\n\n"
        "VERY IMPORTANT:\n"
        "- Analyse the FULL answers, not just short snippets.\n"
        "- Do NOT invent brands that are not clearly implied by the answers.\n"
        "- Keep the analysis grounded in what is observable from prompts + answers.\n"
        "- Output must be clear, scannable Markdown with headings and bullet points.\n"
    )

    # User prompt containing the context
    user_prompt = (
        "Below is a JSON-like context object for one batch of LLM visibility tests.\n"
        "The 'wins' array contains prompts where the client's domain appeared in the answer.\n"
        "The 'losses' array contains prompts where competitors appeared instead.\n\n"
        "Please write a detailed competitor analysis report comparing:\n"
        "- What seems to help the client win in the 'wins' prompts.\n"
        "- What the winning competitors are doing differently in the 'losses' prompts.\n\n"
        "End with a concrete checklist of recommended changes the client should make to its product/service page "
        "and prompt strategy to improve visibility for similar prompts.\n\n"
        "Context object:\n"
        f"{context}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    markdown = response.choices[0].message.content or ""
    return markdown