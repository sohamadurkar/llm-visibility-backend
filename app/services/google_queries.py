import os
from typing import List, Dict, Any, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY")

GOOGLE_API_URL = "https://api.scrapingdog.com/google/"


def _ensure_api_key():
    if not SCRAPINGDOG_API_KEY:
        raise RuntimeError("SCRAPINGDOG_API_KEY not set in environment (.env)")


def fetch_google_serp(
    query: str,
    country: str = "uk",
    results: int = 10,
    page: int = 0,
) -> Dict[str, Any]:
    """
    Calls Scrapingdog Google Search API and returns the JSON response.
    """
    _ensure_api_key()

    params = {
        "api_key": SCRAPINGDOG_API_KEY,
        "query": query,
        "results": str(results),
        "country": country,
        "page": str(page),
    }

    resp = httpx.get(GOOGLE_API_URL, params=params, timeout=40)
    resp.raise_for_status()
    return resp.json()


def extract_people_also_ask_questions(data: Dict[str, Any]) -> List[str]:
    """
    Source A – People Also Ask (PAA)
    From a Scrapingdog Google response, extract 'people_also_ask' questions.
    If your Scrapingdog plan/response uses a different key, extend this function.
    """
    questions: List[str] = []
    paa = data.get("people_also_ask", [])
    for item in paa:
        # Common Scrapingdog PAA shape: { "question": "...", ... }
        q = item.get("question")
        if isinstance(q, str):
            q = q.strip()
            if q:
                questions.append(q)
    return questions


def extract_related_searches(data: Dict[str, Any]) -> List[str]:
    """
    Source B – Related Searches
    Extracts 'related_searches' terms from the SERP JSON.
    Scrapingdog typically exposes these as an array like:
      "related_searches": [{"query": "..."}, ...]
    but we are defensive and check multiple keys.
    """
    related: List[str] = []
    raw_related = data.get("related_searches", [])

    for item in raw_related:
        # Try several likely key names
        q = (
            item.get("query")
            or item.get("keyword")
            or item.get("title")
        )
        if isinstance(q, str):
            q = q.strip()
            if q:
                related.append(q)

    return related


def extract_titles_and_snippets(data: Dict[str, Any]) -> List[str]:
    """
    Source C – Search Result Titles & Snippets
    Extracts phrases from organic result titles/snippets which indicate
    how Google is framing this intent.

    We don't treat them as “queries” 1:1, but we treat them as text
    describing what people are searching for and what pages promise.
    """
    phrases: List[str] = []

    organic = data.get("organic_results", []) or data.get("results", [])
    for item in organic:
        # Title
        t = item.get("title")
        if isinstance(t, str):
            t = t.strip()
            if t:
                phrases.append(t)

        # Snippet / description
        s = item.get("snippet") or item.get("description")
        if isinstance(s, str):
            s = s.strip()
            if s:
                phrases.append(s)

    return phrases


def build_default_seed_queries(
    product_title: str,
    category: Optional[str],
) -> List[str]:
    """
    Simple heuristic to create seed queries for a product.
    These double as a fallback if Google returns no data.
    """
    seeds = [
        product_title,
        f"buy {product_title} online",
        f"{product_title} best price",
        f"where to buy {product_title}",
    ]
    if category:
        seeds.extend([
            f"best {category} like {product_title}",
            f"buy {category} {product_title} uk",
        ])

    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for s in seeds:
        s_norm = s.strip().lower()
        if s_norm and s_norm not in seen:
            seen.add(s_norm)
            uniq.append(s.strip())
    return uniq


def collect_google_queries_for_product(
    product_title: str,
    category: Optional[str],
    max_questions: int = 50,
) -> List[str]:
    """
    Runs multiple seed queries through Scrapingdog and aggregates
    unique signals from:

    - Source A: People Also Ask (PAA)
    - Source B: Related Searches
    - Source C: Organic Result Titles & Snippets

    Fallback behaviour:
    - If no items are found for any seed queries, return the seed
      queries themselves so that the pipeline can still generate prompts.
    """
    seeds = build_default_seed_queries(product_title, category)
    all_items: List[str] = []
    seen = set()

    # Try to collect signals from all three sources
    for seed in seeds:
        try:
            data = fetch_google_serp(seed, country="uk", results=10, page=0)
        except Exception:
            # Fail quietly for individual seeds; continue with others
            continue

        # A: PAA questions
        paa_qs = extract_people_also_ask_questions(data)
        # B: Related searches
        related_qs = extract_related_searches(data)
        # C: Titles & snippets
        title_snippet_phrases = extract_titles_and_snippets(data)

        for q in paa_qs + related_qs + title_snippet_phrases:
            q_norm = q.lower()
            if q_norm not in seen:
                seen.add(q_norm)
                all_items.append(q)
                if len(all_items) >= max_questions:
                    return all_items

    # Fallback: if nothing at all was collected, use the seeds themselves
    if not all_items:
        return seeds

    return all_items