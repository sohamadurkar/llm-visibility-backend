
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


def fetch_google_serp(query: str,
                      country: str = "uk",
                      results: int = 10,
                      page: int = 0) -> Dict[str, Any]:
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
    From a Scrapingdog Google response, extract 'people_also_ask' questions.
    If your Scrapingdog plan/response uses a different key, extend this function.
    """
    questions: List[str] = []
    paa = data.get("people_also_ask", [])
    for item in paa:
        q = item.get("question")
        if isinstance(q, str):
            q = q.strip()
            if q:
                questions.append(q)
    return questions


def build_default_seed_queries(product_title: str,
                               category: Optional[str]) -> List[str]:
    """
    Simple heuristic to create seed queries for a product.
    These double as a fallback if Google returns no PAA.
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


def collect_google_queries_for_product(product_title: str,
                                       category: Optional[str],
                                       max_questions: int = 50) -> List[str]:
    """
    Runs multiple seed queries through Scrapingdog and aggregates
    unique 'people also ask' questions.

    Fallback behaviour:
    - If no PAA questions are found for any seed queries, return the seed
      queries themselves so that the pipeline can still generate prompts.
    """
    seeds = build_default_seed_queries(product_title, category)
    all_questions: List[str] = []
    seen = set()

    # Try to collect PAA questions
    for seed in seeds:
        try:
            data = fetch_google_serp(seed, country="uk", results=10, page=0)
        except Exception:
            # Fail quietly for individual seeds; continue with others
            continue

        qs = extract_people_also_ask_questions(data)
        for q in qs:
            q_norm = q.lower()
            if q_norm not in seen:
                seen.add(q_norm)
                all_questions.append(q)
                if len(all_questions) >= max_questions:
                    return all_questions

    # Fallback: if no questions were collected, use the seeds themselves
    if not all_questions:
        return seeds

    return all_questions