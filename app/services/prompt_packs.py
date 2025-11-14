import os
import json
from typing import List, Dict, Any

# Root dir = project root (llm-visibility)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPT_PACKS_DIR = os.path.join(ROOT_DIR, "prompt_packs")

def list_prompt_packs() -> List[Dict[str, Any]]:
    """
    Returns a list of available prompt packs with basic metadata.
    Each pack is a JSON file in prompt_packs/.
    """
    packs: List[Dict[str, Any]] = []
    if not os.path.isdir(PROMPT_PACKS_DIR):
        return packs

    for fname in os.listdir(PROMPT_PACKS_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(PROMPT_PACKS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            packs.append({
                "id": data.get("id") or os.path.splitext(fname)[0],
                "name": data.get("name") or os.path.splitext(fname)[0],
                "category": data.get("category"),
                "language": data.get("language", "en"),
                "num_prompts": len(data.get("prompts", [])),
            })
        except Exception:
            continue

    return packs

def load_prompt_pack(pack_id: str) -> Dict[str, Any]:
    """
    Loads a specific prompt pack by id (filename without .json or the 'id' field in JSON).
    """
    # Try filename first
    fname = f"{pack_id}.json"
    fpath = os.path.join(PROMPT_PACKS_DIR, fname)
    if os.path.isfile(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: scan all packs and match by 'id'
    if os.path.isdir(PROMPT_PACKS_DIR):
        for fname in os.listdir(PROMPT_PACKS_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(PROMPT_PACKS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("id") == pack_id:
                    return data
            except Exception:
                continue

    raise FileNotFoundError(f"Prompt pack '{pack_id}' not found")