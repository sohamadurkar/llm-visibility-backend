import os
from dotenv import load_dotenv

# Ensure env vars are loaded once here
load_dotenv()

# Base/default chat model (for general LLM checks etc.)
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-5.1")

# Model to use for generating prompt packs (behavioural)
DEFAULT_PROMPT_PACK_MODEL = os.getenv("DEFAULT_PROMPT_PACK_MODEL", DEFAULT_LLM_MODEL)

# Model to use for generating Google-seeded prompt packs
DEFAULT_GOOGLE_PACK_MODEL = os.getenv("DEFAULT_GOOGLE_PACK_MODEL", DEFAULT_LLM_MODEL)

# Model to use for generating visibility reports
DEFAULT_REPORT_MODEL = os.getenv("DEFAULT_REPORT_MODEL", "gpt-5.1")