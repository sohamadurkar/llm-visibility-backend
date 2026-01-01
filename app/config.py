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

# ----- NEW: Auth / JWT settings -----
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change_me_in_env")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ----- NEW: Email settings -----
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
EMAIL_FROM_ADDRESS = os.getenv("EMAIL_FROM_ADDRESS", "soham@neuracite.com")

#  Password policy
PASSWORD_MIN_LENGTH = int(os.getenv("PASSWORD_MIN_LENGTH", "8"))