import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# Expect DATABASE_URL for production (e.g. Railway Postgres)
# Example: postgresql+psycopg2://user:password@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Fallback ONLY for local development if you want it.
    # In Railway you will ALWAYS set DATABASE_URL and this block won't run.
    # If you truly never want SQLite anywhere, remove this fallback and raise instead.
    raise RuntimeError("DATABASE_URL not set – please configure it in Railway")

# For SQLite we need check_same_thread; for others we don't.
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Safe defaults for Railway (tune via env vars if needed)
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "60"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))


engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,         # drops dead connections
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT,
    pool_recycle=DB_POOL_RECYCLE,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()