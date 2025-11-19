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
    DATABASE_URL = "sqlite:///./data/llmvis.db"

# For SQLite we need check_same_thread; for others we don't.
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()