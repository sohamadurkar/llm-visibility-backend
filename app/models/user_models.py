# app/models/user_models.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime

from app.db.engine import Base


class User(Base):
    """
    Per-tenant user.
    This table will exist separately in each tenant schema (and in public if you want).
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)