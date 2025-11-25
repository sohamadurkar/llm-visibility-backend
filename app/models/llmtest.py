from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.engine import Base


class LLMTest(Base):
    """
    Stores the result of a single LLM visibility test.

    - product_id: which product this test relates to.
    - prompt_id: optional FK to the Prompt table (for prompts that belong to a pack or are stored globally).
    - prompt: raw prompt text used for this run (kept for history even if linked to Prompt).
    """
    __tablename__ = "llm_tests"

    id = Column(Integer, primary_key=True, index=True)

    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    # NEW: optional link to a stored Prompt (for pack-based or reusable prompts)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=True)

    model_used = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    appeared = Column(Boolean, default=False)
    matched_domain = Column(String, nullable=True)
    snippet = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    product = relationship("Product", back_populates="llm_tests")
    prompt_obj = relationship("Prompt", back_populates="llm_tests")