from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.engine import Base


class LLMTest(Base):
    __tablename__ = "llm_tests"

    id = Column(Integer, primary_key=True, index=True)

    # Which product this test was run for
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)

    # NEW: link each test back to a specific prompt & pack (if applicable)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=True)
    pack_id = Column(Integer, ForeignKey("prompt_packs.id"), nullable=True)

    model_used = Column(String, nullable=False)

    # The raw prompt text that was sent to the LLM
    prompt = Column(Text, nullable=False)

    appeared = Column(Boolean, default=False)
    matched_domain = Column(String, nullable=True)
    snippet = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    prompt_obj = relationship("Prompt", back_populates="tests")
    pack = relationship("PromptPack")
    # You can optionally add a Product relationship if you want:
    # product = relationship("Product", back_populates="llm_tests")