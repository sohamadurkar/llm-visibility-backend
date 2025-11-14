from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.engine import Base

class LLMTest(Base):
    __tablename__ = "llm_tests"
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    model_used = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    appeared = Column(Boolean, default=False)
    matched_domain = Column(String, nullable=True)
    snippet = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # optional backref if you want it
    # product = relationship("Product", back_populates="llm_tests")