from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from app.db.engine import Base

class LLMBatchRun(Base):
    __tablename__ = "llm_batch_runs"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False)
    pack_id = Column(String, nullable=False)
    model_used = Column(String, nullable=False)

    total_prompts = Column(Integer, nullable=False)
    appeared_count = Column(Integer, nullable=False)
    visibility_score = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)