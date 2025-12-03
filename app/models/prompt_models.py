from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.engine import Base


class PromptPack(Base):
    """
    Represents a prompt pack (the DB version of the JSON pack file).
    `pack_key` should match the JSON pack["id"] and the filename without .json.
    """
    __tablename__ = "prompt_packs"

    id = Column(Integer, primary_key=True, index=True)

    # e.g. "auto_1_velvet_mary_jane_pumps"
    pack_key = Column(String, unique=True, index=True, nullable=False)

    name = Column(String, nullable=True)
    category = Column(String, nullable=True)
    source = Column(String, nullable=True)   # e.g. "auto_generated_high_intent", "google_people_also_ask"
    language = Column(String, default="en")

    #  Optional link to the product the pack was generated for
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", backref="prompt_packs")
    prompts = relationship("Prompt", back_populates="pack", cascade="all, delete-orphan")


class Prompt(Base):
    """
    Individual prompt inside a pack.
    """
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    pack_id = Column(Integer, ForeignKey("prompt_packs.id"), nullable=False)

    # Position in the pack (0-based or 1-based; here we treat it as 0-based)
    index = Column(Integer, nullable=False)

    # The actual text of the prompt
    text = Column(Text, nullable=False)

    # Optional: behaviour tag (e.g. "budget", "fit", "brand", etc.)
    behaviour = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    pack = relationship("PromptPack", back_populates="prompts")
    tests = relationship("LLMTest", back_populates="prompt_obj")