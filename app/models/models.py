from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.db.engine import Base


class Website(Base):
    __tablename__ = "websites"

    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    products = relationship("Product", back_populates="website")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    website_id = Column(Integer, ForeignKey("websites.id"), nullable=False)
    url = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=True)
    has_product_jsonld = Column(Boolean, default=False)
    last_checked = Column(DateTime, default=datetime.utcnow)

    website = relationship("Website", back_populates="products")

    # NEW: prompt packs generated for / around this product
    prompt_packs = relationship(
        "PromptPack",
        back_populates="product",
        cascade="all, delete-orphan",
    )

    # NEW: individual prompts associated with this product
    prompts = relationship(
        "Prompt",
        back_populates="product",
        cascade="all, delete-orphan",
    )


class PromptPack(Base):
    """
    Represents a logical pack of prompts (behavioural or Google-seeded)
    that is used for batch LLM visibility tests.

    `pack_id` mirrors the ID you already use in JSON files, e.g.:
      - auto_123_product-slug
      - google_auto_123_product-slug
    """

    __tablename__ = "prompt_packs"

    id = Column(Integer, primary_key=True, index=True)

    # External-facing identifier (used in API, filenames, etc.)
    pack_id = Column(String, unique=True, index=True, nullable=False)

    name = Column(String, nullable=False)
    category = Column(String, nullable=True)
    language = Column(String, nullable=True, default="en")
    # e.g. "behavioural", "google", "manual"
    source = Column(String, nullable=True)

    # Optional: which product this pack was generated for
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    product = relationship("Product", back_populates="prompt_packs")
    prompts = relationship(
        "Prompt",
        back_populates="pack",
        cascade="all, delete-orphan",
    )


class Prompt(Base):
    """
    Represents a single prompt string, tracked globally.

    This lets you:
      - Avoid generating duplicates in future.
      - Attach multiple LLMTest rows to the same logical prompt.
      - Analyse visibility per prompt, per pack, per product, etc.
    """

    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)

    # Raw prompt text
    text = Column(Text, nullable=False)

    # Normalised version for deduping (e.g. lowercased + trimmed)
    text_normalized = Column(Text, nullable=False, index=True)

    # Where this prompt came from:
    # "behavioural", "google", "manual", etc.
    source = Column(String, nullable=True)

    # Optional link to the pack that contains this prompt
    pack_id = Column(Integer, ForeignKey("prompt_packs.id"), nullable=True)

    # Optional link to the product this prompt is primarily about
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pack = relationship("PromptPack", back_populates="prompts")
    product = relationship("Product", back_populates="prompts")

    __table_args__ = (
      # Example uniqueness constraint:
      # Uncomment if you want to enforce "no duplicate prompt text per product"
      # UniqueConstraint(
      #     "product_id",
      #     "text_normalized",
      #     name="uq_prompt_product_text_normalized",
      # ),
    )
