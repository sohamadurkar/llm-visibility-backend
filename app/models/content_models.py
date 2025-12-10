# app/models/content_models.py

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import relationship

from app.db.engine import Base


class ContentArticle(Base):
    """
    Long-form content article for a specific product and angle.

    One row ~= one article such as:
      - Fit & Comfort guide
      - Occasion-specific guide
      - Budget/value angle
      - etc.
    """
    __tablename__ = "content_articles"

    id = Column(Integer, primary_key=True, index=True)

    # Product this article belongs to
    product_id = Column(
        Integer,
        ForeignKey("products.id"),
        nullable=False,
        index=True,
    )

    # Angle key, e.g. "fit_comfort", "occasion", "budget_value", ...
    angle_key = Column(String, nullable=False, index=True)

    # Product-specific human-readable angle label, e.g.
    # "Supportive heels for all-day weddings"
    angle_label = Column(String, nullable=True)

    # Human-readable title, e.g. "Fit & Comfort Guide for Velvet Mary Jane Pumps"
    title = Column(String, nullable=False)

    # Public slug, unique within this tenant schema
    # e.g. "product-12-fit-comfort-guide"
    slug = Column(String, unique=True, index=True, nullable=False)

    # Optional meta description for SEO
    meta_description = Column(String, nullable=True)

    # Main content body in HTML (we'll ask the LLM to output HTML)
    content_html = Column(Text, nullable=False)

    # Publication flags
    is_published = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    product = relationship("Product", backref="content_articles")