from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
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
