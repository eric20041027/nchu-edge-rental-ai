"""Shared data models for data preparation pipeline."""
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class MergedRental(BaseModel):
    """Validated merged and deduplicated rental record."""
    model_config = ConfigDict(populate_by_name=True)

    url: str = Field(..., min_length=1)
    address: str = Field(default="")
    layout: str = Field(default="")
    property_type: str = Field(default="")
    size: str = Field(default="")
    rent: str = Field(default="")
    available_rooms: str = Field(default="")
    deposit: str = Field(default="")
    safety_cert: str = Field(default="")
    floor: str = Field(default="")
    contact_name: str = Field(default="")
    contact_phone: str = Field(default="")
    furniture: str = Field(default="")
    rent_includes: str = Field(default="")
    extra_fees: str = Field(default="")
    safety_mgmt: str = Field(default="")
    fire_safety: str = Field(default="")
    notes: str = Field(default="")
    image_url: str = Field(default="")
    distance_km: str = Field(default="")
    walk_mins: str = Field(default="")
    scooter_mins: str = Field(default="")


class QueryPropertyPair(BaseModel):
    """Single query-property training example."""
    query: str = Field(..., min_length=1)
    property_id: str = Field(..., min_length=1)
    property_text: str = Field(default="")  # Human-readable property description for cross-encoder
    is_match: bool
    score: Optional[int] = None  # 0-3 graded relevance


class TrainingDataset(BaseModel):
    """Training dataset with query-property pairs."""
    train_pairs: list[QueryPropertyPair]
    val_pairs: list[QueryPropertyPair]
    test_pairs: list[QueryPropertyPair]
    metadata: dict = Field(default_factory=dict)


class PropertyEmbedding(BaseModel):
    """Precomputed property embedding vector."""
    property_id: str = Field(..., min_length=1)
    text: str = Field(default="")
    embedding: list[float]
    model_name: str = Field(default="unknown")


class EmbeddingBatch(BaseModel):
    """Batch of property embeddings."""
    embeddings: list[PropertyEmbedding]
    count: int
    model_name: str = Field(default="unknown")


class HardNegativeExample(BaseModel):
    """Hard negative example for active learning."""
    query: str = Field(..., min_length=1)
    property_id: str = Field(..., min_length=1)
    model_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="")


class BudgetTrap(BaseModel):
    """Budget trap negative example."""
    query: str = Field(..., min_length=1)
    property_id: str = Field(..., min_length=1)
    user_budget: float
    property_rent: float
