"""Shared data models for model training pipeline."""
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any


class TrainingMetrics(BaseModel):
    """Training metrics at each epoch."""
    model_config = ConfigDict(populate_by_name=True)

    epoch: int = Field(..., ge=0)
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_f1: float
    learning_rate: float = Field(default=2e-5)
    timestamp: Optional[str] = None


class EvaluationMetrics(BaseModel):
    """Final evaluation metrics on test set."""
    model_config = ConfigDict(populate_by_name=True)

    accuracy: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    ndcg_at_5: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    auc_roc: Optional[float] = None


class ModelCheckpoint(BaseModel):
    """Saved model checkpoint metadata."""
    model_config = ConfigDict(populate_by_name=True)

    checkpoint_path: str = Field(..., min_length=1)
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    model_size_mb: float = Field(default=0.0)
    timestamp: Optional[str] = None


class ExportedModel(BaseModel):
    """Exported model metadata."""
    model_config = ConfigDict(populate_by_name=True)

    model_path: str = Field(..., min_length=1)
    format: str = Field(default="onnx")  # onnx, safetensors, etc.
    model_name: str = Field(default="unknown")
    input_max_length: int = Field(default=64)
    quantized: bool = Field(default=False)
    file_size_mb: float = Field(default=0.0)


class QuantizationConfig(BaseModel):
    """Configuration for model quantization."""
    model_config = ConfigDict(populate_by_name=True)

    enable_quantization: bool = Field(default=True)
    quant_type: str = Field(default="uint8")  # uint8, int8, etc.
    dynamic: bool = Field(default=False)
    opset_version: int = Field(default=15)


class HardExampleSet(BaseModel):
    """Set of hard examples for adversarial training."""
    model_config = ConfigDict(populate_by_name=True)

    examples: list[Dict[str, Any]] = Field(default_factory=list)
    count: int = Field(default=0)
    source: str = Field(default="unknown")  # mined, augmented, etc.


class TrainingResult(BaseModel):
    """Final training result."""
    model_config = ConfigDict(populate_by_name=True)

    model_path: str
    metrics: EvaluationMetrics
    training_time_seconds: float
    epochs_completed: int
    final_checkpoint: Optional[ModelCheckpoint] = None
    exported_models: list[ExportedModel] = Field(default_factory=list)
