"""Model training pipeline module.

Provides modularized, production-ready model training with:
- Centralized configuration (environment-driven)
- Pydantic validation models
- Abstract base trainer for consistent patterns
- Individual trainer classes for each training step
- Central pipeline orchestrator

Usage:
    from pipeline.model_training import ModelTrainingConfig, ModelPipeline

    cfg = ModelTrainingConfig()
    pipeline = ModelPipeline(cfg)
    pipeline.run()

Or use individual trainers:
    from pipeline.model_training import ModelTrainer, Exporter

    trainer = ModelTrainer(cfg)
    model = trainer.run()

    exporter = Exporter(cfg)
    exporter.run(model)
"""

from .config import ModelTrainingConfig
from .models import (
    TrainingMetrics,
    EvaluationMetrics,
    ModelCheckpoint,
    ExportedModel,
    QuantizationConfig,
    HardExampleSet,
    TrainingResult,
)
from .base import BaseTrainer
from .trainer import ModelTrainer
from .exporter import Exporter
from .quantizer import Quantizer
from .evaluator import Evaluator
from .pipeline import ModelPipeline

__all__ = [
    "ModelTrainingConfig",
    "TrainingMetrics",
    "EvaluationMetrics",
    "ModelCheckpoint",
    "ExportedModel",
    "QuantizationConfig",
    "HardExampleSet",
    "TrainingResult",
    "BaseTrainer",
    "ModelTrainer",
    "Exporter",
    "Quantizer",
    "Evaluator",
    "ModelPipeline",
]
