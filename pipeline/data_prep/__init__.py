"""Data preparation pipeline module.

Provides modularized, production-ready data processing with:
- Centralized configuration (environment-driven)
- Pydantic validation models
- Abstract base processor for consistent patterns
- Individual processor classes for each pipeline step
- Central pipeline orchestrator

Usage:
    from pipeline.data_prep import DataPrepConfig, DataPipeline

    cfg = DataPrepConfig()
    pipeline = DataPipeline(cfg)
    pipeline.run()

Or use individual processors:
    from pipeline.data_prep import DataMerger, DatasetGenerator

    merger = DataMerger(cfg)
    merged_data = merger.run()
"""

from .config import DataPrepConfig
from .models import (
    MergedRental,
    QueryPropertyPair,
    TrainingDataset,
    PropertyEmbedding,
    EmbeddingBatch,
    HardNegativeExample,
    BudgetTrap,
)
from .base import BaseProcessor
from .merger import DataMerger
from .generator import DatasetGenerator
from .augmenter import SemanticAugmenter
from .miner import HardNegativeMiner
from .embedder import EmbeddingPrecomputer
from .pipeline import DataPipeline

__all__ = [
    "DataPrepConfig",
    "MergedRental",
    "QueryPropertyPair",
    "TrainingDataset",
    "PropertyEmbedding",
    "EmbeddingBatch",
    "HardNegativeExample",
    "BudgetTrap",
    "BaseProcessor",
    "DataMerger",
    "DatasetGenerator",
    "SemanticAugmenter",
    "HardNegativeMiner",
    "EmbeddingPrecomputer",
    "DataPipeline",
]
