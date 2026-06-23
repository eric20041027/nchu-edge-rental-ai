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

from .config import DataPrepConfig  # 純 stdlib,輕量,保留 eager

# Lazy import(PEP 562):models(pydantic)與各 processor 延遲載入,讓無 pydantic 的
# 本機 dev box 仍能用輕量子模組(如 precompute_embeddings,純 stdlib)。
_LAZY = {
    "MergedRental": ".models", "QueryPropertyPair": ".models", "TrainingDataset": ".models",
    "PropertyEmbedding": ".models", "EmbeddingBatch": ".models",
    "HardNegativeExample": ".models", "BudgetTrap": ".models",
    "BaseProcessor": ".base", "DataMerger": ".merger", "DatasetGenerator": ".generator",
    "SemanticAugmenter": ".augmenter", "HardNegativeMiner": ".miner", "SilverLabeler": ".labeler",
    "CommuteDataUpdater": ".commute_updater", "BudgetTrapGenerator": ".budget_generator",
    "EmbeddingPrecomputer": ".embedder", "DataPipeline": ".pipeline",
}


def __getattr__(name: str):  # PEP 562
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)


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
    "SilverLabeler",
    "CommuteDataUpdater",
    "BudgetTrapGenerator",
    "EmbeddingPrecomputer",
    "DataPipeline",
]
