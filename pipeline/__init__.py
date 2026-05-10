"""NCHU AI Rental Pipeline - Modularized data processing and model training.

Three-phase modular architecture:
- Phase 1: Web Crawling (pipeline.crawlers)
- Phase 2: Data Preparation (pipeline.data_prep)
- Phase 3: Model Training (pipeline.model_training)
- Master: End-to-end orchestration (pipeline.orchestrator)
"""

from .orchestrator import PipelineOrchestrator
from .runners import run_crawlers, run_data_prep, run_model_training

__all__ = [
    "PipelineOrchestrator",
    "run_crawlers",
    "run_data_prep",
    "run_model_training",
]
