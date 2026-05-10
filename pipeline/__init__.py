"""NCHU AI Rental Pipeline - Modularized data processing and model training.

Three-phase modular architecture:
- Phase 1: Web Crawling (pipeline.crawlers)
- Phase 2: Data Preparation (pipeline.data_prep)
- Phase 3: Model Training (pipeline.model_training)
- Master: End-to-end orchestration (pipeline.orchestrator)

Feature modules:
- pipeline.constraints       — Hard constraint filtering (B3)
- pipeline.data_prep.lifestyle_mapper — Lifestyle intent inference (B2)
- pipeline.osrm_client       — Real road-network commute times (B4)
- pipeline.ner_model         — NER entity extraction (B1)
"""

from .orchestrator import PipelineOrchestrator
from .runners import run_crawlers, run_data_prep, run_model_training
from .osrm_client import OSRMClient
from .constraints import HardConstraintFilter, ParsedQuery
from .data_prep.lifestyle_mapper import LifestyleMapper
from .ner_model import NERConfig, NERTrainer, NERPredictor

__all__ = [
    "PipelineOrchestrator",
    "run_crawlers",
    "run_data_prep",
    "run_model_training",
    "OSRMClient",
    "HardConstraintFilter",
    "ParsedQuery",
    "LifestyleMapper",
    "NERConfig",
    "NERTrainer",
    "NERPredictor",
]
