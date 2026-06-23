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

# Lazy import(PEP 562):重依賴子模組(crawlers→pydantic、ner_model→torch 等)延遲到
# 真正取用時才載入,讓本機 CPU dev box(無 pydantic/torch)仍能用輕量子模組,如
# `python -m pipeline.build_frontend_data`(只需 precompute_embeddings,純 stdlib)。
_LAZY = {
    "PipelineOrchestrator": ".orchestrator",
    "run_crawlers": ".runners",
    "run_data_prep": ".runners",
    "run_model_training": ".runners",
    "OSRMClient": ".osrm_client",
    "HardConstraintFilter": ".constraints",
    "ParsedQuery": ".constraints",
    "LifestyleMapper": ".data_prep.lifestyle_mapper",
    "NERConfig": ".ner_model",
    "NERTrainer": ".ner_model",
    "NERPredictor": ".ner_model",
}


def __getattr__(name: str):  # PEP 562
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)


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
