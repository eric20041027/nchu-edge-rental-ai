"""Wrapper functions for running individual phases."""
from typing import Dict, Any

from .crawlers import CrawlerConfig
from .data_prep import DataPrepConfig, DataPipeline
from .model_training import ModelTrainingConfig, ModelPipeline


def run_crawlers(config: CrawlerConfig) -> Dict[str, Any]:
    """Run Phase 1: Web crawling.

    Args:
        config: CrawlerConfig instance

    Returns:
        Dictionary with crawling results
    """
    # Phase 1 implementation would go here
    # For now, return placeholder
    return {
        "status": "completed",
        "total_properties": 0,
    }


def run_data_prep(config: DataPrepConfig) -> Dict[str, Any]:
    """Run Phase 2: Data preparation.

    Args:
        config: DataPrepConfig instance

    Returns:
        Dictionary with data preparation results
    """
    pipeline = DataPipeline(config)
    result = pipeline.run()

    return {
        "status": "completed",
        "pipeline_result": result,
        "training_samples": 0,
    }


def run_model_training(config: ModelTrainingConfig) -> Dict[str, Any]:
    """Run Phase 3: Model training.

    Args:
        config: ModelTrainingConfig instance

    Returns:
        Dictionary with training results
    """
    pipeline = ModelPipeline(config)
    result = pipeline.run()

    return {
        "status": "completed",
        "pipeline_result": result,
        "epochs_completed": result.epochs_completed,
        "model_path": result.model_path,
    }
