"""Wrapper functions for running individual phases."""
import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any

from .crawlers import CrawlerConfig
from .data_prep import DataPrepConfig, DataPipeline
from .model_training import ModelTrainingConfig, ModelPipeline

logger = logging.getLogger(__name__)


def run_crawlers(config: CrawlerConfig) -> Dict[str, Any]:
    """Run Phase 1: Web crawling (dd-room + NCHU official site).

    Args:
        config: CrawlerConfig instance

    Returns:
        Dictionary with crawling results including total_properties count
    """
    from .crawlers.crawler_591 import main as crawler_591_main
    from .crawlers.crawler_ddroom import main as ddroom_main
    from .crawlers.crawler_nchu import main as nchu_main

    logger.info("Running dd-room crawler...")
    try:
        asyncio.run(ddroom_main())
    except Exception as e:
        logger.error("dd-room crawler failed: %s", e)

    logger.info("Running NCHU official site crawler...")
    try:
        asyncio.run(nchu_main())
    except Exception as e:
        logger.error("NCHU crawler failed: %s", e)

    logger.info("Running 591 crawler...")
    try:
        asyncio.run(crawler_591_main())
    except Exception as e:
        logger.error("591 crawler failed: %s", e)

    # Count rows written
    total = 0
    for csv_path in [config.output_csv, config.nchu_output_csv]:
        try:
            with open(csv_path, encoding="utf-8-sig") as f:
                total += max(0, sum(1 for _ in csv.reader(f)) - 1)
        except FileNotFoundError:
            pass

    return {
        "status": "completed",
        "total_properties": total,
    }


def run_data_prep(config: DataPrepConfig) -> Dict[str, Any]:
    """Run Phase 2: Data preparation.

    Args:
        config: DataPrepConfig instance

    Returns:
        Dictionary with data preparation results including training_samples count
    """
    pipeline = DataPipeline(config)
    pipeline.run()

    # Count actual training samples produced
    training_samples = 0
    train_path = config.processed_data_dir / "training_dataset.json"
    if train_path.exists():
        with open(train_path, encoding="utf-8") as f:
            training_samples = len(json.load(f))

    return {
        "status": "completed",
        "training_samples": training_samples,
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
