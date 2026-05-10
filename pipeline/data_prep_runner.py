"""Unified entry point for data preparation pipeline (Phase 2)."""
import logging
import sys

from pipeline.data_prep import DataPrepConfig, DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data_prep.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Execute data preparation pipeline."""
    try:
        cfg = DataPrepConfig()
        cfg.validate_input_files()

        # Run complete pipeline
        pipeline = DataPipeline(cfg)
        pipeline.run(
            enable_augmentation=cfg.enable_llm_augmentation,
            enable_mining=cfg.enable_hard_mining,
            enable_embeddings=True,
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
