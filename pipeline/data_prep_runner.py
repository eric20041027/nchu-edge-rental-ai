"""Unified entry point for data preparation pipeline (Phase 2)."""
import logging
import sys

from pipeline.data_prep import DataPrepConfig
from pipeline.data_prep.merger import DataMerger
from pipeline.data_prep.generator import DatasetGenerator
from pipeline.data_prep.embedder import EmbeddingPrecomputer

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
    logger.info("=" * 70)
    logger.info("NCHU Rental Pipeline — Data Preparation (Phase 2)")
    logger.info("=" * 70)

    try:
        cfg = DataPrepConfig()
        cfg.validate_input_files()
        logger.info(f"Configuration: merged_csv={cfg.merged_csv}, dataset_json={cfg.dataset_json}")

        # Step 1: Merge sources
        logger.info("\n[Step 1/3] Merging and deduplicating data sources...")
        merger = DataMerger(cfg)
        merged_data = merger.run()

        # Step 2: Generate dataset
        logger.info("\n[Step 2/3] Generating training dataset...")
        generator = DatasetGenerator(cfg)
        dataset = generator.run(merged_data)

        # Step 3: Precompute embeddings
        if cfg.enable_llm_augmentation or True:  # Always precompute embeddings
            logger.info("\n[Step 3/3] Precomputing property embeddings...")
            embedder = EmbeddingPrecomputer(cfg)
            embeddings = embedder.run(merged_data)

        logger.info("\n" + "=" * 70)
        logger.info("✓ Phase 2 (Data Prep) completed successfully")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
