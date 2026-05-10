"""Unified entry point for Phase 3 model training pipeline."""
import logging
from pipeline.model_training import ModelTrainingConfig, ModelPipeline


def main():
    """Run the complete model training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    )

    logger = logging.getLogger("model_training_runner")

    try:
        logger.info("Initializing model training configuration")
        config = ModelTrainingConfig()

        logger.info("Validating input files")
        config.validate_input_files()

        logger.info("Starting model training pipeline")
        pipeline = ModelPipeline(config)

        result = pipeline.run()

        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║" + " " * 15 + "Pipeline Execution Completed" + " " * 15 + "║")
        logger.info("╚" + "═" * 58 + "╝")
        logger.info(f"Model path: {result.model_path}")
        logger.info(f"Training time: {result.training_time_seconds:.2f}s")
        logger.info(f"Epochs completed: {result.epochs_completed}")
        logger.info(f"Exported models: {len(result.exported_models)}")

        if result.metrics:
            logger.info("Final evaluation metrics:")
            logger.info(f"  Accuracy: {result.metrics.accuracy:.4f}")
            logger.info(f"  F1 Score: {result.metrics.f1_score:.4f}")
            logger.info(f"  NDCG@5: {result.metrics.ndcg_at_5:.4f}")
            logger.info(f"  MRR: {result.metrics.mrr:.4f}")

        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
