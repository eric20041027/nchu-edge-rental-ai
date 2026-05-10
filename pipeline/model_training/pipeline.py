"""Model training pipeline orchestrator."""
import time
import logging
from pathlib import Path
from typing import Optional

from .config import ModelTrainingConfig
from .trainer import ModelTrainer
from .exporter import Exporter
from .quantizer import Quantizer
from .evaluator import Evaluator
from .models import TrainingResult


class ModelPipeline:
    """Orchestrates complete model training pipeline.

    Coordinates training, evaluation, export, and quantization steps
    with optional skip capability for each step.
    """

    def __init__(self, config: ModelTrainingConfig):
        """Initialize pipeline with configuration.

        Args:
            config: ModelTrainingConfig instance
        """
        self.config = config
        self.logger = self._create_logger()
        self.model = None
        self.tokenizer = None
        self.training_metrics = None
        self.eval_metrics = None
        self.exported_models = []

    def _create_logger(self) -> logging.Logger:
        """Create logger for pipeline."""
        logger = logging.getLogger("ModelPipeline")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def run(self, skip_steps: Optional[list] = None) -> TrainingResult:
        """Execute complete model training pipeline.

        Args:
            skip_steps: List of steps to skip (trainer, exporter, quantizer, evaluator)

        Returns:
            TrainingResult with all pipeline outputs
        """
        skip_steps = skip_steps or []
        start_time = time.time()

        self._log_pipeline_start()

        try:
            if "trainer" not in skip_steps:
                self._run_trainer()

            if "evaluator" not in skip_steps:
                self._run_evaluator()

            if "exporter" not in skip_steps:
                self._run_exporter()

            if "quantizer" not in skip_steps:
                self._run_quantizer()

            elapsed_time = time.time() - start_time
            self.logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

            return self._build_result(elapsed_time)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def _run_trainer(self) -> None:
        """Run model training step."""
        self.logger.info("═" * 60)
        self.logger.info("→ STEP 1: Model Training")
        self.logger.info("═" * 60)

        trainer = ModelTrainer(self.config)
        result = trainer.run()

        self.model = result["model"]
        self.tokenizer = result["tokenizer"]
        self.training_metrics = result["training_metrics"]

        self.logger.info(f"Model checkpoint: {result['model_path']}")

    def _run_evaluator(self) -> None:
        """Run evaluation step."""
        if self.model is None or self.tokenizer is None:
            self.logger.warning("Skipping evaluator: model not trained")
            return

        self.logger.info("═" * 60)
        self.logger.info("→ STEP 2: Model Evaluation")
        self.logger.info("═" * 60)

        evaluator = Evaluator(self.config)
        self.eval_metrics = evaluator.run(self.model, self.tokenizer)

    def _run_exporter(self) -> None:
        """Run ONNX export step."""
        if self.model is None or self.tokenizer is None:
            self.logger.warning("Skipping exporter: model not trained")
            return

        self.logger.info("═" * 60)
        self.logger.info("→ STEP 3: ONNX Export")
        self.logger.info("═" * 60)

        exporter = Exporter(self.config)
        exported_model = exporter.run(self.model, self.tokenizer)
        self.exported_models.append(exported_model)

    def _run_quantizer(self) -> None:
        """Run model quantization step."""
        if not self.config.onnx_output_path.exists():
            self.logger.warning("Skipping quantizer: ONNX model not found")
            return

        if not self.config.enable_quantization:
            self.logger.info("Quantization disabled in config")
            return

        self.logger.info("═" * 60)
        self.logger.info("→ STEP 4: Model Quantization")
        self.logger.info("═" * 60)

        quantizer = Quantizer(self.config)
        quantized_model = quantizer.run(str(self.config.onnx_output_path))
        self.exported_models.append(quantized_model)

    def _build_result(self, elapsed_time: float) -> TrainingResult:
        """Build final training result.

        Args:
            elapsed_time: Total pipeline execution time in seconds

        Returns:
            TrainingResult with all pipeline outputs
        """
        return TrainingResult(
            model_path=str(self.config.saved_model_dir),
            metrics=self.eval_metrics or self._get_default_metrics(),
            training_time_seconds=elapsed_time,
            epochs_completed=self.config.num_epochs,
            final_checkpoint=None,
            exported_models=self.exported_models,
        )

    def _get_default_metrics(self):
        """Get default evaluation metrics if not computed."""
        from .models import EvaluationMetrics

        return EvaluationMetrics(
            accuracy=0.0,
            f1_score=0.0,
            precision=0.0,
            recall=0.0,
            ndcg_at_5=0.0,
            mrr=0.0,
        )

    def _log_pipeline_start(self) -> None:
        """Log pipeline start information."""
        self.logger.info("╔" + "═" * 58 + "╗")
        self.logger.info("║" + " " * 18 + "Model Training Pipeline" + " " * 18 + "║")
        self.logger.info("╚" + "═" * 58 + "╝")
        self.logger.info(f"Model: {self.config.model_checkpoint}")
        self.logger.info(f"Epochs: {self.config.num_epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        self.logger.info("")
