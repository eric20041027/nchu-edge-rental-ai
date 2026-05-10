"""Master orchestrator coordinating all three phases of the pipeline."""
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .crawlers import CrawlerConfig
from .data_prep import DataPrepConfig
from .model_training import ModelTrainingConfig
from .runners import run_crawlers, run_data_prep, run_model_training


class PipelineOrchestrator:
    """Orchestrates complete end-to-end pipeline across all phases.

    Coordinates data crawling (Phase 1), data preparation (Phase 2),
    and model training (Phase 3) into a single unified workflow.
    """

    def __init__(self, skip_phases: Optional[list] = None):
        """Initialize orchestrator.

        Args:
            skip_phases: List of phases to skip (1, 2, 3)
        """
        self.skip_phases = skip_phases or []
        self.logger = self._create_logger()
        self.results = {}
        self.start_time = None
        self.phase_times = {}

    def _create_logger(self) -> logging.Logger:
        """Create logger for orchestrator."""
        logger = logging.getLogger("PipelineOrchestrator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def run(self) -> Dict[str, Any]:
        """Execute complete end-to-end pipeline.

        Returns:
            Dictionary with results from all phases
        """
        self.start_time = time.time()
        self._log_pipeline_start()

        try:
            if 1 not in self.skip_phases:
                self._run_phase_1()

            if 2 not in self.skip_phases:
                self._run_phase_2()

            if 3 not in self.skip_phases:
                self._run_phase_3()

            total_time = time.time() - self.start_time
            self._log_pipeline_completion(total_time)

            return self._build_final_results(total_time)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def _run_phase_1(self) -> None:
        """Run Phase 1: Web Crawling."""
        self.logger.info("╔" + "═" * 58 + "╗")
        self.logger.info("║" + " " * 20 + "PHASE 1: WEB CRAWLING" + " " * 18 + "║")
        self.logger.info("╚" + "═" * 58 + "╝")

        phase_start = time.time()

        try:
            self.logger.info("Initializing crawler configuration")
            config = CrawlerConfig()

            self.logger.info("Starting web crawling process")
            result = run_crawlers(config)

            self.results["phase_1"] = result
            phase_time = time.time() - phase_start
            self.phase_times["phase_1"] = phase_time

            self.logger.info(f"Phase 1 completed in {phase_time:.2f}s")
            self.logger.info(f"Crawled properties: {result.get('total_properties', 0)}")

        except Exception as e:
            self.logger.error(f"Phase 1 failed: {e}")
            raise

    def _run_phase_2(self) -> None:
        """Run Phase 2: Data Preparation."""
        self.logger.info("╔" + "═" * 58 + "╗")
        self.logger.info("║" + " " * 17 + "PHASE 2: DATA PREPARATION" + " " * 17 + "║")
        self.logger.info("╚" + "═" * 58 + "╝")

        phase_start = time.time()

        try:
            self.logger.info("Initializing data preparation configuration")
            config = DataPrepConfig()

            self.logger.info("Validating input files")
            config.validate_input_files()

            self.logger.info("Starting data preparation pipeline")
            result = run_data_prep(config)

            self.results["phase_2"] = result
            phase_time = time.time() - phase_start
            self.phase_times["phase_2"] = phase_time

            self.logger.info(f"Phase 2 completed in {phase_time:.2f}s")
            self.logger.info(f"Training pairs generated: {result.get('training_samples', 0)}")

        except Exception as e:
            self.logger.error(f"Phase 2 failed: {e}")
            raise

    def _run_phase_3(self) -> None:
        """Run Phase 3: Model Training."""
        self.logger.info("╔" + "═" * 58 + "╗")
        self.logger.info("║" + " " * 17 + "PHASE 3: MODEL TRAINING" + " " * 18 + "║")
        self.logger.info("╚" + "═" * 58 + "╝")

        phase_start = time.time()

        try:
            self.logger.info("Initializing model training configuration")
            config = ModelTrainingConfig()

            self.logger.info("Validating training data files")
            config.validate_input_files()

            self.logger.info("Starting model training pipeline")
            result = run_model_training(config)

            self.results["phase_3"] = result
            phase_time = time.time() - phase_start
            self.phase_times["phase_3"] = phase_time

            self.logger.info(f"Phase 3 completed in {phase_time:.2f}s")
            self.logger.info(f"Training epochs: {result.get('epochs_completed', 0)}")

        except Exception as e:
            self.logger.error(f"Phase 3 failed: {e}")
            raise

    def _build_final_results(self, total_time: float) -> Dict[str, Any]:
        """Build final results object.

        Args:
            total_time: Total pipeline execution time

        Returns:
            Dictionary with all results
        """
        return {
            "status": "completed",
            "total_time_seconds": total_time,
            "phase_times": self.phase_times,
            "phase_1_result": self.results.get("phase_1"),
            "phase_2_result": self.results.get("phase_2"),
            "phase_3_result": self.results.get("phase_3"),
        }

    def _log_pipeline_start(self) -> None:
        """Log pipeline start information."""
        self.logger.info("")
        self.logger.info("╔" + "═" * 58 + "╗")
        self.logger.info("║" + " " * 12 + "END-TO-END PIPELINE (PHASE 1-2-3)" + " " * 13 + "║")
        self.logger.info("╚" + "═" * 58 + "╝")
        self.logger.info("")
        self.logger.info("Phases to execute:")
        if 1 not in self.skip_phases:
            self.logger.info("  ✓ Phase 1: Web Crawling")
        else:
            self.logger.info("  ✗ Phase 1: Web Crawling (SKIPPED)")
        if 2 not in self.skip_phases:
            self.logger.info("  ✓ Phase 2: Data Preparation")
        else:
            self.logger.info("  ✗ Phase 2: Data Preparation (SKIPPED)")
        if 3 not in self.skip_phases:
            self.logger.info("  ✓ Phase 3: Model Training")
        else:
            self.logger.info("  ✗ Phase 3: Model Training (SKIPPED)")
        self.logger.info("")

    def _log_pipeline_completion(self, total_time: float) -> None:
        """Log pipeline completion information."""
        self.logger.info("")
        self.logger.info("╔" + "═" * 58 + "╗")
        self.logger.info("║" + " " * 13 + "PIPELINE EXECUTION COMPLETED" + " " * 17 + "║")
        self.logger.info("╚" + "═" * 58 + "╝")
        self.logger.info("")
        self.logger.info("Timing Breakdown:")
        for phase, duration in self.phase_times.items():
            self.logger.info(f"  {phase}: {duration:.2f}s")
        self.logger.info(f"  Total: {total_time:.2f}s")
        self.logger.info("")
