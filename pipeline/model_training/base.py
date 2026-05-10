"""Base trainer class for model training pipeline."""
import abc
import logging
import pickle
from pathlib import Path
from typing import Any, TypeVar, Optional

from .config import ModelTrainingConfig

T = TypeVar("T")


class BaseTrainer(abc.ABC):
    """Abstract base class for all model training components."""

    def __init__(self, config: ModelTrainingConfig, logger: Optional[logging.Logger] = None):
        """Initialize trainer with config and logger.

        Args:
            config: ModelTrainingConfig instance with all settings
            logger: Optional logger instance; creates default if not provided
        """
        self.config = config
        self.logger = logger or self._create_logger()

    def _create_logger(self) -> logging.Logger:
        """Create a logger for this trainer."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @abc.abstractmethod
    def run(self) -> Any:
        """Execute this training step.

        Returns:
            Result of training (type depends on specific trainer)
        """
        pass

    def save_checkpoint(self, data: Any, name: str) -> Path:
        """Save intermediate checkpoint to disk.

        Args:
            data: Data to save (model, metrics, etc.)
            name: Checkpoint name (e.g., "epoch_5")

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.config.checkpoint_dir / f"{name}.pkl"
        self.logger.info(f"Saving checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(data, f)
        return checkpoint_path

    def load_checkpoint(self, name: str) -> Any:
        """Load intermediate checkpoint from disk.

        Args:
            name: Checkpoint name (e.g., "epoch_5")

        Returns:
            Loaded data

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = self.config.checkpoint_dir / f"{name}.pkl"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

    def checkpoint_exists(self, name: str) -> bool:
        """Check if checkpoint exists.

        Args:
            name: Checkpoint name

        Returns:
            True if checkpoint file exists
        """
        return (self.config.checkpoint_dir / f"{name}.pkl").exists()

    def log_step(self, message: str) -> None:
        """Log a training step."""
        self.logger.info(f"→ {message}")

    def log_result(self, key: str, value: Any) -> None:
        """Log a training result."""
        self.logger.info(f"  {key}: {value}")

    def log_metrics(self, metrics: dict) -> None:
        """Log training metrics."""
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
