"""Base processor class for data preparation pipeline."""
import abc
import logging
import pickle
from pathlib import Path
from typing import Any, TypeVar, Optional

from .config import DataPrepConfig

T = TypeVar("T")


class BaseProcessor(abc.ABC):
    """Abstract base class for all data prep processors."""

    def __init__(self, config: DataPrepConfig, logger: Optional[logging.Logger] = None):
        """Initialize processor with config and logger.

        Args:
            config: DataPrepConfig instance with all settings
            logger: Optional logger instance; creates default if not provided
        """
        self.config = config
        self.logger = logger or self._create_logger()

    def _create_logger(self) -> logging.Logger:
        """Create a logger for this processor."""
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
        """Execute this processing step.

        Returns:
            Processed data (type depends on specific processor)
        """
        pass

    def save_checkpoint(self, data: Any, name: str) -> Path:
        """Save intermediate checkpoint to disk.

        Args:
            data: Data to save
            name: Checkpoint name (e.g., "merger_output")

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
            name: Checkpoint name (e.g., "merger_output")

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
        """Log a processing step."""
        self.logger.info(f"→ {message}")

    def log_result(self, key: str, value: Any) -> None:
        """Log a processing result."""
        self.logger.info(f"  {key}: {value}")
