"""Configuration management for model training pipeline."""
import os
from pathlib import Path
from typing import Optional

from .models import QuantizationConfig


class ModelTrainingConfig:
    """Centralized configuration for all model training steps."""

    def __init__(self):
        # Project root
        self.project_root = Path(__file__).parent.parent.parent
        self.data_root = self.project_root / "data"
        self.processed_data_dir = self.data_root / "processed"
        self.checkpoint_dir = self.project_root / ".checkpoints" / "model_training"

        # Use D drive if available (has more space), otherwise fall back to project root
        d_renting_models = Path("D:/renting_models")
        self.saved_models_dir = d_renting_models if d_renting_models.exists() else self.project_root / "saved_models"

        self.frontend_models_dir = self.project_root / "frontend" / "models" / "custom_onnx_model_dir"

        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_models_dir.mkdir(parents=True, exist_ok=True)
        self.frontend_models_dir.mkdir(parents=True, exist_ok=True)

        # Model checkpoint paths
        self.model_checkpoint = self._env_string("MODEL_CHECKPOINT", "hfl/rbt6")
        self.saved_model_dir = self._env_path("SAVED_MODEL_DIR", self.saved_models_dir / "rbt6_finetuned")

        # Bi-encoder (T2) paths/settings — CE 同源 base, separate save dir so the
        # cross-encoder checkpoint above is never overwritten. The query encoder
        # exported by T3 reads from bi_encoder_saved_dir.
        self.bi_encoder_saved_dir = self._env_path(
            "BI_ENCODER_SAVED_DIR", self.saved_models_dir / "rbt6_bi_encoder"
        )
        # Contrastive-learning temperature for the InfoNCE / MNRL objective.
        # Lower temperature = sharper softmax over candidates. 0.05 is the
        # sentence-transformers MNRL default scaled cosine convention (scale=20).
        self.bi_encoder_temperature = self._env_float("BI_ENCODER_TEMPERATURE", 0.05)
        # T3 query-encoder ONNX output dir (FP32 + quantized + tokenizer). Kept
        # SEPARATE from custom_onnx_model_dir (the cross-encoder) so the CE files
        # are never touched. The frontend loads the query encoder from here; the
        # tokenizer (tokenizer.json / vocab.txt) is co-located so on-device query
        # encoding uses the SAME vocab the bi-encoder was trained against.
        self.bi_encoder_onnx_dir = self._env_path(
            "BI_ENCODER_ONNX_DIR", self.project_root / "frontend" / "models" / "bi_encoder_dir"
        )
        self.bi_encoder_onnx_path = self.bi_encoder_onnx_dir / "bi_encoder.onnx"
        self.bi_encoder_quant_path = self.bi_encoder_onnx_dir / "bi_encoder_quant.onnx"
        self.onnx_output_path = self._env_path("ONNX_OUTPUT_PATH", self.frontend_models_dir / "my_custom_model.onnx")
        self.quantized_model_path = self._env_path("QUANTIZED_MODEL_PATH", self.frontend_models_dir / "my_custom_model_quant.onnx")

        # Input data paths — prefer recommendation_*.json (has property text) over generated datasets
        _default_train = (
            self.processed_data_dir / "recommendation_train.json"
            if (self.processed_data_dir / "recommendation_train.json").exists()
            else self.processed_data_dir / "training_dataset.json"
        )
        _default_val = (
            self.processed_data_dir / "recommendation_dev.json"
            if (self.processed_data_dir / "recommendation_dev.json").exists()
            else self.processed_data_dir / "validation_dataset.json"
        )
        _default_test = (
            self.processed_data_dir / "recommendation_test.json"
            if (self.processed_data_dir / "recommendation_test.json").exists()
            else self.processed_data_dir / "test_dataset.json"
        )
        self.train_data_path = self._env_path("TRAIN_DATA_PATH", _default_train)
        self.val_data_path = self._env_path("VAL_DATA_PATH", _default_val)
        self.test_data_path = self._env_path("TEST_DATA_PATH", _default_test)

        # Training hyperparameters
        self.max_length = self._env_int("MAX_LENGTH", 64)
        self.batch_size = self._env_int("BATCH_SIZE", 32)
        self.num_epochs = self._env_int("NUM_EPOCHS", 10)
        self.learning_rate = self._env_float("LEARNING_RATE", 2e-5)
        self.warmup_steps = self._env_int("WARMUP_STEPS", 500)
        self.early_stopping_patience = self._env_int("EARLY_STOPPING_PATIENCE", 3)
        self.fp16 = self._env_bool("FP16", False)  # Disabled: FGMTrainer incompatible with fp16 scaler
        self.random_seed = self._env_int("RANDOM_SEED", 42)

        # Training features
        self.enable_adversarial_training = self._env_bool("ENABLE_ADVERSARIAL_TRAINING", True)
        self.adversarial_epsilon = self._env_float("ADVERSARIAL_EPSILON", 1.0)
        self.enable_hard_example_mining = self._env_bool("ENABLE_HARD_MINING", True)

        # Evaluation parameters
        self.eval_batch_size = self._env_int("EVAL_BATCH_SIZE", 64)
        self.eval_sample_size = self._env_int("EVAL_SAMPLE_SIZE", 10000)  # cover full test set

        # ONNX export parameters
        self.onnx_opset_version = self._env_int("ONNX_OPSET_VERSION", 15)
        self.enable_quantization = self._env_bool("ENABLE_QUANTIZATION", True)

        # Quantization config
        self.quantization_config = QuantizationConfig(
            enable_quantization=self.enable_quantization,
            quant_type=self._env_string("QUANT_TYPE", "uint8"),
            dynamic=self._env_bool("QUANT_DYNAMIC", False),
            opset_version=self.onnx_opset_version,
        )

    @staticmethod
    def _env_path(key: str, default: Path) -> Path:
        """Read Path from environment or use default."""
        return Path(os.getenv(key, str(default)))

    @staticmethod
    def _env_string(key: str, default: str) -> str:
        """Read string from environment or use default."""
        return os.getenv(key, default)

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        """Read int from environment or use default."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _env_float(key: str, default: float) -> float:
        """Read float from environment or use default."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _env_bool(key: str, default: bool) -> bool:
        """Read bool from environment or use default."""
        val = os.getenv(key, str(default)).lower()
        return val in ("true", "1", "yes", "on")

    def validate_input_files(self) -> bool:
        """Check that required input files exist."""
        required_files = [self.train_data_path, self.val_data_path, self.test_data_path]
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        return True
