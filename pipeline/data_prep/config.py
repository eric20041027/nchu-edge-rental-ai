"""Configuration management for data preparation pipeline."""
import os
from pathlib import Path
from typing import Optional


class DataPrepConfig:
    """Centralized configuration for all data prep steps."""

    def __init__(self):
        # Project root
        self.project_root = Path(__file__).parent.parent.parent
        self.data_root = self.project_root / "data"
        self.raw_data_dir = self.data_root / "raw"
        self.processed_data_dir = self.data_root / "processed"
        self.checkpoint_dir = self.project_root / ".checkpoints" / "data_prep"

        # Ensure directories exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Input files for merge
        self.main_csv = self._env_path("DATA_PREP_MAIN_CSV", self.raw_data_dir / "nchu_rental_info.csv")
        self.official_csv = self._env_path("DATA_PREP_OFFICIAL_CSV", self.raw_data_dir / "nchu_official_raw.csv")

        # Output files
        self.merged_csv = self._env_path("DATA_PREP_MERGED_CSV", self.processed_data_dir / "merged_rentals.csv")
        self.dataset_json = self._env_path("DATA_PREP_DATASET_JSON", self.processed_data_dir / "training_dataset.json")
        self.embeddings_pkl = self._env_path("DATA_PREP_EMBEDDINGS_PKL", self.processed_data_dir / "property_embeddings.pkl")

        # Dataset generation params
        self.queries_per_property = self._env_int("DATA_PREP_QUERIES_PER_PROPERTY", 3)
        self.train_split = self._env_float("DATA_PREP_TRAIN_SPLIT", 0.7)
        self.val_split = self._env_float("DATA_PREP_VAL_SPLIT", 0.15)
        self.test_split = self._env_float("DATA_PREP_TEST_SPLIT", 0.15)
        self.random_seed = self._env_int("DATA_PREP_RANDOM_SEED", 42)
        self.query_dedup = self._env_bool("DATA_PREP_QUERY_DEDUP", True)

        # Embedding params
        self.embedding_model = self._env_string("DATA_PREP_EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L6-v2")
        self.embedding_batch_size = self._env_int("DATA_PREP_EMBEDDING_BATCH_SIZE", 32)
        self.embedding_device = self._env_string("DATA_PREP_EMBEDDING_DEVICE", "cpu")

        # LLM augmentation params
        self.enable_llm_augmentation = self._env_bool("DATA_PREP_ENABLE_LLM", True)
        self.llm_batch_size = self._env_int("DATA_PREP_LLM_BATCH_SIZE", 5)
        self.llm_api_key = os.getenv("ANTHROPIC_API_KEY", "")

        # Hard negative mining params
        self.enable_hard_mining = self._env_bool("DATA_PREP_ENABLE_HARD_MINING", True)
        self.hard_negative_ratio = self._env_float("DATA_PREP_HARD_NEGATIVE_RATIO", 0.2)

        # Commute data params
        self.enable_commute_update = self._env_bool("DATA_PREP_ENABLE_COMMUTE", True)
        self.commute_cache_dir = self.checkpoint_dir / "commute_cache"
        self.commute_cache_dir.mkdir(parents=True, exist_ok=True)

        # Budget trap params
        self.enable_budget_traps = self._env_bool("DATA_PREP_ENABLE_BUDGET_TRAPS", True)
        self.budget_trap_count = self._env_int("DATA_PREP_BUDGET_TRAP_COUNT", 500)

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
        if not self.main_csv.exists():
            raise FileNotFoundError(f"Main CSV not found: {self.main_csv}")
        if not self.official_csv.exists():
            raise FileNotFoundError(f"Official CSV not found: {self.official_csv}")
        return True
