"""Orchestrator for complete data preparation pipeline."""
import logging
from typing import Optional

import pandas as pd

from .config import DataPrepConfig
from .merger import DataMerger
from .generator import DatasetGenerator
from .augmenter import SemanticAugmenter
from .miner import HardNegativeMiner
from .embedder import EmbeddingPrecomputer


class DataPipeline:
    """Coordinates all data preparation steps."""

    def __init__(self, config: Optional[DataPrepConfig] = None):
        """Initialize pipeline with config.

        Args:
            config: DataPrepConfig instance (creates default if None)
        """
        self.config = config or DataPrepConfig()
        self.logger = logging.getLogger(__name__)
        self.merged_data = None
        self.dataset = None

    def run(
        self,
        steps: Optional[list[str]] = None,
        enable_augmentation: bool = True,
        enable_mining: bool = True,
        enable_embeddings: bool = True,
    ) -> None:
        """Execute data prep pipeline.

        Args:
            steps: List of steps to run (default: all)
            enable_augmentation: Whether to run LLM augmentation
            enable_mining: Whether to run hard negative mining
            enable_embeddings: Whether to precompute embeddings
        """
        if steps is None:
            steps = ["merge", "generate", "augment", "mine", "embed"]

        self.logger.info("=" * 70)
        self.logger.info("Data Preparation Pipeline")
        self.logger.info("=" * 70)

        try:
            # Step 1: Merge
            if "merge" in steps:
                self._run_merge()

            # Step 2: Generate
            if "generate" in steps:
                self._run_generate()

            # Step 3: Augment (Optional)
            if "augment" in steps and enable_augmentation:
                self._run_augment()

            # Step 4: Mine hard negatives (Optional)
            if "mine" in steps and enable_mining:
                self._run_mine()

            # Step 5: Embeddings (Optional)
            if "embed" in steps and enable_embeddings:
                self._run_embed()

            self.logger.info("\n" + "=" * 70)
            self.logger.info("✓ Data Preparation Complete")
            self.logger.info("=" * 70)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _run_merge(self) -> None:
        """Execute merge step."""
        self.logger.info("\n[Step 1/5] Merging data sources...")
        merger = DataMerger(self.config)
        self.merged_data = merger.run()

    def _run_generate(self) -> None:
        """Execute dataset generation step."""
        self.logger.info("\n[Step 2/5] Generating training dataset...")
        generator = DatasetGenerator(self.config)
        self.dataset = generator.run(self.merged_data)

    def _run_augment(self) -> None:
        """Execute LLM augmentation step."""
        self.logger.info("\n[Step 3/5] Augmenting with LLM...")
        if not self.config.llm_api_key:
            self.logger.warning("LLM API key not set. Skipping augmentation.")
            return

        augmenter = SemanticAugmenter(self.config)
        augmented = augmenter.run(target_count=1000)
        self.logger.info(f"Generated {len(augmented)} augmented samples")

    def _run_mine(self) -> None:
        """Execute hard negative mining step."""
        self.logger.info("\n[Step 4/5] Mining hard negatives...")
        miner = HardNegativeMiner(self.config)
        mined = miner.run()
        self.logger.info(f"Mined {len(mined)} hard negatives")

    def _run_embed(self) -> None:
        """Execute embedding precomputation step."""
        self.logger.info("\n[Step 5/5] Precomputing embeddings...")
        embedder = EmbeddingPrecomputer(self.config)
        batch = embedder.run(self.merged_data)
        self.logger.info(f"Precomputed {batch.count} embeddings")

    def run_step(self, step_name: str) -> None:
        """Run a single step by name.

        Args:
            step_name: One of 'merge', 'generate', 'augment', 'mine', 'embed'
        """
        if step_name == "merge":
            self._run_merge()
        elif step_name == "generate":
            self._run_generate()
        elif step_name == "augment":
            self._run_augment()
        elif step_name == "mine":
            self._run_mine()
        elif step_name == "embed":
            self._run_embed()
        else:
            raise ValueError(f"Unknown step: {step_name}")
