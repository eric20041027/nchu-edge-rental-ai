"""Orchestrator for complete data preparation pipeline."""
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DataPrepConfig
from .merger import DataMerger
from .generator import DatasetGenerator
from .augmenter import SemanticAugmenter
from .miner import HardNegativeMiner
from .commute_updater import CommuteDataUpdater
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
            steps = ["merge", "commute", "generate", "augment", "mine", "embed"]

        self.logger.info("=" * 70)
        self.logger.info("Data Preparation Pipeline")
        self.logger.info("=" * 70)

        try:
            # Step 1: Merge
            if "merge" in steps:
                self._run_merge()

            # Step 2: Commute data update (after merge, before generate)
            if "commute" in steps and self.config.enable_commute_update:
                self._run_commute()

            # Step 3: Generate
            if "generate" in steps:
                self._run_generate()

            # Step 4: Augment (Optional)
            if "augment" in steps and enable_augmentation:
                self._run_augment()

            # Step 5: Mine hard negatives (Optional)
            if "mine" in steps and enable_mining:
                self._run_mine()

            # Step 6: Embeddings (Optional)
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
        self.logger.info("\n[Step 1/6] Merging data sources...")
        merger = DataMerger(self.config)
        self.merged_data = merger.run()

    def _run_commute(self) -> None:
        """Execute commute data update step (enriches merged CSV with walk/ride times)."""
        self.logger.info("\n[Step 2/6] Updating commute data...")
        updater = CommuteDataUpdater(self.config)
        updated_df = updater.run(self.merged_data)
        # Persist enriched data back so downstream steps use the updated values
        updated_df.to_csv(self.config.merged_csv, index=False, encoding="utf-8-sig")
        self.merged_data = updated_df
        self.logger.info("Commute data updated and saved to merged CSV")

    def _run_generate(self) -> None:
        """Execute dataset generation step."""
        self.logger.info("\n[Step 3/6] Generating training dataset...")
        generator = DatasetGenerator(self.config)
        self.dataset = generator.run(self.merged_data)

    def _run_augment(self) -> None:
        """Execute LLM augmentation step."""
        self.logger.info("\n[Step 4/6] Augmenting with LLM...")
        if not self.config.llm_api_key:
            self.logger.warning("LLM API key not set. Skipping augmentation.")
            return

        augmenter = SemanticAugmenter(self.config)
        augmented = augmenter.run(target_count=1000)
        self.logger.info(f"Generated {len(augmented)} augmented samples")

        if augmented:
            train_path = Path(self.config.dataset_json).parent / "training_dataset.json"
            self._append_samples_to_json(
                train_path,
                [
                    {"query": p.query, "property_id": p.property_id, "label": p.is_match, "score": p.score}
                    for p in augmented
                ],
            )
            self.logger.info(f"Merged {len(augmented)} augmented samples into {train_path}")

    def _run_mine(self) -> None:
        """Execute hard negative mining step."""
        self.logger.info("\n[Step 5/6] Mining hard negatives...")
        miner = HardNegativeMiner(self.config)
        mined = miner.run()
        self.logger.info(f"Mined {len(mined)} hard negatives")

        if mined:
            train_path = Path(self.config.dataset_json).parent / "training_dataset.json"
            self._append_samples_to_json(
                train_path,
                [
                    {"query": ex.query, "property_id": ex.property_id, "label": False, "score": 0}
                    for ex in mined
                ],
            )
            self.logger.info(f"Merged {len(mined)} hard negatives into {train_path}")

    @staticmethod
    def _append_samples_to_json(path: Path, new_samples: list) -> None:
        """Append new_samples to an existing JSON array file (creates if missing)."""
        existing = []
        if path.exists():
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
        existing.extend(new_samples)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def _run_embed(self) -> None:
        """Execute embedding precomputation step."""
        self.logger.info("\n[Step 6/6] Precomputing embeddings...")
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
        elif step_name == "commute":
            self._run_commute()
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
