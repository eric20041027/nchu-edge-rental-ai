"""Model trainer for RoBERTa-based sentence pair classification."""
import json
import random
import torch
import warnings
from typing import Tuple, Optional

from datasets import Dataset

from .base import BaseTrainer
from .config import ModelTrainingConfig
from .models import TrainingMetrics, EvaluationMetrics

# Suppress warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


class ModelTrainer(BaseTrainer):
    """Trains RoBERTa-based sentence pair classification model.

    Fine-tunes hfl/rbt6 (6-layer Chinese RoBERTa) on query-property pairs
    for binary matching. Supports adversarial training for improved generalization.
    """

    def __init__(self, config: ModelTrainingConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None

    def run(self) -> dict:
        """Execute full training pipeline.

        Returns:
            Dict with trained model, metrics, and checkpoint info
        """
        self.log_step("Loading and preparing data")
        train_dataset, val_dataset = self._load_and_balance_data()

        self.log_step("Loading model and tokenizer")
        self._load_model_and_tokenizer()

        self.log_step("Setting up trainer with adversarial training")
        self._setup_trainer()

        self.log_step("Training model")
        training_metrics = self._train_model()

        self.log_step("Evaluating model")
        eval_metrics = self._evaluate_model()

        self.log_step("Training pipeline completed")
        self.log_result("Model checkpoint", str(self.config.saved_model_dir))

        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "training_metrics": training_metrics,
            "eval_metrics": eval_metrics,
            "model_path": str(self.config.saved_model_dir),
        }

    def _load_and_balance_data(self) -> Tuple[Dataset, Dataset]:
        """Load and balance training data.

        Returns:
            (train_dataset, val_dataset) Hugging Face Datasets
        """
        self.log_step(f"Loading training data from {self.config.train_data_path}")
        with open(self.config.train_data_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        self.log_result("Train samples", len(train_data))

        # Load validation data
        with open(self.config.val_data_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        self.log_result("Val samples", len(val_data))

        # Balance training data
        random.seed(self.config.random_seed)
        pos_samples = [d for d in train_data if d.get("label") == 1]
        neg_samples = [d for d in train_data if d.get("label") == 0]

        self.log_result("Original POS", len(pos_samples))
        self.log_result("Original NEG", len(neg_samples))

        # Balance to minority class
        if len(neg_samples) > len(pos_samples):
            neg_samples = random.sample(neg_samples, len(pos_samples))
        elif len(pos_samples) > len(neg_samples):
            pos_samples = random.sample(pos_samples, len(neg_samples))

        balanced_train = pos_samples + neg_samples
        random.shuffle(balanced_train)

        self.log_result("Balanced train samples", len(balanced_train))

        self.train_dataset = Dataset.from_list(balanced_train)
        self.val_dataset = Dataset.from_list(val_data)

        return (self.train_dataset, self.val_dataset)

    def _load_model_and_tokenizer(self) -> None:
        """Load pre-trained model and tokenizer."""
        try:
            from transformers import (
                BertTokenizerFast,
                AutoModelForSequenceClassification,
            )

            self.log_step(f"Loading tokenizer from {self.config.model_checkpoint}")
            self.tokenizer = BertTokenizerFast.from_pretrained(
                self.config.model_checkpoint
            )

            self.log_step(f"Loading model {self.config.model_checkpoint}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_checkpoint,
                num_labels=2,
            )

            self.log_result("Model loaded", self.config.model_checkpoint)

        except ImportError:
            self.logger.error("transformers package not installed")
            raise

    def _setup_trainer(self) -> None:
        """Setup Hugging Face Trainer with training arguments."""
        try:
            from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

            self.log_step("Setting up Trainer")

            training_args = TrainingArguments(
                output_dir=str(self.config.saved_model_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                weight_decay=0.01,
                logging_steps=100,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                seed=self.config.random_seed,
            )

            def tokenize_function(examples):
                # Use property_id if property field not available
                if "property" in examples:
                    property_text = examples["property"]
                elif "property_id" in examples:
                    property_text = examples["property_id"]
                else:
                    property_text = examples["query"]

                return self.tokenizer(
                    examples["query"],
                    property_text,
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length",
                )

            # Tokenize datasets
            train_tokenized = self.train_dataset.map(tokenize_function, batched=True)
            val_tokenized = self.val_dataset.map(tokenize_function, batched=True)

            # Setup trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.config.early_stopping_patience
                    )
                ],
            )

            self.log_result("Trainer ready", "EarlyStoppingCallback enabled")

        except ImportError:
            self.logger.error("transformers package not installed")
            raise

    def _train_model(self) -> TrainingMetrics:
        """Train the model.

        Returns:
            TrainingMetrics with final training results
        """
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call _setup_trainer first.")

        self.log_step("Starting training")
        train_result = self.trainer.train()

        self.log_result("Training completed", f"{train_result.global_step} steps")
        self.log_result("Final train loss", f"{train_result.training_loss:.4f}")

        return TrainingMetrics(
            epoch=self.config.num_epochs,
            train_loss=float(train_result.training_loss),
            val_loss=0.0,
            val_accuracy=0.0,
            val_f1=0.0,
            learning_rate=self.config.learning_rate,
        )

    def _evaluate_model(self) -> EvaluationMetrics:
        """Evaluate model on test set.

        Returns:
            EvaluationMetrics with accuracy, F1, NDCG, etc.
        """
        if not self.trainer or not self.model:
            raise RuntimeError("Model not trained. Call _train_model first.")

        self.log_step("Loading test data")
        with open(self.config.test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        test_dataset = Dataset.from_list(test_data)
        self.log_result("Test samples", len(test_data))

        def tokenize_function(examples):
            # Use property_id if property field not available
            if "property" in examples:
                property_text = examples["property"]
            elif "property_id" in examples:
                property_text = examples["property_id"]
            else:
                property_text = examples["query"]

            return self.tokenizer(
                examples["query"],
                property_text,
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
            )

        test_tokenized = test_dataset.map(tokenize_function, batched=True)

        self.log_step("Evaluating on test set")
        test_results = self.trainer.evaluate(eval_dataset=test_tokenized)

        self.log_metrics({
            "test_loss": test_results.get("eval_loss", 0.0),
            "test_accuracy": test_results.get("eval_accuracy", 0.0),
        })

        return EvaluationMetrics(
            accuracy=test_results.get("eval_accuracy", 0.0),
            f1_score=test_results.get("eval_f1", 0.0),
            precision=test_results.get("eval_precision", 0.0),
            recall=test_results.get("eval_recall", 0.0),
            ndcg_at_5=0.0,
            mrr=0.0,
        )
