"""Model trainer for RoBERTa-based sentence pair classification."""
import json
import math
import random
import torch
import warnings
from collections import defaultdict
from typing import Tuple, Optional

from datasets import Dataset
from transformers import Trainer

from .base import BaseTrainer
from .config import ModelTrainingConfig
from .models import TrainingMetrics, EvaluationMetrics

# Suppress warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


class FGM:
    """Fast Gradient Method for adversarial training on embedding layer."""

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGMTrainer(Trainer):
    """HuggingFace Trainer with FGM adversarial training and soft-label ranking loss.

    Combined loss = 0.5 * CrossEntropy(hard_labels) + 0.5 * BCE(soft_labels)
    Soft labels encode graded relevance (-1→0.0, 0→0.0, 1→0.4, 2→0.7, 3→1.0)
    so the model learns to rank high-relevance properties above low-relevance ones.
    """

    @staticmethod
    def _compute_combined_loss(outputs, inputs_with_labels, soft_labels):
        """Compute 0.5 * CE(hard) + 0.5 * BCE(soft) combined loss."""
        import torch.nn.functional as F

        ce_loss = outputs.loss  # cross-entropy from model's own computation
        if soft_labels is None:
            return ce_loss

        logits = outputs.logits
        # Log-odds score for the positive class
        pos_logit = logits[:, 1] - logits[:, 0]
        ranking_loss = F.binary_cross_entropy_with_logits(pos_logit, soft_labels.float())
        return 0.5 * ce_loss + 0.5 * ranking_loss

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Extract soft relevance labels — pop so the model never sees them
        soft_labels = inputs.pop("soft_labels", None)

        outputs = model(**inputs)
        loss = self._compute_combined_loss(outputs, inputs, soft_labels)
        loss.backward()

        fgm = FGM(model)
        fgm.attack()
        outputs_adv = model(**inputs)
        loss_adv = self._compute_combined_loss(outputs_adv, inputs, soft_labels)
        loss_adv.backward()
        fgm.restore()

        return loss.detach()


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

        # Balance training data — use 1:2 pos:neg for ranking (more negatives = better contrast)
        random.seed(self.config.random_seed)
        pos_samples = [d for d in train_data if d.get("label") == 1]
        neg_samples = [d for d in train_data if d.get("label") == 0]

        self.log_result("Original POS", len(pos_samples))
        self.log_result("Original NEG", len(neg_samples))

        # Keep up to 2× negatives relative to positives for better ranking signal
        target_neg = min(len(neg_samples), 2 * len(pos_samples))
        if len(neg_samples) > target_neg:
            neg_samples = random.sample(neg_samples, target_neg)
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
            from transformers import TrainingArguments, EarlyStoppingCallback

            self.log_step("Setting up Trainer with FGM adversarial training")

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
                fp16=getattr(self.config, "fp16", False),
                dataloader_num_workers=0,  # Windows compatibility
            )

            # Graded relevance → soft label: -1/0→0.0, 1→0.4, 2→0.7, 3→1.0
            # label=1 with relevance=0 gets 0.15 (minimum positive)
            _REL_SOFT = {-1: 0.0, 0: 0.0, 1: 0.4, 2: 0.7, 3: 1.0}
            _LABEL1_REL0_SOFT = 0.15  # positive but zero-graded relevance

            def tokenize_function(examples):
                # Use property text field (cross-encoder needs actual text, not URLs)
                if "property" in examples:
                    property_text = examples["property"]
                else:
                    # Fallback: empty string is better than a URL
                    property_text = [""] * len(examples["query"])

                result = self.tokenizer(
                    examples["query"],
                    property_text,
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length",
                )
                # HuggingFace Trainer expects "labels" column (hard binary)
                if "label" in examples:
                    result["labels"] = [int(bool(l)) for l in examples["label"]]

                # Soft labels encode graded relevance for ranking-aware loss
                labels = examples.get("label", [0] * len(examples["query"]))
                relevances = examples.get("relevance", [0] * len(examples["query"]))
                soft = []
                for lbl, rel in zip(labels, relevances):
                    try:
                        rel_int = int(rel) if rel is not None else 0
                    except (ValueError, TypeError):
                        rel_int = 0
                    if int(bool(lbl)) == 0:
                        soft.append(0.0)
                    elif rel_int == 0:
                        soft.append(_LABEL1_REL0_SOFT)
                    else:
                        soft.append(_REL_SOFT.get(rel_int, 0.15))
                result["soft_labels"] = soft
                return result

            # Tokenize datasets
            train_tokenized = self.train_dataset.map(tokenize_function, batched=True)
            val_tokenized = self.val_dataset.map(tokenize_function, batched=True)

            # Setup trainer with FGM adversarial training
            self.trainer = FGMTrainer(
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

        # Explicitly save best model + tokenizer so exporter can load from saved_model_dir
        self.log_step("Saving best model to disk")
        self.trainer.save_model(str(self.config.saved_model_dir))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(self.config.saved_model_dir))
        self.log_result("Model saved", str(self.config.saved_model_dir))

        # Pull best val metrics from trainer log history
        val_loss = 0.0
        val_accuracy = 0.0
        val_f1 = 0.0
        for entry in reversed(self.trainer.state.log_history):
            if "eval_loss" in entry:
                val_loss = entry.get("eval_loss", 0.0)
                val_accuracy = entry.get("eval_accuracy", 0.0)
                val_f1 = entry.get("eval_f1", 0.0)
                break

        return TrainingMetrics(
            epoch=self.config.num_epochs,
            train_loss=float(train_result.training_loss),
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_f1=val_f1,
            learning_rate=self.config.learning_rate,
        )

    def _evaluate_model(self) -> EvaluationMetrics:
        """Evaluate model on test set with real NDCG@5 and MRR.

        Returns:
            EvaluationMetrics with accuracy, F1, NDCG@5, MRR.
        """
        if not self.trainer or not self.model:
            raise RuntimeError("Model not trained. Call _train_model first.")

        self.log_step("Loading test data")
        with open(self.config.test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        test_dataset = Dataset.from_list(test_data)
        self.log_result("Test samples", len(test_data))

        def tokenize_function(examples):
            property_text = (
                examples["property"] if "property" in examples
                else examples.get("property_id", examples["query"])
            )
            return self.tokenizer(
                examples["query"],
                property_text,
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
            )

        test_tokenized = test_dataset.map(tokenize_function, batched=True)

        self.log_step("Running predictions on test set")
        predictions = self.trainer.predict(test_tokenized)
        scores = predictions.predictions[:, 1].tolist()  # logit for positive class

        # Standard eval metrics
        test_results = predictions.metrics or {}
        self.log_metrics({
            "test_loss": test_results.get("test_loss", 0.0),
        })

        # --- NDCG@5 and MRR (per-query ranking metrics) ---
        query_groups: dict = defaultdict(list)
        for i, item in enumerate(test_data):
            relevance = item.get("score") if item.get("score") is not None else (1 if item.get("label") else 0)
            query_groups[item["query"]].append({
                "score": scores[i],
                "label": bool(item.get("label", False)),
                "relevance": float(relevance),
            })

        ndcg_scores, mrr_scores = [], []
        for items in query_groups.values():
            items.sort(key=lambda x: x["score"], reverse=True)
            top5 = items[:5]

            # MRR: reciprocal rank of first relevant item in top-5
            mrr_val = 0.0
            for rank, item in enumerate(top5, 1):
                if item["label"]:
                    mrr_val = 1.0 / rank
                    break
            mrr_scores.append(mrr_val)

            # NDCG@5: graded relevance DCG / IDCG
            dcg = sum(item["relevance"] / math.log2(rank + 1) for rank, item in enumerate(top5, 1))
            ideal = sorted([x["relevance"] for x in items], reverse=True)[:5]
            idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        ndcg_at_5 = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

        self.log_metrics({"NDCG@5": ndcg_at_5, "MRR": mrr})

        return EvaluationMetrics(
            accuracy=test_results.get("test_accuracy", 0.0),
            f1_score=test_results.get("test_f1", 0.0),
            precision=test_results.get("test_precision", 0.0),
            recall=test_results.get("test_recall", 0.0),
            ndcg_at_5=ndcg_at_5,
            mrr=mrr,
        )
