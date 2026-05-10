"""Model evaluation and ranking metrics computation."""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from datasets import Dataset

from .base import BaseTrainer
from .config import ModelTrainingConfig
from .models import EvaluationMetrics


class Evaluator(BaseTrainer):
    """Computes evaluation metrics including NDCG and MRR for ranking tasks.

    Evaluates model on query-property matching with ranking-specific metrics
    (NDCG@5, MRR) in addition to classification metrics (accuracy, F1).
    """

    def __init__(self, config: ModelTrainingConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.test_data = None

    def run(self, model, tokenizer) -> EvaluationMetrics:
        """Evaluate model on test set with ranking metrics.

        Args:
            model: Trained transformer model
            tokenizer: Associated tokenizer

        Returns:
            EvaluationMetrics with all evaluation results
        """
        self.model = model
        self.tokenizer = tokenizer

        self.log_step("Loading test data")
        self._load_test_data()

        self.log_step("Computing evaluation metrics")
        metrics = self._compute_metrics()

        self.log_metrics({
            "accuracy": metrics.accuracy,
            "f1_score": metrics.f1_score,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "ndcg_at_5": metrics.ndcg_at_5,
            "mrr": metrics.mrr,
        })

        return metrics

    def _load_test_data(self) -> None:
        """Load test dataset."""
        if not self.config.test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.config.test_data_path}")

        with open(self.config.test_data_path, "r", encoding="utf-8") as f:
            self.test_data = json.load(f)

        self.log_result("Test samples", len(self.test_data))

    def _compute_metrics(self) -> EvaluationMetrics:
        """Compute all evaluation metrics.

        Returns:
            EvaluationMetrics with computed values
        """
        predictions = self._get_predictions()

        accuracy = self._compute_accuracy(predictions)
        f1_score = self._compute_f1(predictions)
        precision = self._compute_precision(predictions)
        recall = self._compute_recall(predictions)
        ndcg_at_5 = self._compute_ndcg_at_k(predictions, k=5)
        mrr = self._compute_mrr(predictions)

        return EvaluationMetrics(
            accuracy=accuracy,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
            ndcg_at_5=ndcg_at_5,
            mrr=mrr,
            auc_roc=None,
        )

    def _get_predictions(self) -> List[Dict]:
        """Get model predictions on test set.

        Returns:
            List of predictions with scores
        """
        import torch

        predictions = []

        for sample in self.test_data[: self.config.eval_sample_size]:
            inputs = self.tokenizer(
                sample.get("query", ""),
                sample.get("property", ""),
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                pred_label = torch.argmax(logits).item()
                pred_score = torch.softmax(logits, dim=-1)[1].item()

            predictions.append({
                "true_label": sample.get("label", 0),
                "pred_label": pred_label,
                "pred_score": pred_score,
            })

        return predictions

    def _compute_accuracy(self, predictions: List[Dict]) -> float:
        """Compute classification accuracy."""
        if not predictions:
            return 0.0

        correct = sum(1 for p in predictions if p["true_label"] == p["pred_label"])
        return correct / len(predictions)

    def _compute_f1(self, predictions: List[Dict]) -> float:
        """Compute F1 score."""
        tp = sum(1 for p in predictions if p["true_label"] == 1 and p["pred_label"] == 1)
        fp = sum(1 for p in predictions if p["true_label"] == 0 and p["pred_label"] == 1)
        fn = sum(1 for p in predictions if p["true_label"] == 1 and p["pred_label"] == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def _compute_precision(self, predictions: List[Dict]) -> float:
        """Compute precision."""
        tp = sum(1 for p in predictions if p["true_label"] == 1 and p["pred_label"] == 1)
        fp = sum(1 for p in predictions if p["true_label"] == 0 and p["pred_label"] == 1)

        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _compute_recall(self, predictions: List[Dict]) -> float:
        """Compute recall."""
        tp = sum(1 for p in predictions if p["true_label"] == 1 and p["pred_label"] == 1)
        fn = sum(1 for p in predictions if p["true_label"] == 1 and p["pred_label"] == 0)

        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _compute_ndcg_at_k(self, predictions: List[Dict], k: int = 5) -> float:
        """Compute NDCG@k metric for ranking evaluation.

        NDCG measures ranking quality considering relevance grades.
        """
        if not predictions or k == 0:
            return 0.0

        dcg = sum(
            (2 ** p["pred_label"] - 1) / np.log2(i + 2)
            for i, p in enumerate(predictions[:k])
        )

        ideal_order = sorted(predictions, key=lambda p: p["true_label"], reverse=True)
        idcg = sum(
            (2 ** p["true_label"] - 1) / np.log2(i + 2)
            for i, p in enumerate(ideal_order[:k])
        )

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_mrr(self, predictions: List[Dict]) -> float:
        """Compute Mean Reciprocal Rank for ranking evaluation.

        MRR measures how quickly the first relevant result appears.
        """
        if not predictions:
            return 0.0

        for rank, pred in enumerate(predictions, start=1):
            if pred["true_label"] == 1:
                return 1.0 / rank

        return 0.0
