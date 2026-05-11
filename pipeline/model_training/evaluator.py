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
        """Get model predictions on test set with query/property text pairing.

        Returns:
            List of predictions with scores and query grouping info
        """
        import torch

        predictions = []

        original_device = next(self.model.parameters()).device
        self.log_step(f"Moving model from {original_device} to CPU for inference")
        self.model = self.model.to('cpu')

        try:
            for sample in self.test_data[: self.config.eval_sample_size]:
                query = sample.get("query", "")
                # Require actual property text — URLs are not useful for cross-encoder
                property_text = sample.get("property", "")
                if not property_text or property_text.startswith("http"):
                    property_text = ""

                inputs = self.tokenizer(
                    query,
                    property_text,
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                inputs = {k: v.to('cpu') for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0]
                    pred_label = torch.argmax(logits).item()
                    pred_score = torch.softmax(logits, dim=-1)[1].item()

                true_label = sample.get("label", 0)
                if isinstance(true_label, bool):
                    true_label = int(true_label)

                predictions.append({
                    "query": query,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pred_score": pred_score,
                })

        finally:
            self.model = self.model.to(original_device)
            self.log_step(f"Model restored to {original_device}")

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
        """Compute NDCG@k averaged over queries.

        Groups predictions by query, sorts each group by model score (pred_score),
        then evaluates ranking quality using true relevance labels.
        """
        if not predictions or k == 0:
            return 0.0

        # Group by query
        from collections import defaultdict
        query_groups: dict = defaultdict(list)
        for p in predictions:
            query_groups[p.get("query", "__all__")].append(p)

        ndcg_scores = []
        for group in query_groups.values():
            if not any(p["true_label"] == 1 for p in group):
                continue  # Skip queries with no relevant docs

            # Sort by model score (descending) — this is the ranking
            ranked = sorted(group, key=lambda p: p["pred_score"], reverse=True)

            dcg = sum(
                (2 ** p["true_label"] - 1) / np.log2(i + 2)
                for i, p in enumerate(ranked[:k])
            )
            ideal = sorted(group, key=lambda p: p["true_label"], reverse=True)
            idcg = sum(
                (2 ** p["true_label"] - 1) / np.log2(i + 2)
                for i, p in enumerate(ideal[:k])
            )
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    def _compute_mrr(self, predictions: List[Dict]) -> float:
        """Compute Mean Reciprocal Rank averaged over queries.

        Groups by query, sorts by model score, finds rank of first relevant result.
        """
        if not predictions:
            return 0.0

        from collections import defaultdict
        query_groups: dict = defaultdict(list)
        for p in predictions:
            query_groups[p.get("query", "__all__")].append(p)

        rr_scores = []
        for group in query_groups.values():
            if not any(p["true_label"] == 1 for p in group):
                continue
            ranked = sorted(group, key=lambda p: p["pred_score"], reverse=True)
            for rank, pred in enumerate(ranked, start=1):
                if pred["true_label"] == 1:
                    rr_scores.append(1.0 / rank)
                    break

        return float(np.mean(rr_scores)) if rr_scores else 0.0


if __name__ == "__main__":
    """Standalone evaluation script."""
    from transformers import BertForSequenceClassification, BertTokenizerFast

    config = ModelTrainingConfig()
    print(f"Loading model from: {config.saved_model_dir}")

    try:
        tokenizer = BertTokenizerFast.from_pretrained(str(config.saved_model_dir))
        model = BertForSequenceClassification.from_pretrained(str(config.saved_model_dir), num_labels=2)

        evaluator = Evaluator(config)
        metrics = evaluator.run(model, tokenizer)

        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics.accuracy:.4f}")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"Recall:    {metrics.recall:.4f}")
        print(f"F1-Score:  {metrics.f1_score:.4f}")
        print(f"NDCG@5:    {metrics.ndcg_at_5:.4f}")
        print(f"MRR:       {metrics.mrr:.4f}")
        print(f"{'='*50}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
