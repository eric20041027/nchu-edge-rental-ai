"""
training_utils.py — Shared utilities for teacher and student training scripts.

Extracted from train_teacher.py and train_and_export_onnx.py to avoid duplicate
definitions. Both scripts import from here.
"""
import numpy as np
import torch
from transformers import EarlyStoppingCallback, TrainerCallback


# ── Adversarial Training (FGM) ────────────────────────────────────────────────

class FGM:
    """Fast Gradient Method adversarial perturbation on word embeddings.

    Usage (in training_step):
        fgm = FGM(model)
        fgm.attack()
        try:
            loss_adv = compute_loss(...)
            backward(loss_adv)
        finally:
            fgm.restore()   # always restore, even on exception
    """

    def __init__(self, model):
        self.model  = model
        self.backup = {}

    def attack(self, epsilon: float = 1.0, emb_name: str = "word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(epsilon * param.grad / norm)

    def restore(self, emb_name: str = "word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup, f"FGM: no backup for {name}"
                param.data = self.backup[name]
        self.backup = {}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(p):
    """Binary classification metrics: accuracy, precision, recall, F1."""
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    acc   = (preds == labels).mean()
    tp    = ((preds == 1) & (labels == 1)).sum()
    fp    = ((preds == 1) & (labels == 0)).sum()
    fn    = ((preds == 0) & (labels == 1)).sum()
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {
        "accuracy":  round(float(acc),  3),
        "precision": round(float(prec), 3),
        "recall":    round(float(rec),  3),
        "f1":        round(float(f1),   3),
    }


# ── Callbacks ─────────────────────────────────────────────────────────────────

class CleanLogCallback(TrainerCallback):
    """Pretty-prints training and validation logs to stdout."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            print(f"  {'Epoch':>5} {logs.get('epoch', 0):>5.2f} | "
                  f"Loss: {logs.get('loss', 0):>8.5f} | "
                  f"LR: {logs.get('learning_rate', 0):>9.2e}")
        elif "eval_loss" in logs:
            print("-" * 70)
            print(f"  VALIDATION | Loss: {logs.get('eval_loss', 0):>8.5f} | "
                  f"Acc: {logs.get('eval_accuracy', 0):>7.5f} | "
                  f"Prec: {logs.get('eval_precision', 0):>7.5f} | "
                  f"Rec: {logs.get('eval_recall', 0):>7.5f} | "
                  f"F1: {logs.get('eval_f1', 0):>7.5f}")
            print("-" * 70)


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    """EarlyStoppingCallback with verbose patience counter output."""

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        super().on_evaluate(args, state, control, metrics, **kwargs)
        c          = self.early_stopping_patience_counter
        p          = self.early_stopping_patience
        metric_val = metrics.get(f"eval_{args.metric_for_best_model}", 0)
        if c > 0:
            print(f"  [EarlyStopping] Patience: {c}/{p}  "
                  f"(best {args.metric_for_best_model}={state.best_metric:.5f})")
        else:
            print(f"  [EarlyStopping] New best {args.metric_for_best_model}={metric_val:.5f}  "
                  f"Patience reset.")
