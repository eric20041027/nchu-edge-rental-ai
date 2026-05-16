"""
ndcg_callback.py — HuggingFace TrainerCallback that computes NDCG@5 on the
dev set after each evaluation epoch and appends it to a convergence JSON file.

NDCG formula (exponential gain, 0-indexed rank):
    dcg  = sum((2**max(rel, 0) - 1) / log2(rank + 2) for rank, rel in top5)
    ndcg = dcg / idcg  if idcg > 0 else 0.0

rel=-1 is treated as 0 via max(rel, 0).
"""
import json
import math
import os
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import TrainerCallback


# ── NDCG helpers ──────────────────────────────────────────────────────────────

def _dcg(rels: List[int], k: int = 5) -> float:
    """Discounted Cumulative Gain with exponential gain."""
    total = 0.0
    for rank, rel in enumerate(rels[:k]):
        gain = (2 ** max(rel, 0)) - 1
        total += gain / math.log2(rank + 2)   # rank is 0-indexed → rank+2
    return total


def compute_ndcg5_from_groups(
    query_to_items: dict,   # {query_key: [(score, rel), ...]}
) -> float:
    """Given per-query score/rel lists, compute mean NDCG@5."""
    ndcg_scores = []
    for items in query_to_items.values():
        # Sort by predicted score descending
        sorted_by_score = sorted(items, key=lambda x: x[0], reverse=True)
        pred_rels = [rel for _, rel in sorted_by_score]

        # Ideal: sort by relevance descending
        ideal_rels = sorted(pred_rels, reverse=True)

        dcg  = _dcg(pred_rels, k=5)
        idcg = _dcg(ideal_rels, k=5)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(sum(ndcg_scores) / len(ndcg_scores)) if ndcg_scores else 0.0


# ── Callback ──────────────────────────────────────────────────────────────────

class NDCGLogCallback(TrainerCallback):
    """
    Computes NDCG@5 on the dev set after each evaluation epoch.

    Args:
        model:       The student model (already on the correct device during training).
        tokenizer:   BertTokenizerFast used for the training run.
        dev_data:    Raw list of dicts with keys: query, property, label, relevance.
        device:      torch device (typically "cuda").
        output_path: Path to the convergence JSON file to write/append.
        max_length:  Tokenizer max_length (default 64, matches train_and_export_onnx.py).
        batch_size:  Inference chunk size (default 64).
    """

    def __init__(
        self,
        model,
        tokenizer,
        dev_data: List[dict],
        device,
        output_path: str,
        max_length: int = 64,
        batch_size: int = 64,
    ):
        self.model       = model
        self.tokenizer   = tokenizer
        self.dev_data    = dev_data
        self.device      = device
        self.output_path = output_path
        self.max_length  = max_length
        self.batch_size  = batch_size
        self.history: List[dict] = []

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self) -> dict:
        """Tokenize all dev pairs and run batched inference.

        Returns:
            query_to_items: {query_str: [(score, rel_int), ...]}
        """
        self.model.eval()
        device = next(self.model.parameters()).device   # honour actual device

        queries    = [d["query"]    for d in self.dev_data]
        properties = [d["property"] for d in self.dev_data]
        relevances = [int(d.get("relevance", d.get("label", 0))) for d in self.dev_data]

        all_scores = []
        n = len(queries)

        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                end   = min(start + self.batch_size, n)
                batch_q = queries[start:end]
                batch_p = properties[start:end]

                enc = self.tokenizer(
                    batch_q, batch_p,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                out = self.model(**enc)
                # Score = logit for class 1 (MATCH)
                scores = out.logits[:, 1].cpu().tolist()
                all_scores.extend(scores)

        # Group by query
        query_to_items: dict = defaultdict(list)
        for score, rel, q in zip(all_scores, relevances, queries):
            query_to_items[q].append((score, rel))

        return dict(query_to_items)

    # ── Callback hook ─────────────────────────────────────────────────────────

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called by Trainer after each evaluation pass."""
        query_to_items = self._run_inference()
        ndcg5 = compute_ndcg5_from_groups(query_to_items)

        epoch    = state.epoch if state.epoch is not None else len(self.history) + 1
        eval_f1  = (metrics or {}).get("eval_f1",   None)
        eval_loss= (metrics or {}).get("eval_loss", None)

        entry = {
            "epoch":     round(epoch, 2),
            "ndcg_at_5": round(ndcg5, 6),
        }
        if eval_f1   is not None:
            entry["eval_f1"]   = round(float(eval_f1),   6)
        if eval_loss is not None:
            entry["eval_loss"] = round(float(eval_loss), 6)

        self.history.append(entry)
        print(f"  [NDCG Callback] Epoch {epoch:.1f} -> NDCG@5 = {ndcg5:.4f}")

        # Write full history to disk
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
