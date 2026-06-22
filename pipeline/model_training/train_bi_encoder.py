"""Bi-encoder contrastive trainer for vector retrieval (T2).

Mirrors ``trainer.py`` (ModelTrainer / BaseTrainer / log_step) but trains a
BI-ENCODER instead of a cross-encoder. Where the cross-encoder feeds
``(query, property)`` as one sequence into a classification head, the bi-encoder
encodes query and property SEPARATELY with a SHARED hfl/rbt6 encoder, mean-pools
(mask-aware) over tokens, L2-normalizes, and compares with cosine similarity.

=== Why this exact architecture (do not deviate — spec Resolved Decisions) ===
- Base model: hfl/rbt6 (SAME 6-layer Chinese RoBERTa as the CE — "CE 同源"), so
  the tokenizer/vocab are already validated and the quantization pipeline is reused.
- Shared-weight encoder: one encoder used for both sides. Standard for bi-encoders
  and keeps the on-device model small (edge-first).
- Pooling: mask-aware MEAN pool -> L2 normalize. This MUST match the frontend
  convention (同源: same model, mean-pool, L2-norm) so the query embedding produced
  on-device and the property embeddings pre-computed offline are directly comparable
  (cosine == dot of normalized vectors).
- Loss: InfoNCE / MultipleNegativesRankingLoss (MNRL). Temperature-scaled cosine
  similarity, cross-entropy over the candidate batch.

=== Negative construction (the exact strategy implemented here) ===
For each training step we build a batch of B anchor queries. For anchor i:
  * POSITIVE: the property paired with the query under label==1.
  * IN-BATCH NEGATIVES: the positive properties of the OTHER B-1 anchors in the
    batch (MNRL — every other row's positive is a negative for this row). This is
    free and scales the number of negatives with batch size.
  * HARD NEGATIVES: properties marked ``is_hard==true`` for the SAME query are
    appended to a shared candidate pool. So the candidate matrix is
    ``[B positives ; H hard-negatives]`` and each anchor scores against all of
    them. Hard negatives are shared across the batch (any anchor's hard negative
    is also a negative for the others — they are by construction non-matching
    properties), which is the standard MNRL-with-hard-negatives formulation.
The similarity matrix is ``sim[i, j] = cos(q_i, cand_j) / temperature`` and the
target for anchor i is column i (its own positive). Cross-entropy over each row.

Dataset construction from ``recommendation_train.json``:
  anchors = distinct queries that have at least one label==1 property.
  positive = that property (if a query has multiple positives, each
    (query, positive) becomes its own anchor row — handled sensibly, not dropped).
  hard negatives = the is_hard==true properties for that same query (when present).
Note: labels / is_hard in the JSON are stored as STRINGS ("0"/"1"/"True"/"False")
— coerced robustly via ``_as_bool`` / ``_as_int``.

=== NOTE FOR T3 (ONNX export of the QUERY encoder) ===
T3 must export the QUERY-encoding path only:
    input_ids + attention_mask  ->  AutoModel encoder  ->  mask-aware mean-pool
    ->  L2-normalize  ->  embedding (float32, dim = hidden_size, default 768).
Requirements (mirror exporter.py):
  * dynamo=False (legacy TorchScript tracer) so weights embed in a single .onnx.
  * opset 15 (config.onnx_opset_version), do_constant_folding, export_params.
  * Apply ``Exporter._apply_onnx_monkey_patch()`` before tracing (SDPA/ONNX fix).
  * token_type_ids are NOT needed for a single-text query encode; export with
    input_ids + attention_mask only. The pooling+normalize MUST be inside the
    exported graph so the on-device output equals what this trainer produces and
    matches the offline property embeddings (同源). ``BiEncoder.encode`` /
    ``BiEncoder.forward`` below already produce exactly that embedding — wrap the
    saved encoder in this module (or re-implement pool+norm identically) for export.
The encoder + tokenizer are saved via ``save_pretrained`` to
``config.bi_encoder_saved_dir`` so T3 can load them with ``AutoModel.from_pretrained``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTrainer
from .config import ModelTrainingConfig

warnings.filterwarnings("ignore", message=".*pin_memory.*")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


# --------------------------------------------------------------------------- #
# Robust coercion — recommendation_*.json stores label/is_hard as strings.
# --------------------------------------------------------------------------- #
def _as_int(value) -> int:
    """Coerce '0'/'1'/0/1/True to int, defaulting to 0 on garbage."""
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_bool(value) -> bool:
    """Coerce 'True'/'true'/'1'/1/True to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return False


# --------------------------------------------------------------------------- #
# Pooling + bi-encoder module
# --------------------------------------------------------------------------- #
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-aware mean pooling over the token dimension.

    Args:
        last_hidden_state: (B, T, H) encoder output.
        attention_mask: (B, T) 1 for real tokens, 0 for padding.

    Returns:
        (B, H) pooled embedding — padding tokens excluded from the mean.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)  # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1) avoid div-by-zero
    return summed / counts


class BiEncoder(nn.Module):
    """Shared-weight bi-encoder: encode -> mask-aware mean-pool -> L2-normalize.

    The forward/encode output is the SAME embedding the frontend must reproduce
    (同源). T3 exports exactly this path for the query side.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized mean-pooled embedding. (B, H)"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=1)

    # encode is an alias used by eval / export call-sites for readability.
    encode = forward


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class BiEncoderTrainer(BaseTrainer):
    """Trains a shared-weight hfl/rbt6 bi-encoder with InfoNCE / MNRL loss.

    Mirrors ModelTrainer's structure (run / log_step / save pattern) but uses a
    hand-written contrastive loop instead of HuggingFace Trainer, because the
    candidate matrix (positives + shared hard negatives) is not a per-row label
    that the stock Trainer/compute_loss interface expresses cleanly.
    """

    def __init__(
        self,
        config: ModelTrainingConfig,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        max_length: int,
        sample: Optional[int] = None,
        output_dir: Optional[Path] = None,
        bf16: bool = False,
        tf32: bool = False,
    ):
        super().__init__(config)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.sample = sample
        self.output_dir = Path(output_dir) if output_dir else config.bi_encoder_saved_dir

        # A100 pure-acceleration (speed only; training dynamics unchanged).
        # bf16 engages only if the GPU supports it -> T4 falls back to fp32.
        if tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.use_bf16 = bool(
            bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )

        self.tokenizer = None
        self.model: Optional[BiEncoder] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.used_fallback = False  # True if random-init fallback was used

    # ----------------------------- pipeline ------------------------------- #
    def run(self) -> dict:
        """Execute the full bi-encoder training pipeline."""
        self.log_step("Loading and preparing contrastive pairs")
        anchors = self._load_pairs(self.config.train_data_path)
        self.log_result("Anchor (query, positive) rows", len(anchors))

        self.log_step("Loading shared encoder and tokenizer")
        self._load_model_and_tokenizer()

        self.log_step("Training bi-encoder (InfoNCE / MNRL)")
        step_losses = self._train(anchors)

        self.log_step("Saving encoder + tokenizer")
        self._save()

        self.log_step("Offline retrieval eval on dev set")
        eval_metrics = self._evaluate()

        self.log_step("Bi-encoder training pipeline completed")
        self.log_result("Model path", str(self.output_dir))
        self.log_result("Used random-init fallback", self.used_fallback)

        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "step_losses": step_losses,
            "eval_metrics": eval_metrics,
            "model_path": str(self.output_dir),
            "used_fallback": self.used_fallback,
        }

    # ----------------------------- data ----------------------------------- #
    def _load_pairs(self, path) -> List[dict]:
        """Build anchor rows: {query, positive, hard_negs:[...]}.

        anchors = each (query, label==1 property) pair (multi-positive queries
        contribute one row per positive). hard_negs = is_hard==true properties
        for the same query.
        """
        self.log_step(f"Loading training data from {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.log_result("Raw rows", len(data))

        # Group hard negatives per query for fast lookup.
        hard_by_query: dict = {}
        for row in data:
            if _as_bool(row.get("is_hard")) and _as_int(row.get("label")) == 0:
                hard_by_query.setdefault(row["query"], []).append(row["property"])

        anchors: List[dict] = []
        for row in data:
            if _as_int(row.get("label")) != 1:
                continue
            query = row["query"]
            anchors.append(
                {
                    "query": query,
                    "positive": row["property"],
                    "hard_negs": hard_by_query.get(query, []),
                }
            )

        random.seed(self.config.random_seed)
        random.shuffle(anchors)

        if self.sample is not None:
            anchors = anchors[: self.sample]
            self.log_result("Sampled to", len(anchors))

        n_with_hard = sum(1 for a in anchors if a["hard_negs"])
        self.log_result("Anchors with hard negatives", n_with_hard)
        return anchors

    # ----------------------------- model ---------------------------------- #
    def _load_model_and_tokenizer(self) -> None:
        """Load hfl/rbt6 encoder; fall back to random-init config on no network."""
        from transformers import AutoConfig, AutoModel, BertTokenizerFast

        checkpoint = self.config.model_checkpoint
        try:
            self.log_step(f"Loading tokenizer + encoder from {checkpoint}")
            self.tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
            encoder = AutoModel.from_pretrained(checkpoint)
            self.log_result("Encoder loaded", checkpoint)
        except Exception as e:  # network / download failure -> graceful fallback
            self.logger.warning(
                f"Could not download '{checkpoint}' ({type(e).__name__}: {e}). "
                "Falling back to a TINY random-init config of the same architecture "
                "(transformers can init from config without downloading) so the "
                "sanity check can still prove the loss computes and decreases."
            )
            self.used_fallback = True
            self.tokenizer = self._build_fallback_tokenizer()
            cfg = AutoConfig.for_model(
                "bert",
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=128,
                max_position_embeddings=128,
            )
            encoder = AutoModel.from_config(cfg)
            self.log_result("Encoder", "random-init fallback (bert tiny)")

        self.model = BiEncoder(encoder).to(self.device)

    @staticmethod
    def _build_fallback_tokenizer():
        """Minimal char-level BERT tokenizer for the offline fallback path."""
        from tokenizers import BertWordPieceTokenizer
        from transformers import BertTokenizerFast

        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        # Small Chinese/ASCII vocab — enough for the sanity pass.
        vocab += [chr(c) for c in range(0x4E00, 0x4E00 + 500)]  # common CJK block
        vocab += [chr(c) for c in range(0x20, 0x7F)]  # ASCII printable
        vocab += [str(d) for d in range(10)]
        vocab = list(dict.fromkeys(vocab))  # dedupe, keep order

        tmp = Path(os.getenv("TMPDIR", "/tmp")) / "bi_encoder_fallback_vocab.txt"
        tmp.write_text("\n".join(vocab), encoding="utf-8")
        _ = BertWordPieceTokenizer  # imported to assert availability
        return BertTokenizerFast(vocab_file=str(tmp), do_lower_case=False)

    # ----------------------------- training ------------------------------- #
    def _encode_texts(self, texts: List[str]) -> dict:
        """Tokenize a list of texts to device tensors."""
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
        }

    def _train(self, anchors: List[dict]) -> List[float]:
        """Manual contrastive training loop. Returns list of step losses."""
        if not anchors:
            raise RuntimeError("No anchor pairs found — check label==1 rows in data.")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        temperature = max(self.config.bi_encoder_temperature, 1e-6)
        step_losses: List[float] = []
        global_step = 0

        for epoch in range(self.epochs):
            random.shuffle(anchors)
            for start in range(0, len(anchors), self.batch_size):
                batch = anchors[start : start + self.batch_size]
                if len(batch) < 2:
                    continue  # MNRL needs >=2 rows for in-batch negatives

                queries = [b["query"] for b in batch]
                positives = [b["positive"] for b in batch]
                # Shared hard-negative pool across the batch (dedup, cap for memory).
                hard_pool: List[str] = []
                seen = set(positives)
                for b in batch:
                    for hn in b["hard_negs"]:
                        if hn not in seen:
                            seen.add(hn)
                            hard_pool.append(hn)
                hard_pool = hard_pool[: 2 * len(batch)]  # cap candidate growth

                self.model.train()
                # bf16 autocast on A100 (no GradScaler needed for bf16); fp32 otherwise.
                actx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self.use_bf16
                    else contextlib.nullcontext()
                )
                with actx:
                    q_emb = self.model(**self._encode_texts(queries))          # (B, H)
                    cand_texts = positives + hard_pool
                    cand_emb = self.model(**self._encode_texts(cand_texts))    # (B+H, H)

                    # sim[i,j] = cos(q_i, cand_j) / T ; target = own positive column i
                    sim = (q_emb @ cand_emb.t()) / temperature                  # (B, B+H)
                    targets = torch.arange(len(batch), device=self.device)
                    loss = F.cross_entropy(sim, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_losses.append(float(loss.item()))
                global_step += 1
                if global_step % 10 == 0 or global_step <= 5:
                    self.log_result(
                        f"step {global_step} (epoch {epoch})",
                        f"loss={loss.item():.4f}",
                    )

        if step_losses:
            self.log_result("First step loss", f"{step_losses[0]:.4f}")
            self.log_result("Last step loss", f"{step_losses[-1]:.4f}")
        return step_losses

    # ----------------------------- save ----------------------------------- #
    def _save(self) -> None:
        """Save the bare encoder + tokenizer via save_pretrained for T3."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Save the underlying AutoModel (encoder) so T3 can AutoModel.from_pretrained.
        self.model.encoder.save_pretrained(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        self.log_result("Saved encoder + tokenizer", str(self.output_dir))

    # ----------------------------- eval ----------------------------------- #
    @torch.no_grad()
    def _evaluate(self, max_queries: int = 200, neg_pool: int = 50) -> dict:
        """Lightweight retrieval eval on the dev set.

        For each eval query with a positive, score its positive against a random
        pool of negative properties and report: (a) mean cosine of positives,
        (b) mean cosine of sampled negatives, (c) Recall@1 (positive ranked #1
        against the negative pool). A learned encoder should push pos > neg.
        """
        val_path = self.config.val_data_path
        if not Path(val_path).exists():
            self.log_result("Dev eval", "skipped (no val file)")
            return {}

        with open(val_path, "r", encoding="utf-8") as f:
            dev = json.load(f)

        positives = [d for d in dev if _as_int(d.get("label")) == 1]
        all_props = list({d["property"] for d in dev})
        if not positives or len(all_props) < 2:
            self.log_result("Dev eval", "skipped (insufficient dev positives)")
            return {}

        random.seed(self.config.random_seed)
        random.shuffle(positives)
        positives = positives[:max_queries]

        self.model.eval()
        pos_sims, neg_sims, hits = [], [], 0
        for item in positives:
            q_emb = self.model(**self._encode_texts([item["query"]]))  # (1, H)
            pos_emb = self.model(**self._encode_texts([item["property"]]))  # (1, H)
            pos_score = float((q_emb @ pos_emb.t()).item())

            negs = [p for p in random.sample(all_props, min(neg_pool, len(all_props)))
                    if p != item["property"]]
            if not negs:
                continue
            neg_emb = self.model(**self._encode_texts(negs))  # (N, H)
            neg_scores = (q_emb @ neg_emb.t()).squeeze(0)  # (N,)
            mean_neg = float(neg_scores.mean().item())

            pos_sims.append(pos_score)
            neg_sims.append(mean_neg)
            if pos_score > float(neg_scores.max().item()):
                hits += 1

        n = len(pos_sims)
        metrics = {
            "dev_queries": n,
            "mean_pos_cosine": sum(pos_sims) / n if n else 0.0,
            "mean_neg_cosine": sum(neg_sims) / n if n else 0.0,
            "recall_at_1_vs_pool": hits / n if n else 0.0,
        }
        self.log_metrics(metrics)
        return metrics


# --------------------------------------------------------------------------- #
# Sanity helpers (also exercised by `python -m ... --sample`)
# --------------------------------------------------------------------------- #
def assert_unit_norm(model: BiEncoder, tokenizer, max_length: int, device) -> float:
    """Assert mean-pool + L2-normalize yields unit-norm vectors. Returns max |‖v‖-1|."""
    enc = tokenizer(
        ["這是一個測試查詢", "套房 南區 8000元 有冷氣"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        emb = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
    norms = emb.norm(p=2, dim=1)
    max_dev = float((norms - 1.0).abs().max().item())
    assert max_dev < 1e-4, f"embeddings not unit-norm (max deviation {max_dev})"
    return max_dev


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train hfl/rbt6 bi-encoder (T2).")
    parser.add_argument("--epochs", type=int, default=3, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size (= # in-batch negatives + 1)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--max-length", type=int, default=64, help="max token length")
    parser.add_argument("--sample", type=int, default=None, help="use only N anchors (quick sanity run)")
    parser.add_argument("--output-dir", type=str, default=None, help="where to save encoder + tokenizer")
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast (A100; T4 auto-fallback to fp32)")
    parser.add_argument("--tf32", action="store_true", help="enable TF32 matmul/cudnn (A100 speed)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ModelTrainingConfig()
    trainer = BiEncoderTrainer(
        config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        sample=args.sample,
        output_dir=args.output_dir,
        bf16=args.bf16,
        tf32=args.tf32,
    )
    result = trainer.run()

    # Sanity assertions (loss decreased, embeddings unit-norm).
    losses = result["step_losses"]
    if len(losses) >= 2:
        decreased = losses[-1] < losses[0]
        trainer.log_result(
            "Loss decreased over run",
            f"{decreased} (first={losses[0]:.4f} -> last={losses[-1]:.4f})",
        )
    max_dev = assert_unit_norm(
        trainer.model, trainer.tokenizer, args.max_length, trainer.device
    )
    trainer.log_result("Unit-norm check passed", f"max |‖v‖-1| = {max_dev:.2e}")
    trainer.log_result(
        "Path taken",
        "random-init fallback" if result["used_fallback"] else "real hfl/rbt6",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
