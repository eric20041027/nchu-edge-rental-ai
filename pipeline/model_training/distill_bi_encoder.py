"""Distill the rbt6 bi-encoder into a smaller rbt3 student (T2-distill).

Mirrors ``train_bi_encoder.py`` but instead of fine-tuning rbt6 from the HF
checkpoint, it TRANSFERS the deployed 6-layer bi-encoder into a 3-layer student
so the on-device model halves in size (≈57 MB → ≈38 MB INT8, same as the CE which
was already rbt6→rbt3 distilled). hidden_size stays 768, so the embedding dim and
the frontend ``property_embeddings.json`` schema are UNCHANGED — only the layer
count (and thus weights) shrink.

=== Why this exact design (do not deviate — 同源 constraint) ===
- Teacher  : the CURRENT production weights at ``config.bi_encoder_saved_dir``
  (``rbt6_bi_encoder``), NOT a fresh hfl/rbt6. We distill what is actually shipped.
- Student  : rbt6's FIRST 3 encoder layers (layer-truncation init — embeddings +
  layers[0:3] copied from the teacher). Standard, far stronger than random init;
  this is how rbt6→rbt3 is bootstrapped before fine-tuning.
- Pooling / norm : identical mask-aware mean-pool → L2-normalize (reuses
  ``BiEncoder`` / ``mean_pool`` from train_bi_encoder.py verbatim). The student
  output MUST be the same kind of vector as the teacher so query embeddings
  (on-device) stay comparable to property embeddings — which T4 RE-ENCODES with
  this student (a distilled student lives in a NEW vector space, so the 704
  property vectors are stale until re-built with these weights).
- Loss = α · DISTILL + (1−α) · MNRL :
    DISTILL = (1 − cos(student_emb, teacher_emb)) over query AND candidate texts
              — pulls the student's geometry toward the teacher's.
    MNRL    = the same InfoNCE/MNRL contrastive loss as train_bi_encoder.py
              (in-batch + shared hard negatives) — keeps the student anchored to
              ground-truth (query, positive) pairs, not just the teacher.
  α defaults to 0.5; raise it to lean on the teacher, lower it to lean on labels.

=== Pipeline AFTER this script (unchanged — reuses existing T3/T4) ===
  1. python -m pipeline.model_training.distill_bi_encoder        # → rbt3 student weights
  2. python -m pipeline.model_training.export_bi_encoder --saved-dir <student>   # ONNX + INT8
  3. python -m pipeline.data_prep.build_property_embeddings --saved-dir <student> # RE-ENCODE props
  4. python -m pipeline.model_training.semantic_benchmark        # recall@K gate (守天花板)
The student is saved via ``save_pretrained`` to ``--output-dir`` (default
``saved_models/rbt3_bi_encoder``) so steps 2–3 load it with ``--saved-dir``.

Run on Colab/GPU. Locally (no torch / tiny CPU) use ``--sample`` for a smoke test
that only proves the loss computes + decreases and the student is unit-norm.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import warnings
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from .base import BaseTrainer
from .config import ModelTrainingConfig
# DRY: reuse the exact module + helpers the production trainer/export use (同源).
from .train_bi_encoder import (
    BiEncoder,
    _as_bool,
    _as_int,
    assert_unit_norm,
)

warnings.filterwarnings("ignore", message=".*pin_memory.*")

STUDENT_LAYERS = 3  # rbt6 → rbt3 (mirrors the CE distillation depth)


class BiEncoderDistiller(BaseTrainer):
    """Distill the deployed rbt6 bi-encoder into an rbt3 student."""

    def __init__(
        self,
        config: ModelTrainingConfig,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        max_length: int,
        alpha: float,
        teacher_dir: Optional[Path],
        output_dir: Optional[Path],
        student_layers: int = STUDENT_LAYERS,
        sample: Optional[int] = None,
        bf16: bool = False,
        tf32: bool = False,
    ):
        super().__init__(config)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.alpha = alpha
        self.student_layers = student_layers
        self.sample = sample
        self.teacher_dir = Path(teacher_dir) if teacher_dir else config.bi_encoder_saved_dir
        self.output_dir = (
            Path(output_dir) if output_dir
            else config.saved_models_dir / "rbt3_bi_encoder"
        )

        if tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.use_bf16 = bool(
            bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )

        self.tokenizer = None
        self.teacher: Optional[BiEncoder] = None
        self.student: Optional[BiEncoder] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------- pipeline ------------------------------- #
    def _seed_everything(self) -> None:
        seed = self.config.random_seed
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.log_result("Seeded (python/torch/cuda)", seed)

    def run(self) -> dict:
        self._seed_everything()

        self.log_step("Loading contrastive pairs")
        anchors = self._load_pairs(self.config.train_data_path)
        self.log_result("Anchor (query, positive) rows", len(anchors))

        self.log_step("Loading teacher (rbt6) + building student (rbt3)")
        self._load_teacher_and_build_student()

        self.log_step(f"Distilling (α={self.alpha} · cos-distill + {1 - self.alpha:.2f} · MNRL)")
        step_losses = self._train(anchors)

        self.log_step("Saving student encoder + tokenizer")
        self._save()

        self.log_step("Offline retrieval eval on dev set (student)")
        eval_metrics = self._evaluate()

        self.log_step("Bi-encoder distillation pipeline completed")
        self.log_result("Student path", str(self.output_dir))
        return {
            "student": self.student,
            "tokenizer": self.tokenizer,
            "step_losses": step_losses,
            "eval_metrics": eval_metrics,
            "model_path": str(self.output_dir),
        }

    # ----------------------------- data ----------------------------------- #
    def _load_pairs(self, path) -> List[dict]:
        """Build anchor rows {query, positive, hard_negs} — identical to T2."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        hard_by_query: dict = {}
        for row in data:
            if _as_bool(row.get("is_hard")) and _as_int(row.get("label")) == 0:
                hard_by_query.setdefault(row["query"], []).append(row["property"])

        anchors: List[dict] = []
        for row in data:
            if _as_int(row.get("label")) != 1:
                continue
            anchors.append({
                "query": row["query"],
                "positive": row["property"],
                "hard_negs": hard_by_query.get(row["query"], []),
            })

        random.seed(self.config.random_seed)
        random.shuffle(anchors)
        if self.sample is not None:
            anchors = anchors[: self.sample]
            self.log_result("Sampled to", len(anchors))
        return anchors

    # ----------------------------- model ---------------------------------- #
    def _load_teacher_and_build_student(self) -> None:
        """Load deployed rbt6 as frozen teacher; truncate to rbt3 for the student."""
        from transformers import AutoModel, BertTokenizerFast

        if not (self.teacher_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No teacher weights at {self.teacher_dir}. Expected the deployed "
                f"rbt6 bi-encoder (config.json + model weights). Train T2 first or "
                f"pass --teacher-dir."
            )

        self.tokenizer = BertTokenizerFast.from_pretrained(str(self.teacher_dir))
        teacher_encoder = AutoModel.from_pretrained(str(self.teacher_dir))
        n_teacher = teacher_encoder.config.num_hidden_layers
        self.log_result("Teacher layers", n_teacher)
        if self.student_layers >= n_teacher:
            raise ValueError(
                f"student_layers ({self.student_layers}) must be < teacher layers "
                f"({n_teacher}) — nothing to distill."
            )

        # Student = same config but fewer layers; copy embeddings + first K layers.
        from transformers import AutoConfig
        student_cfg = AutoConfig.from_pretrained(str(self.teacher_dir))
        student_cfg.num_hidden_layers = self.student_layers
        student_encoder = AutoModel.from_config(student_cfg)

        # Layer-truncation init: copy embeddings + the first K encoder layers.
        missing = student_encoder.load_state_dict(
            self._truncated_state_dict(teacher_encoder, self.student_layers),
            strict=False,
        )
        # `missing.unexpected_keys` should be empty; `missing_keys` only the pooler
        # (unused by mean-pool) — log so a real mismatch is visible, not silent.
        self.log_result("Student init missing keys", len(missing.missing_keys))
        self.log_result("Student init unexpected keys", len(missing.unexpected_keys))

        self.teacher = BiEncoder(teacher_encoder).to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = BiEncoder(student_encoder).to(self.device)

    @staticmethod
    def _truncated_state_dict(teacher_encoder, k: int) -> dict:
        """Teacher state_dict keeping embeddings + encoder layers [0:k]."""
        out = {}
        for name, tensor in teacher_encoder.state_dict().items():
            if name.startswith("encoder.layer."):
                layer_idx = int(name.split(".")[2])
                if layer_idx >= k:
                    continue
            out[name] = tensor.clone()
        return out

    # ----------------------------- training ------------------------------- #
    def _encode_texts(self, texts: List[str]) -> dict:
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
        if not anchors:
            raise RuntimeError("No anchor pairs found — check label==1 rows in data.")

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.learning_rate)
        temperature = max(self.config.bi_encoder_temperature, 1e-6)
        step_losses: List[float] = []
        global_step = 0

        for epoch in range(self.epochs):
            random.shuffle(anchors)
            for start in range(0, len(anchors), self.batch_size):
                batch = anchors[start : start + self.batch_size]
                if len(batch) < 2:
                    continue

                queries = [b["query"] for b in batch]
                positives = [b["positive"] for b in batch]
                hard_pool: List[str] = []
                seen = set(positives)
                for b in batch:
                    for hn in b["hard_negs"]:
                        if hn not in seen:
                            seen.add(hn)
                            hard_pool.append(hn)
                hard_pool = hard_pool[: 2 * len(batch)]
                cand_texts = positives + hard_pool

                q_enc = self._encode_texts(queries)
                c_enc = self._encode_texts(cand_texts)

                # Teacher embeddings (no grad) for the distillation target.
                with torch.no_grad():
                    t_q = self.teacher(**q_enc)
                    t_c = self.teacher(**c_enc)

                self.student.train()
                actx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self.use_bf16
                    else contextlib.nullcontext()
                )
                with actx:
                    s_q = self.student(**q_enc)                       # (B, H)
                    s_c = self.student(**c_enc)                       # (B+H, H)

                    # MNRL contrastive (same as T2): target = own positive column.
                    sim = (s_q @ s_c.t()) / temperature
                    targets = torch.arange(len(batch), device=self.device)
                    mnrl = F.cross_entropy(sim, targets)

                    # Cosine distillation toward the teacher geometry (q + cands).
                    # Embeddings are already L2-normalized → cos == dot.
                    distill = (1.0 - (s_q * t_q).sum(dim=1)).mean() \
                            + (1.0 - (s_c * t_c).sum(dim=1)).mean()
                    distill = distill * 0.5  # average over the two text groups

                    loss = self.alpha * distill + (1.0 - self.alpha) * mnrl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_losses.append(float(loss.item()))
                global_step += 1
                if global_step % 10 == 0 or global_step <= 5:
                    self.log_result(
                        f"step {global_step} (epoch {epoch})",
                        f"loss={loss.item():.4f} (mnrl={mnrl.item():.4f} distill={distill.item():.4f})",
                    )

        if step_losses:
            self.log_result("First step loss", f"{step_losses[0]:.4f}")
            self.log_result("Last step loss", f"{step_losses[-1]:.4f}")
        return step_losses

    # ----------------------------- save ----------------------------------- #
    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.student.encoder.save_pretrained(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        self.log_result("Saved student encoder + tokenizer", str(self.output_dir))

    # ----------------------------- eval ----------------------------------- #
    @torch.no_grad()
    def _evaluate(self, max_queries: int = 200, neg_pool: int = 50) -> dict:
        """Same lightweight retrieval eval as T2 — reports pos/neg cosine + R@1.

        Also reports mean cos(student, teacher) on dev queries so a distillation
        collapse (student drifting away from the teacher geometry) is visible.
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

        self.student.eval()
        pos_sims, neg_sims, st_sims, hits = [], [], [], 0
        for item in positives:
            q_enc = self._encode_texts([item["query"]])
            p_enc = self._encode_texts([item["property"]])
            q_emb = self.student(**q_enc)
            pos_emb = self.student(**p_enc)
            pos_score = float((q_emb @ pos_emb.t()).item())

            # student-vs-teacher agreement on the query embedding (collapse guard).
            t_q = self.teacher(**q_enc)
            st_sims.append(float((q_emb * t_q).sum().item()))

            negs = [p for p in random.sample(all_props, min(neg_pool, len(all_props)))
                    if p != item["property"]]
            if not negs:
                continue
            neg_emb = self.student(**self._encode_texts(negs))
            neg_scores = (q_emb @ neg_emb.t()).squeeze(0)
            pos_sims.append(pos_score)
            neg_sims.append(float(neg_scores.mean().item()))
            if pos_score > float(neg_scores.max().item()):
                hits += 1

        n = len(pos_sims)
        metrics = {
            "dev_queries": n,
            "mean_pos_cosine": sum(pos_sims) / n if n else 0.0,
            "mean_neg_cosine": sum(neg_sims) / n if n else 0.0,
            "recall_at_1_vs_pool": hits / n if n else 0.0,
            "mean_student_teacher_cosine": sum(st_sims) / len(st_sims) if st_sims else 0.0,
        }
        self.log_metrics(metrics)
        return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill rbt6 bi-encoder → rbt3 student.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="distill weight: loss = α·cos-distill + (1−α)·MNRL")
    parser.add_argument("--student-layers", type=int, default=STUDENT_LAYERS,
                        help="student encoder layer count (rbt6→3 by default)")
    parser.add_argument("--teacher-dir", type=str, default=None,
                        help="deployed rbt6 bi-encoder dir (default: config.bi_encoder_saved_dir)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="where to save the student (default: saved_models/rbt3_bi_encoder)")
    parser.add_argument("--sample", type=int, default=None,
                        help="use only N anchors (CPU smoke test)")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ModelTrainingConfig()
    distiller = BiEncoderDistiller(
        config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        alpha=args.alpha,
        teacher_dir=args.teacher_dir,
        output_dir=args.output_dir,
        student_layers=args.student_layers,
        sample=args.sample,
        bf16=args.bf16,
        tf32=args.tf32,
    )
    result = distiller.run()

    losses = result["step_losses"]
    if len(losses) >= 2:
        decreased = losses[-1] < losses[0]
        distiller.log_result(
            "Loss decreased over run",
            f"{decreased} (first={losses[0]:.4f} -> last={losses[-1]:.4f})",
        )
    max_dev = assert_unit_norm(
        distiller.student, distiller.tokenizer, args.max_length, distiller.device
    )
    distiller.log_result("Unit-norm check passed", f"max |‖v‖-1| = {max_dev:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
