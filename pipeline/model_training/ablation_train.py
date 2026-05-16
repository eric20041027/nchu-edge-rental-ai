"""
ablation_train.py — Standalone training module for ablation experiments.

Each run saves:
  results_root/{run_id}/config.json        — AblationConfig serialized
  results_root/{run_id}/metrics.json       — NDCG@5, F1, precision, recall, accuracy
  results_root/{run_id}/convergence.json   — per-epoch NDCG (written by NDCGLogCallback)
  models_root/{run_id}/                    — PyTorch model (trainer.save_model())

Training hyperparams match train_and_export_onnx.py:
  LR=3e-5, batch=32, epochs=10, patience=6, warmup_ratio=0.08, cosine LR, fp16=True.

ONNX export is intentionally skipped (too slow per ablation run).
"""
import json
import math
import os
import random
import warnings
from collections import defaultdict
from typing import List, Optional, Tuple

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]   = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]          = "1"

warnings.filterwarnings("ignore", message=".*pin_memory.*")

import torch
import torch.nn.functional as F
import numpy as np

from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()

from .ablation_config import AblationConfig
from .ndcg_callback import NDCGLogCallback, compute_ndcg5_from_groups
from .training_utils import FGM, compute_metrics, CleanLogCallback, CustomEarlyStoppingCallback


# ── Constants ─────────────────────────────────────────────────────────────────

STUDENT_CHECKPOINT  = "hfl/rbt3"
DISTILL_TEMPERATURE = 4.0
FOCAL_GAMMA         = 0.0          # Disabled (kept for numerical parity with v2.9)
LABEL_SMOOTHING     = 0.05
MAX_LENGTH          = 64

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

_RBT6_TEACHER_DIR  = os.path.join(_PROJECT_ROOT, "saved_models", "rbt6_teacher")
TEACHER_MODEL_PATH = (_RBT6_TEACHER_DIR if os.path.isdir(_RBT6_TEACHER_DIR)
                      else "hfl/rbt6")


# ── AblationDistillTrainer ────────────────────────────────────────────────────

class AblationDistillTrainer(Trainer):
    """
    Parameterized version of DistillTrainer controlled by AblationConfig.
    Respects cfg.enable_ranknet, cfg.enable_listnet, cfg.rdrop_alpha,
    cfg.distill_alpha_max/min, and cfg.enable_fgm.
    """

    def __init__(
        self,
        teacher_model: PreTrainedModel,
        cfg: AblationConfig,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.cfg = cfg

    # ── Dynamic KD alpha ──────────────────────────────────────────────────────

    def _get_current_alpha(self) -> float:
        """Cosine annealing from cfg.distill_alpha_max -> cfg.distill_alpha_min."""
        try:
            if self.state.epoch is not None:
                t = min(float(self.state.epoch) / float(self.args.num_train_epochs), 1.0)
                cos_val = (1 + math.cos(math.pi * t)) / 2   # 1.0→0.0
                return (self.cfg.distill_alpha_min
                        + cos_val * (self.cfg.distill_alpha_max - self.cfg.distill_alpha_min))
        except Exception:
            pass
        return self.cfg.distill_alpha_max

    # ── Teacher forward ───────────────────────────────────────────────────────

    def _teacher_logits(self, model_inputs: dict) -> torch.Tensor:
        teacher_inputs = {
            k: v for k, v in model_inputs.items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
        }
        with torch.no_grad():
            out = self.teacher(**teacher_inputs)
        return out.logits   # [B, 2]

    # ── Loss computation ──────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False,
                     use_rdrop: bool = True, **kwargs):
        cfg       = self.cfg
        labels    = inputs.get("labels")
        weights   = inputs.get("sample_weight", None)
        relevance = inputs.get("relevance", labels.float())

        _EXTRA_KEYS  = {"sample_weight", "relevance"}
        model_inputs = {k: v for k, v in inputs.items() if k not in _EXTRA_KEYS}

        # ── Forward pass 1 ────────────────────────────────────────────────────
        outputs1 = model(**model_inputs)
        logits1  = outputs1.get("logits")   # [B, 2]

        # ── R-Drop ───────────────────────────────────────────────────────────
        if model.training and use_rdrop and cfg.rdrop_alpha > 0:
            outputs2   = model(**model_inputs)
            logits2    = outputs2.get("logits")
            p1 = F.softmax(logits1, dim=-1)
            p2 = F.softmax(logits2, dim=-1)
            rdrop_loss = (
                F.kl_div(F.log_softmax(logits1, dim=-1), p2.detach(), reduction="batchmean") +
                F.kl_div(F.log_softmax(logits2, dim=-1), p1.detach(), reduction="batchmean")
            ) / 2.0
            logits = (logits1 + logits2) / 2.0
        else:
            logits     = logits1
            rdrop_loss = torch.tensor(0.0, device=logits1.device)

        T_task     = 2.0
        rel_logits = logits[:, 1] / T_task   # [B]

        # ── 1. Focal CE (gamma=0 → plain CE with label smoothing) ─────────────
        with torch.no_grad():
            p_t = torch.exp(
                -F.cross_entropy(logits.detach(), labels, reduction="none")
            )
        focal_weight = (1.0 - p_t) ** FOCAL_GAMMA

        ce_loss = F.cross_entropy(
            logits, labels, reduction="none",
            label_smoothing=LABEL_SMOOTHING,
        )
        focal_ce = focal_weight * ce_loss

        pp = torch.where(labels == 0,
                         torch.tensor(1.5, device=labels.device),
                         torch.ones(labels.shape, device=labels.device))

        task_loss = (focal_ce * weights * pp).mean() if weights is not None else (focal_ce * pp).mean()

        # ── 2. RankNet (pairwise) ─────────────────────────────────────────────
        if cfg.enable_ranknet:
            s_i = rel_logits.unsqueeze(1)
            s_j = rel_logits.unsqueeze(0)
            r_i = relevance.unsqueeze(1)
            r_j = relevance.unsqueeze(0)
            mask = (r_i > r_j).float()
            if mask.sum() > 0:
                ranknet   = F.softplus(-(s_i - s_j)) * mask
                task_loss = task_loss + (ranknet.sum() / mask.sum()) * 1.5

        # ── 3. ListNet (listwise) ─────────────────────────────────────────────
        if cfg.enable_listnet:
            target_dist = torch.softmax(relevance, dim=0)
            pred_dist   = torch.log_softmax(rel_logits, dim=0)
            task_loss   = task_loss + (-torch.sum(target_dist * pred_dist)) * 1.0

        # ── 4. Knowledge Distillation ─────────────────────────────────────────
        alpha = self._get_current_alpha()
        if alpha > 0.0:
            teacher_device = next(self.teacher.parameters()).device
            if teacher_device != logits.device:
                self.teacher.to(logits.device)
            teacher_logits = self._teacher_logits(model_inputs)
            T = DISTILL_TEMPERATURE
            kl_loss = F.kl_div(
                F.log_softmax(logits         / T, dim=-1),
                F.softmax    (teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
        else:
            kl_loss = torch.tensor(0.0, device=logits.device)

        # ── Combine ───────────────────────────────────────────────────────────
        loss = ((1.0 - alpha) * task_loss
                + alpha * kl_loss
                + cfg.rdrop_alpha * rdrop_loss)

        return (loss, outputs1) if return_outputs else loss

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Normal forward + backward (with R-Drop if configured)
        loss = self.compute_loss(model, inputs, use_rdrop=True)
        self.accelerator.backward(loss)

        if self.cfg.enable_fgm:
            fgm = FGM(model)
            fgm.attack()
            try:
                loss_adv = self.compute_loss(model, inputs, use_rdrop=False)
                self.accelerator.backward(loss_adv)
            finally:
                fgm.restore()

        return loss.detach()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_and_balance(train_path: str, dev_path: str) -> Tuple[Dataset, Dataset]:
    """Same negative-sampling logic as train_and_export_onnx.py v2.9."""
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    hard_examples_path = os.path.join(os.path.dirname(train_path), "hard_examples.json")
    hard_examples: list = []
    if os.path.exists(hard_examples_path):
        with open(hard_examples_path, "r", encoding="utf-8") as f:
            hard_examples = json.load(f)
        for d in hard_examples:
            d["is_hard"] = True

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    random.seed(42)
    pos_samples = [d for d in train_data if d["label"] == 1]
    neg_all     = [d for d in train_data if d["label"] == 0]
    target_neg  = len(pos_samples)
    neg_samples = random.sample(neg_all, min(target_neg, len(neg_all)))

    combined = pos_samples + neg_samples + hard_examples
    random.shuffle(combined)
    return Dataset.from_list(combined), Dataset.from_list(dev_data)


def _tokenize(
    train_ds: Dataset,
    eval_ds:  Dataset,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Dataset, Dataset]:
    rel_map = {3: 12.0, 2: 4.0, 1: 0.8, 0: 5.0, -1: 0.3}

    def tok_fn(examples):
        tok = tokenizer(
            examples["query"], examples["property"],
            padding="max_length", max_length=MAX_LENGTH, truncation=True,
        )
        tok["labels"] = examples["label"]
        is_hard_list  = examples.get("is_hard", [False] * len(examples["query"]))
        weights = []
        for i in range(len(examples["query"])):
            w = float(rel_map.get(examples["relevance"][i], 1.0))
            if is_hard_list[i]:
                w *= 2.0
            weights.append(w)
        tok["sample_weight"] = weights
        tok["relevance"]     = [float(r) for r in examples["relevance"]]
        return tok

    return (
        train_ds.map(tok_fn, batched=True),
        eval_ds .map(tok_fn, batched=True),
    )


# ── NDCG@5 evaluation (flat, no grouping by query struct) ────────────────────

def compute_flat_ndcg5(
    model:     PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data:      List[dict],
    device,
    batch_size: int = 64,
) -> float:
    """
    Run model inference on all (query, property) pairs in data, group by query,
    compute NDCG@5 with exponential gain. Returns mean NDCG@5.
    """
    model.eval()
    queries    = [d["query"]    for d in data]
    properties = [d["property"] for d in data]
    relevances = [int(d.get("relevance", d.get("label", 0))) for d in data]

    all_scores = []
    with torch.no_grad():
        for start in range(0, len(queries), batch_size):
            end   = min(start + batch_size, len(queries))
            enc   = tokenizer(
                queries[start:end], properties[start:end],
                padding="max_length", max_length=MAX_LENGTH,
                truncation=True, return_tensors="pt",
            )
            enc   = {k: v.to(device) for k, v in enc.items()}
            out   = model(**enc)
            all_scores.extend(out.logits[:, 1].cpu().tolist())

    query_to_items: dict = defaultdict(list)
    for score, rel, q in zip(all_scores, relevances, queries):
        query_to_items[q].append((score, rel))

    return compute_ndcg5_from_groups(dict(query_to_items))


# ── KD alpha logger ───────────────────────────────────────────────────────────

class _AlphaLogCallback(TrainerCallback):
    def __init__(self, trainer_ref: AblationDistillTrainer):
        self._trainer = trainer_ref

    def on_epoch_begin(self, args, state, control, **kwargs):
        cfg = self._trainer.cfg
        if cfg.distill_alpha_max > 0.0:
            alpha = self._trainer._get_current_alpha()
            print(f"  [KD] Epoch {state.epoch:.0f} | alpha={alpha:.3f} "
                  f"(range [{cfg.distill_alpha_min}, {cfg.distill_alpha_max}])")


# ── Main run_ablation function ────────────────────────────────────────────────

def run_ablation(
    cfg:          AblationConfig,
    results_root: str,
    models_root:  str,
) -> dict:
    """
    Execute one ablation run.

    Returns a metrics dict:
        {run_id, ndcg_at_5, f1, precision, recall, accuracy}

    Skips and returns existing metrics if metrics.json already present.
    """
    run_results_dir = os.path.join(results_root, cfg.run_id)
    metrics_path    = os.path.join(run_results_dir, "metrics.json")

    # ── Skip / resume ─────────────────────────────────────────────────────────
    if os.path.exists(metrics_path):
        print(f"  [skip] {cfg.run_id}: metrics.json found — skipping training.")
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(run_results_dir, exist_ok=True)
    run_models_dir = os.path.join(models_root, cfg.run_id)
    os.makedirs(run_models_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Run: {cfg.run_id} | {cfg.description}")
    print(f"  RankNet={cfg.enable_ranknet}, ListNet={cfg.enable_listnet}, "
          f"FGM={cfg.enable_fgm}, RDrop={cfg.rdrop_alpha}, "
          f"alpha=[{cfg.distill_alpha_min},{cfg.distill_alpha_max}]")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"[{cfg.run_id}] CUDA not available. Ablation training requires GPU."
        )

    # ── Save config ───────────────────────────────────────────────────────────
    with open(os.path.join(run_results_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

    # ── Load teacher ──────────────────────────────────────────────────────────
    teacher_path = TEACHER_MODEL_PATH
    print(f"  Loading teacher from: {teacher_path}")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        teacher_path,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── Load student ──────────────────────────────────────────────────────────
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_CHECKPOINT,
        num_labels=2,
        id2label={0: "NOT_MATCH", 1: "MATCH"},
        label2id={"NOT_MATCH": 0, "MATCH": 1},
        hidden_dropout_prob=0.15,
        attention_probs_dropout_prob=0.15,
    )

    # ── Load & tokenize data ──────────────────────────────────────────────────
    tokenizer = BertTokenizerFast.from_pretrained(STUDENT_CHECKPOINT)

    data_dir  = os.path.join(_PROJECT_ROOT, "data", "processed")
    train_raw, dev_raw = _load_and_balance(
        os.path.join(data_dir, "recommendation_train.json"),
        os.path.join(data_dir, "recommendation_dev.json"),
    )
    train_ds, eval_ds = _tokenize(train_raw, dev_raw, tokenizer)

    # Raw dev list for NDCG callback
    with open(os.path.join(data_dir, "recommendation_dev.json"), "r", encoding="utf-8") as f:
        dev_data_raw = json.load(f)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    output_tmp = os.path.join(run_models_dir, "_tmp_checkpoints")
    training_args = TrainingArguments(
        output_dir=output_tmp,
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_ratio=0.08,
        label_smoothing_factor=0.0,   # applied manually in compute_loss
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
        log_level="error",
        max_grad_norm=1.0,
    )

    # ── Convergence callback ──────────────────────────────────────────────────
    convergence_path = os.path.join(run_results_dir, "convergence.json")
    ndcg_callback = NDCGLogCallback(
        model=student,
        tokenizer=tokenizer,
        dev_data=dev_data_raw,
        device=device,
        output_path=convergence_path,
        max_length=MAX_LENGTH,
        batch_size=64,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = AblationDistillTrainer(
        teacher_model=teacher,
        cfg=cfg,
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            CustomEarlyStoppingCallback(
                early_stopping_patience=6,
                early_stopping_threshold=0.001,
            ),
            CleanLogCallback(),
            ndcg_callback,
        ],
    )
    trainer.add_callback(_AlphaLogCallback(trainer))
    trainer.train()

    # ── Save best model ───────────────────────────────────────────────────────
    trainer.save_model(run_models_dir)
    tokenizer.save_pretrained(run_models_dir)
    print(f"  Model saved to: {run_models_dir}")

    # ── Clean up _tmp_checkpoints (optimizer states, RNG, ~880 MB per run) ───
    import shutil
    tmp_ckpt_dir = os.path.join(run_models_dir, "_tmp_checkpoints")
    if os.path.isdir(tmp_ckpt_dir):
        shutil.rmtree(tmp_ckpt_dir)
        print(f"  Cleaned up: {tmp_ckpt_dir}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    with open(os.path.join(data_dir, "recommendation_test.json"), "r", encoding="utf-8") as f:
        test_data_raw = json.load(f)

    # Tokenize test set
    test_raw = Dataset.from_list(test_data_raw)

    def tok_test(examples):
        t = tokenizer(
            examples["query"], examples["property"],
            padding="max_length", max_length=MAX_LENGTH, truncation=True,
        )
        t["labels"] = examples["label"]
        return t

    test_ds     = test_raw.map(tok_test, batched=True, remove_columns=test_raw.column_names)
    test_result = trainer.evaluate(test_ds)

    # NDCG@5 on test set
    best_model = trainer.model
    best_model.to(device)
    ndcg5 = compute_flat_ndcg5(best_model, tokenizer, test_data_raw, device)
    print(f"  Test NDCG@5: {ndcg5:.4f}")

    # ── Build & save metrics ──────────────────────────────────────────────────
    metrics = {
        "run_id":    cfg.run_id,
        "ndcg_at_5": round(ndcg5, 6),
        "f1":        round(float(test_result.get("eval_f1",        0.0)), 6),
        "precision": round(float(test_result.get("eval_precision", 0.0)), 6),
        "recall":    round(float(test_result.get("eval_recall",    0.0)), 6),
        "accuracy":  round(float(test_result.get("eval_accuracy",  0.0)), 6),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    return metrics
