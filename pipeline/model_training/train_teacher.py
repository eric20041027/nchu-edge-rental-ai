"""
train_teacher.py - Dedicated rbt6 Teacher Training (for v2.6 Knowledge Distillation)

Trains hfl/rbt6 (6-layer Chinese RoBERTa) on the rental dataset, then saves to
saved_models/rbt6_teacher/.

v2.6 change: RankNet + ListNet REMOVED from teacher loss.
Rationale: ranking losses push all positive scores higher, inflating Recall but
crushing Precision (v2.5 teacher: Rec=0.985, Prec=0.652). The teacher's role is
to provide well-calibrated soft labels, not to be a ranking model. Pure CE + R-Drop
+ FGM gives cleaner probability distributions for distillation.

Loss = CE(label_smoothing=0.05) + 0.05 × SymKL(pass1 ‖ pass2) + FGM
     (RankNet and ListNet intentionally excluded)

Expected: F1 ≈ 83-88% with higher Precision ceiling.
"""

import math
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]   = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]          = "1"
os.environ["HF_HUB_OFFLINE"]                    = "0"

import json
import random
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

from typing import Tuple
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()

from datasets import Dataset
from torch import nn
import torch.nn.functional as F


# ── Configuration ──────────────────────────────────────────────────────────────

TEACHER_CHECKPOINT = "hfl/rbt6"   # 6-layer Chinese RoBERTa pre-trained

BASE_DIR         = os.path.dirname(__file__)
TEACHER_SAVE_DIR = os.path.join(BASE_DIR, "../../saved_models/rbt6_teacher")
DATA_DIR         = os.path.join(BASE_DIR, "../../data/processed")

# Training hyperparams — v2.6: LR=5e-6 for stable convergence, longer training
LEARNING_RATE   = 5e-6
NUM_EPOCHS      = 15
PATIENCE        = 7
BATCH_SIZE      = 32
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.08       # Slightly longer warmup at very low LR
MAX_LENGTH      = 64

# Loss config — RankNet/ListNet removed; teacher only needs calibrated CE + R-Drop
RDROP_ALPHA     = 0.05
FOCAL_GAMMA     = 0.0        # Disabled
LABEL_SMOOTHING = 0.05
T_TASK          = 2.0        # Kept for compatibility (unused without ranking losses)

# Sample weights (same as student)
REL_MAP = {3: 12.0, 2: 4.0, 1: 0.8, 0: 5.0, -1: 0.3}


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_and_balance_data() -> Tuple[Dataset, Dataset]:
    print(f"\n[Data] Loading training data...")
    with open(os.path.join(DATA_DIR, "recommendation_train.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)

    hard_path = os.path.join(DATA_DIR, "hard_examples.json")
    hard_examples = []
    if os.path.exists(hard_path):
        with open(hard_path, "r", encoding="utf-8") as f:
            hard_examples = json.load(f)
        for d in hard_examples:
            d["is_hard"] = True
        print(f"  Loaded {len(hard_examples)} hard examples.")

    with open(os.path.join(DATA_DIR, "recommendation_dev.json"), "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    random.seed(42)
    pos_samples = [d for d in train_data if d["label"] == 1]
    neg_all  = [d for d in train_data if d["label"] == 0]
    neg_hard = [d for d in neg_all if d.get("relevance", -1) == 0]
    neg_easy = [d for d in neg_all if d.get("relevance", -1) == -1]

    # Stratified negative sampling: fill quota with hard-conflict first
    target_neg = len(pos_samples)
    if len(neg_hard) >= target_neg:
        neg_samples = random.sample(neg_hard, target_neg)
    else:
        n_easy = min(target_neg - len(neg_hard), len(neg_easy))
        neg_samples = neg_hard + random.sample(neg_easy, n_easy)
        if len(neg_samples) < target_neg:
            remaining = [d for d in neg_all if d not in set(neg_samples)]
            neg_samples += random.sample(remaining,
                                         min(target_neg - len(neg_samples), len(remaining)))

    train_balanced = pos_samples + neg_samples + hard_examples
    random.shuffle(train_balanced)

    print(f"  POS={len(pos_samples)}, NEG={len(neg_samples)} "
          f"(hard={sum(1 for d in neg_samples if d.get('relevance',-1)==0)}, "
          f"random={sum(1 for d in neg_samples if d.get('relevance',-1)==-1)})")
    print(f"  Final train: {len(train_balanced)}  |  Dev: {len(dev_data)}")
    return Dataset.from_list(train_balanced), Dataset.from_list(dev_data)


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenize_datasets(train_ds: Dataset, eval_ds: Dataset,
                      tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset]:
    print("\n[Tokenize] Encoding sentence pairs...")

    def tokenize_fn(examples):
        tok = tokenizer(
            examples["query"], examples["property"],
            padding="max_length", max_length=MAX_LENGTH, truncation=True,
        )
        tok["labels"] = examples["label"]
        is_hard = examples.get("is_hard", [False] * len(examples["query"]))
        weights = []
        for i in range(len(examples["query"])):
            w = float(REL_MAP.get(examples["relevance"][i], 1.0))
            if is_hard[i]:
                w *= 2.0
            weights.append(w)
        tok["sample_weight"] = weights
        tok["relevance"]     = [float(r) for r in examples["relevance"]]
        return tok

    return (
        train_ds.map(tokenize_fn, batched=True),
        eval_ds .map(tokenize_fn, batched=True),
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(p):
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


# ── FGM Adversarial Training ───────────────────────────────────────────────────

class FGM:
    def __init__(self, model):
        self.model  = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(epsilon * param.grad / norm)

    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                param.data = self.backup[name]
        self.backup = {}


# ── Teacher Trainer (CE + RankNet + ListNet + R-Drop + FGM) ───────────────────

class TeacherTrainer(Trainer):
    """
    Multi-task trainer for the rbt6 teacher.
    No KD (the teacher has no teacher of its own).
    Identical loss structure to the student so soft labels will be
    well-calibrated on this task's distribution.
    """

    def compute_loss(self, model, inputs, return_outputs=False,
                     use_rdrop: bool = True, **kwargs):
        labels    = inputs.get("labels")
        weights   = inputs.pop("sample_weight", None)
        relevance = inputs.get("relevance", labels.float())

        # ── Forward pass 1 ────────────────────────────────────────────────────
        outputs1 = model(**inputs)
        logits1  = outputs1.get("logits")   # [B, 2]

        # ── R-Drop: forward pass 2 ────────────────────────────────────────────
        if model.training and use_rdrop and RDROP_ALPHA > 0:
            outputs2 = model(**inputs)
            logits2  = outputs2.get("logits")
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

        # ── CE loss only (no RankNet/ListNet — teacher needs clean calibration) ──
        ce_loss = F.cross_entropy(logits, labels, reduction="none",
                                  label_smoothing=LABEL_SMOOTHING)
        pp = torch.where(labels == 0,
                         torch.tensor(1.5, device=labels.device),
                         torch.ones(labels.shape, device=labels.device))
        task_loss = (ce_loss * weights * pp).mean() if weights is not None \
                    else (ce_loss * pp).mean()

        # ── Combine ───────────────────────────────────────────────────────────
        loss = task_loss + RDROP_ALPHA * rdrop_loss
        return (loss, outputs1) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Normal backward + FGM adversarial backward.
        Uses self.accelerator.backward() so FP16 GradScaler is initialised
        correctly before Trainer calls _clip_grad_norm.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs, use_rdrop=True)
        self.accelerator.backward(loss)   # ← FP16-safe (initialises GradScaler)

        fgm = FGM(model)
        fgm.attack()
        loss_adv = self.compute_loss(model, inputs, use_rdrop=False)
        self.accelerator.backward(loss_adv)   # ← accumulates adversarial grads
        fgm.restore()

        return loss.detach()


# ── Callbacks ──────────────────────────────────────────────────────────────────

class CleanLogCallback(TrainerCallback):
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
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        super().on_evaluate(args, state, control, metrics, **kwargs)
        c = self.early_stopping_patience_counter
        p = self.early_stopping_patience
        metric_val = metrics.get(f"eval_{args.metric_for_best_model}", 0)
        if c > 0:
            print(f"  [EarlyStopping] Patience: {c}/{p}  "
                  f"(best {args.metric_for_best_model}={state.best_metric:.5f})")
        else:
            print(f"  [EarlyStopping] New best "
                  f"{args.metric_for_best_model}={metric_val:.5f}  Patience reset.")


# ── Main Training ──────────────────────────────────────────────────────────────

def train_teacher():
    # ── CUDA check ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[Train] CUDA GPU not available. Training requires a CUDA-capable GPU.\n"
            "  Check: python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "  If using WSL/remote, ensure CUDA driver and PyTorch CUDA build match."
        )
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n[Train] Device: CUDA — {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"[Train] Teacher checkpoint: {TEACHER_CHECKPOINT}")
    print(f"[Train] Save path:          {TEACHER_SAVE_DIR}")

    # ── Guard: don't accidentally overwrite a good teacher ────────────────────
    if os.path.isdir(TEACHER_SAVE_DIR):
        ckpt_files = [f for f in os.listdir(TEACHER_SAVE_DIR)
                      if f.endswith(".safetensors") or f.endswith(".bin")]
        if ckpt_files:
            print(f"\n[WARN] {TEACHER_SAVE_DIR} already has weights: {ckpt_files}")
            ans = input("  Overwrite existing teacher? (yes/no): ").strip().lower()
            if ans != "yes":
                print("  Aborted. Existing teacher preserved.")
                return

    # ── Load data ─────────────────────────────────────────────────────────────
    tokenizer = BertTokenizerFast.from_pretrained(TEACHER_CHECKPOINT)
    train_raw, eval_raw = load_and_balance_data()
    train_ds, eval_ds   = tokenize_datasets(train_raw, eval_raw, tokenizer)

    # ── Load model ────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_CHECKPOINT,
        num_labels=2,
        id2label={0: "NOT_MATCH", 1: "MATCH"},
        label2id={"NOT_MATCH": 0, "MATCH": 1},
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        ignore_mismatched_sizes=True,   # rbt6 has no classifier head pre-trained
    )

    output_dir = os.path.join(BASE_DIR, "../../saved_models/rbt6_teacher_output")
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        label_smoothing_factor=0.0,    # Applied manually in loss
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

    trainer = TeacherTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            CustomEarlyStoppingCallback(
                early_stopping_patience=PATIENCE,
                early_stopping_threshold=0.001,
            ),
            CleanLogCallback(),
        ],
    )

    print(f"\n{'='*60}")
    print(f"  rbt6 Teacher Training")
    print(f"  LR={LEARNING_RATE}, Epochs={NUM_EPOCHS}, Patience={PATIENCE}")
    print(f"  Loss: CE(ls=0.05) + RankNet(T={T_TASK})×1.5 + ListNet + R-Drop + FGM")
    print(f"{'='*60}")

    trainer.train()

    # ── Holdout evaluation ────────────────────────────────────────────────────
    print("\n[Evaluate] Testing on holdout set...")
    with open(os.path.join(DATA_DIR, "recommendation_test.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)

    def tok_fn(examples):
        t = tokenizer(
            examples["query"], examples["property"],
            padding="max_length", max_length=MAX_LENGTH, truncation=True,
        )
        t["labels"] = examples["label"]
        return t

    test_ds = Dataset.from_list(test_data).map(tok_fn, batched=True)
    results = trainer.evaluate(test_ds)
    print(f"  Accuracy:  {results.get('eval_accuracy',  0):.5f}")
    print(f"  Precision: {results.get('eval_precision', 0):.5f}")
    print(f"  Recall:    {results.get('eval_recall',    0):.5f}")
    print(f"  F1 Score:  {results.get('eval_f1',        0):.5f}")

    # ── Save teacher to fixed path ─────────────────────────────────────────────
    print(f"\n[Save] Saving teacher to: {TEACHER_SAVE_DIR}")
    os.makedirs(TEACHER_SAVE_DIR, exist_ok=True)
    trainer.save_model(TEACHER_SAVE_DIR)
    tokenizer.save_pretrained(TEACHER_SAVE_DIR)
    print(f"  Teacher saved. This path is the fixed teacher for v2.5 distillation.")
    print(f"  DO NOT overwrite {TEACHER_SAVE_DIR} — it is the permanent teacher.")
    print("=" * 60)


if __name__ == "__main__":
    train_teacher()
