"""
train_and_export_onnx.py - R-Drop + FGM + KD + Multi-Task Ranking  (v2.5)

Architecture: hfl/rbt3 (3-layer Chinese RoBERTa) ← distilled from rbt6 teacher

Loss = (1-α) × [CE(label_smoothing=0.05) + RankNet(T=2.0)×1.5 + ListNet(T=2.0)]
     + α × T² × KL(student/T ‖ teacher/T)    ← Knowledge Distillation
     + 0.05 × SymKL(pass1 ‖ pass2)            ← R-Drop regularisation
     + FGM adversarial perturbation on word embeddings

Improvements over v2.4 (no-KD baseline):
  1. KD re-enabled (DISTILL_ALPHA_MAX=0.38) — teacher is the dedicated
     rbt6_teacher model trained by train_teacher.py and stored at a
     SEPARATE path (saved_models/rbt6_teacher/) that student training
     NEVER overwrites.  Root cause of v2.4 KD failure:
       - v2.3 teacher overwritten by v2.4α (same SAVED_MODEL_DIR path)
       - pre-trained rbt6 has random classifier head → soft label noise
     Both fixed: teacher path is now immutable.
  2. Dynamic KD α: cosine annealing 0.38 → 0.12 (same as v2.3 plan).
     Conservative upper bound (0.38 vs originally-planned 0.50) accounts
     for rbt6 being a capacity-compressed teacher vs full rbt12.
  3. All v2.4 fixes retained: metric_for_best_model="f1", T_task=2.0,
     label smoothing, cosine LR, stratified negatives, rebalanced weights,
     FGM, R-Drop.

Design decisions (Focal Loss still disabled):
  - Focal Loss (γ=2.0) remains disabled (γ=0.0): v2.4α experiment showed
    Precision −10%, NDCG −0.044.  Chinese rental task has diverse positives
    — no systematic "easy positive" class that Focal Loss is designed for.
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


# ── Configuration ─────────────────────────────────────────────────────────────

STUDENT_CHECKPOINT  = "hfl/rbt3"

# ── Knowledge Distillation (re-enabled in v2.5) ───────────────────────────────
# Teacher: saved_models/rbt6_teacher/ — trained by train_teacher.py.
# This path is INTENTIONALLY different from SAVED_MODEL_DIR so student
# training NEVER overwrites the teacher (root cause of v2.4 KD failure).
#
# If rbt6_teacher/ does not exist yet, run train_teacher.py first:
#   python -m pipeline.model_training.train_teacher
#
# Falls back to "hfl/rbt6" (pre-trained) only as a safety net — but
# pre-trained rbt6 has a random classifier head and makes a poor teacher.
# Always train the teacher before running student training.
_RBT6_TEACHER_DIR   = os.path.join(os.path.dirname(__file__),
                                    "../../saved_models/rbt6_teacher")
TEACHER_MODEL_PATH  = (_RBT6_TEACHER_DIR if os.path.isdir(_RBT6_TEACHER_DIR)
                       else "hfl/rbt6")

DISTILL_TEMPERATURE = 4.0
DISTILL_ALPHA_MAX   = 0.38   # ← KD re-enabled: rbt6_teacher provides good
DISTILL_ALPHA_MIN   = 0.12   #   soft labels; cosine anneal 0.38→0.12

RDROP_ALPHA  = 0.05   # R-Drop symmetric KL weight — conservative to
                      # avoid conflicting with FGM adversarial gradients
FOCAL_GAMMA  = 0.0    # ← Disabled. Focal Loss hurt v2.4 (precision -10%).
                      # CE handles this task fine; kept as 0 (no-op).
LABEL_SMOOTHING = 0.05   # Mild label smoothing for CE calibration

MAX_LENGTH     = 64
ONNX_OUTPUT_PATH = os.path.join(os.path.dirname(__file__),
                                "../../frontend/models/custom_onnx_model_dir/my_custom_model.onnx")
SAVED_MODEL_DIR  = os.path.join(os.path.dirname(__file__),
                                "../../saved_models/rbt3_finetuned")


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_and_balance_data(train_path: str, dev_path: str) -> Tuple[Dataset, Dataset]:
    print(f"\n[Step 1] Loading and balancing data...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"  Raw Train: {len(train_data)} samples")

    hard_examples_path = os.path.join(os.path.dirname(train_path), "hard_examples.json")
    hard_examples = []
    if os.path.exists(hard_examples_path):
        with open(hard_examples_path, "r", encoding="utf-8") as f:
            hard_examples = json.load(f)
        for d in hard_examples:
            d["is_hard"] = True
        print(f"  Loaded {len(hard_examples)} hard examples.")

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    print(f"  Raw Dev:   {len(dev_data)} samples")

    random.seed(42)
    pos_samples = [d for d in train_data if d["label"] == 1]

    # ── Stratified negative sampling ──────────────────────────────────────────
    # Prioritise hard-conflict negatives (rel=0) over random negatives (rel=-1).
    # Hard conflicts teach the model genuine constraint violations; random
    # negatives carry little semantic signal once the model is well-trained.
    neg_all     = [d for d in train_data if d["label"] == 0]
    neg_hard    = [d for d in neg_all if d.get("relevance", -1) == 0]
    neg_easy    = [d for d in neg_all if d.get("relevance", -1) == -1]

    target_neg = len(pos_samples)
    if len(neg_hard) >= target_neg:
        # Enough hard conflicts to fill the quota alone
        neg_samples = random.sample(neg_hard, target_neg)
    else:
        # Take all hard conflicts and fill the remainder with easy negatives
        n_easy = min(target_neg - len(neg_hard), len(neg_easy))
        neg_samples = neg_hard + random.sample(neg_easy, n_easy)
        # If still short (unlikely), top up with any remaining negatives
        if len(neg_samples) < target_neg:
            remaining = [d for d in neg_all if d not in set(neg_samples)]
            neg_samples += random.sample(remaining,
                                         min(target_neg - len(neg_samples), len(remaining)))

    print(f"  Distribution: POS={len(pos_samples)}, NEG={len(neg_samples)} "
          f"(hard-conflict={sum(1 for d in neg_samples if d.get('relevance',-1)==0)}, "
          f"random={sum(1 for d in neg_samples if d.get('relevance',-1)==-1)})")

    train_data = pos_samples + neg_samples + hard_examples
    random.shuffle(train_data)
    print(f"  Final train: {len(train_data)} samples")

    return Dataset.from_list(train_data), Dataset.from_list(dev_data)


# ── Adversarial Training (FGM) ────────────────────────────────────────────────

class FGM:
    def __init__(self, model):
        self.model = model
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


# ── Custom Trainer with Distillation + R-Drop + Focal Loss ────────────────────

class DistillTrainer(Trainer):
    """
    Combined losses:
      - Focal CE       : down-weights easy examples
      - RankNet        : pairwise ranking on graded relevance
      - ListNet        : list-wise KL on relevance distribution
      - KD (dynamic α) : cosine-annealed knowledge distillation from teacher
      - R-Drop         : symmetric KL between two dropout-sampled outputs
      - FGM            : adversarial perturbation on embeddings
    """

    def __init__(self, teacher_model: PreTrainedModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    # ── Dynamic KD alpha ──────────────────────────────────────────────────────

    def _get_current_alpha(self) -> float:
        """Cosine annealing from DISTILL_ALPHA_MAX → DISTILL_ALPHA_MIN."""
        try:
            if self.state.epoch is not None:
                t = min(float(self.state.epoch) / float(self.args.num_train_epochs), 1.0)
                cos_val = (1 + math.cos(math.pi * t)) / 2   # 1.0 at t=0, 0.0 at t=1
                return DISTILL_ALPHA_MIN + cos_val * (DISTILL_ALPHA_MAX - DISTILL_ALPHA_MIN)
        except Exception:
            pass
        return DISTILL_ALPHA_MAX

    # ── Teacher forward ───────────────────────────────────────────────────────

    def _teacher_logits(self, inputs: dict) -> torch.Tensor:
        teacher_inputs = {
            k: v for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "token_type_ids", "labels")
        }
        with torch.no_grad():
            out = self.teacher(**teacher_inputs)
        return out.logits   # [B, 2]

    # ── Core loss computation ─────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False,
                     use_rdrop: bool = True, **kwargs):
        labels    = inputs.get("labels")
        weights   = inputs.pop("sample_weight", None)
        relevance = inputs.get("relevance", labels.float())

        # ── Forward pass 1 ────────────────────────────────────────────────────
        outputs1 = model(**inputs)
        logits1  = outputs1.get("logits")   # [B, 2]

        # ── R-Drop: forward pass 2 (different dropout mask) ───────────────────
        if model.training and use_rdrop and RDROP_ALPHA > 0:
            outputs2 = model(**inputs)
            logits2  = outputs2.get("logits")
            # Symmetric KL
            p1 = F.softmax(logits1, dim=-1)
            p2 = F.softmax(logits2, dim=-1)
            rdrop_loss = (
                F.kl_div(F.log_softmax(logits1, dim=-1), p2.detach(), reduction="batchmean") +
                F.kl_div(F.log_softmax(logits2, dim=-1), p1.detach(), reduction="batchmean")
            ) / 2.0
            logits = (logits1 + logits2) / 2.0   # Average for task/KD losses
        else:
            logits     = logits1
            rdrop_loss = torch.tensor(0.0, device=logits1.device)

            # Temperature scaling for ranking losses (same as v2.3 T_task=2.0;
        # softens logits so RankNet/ListNet gradients are numerically stable)
        T_task     = 2.0
        rel_logits = logits[:, 1] / T_task  # [B]

        # ── 1. Focal Cross-Entropy ─────────────────────────────────────────────
        # Focal weight: (1 - p_t)^γ focuses loss on examples the model got wrong
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

        # Precision penalty: 1.5× for false positives (same as v2.3; keeps
        # recall high while gently discouraging over-prediction of MATCH)
        pp = torch.where(labels == 0,
                         torch.tensor(1.5, device=labels.device),
                         torch.ones(labels.shape, device=labels.device))

        if weights is not None:
            task_loss = (focal_ce * weights * pp).mean()
        else:
            task_loss = (focal_ce * pp).mean()

        # ── 2. RankNet (pairwise) ──────────────────────────────────────────────
        s_i = rel_logits.unsqueeze(1)
        s_j = rel_logits.unsqueeze(0)
        r_i = relevance.unsqueeze(1)
        r_j = relevance.unsqueeze(0)
        mask = (r_i > r_j).float()
        if mask.sum() > 0:
            ranknet = torch.log(1 + torch.exp(-(s_i - s_j))) * mask
            task_loss = task_loss + (ranknet.sum() / mask.sum()) * 1.5

        # ── 3. ListNet (listwise) ─────────────────────────────────────────────
        target_dist = torch.softmax(relevance, dim=0)
        pred_dist   = torch.log_softmax(rel_logits, dim=0)
        task_loss   = task_loss + (-torch.sum(target_dist * pred_dist)) * 1.0

        # ── 4. Knowledge Distillation (disabled when DISTILL_ALPHA_MAX=0.0) ─────
        alpha = self._get_current_alpha()
        if alpha > 0.0:
            teacher_device = next(self.teacher.parameters()).device
            if teacher_device != logits.device:
                self.teacher.to(logits.device)
            teacher_logits = self._teacher_logits(inputs)
            T = DISTILL_TEMPERATURE
            kl_loss = F.kl_div(
                F.log_softmax(logits         / T, dim=-1),
                F.softmax    (teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
        else:
            kl_loss = torch.tensor(0.0, device=logits.device)

        # ── Combine all losses ────────────────────────────────────────────────
        loss = ((1.0 - alpha) * task_loss
                + alpha * kl_loss
                + RDROP_ALPHA * rdrop_loss)

        return (loss, outputs1) if return_outputs else loss

    # ── Training step: normal + FGM adversarial ───────────────────────────────

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Normal backward → FGM attack → adversarial backward → restore.
        Uses self.accelerator.backward() so FP16 GradScaler is initialised
        correctly before Trainer calls _clip_grad_norm.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 1. Normal forward (with R-Drop) + backward
        loss = self.compute_loss(model, inputs, use_rdrop=True)
        self.accelerator.backward(loss)   # ← FP16-safe (initialises GradScaler)

        # 2. FGM attack on student embeddings; adversarial pass without R-Drop
        fgm = FGM(model)
        fgm.attack()
        loss_adv = self.compute_loss(model, inputs, use_rdrop=False)
        self.accelerator.backward(loss_adv)   # ← accumulates adversarial grads
        fgm.restore()

        return loss.detach()


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenize_datasets(
    train_dataset: Dataset,
    eval_dataset:  Dataset,
    tokenizer:     PreTrainedTokenizer,
) -> Tuple[Dataset, Dataset]:
    print("\n[Step 2] Tokenizing sentence pairs...")

    # Sample weights close to v2.3 (rel=3 stays high — perfect pairs carry the
    # strongest ranking signal; stratified sampling already handles negatives).
    rel_map = {3: 12.0, 2: 4.0, 1: 0.8, 0: 5.0, -1: 0.3}

    def tokenize_fn(examples):
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
        train_dataset.map(tokenize_fn, batched=True),
        eval_dataset .map(tokenize_fn, batched=True),
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


class AlphaLogCallback(TrainerCallback):
    """Prints the current KD alpha at the start of each epoch (no-op when KD disabled)."""
    def __init__(self, trainer_ref):
        self._trainer = trainer_ref

    def on_epoch_begin(self, args, state, control, **kwargs):
        if DISTILL_ALPHA_MAX > 0.0:
            alpha = self._trainer._get_current_alpha()
            print(f"  [KD] Epoch {state.epoch:.0f} | alpha={alpha:.3f} "
                  f"(range [{DISTILL_ALPHA_MIN}, {DISTILL_ALPHA_MAX}])")


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
            print(f"  [EarlyStopping] New best {args.metric_for_best_model}={metric_val:.5f}  "
                  f"Patience reset.")


# ── Train ──────────────────────────────────────────────────────────────────────

def train_model(
    train_dataset: Dataset,
    eval_dataset:  Dataset,
) -> Tuple[Trainer, PreTrainedModel]:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "[Train] CUDA GPU not available. Training requires a CUDA-capable GPU.\n"
            "  Check: python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "  If using WSL/remote, ensure the CUDA driver and PyTorch CUDA build match."
        )
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n[Train] Device: CUDA — {gpu_name} ({gpu_mem:.1f} GB)")

    # ── Load teacher ──────────────────────────────────────────────────────────
    if DISTILL_ALPHA_MAX > 0.0:
        print(f"[Distill] Loading teacher from: {TEACHER_MODEL_PATH}")
        teacher = AutoModelForSequenceClassification.from_pretrained(
            TEACHER_MODEL_PATH,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[Distill] Teacher loaded  "
              f"(α: {DISTILL_ALPHA_MAX}→{DISTILL_ALPHA_MIN} cosine, T={DISTILL_TEMPERATURE})")
    else:
        print(f"[Train] KD disabled (α=0); creating dummy teacher (unused)")
        teacher = AutoModelForSequenceClassification.from_pretrained(
            "hfl/rbt3", num_labels=2, ignore_mismatched_sizes=True,
        )
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # ── Load student (fresh rbt3 from HF) ─────────────────────────────────────
    print(f"[Train]   Student: {STUDENT_CHECKPOINT}")
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_CHECKPOINT,
        num_labels=2,
        id2label={0: "NOT_MATCH", 1: "MATCH"},
        label2id={"NOT_MATCH": 0, "MATCH": 1},
        hidden_dropout_prob=0.15,
        attention_probs_dropout_prob=0.15,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__),
                                "../../saved_models/recommendation_model_output"),
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,              # Extra epochs; early stopping on F1
        weight_decay=0.01,                # Same as v2.3 (0.05 over-regularises)
        warmup_ratio=0.08,
        label_smoothing_factor=0.0,       # Smoothing applied manually in Focal CE
        lr_scheduler_type="cosine",       # Cosine decay for smoother convergence
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",       # ← FIX: was "loss" (bug)
        greater_is_better=True,           # ← FIX: was False
        report_to="none",
        save_total_limit=2,
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),   # FP16 mixed-precision on CUDA (faster, same accuracy)
        log_level="error",
        max_grad_norm=1.0,                # Explicit gradient clipping
    )

    trainer = DistillTrainer(
        teacher_model=teacher,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            CustomEarlyStoppingCallback(
                early_stopping_patience=6,        # Tighter patience on F1
                early_stopping_threshold=0.001,   # F1 must improve by ≥0.1%
            ),
            CleanLogCallback(),
        ],
    )
    # Alpha-logger needs a reference to the trainer (added after construction)
    trainer.add_callback(AlphaLogCallback(trainer))

    trainer.train()
    return trainer, student


# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate_on_test(trainer: Trainer, tokenizer: PreTrainedTokenizer, test_path: str):
    print("\n[Evaluate] Testing on holdout set...")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    raw = Dataset.from_list(test_data)

    def tok_fn(examples):
        t = tokenizer(
            examples["query"], examples["property"],
            padding="max_length", max_length=MAX_LENGTH, truncation=True,
        )
        t["labels"] = examples["label"]
        return t

    test_ds = raw.map(tok_fn, batched=True, remove_columns=raw.column_names)
    results = trainer.evaluate(test_ds)

    print(f"  Accuracy:  {results.get('eval_accuracy',  0):.5f}")
    print(f"  Precision: {results.get('eval_precision', 0):.5f}")
    print(f"  Recall:    {results.get('eval_recall',    0):.5f}")
    print(f"  F1 Score:  {results.get('eval_f1',        0):.5f}")


# ── ONNX Export + Quantization ────────────────────────────────────────────────

def export_to_onnx(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    print("\n[Export] Saving model and exporting to ONNX...")

    model.save_pretrained(SAVED_MODEL_DIR)
    tokenizer.save_pretrained(SAVED_MODEL_DIR)

    dummy_query    = "預算五千套房"
    dummy_property = "套房 南區 5000元"
    inputs = tokenizer(
        dummy_query, dummy_property,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )

    model.to("cpu")
    model.eval()
    model.config.attn_implementation = "eager"

    from transformers import AutoModelForSequenceClassification as AMSC
    model = AMSC.from_pretrained(SAVED_MODEL_DIR, attn_implementation="eager")
    model.to("cpu")
    model.eval()

    import logging as _logging
    _logging.getLogger("torch.onnx").setLevel(_logging.ERROR)
    # Force UTF-8 stdout to prevent UnicodeEncodeError on Windows (torch.onnx prints ✅)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    torch.onnx.export(
        model,
        (
            inputs["input_ids"].to("cpu"),
            inputs["attention_mask"].to("cpu"),
            inputs["token_type_ids"].to("cpu"),
        ),
        ONNX_OUTPUT_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits":         {0: "batch_size"},
        },
    )
    print(f"  FP32 model exported to: {ONNX_OUTPUT_PATH}")

    # ── Sync tokenizer + config to frontend directory ─────────────────────────
    import shutil
    frontend_dir = os.path.dirname(ONNX_OUTPUT_PATH)
    for fname in ("tokenizer.json", "tokenizer_config.json", "config.json",
                  "special_tokens_map.json", "vocab.txt"):
        src = os.path.join(SAVED_MODEL_DIR, fname)
        dst = os.path.join(frontend_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Synced {fname} -> frontend model dir")

    # ── INT8 Dynamic Quantization ─────────────────────────────────────────────
    print("\n[Quantize] Applying INT8 dynamic quantization...")
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process

        quant_path = ONNX_OUTPUT_PATH.replace(".onnx", "_quant.onnx")

        # Strip value_info to avoid shape conflicts
        fp32_model = onnx.load(ONNX_OUTPUT_PATH)
        for _ in range(len(fp32_model.graph.value_info)):
            fp32_model.graph.value_info.pop()
        onnx.save(fp32_model, ONNX_OUTPUT_PATH)

        # Optional preprocessing pass
        pre_path = ONNX_OUTPUT_PATH.replace(".onnx", ".pre.onnx")
        try:
            quant_pre_process(ONNX_OUTPUT_PATH, pre_path)
            input_path = pre_path
        except Exception as e:
            print(f"  Preprocess skipped: {e}")
            input_path = ONNX_OUTPUT_PATH

        quantize_dynamic(
            model_input=input_path,
            model_output=quant_path,
            op_types_to_quantize=["MatMul", "Gemm", "Gather"],
            weight_type=QuantType.QInt8,
            use_external_data_format=False,
            per_channel=False,
            reduce_range=True,
            extra_options={"MatMulConstBOnly": True},
        )

        # Ensure single-file output (no external tensors)
        final = onnx.load(quant_path)
        onnx.save(final, quant_path, save_as_external_data=False)

        if os.path.exists(pre_path):
            os.remove(pre_path)

        fp32_mb  = os.path.getsize(ONNX_OUTPUT_PATH) / (1024 * 1024)
        quant_mb = os.path.getsize(quant_path)        / (1024 * 1024)
        print(f"  INT8 quantized: {fp32_mb:.1f} MB -> {quant_mb:.1f} MB "
              f"({100*(1 - quant_mb/fp32_mb):.0f}% reduction)")
        print(f"  Quantized model: {quant_path}")
    except Exception as e:
        print(f"  Quantization failed (non-fatal): {e}")

    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    kd_status = "disabled" if DISTILL_ALPHA_MAX == 0.0 else f"{DISTILL_ALPHA_MAX}→{DISTILL_ALPHA_MIN} (cosine)"
    print("=" * 60)
    print(f"  v2.5: CE + RankNet + ListNet + KD + R-Drop + FGM")
    print(f"  KD α: {kd_status},  T={DISTILL_TEMPERATURE}")
    print(f"  R-Drop α={RDROP_ALPHA},  Focal γ={FOCAL_GAMMA}")
    print(f"  Teacher: {TEACHER_MODEL_PATH}")
    print("=" * 60)

    # Warn if teacher path doesn't exist or is the pre-trained fallback
    if not os.path.isdir(_RBT6_TEACHER_DIR):
        print(f"\n[WARN] rbt6_teacher not found at {_RBT6_TEACHER_DIR}")
        print(f"  Using pre-trained fallback: {TEACHER_MODEL_PATH}")
        print(f"  Pre-trained rbt6 has random classifier head → poor teacher.")
        print(f"  Run train_teacher.py first for best results.\n")

    tokenizer = BertTokenizerFast.from_pretrained(STUDENT_CHECKPOINT)

    base = os.path.join(os.path.dirname(__file__), "../../data/processed")
    train_raw, eval_raw = load_and_balance_data(
        os.path.join(base, "recommendation_train.json"),
        os.path.join(base, "recommendation_dev.json"),
    )

    train_ds, eval_ds = tokenize_datasets(train_raw, eval_raw, tokenizer)
    trainer, model    = train_model(train_ds, eval_ds)

    evaluate_on_test(trainer, tokenizer,
                     os.path.join(base, "recommendation_test.json"))
    export_to_onnx(model, tokenizer)


if __name__ == "__main__":
    main()
