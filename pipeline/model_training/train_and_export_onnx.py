"""
train_and_export_onnx.py - Knowledge Distillation: rbt6 (Teacher) → rbt3 (Student)

Teacher : hfl/rbt6 fine-tuned on rental data  (F1 84.8%, 58 MB quantized)
Student : hfl/rbt3 (3-layer Chinese RoBERTa)  (target ~20 MB quantized)

Loss = (1-α) × [CE + RankNet + ListNet]   ← task losses on student
     +    α  × T² × KL(student/T ‖ teacher/T)  ← distillation loss

α = 0.40,  T = 4.0  (temperature)
"""
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

from typing import Tuple, Dict, List
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

STUDENT_CHECKPOINT  = "hfl/rbt3"    # 3-layer student (target ~20 MB quantized)
TEACHER_MODEL_PATH  = os.path.join(os.path.dirname(__file__),
                                   "../../saved_models/rbt3_finetuned")  # trained rbt6

DISTILL_TEMPERATURE = 4.0   # Higher T → softer distribution → more info transfer
DISTILL_ALPHA       = 0.40  # Weight for KL distillation loss (0 = no distill, 1 = only distill)

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
    neg_samples = [d for d in train_data if d["label"] == 0]
    print(f"  Distribution: POS={len(pos_samples)}, NEG={len(neg_samples)}")

    if len(neg_samples) > len(pos_samples):
        neg_samples = random.sample(neg_samples, len(pos_samples))
    elif len(pos_samples) > len(neg_samples):
        pos_samples = random.sample(pos_samples, len(neg_samples))

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


# ── Custom Trainer with Distillation ──────────────────────────────────────────

class DistillTrainer(Trainer):
    """
    Trainer that combines task losses (CE + RankNet + ListNet) with
    knowledge distillation from a frozen teacher model.
    """

    def __init__(self, teacher_model: PreTrainedModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def _teacher_logits(self, inputs: dict) -> torch.Tensor:
        """Run the frozen teacher on the same inputs (no gradient)."""
        teacher_inputs = {
            k: v for k, v in inputs.items()
            if k in ("input_ids", "attention_mask", "token_type_ids", "labels")
        }
        with torch.no_grad():
            out = self.teacher(**teacher_inputs)
        return out.logits  # [B, 2]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels    = inputs.get("labels")
        weights   = inputs.pop("sample_weight", None)
        relevance = inputs.get("relevance", labels.float())

        # ── Student forward ────────────────────────────────────────────────────
        outputs = model(**inputs)
        logits  = outputs.get("logits")          # [B, 2]

        T_task = 2.0                              # task calibration temperature
        rel_logits = logits[:, 1] / T_task       # [B]

        # 1. Cross-Entropy (weighted, with precision penalty)
        precision_penalty = 1.5
        penalty_w = torch.where(labels == 0, precision_penalty, 1.0)
        ce_loss = nn.CrossEntropyLoss(reduction="none")(logits / T_task, labels)
        if weights is not None:
            task_loss = (ce_loss * weights * penalty_w).mean()
        else:
            task_loss = (ce_loss * penalty_w).mean()

        # 2. RankNet (pairwise)
        s_i = rel_logits.unsqueeze(1)
        s_j = rel_logits.unsqueeze(0)
        r_i = relevance.unsqueeze(1)
        r_j = relevance.unsqueeze(0)
        mask = (r_i > r_j).float()
        if mask.sum() > 0:
            ranknet = torch.log(1 + torch.exp(-(s_i - s_j))) * mask
            task_loss = task_loss + (ranknet.sum() / mask.sum()) * 1.5

        # 3. ListNet (listwise)
        target_dist = torch.softmax(relevance, dim=0)
        pred_dist   = torch.log_softmax(rel_logits, dim=0)
        task_loss   = task_loss + (-torch.sum(target_dist * pred_dist)) * 1.0

        # ── Knowledge Distillation loss ────────────────────────────────────────
        # Move teacher to same device as student output (handles GPU/CPU)
        teacher_device = next(self.teacher.parameters()).device
        student_device = logits.device
        if teacher_device != student_device:
            self.teacher.to(student_device)

        teacher_logits = self._teacher_logits(inputs)       # [B, 2], no grad

        T = DISTILL_TEMPERATURE
        kl_loss = F.kl_div(
            F.log_softmax(logits        / T, dim=-1),       # student log-probs
            F.softmax    (teacher_logits / T, dim=-1),       # teacher probs (soft targets)
            reduction="batchmean",
        ) * (T * T)   # scale by T² to maintain gradient magnitude

        # ── Combine ────────────────────────────────────────────────────────────
        loss = (1.0 - DISTILL_ALPHA) * task_loss + DISTILL_ALPHA * kl_loss

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Standard training step + FGM adversarial perturbation on student only."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 1. Normal forward + backward
        loss = self.compute_loss(model, inputs)
        loss.backward()

        # 2. FGM attack on student embeddings → adversarial backward
        fgm = FGM(model)
        fgm.attack()
        loss_adv = self.compute_loss(model, inputs)
        loss_adv.backward()
        fgm.restore()

        return loss.detach()


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenize_datasets(
    train_dataset: Dataset,
    eval_dataset:  Dataset,
    tokenizer:     PreTrainedTokenizer,
) -> Tuple[Dataset, Dataset]:
    print("\n[Step 2] Tokenizing sentence pairs...")

    rel_map = {3: 15.0, 2: 4.0, 1: 0.8, 0: 6.0, -1: 0.5}

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
    acc  = (preds == labels).mean()
    tp   = ((preds == 1) & (labels == 1)).sum()
    fp   = ((preds == 1) & (labels == 0)).sum()
    fn   = ((preds == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
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
            print("-" * 65)
            print(f"  VALIDATION | Loss: {logs.get('eval_loss', 0):>8.5f} | "
                  f"Acc: {logs.get('eval_accuracy', 0):>7.5f} | "
                  f"Prec: {logs.get('eval_precision', 0):>7.5f} | "
                  f"F1: {logs.get('eval_f1', 0):>7.5f}")
            print("-" * 65)


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        super().on_evaluate(args, state, control, metrics, **kwargs)
        c = self.early_stopping_patience_counter
        p = self.early_stopping_patience
        if c > 0:
            print(f"  [Early Stopping] Patience: {c}/{p}")
        else:
            print(f"  [Early Stopping] New Best! Patience reset to 0/{p}")


# ── Train ──────────────────────────────────────────────────────────────────────

def train_model(
    train_dataset: Dataset,
    eval_dataset:  Dataset,
) -> Tuple[Trainer, PreTrainedModel]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Train] Device: {device.upper()}")

    # ── Load teacher (frozen rbt6) ─────────────────────────────────────────────
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
    print(f"[Distill] Teacher loaded  (α={DISTILL_ALPHA}, T={DISTILL_TEMPERATURE})")

    # ── Load student (rbt3) ────────────────────────────────────────────────────
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
        learning_rate=3e-5,              # slightly higher than rbt6 run (rbt3 needs bigger LR)
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=7,              # extra epochs to compensate for smaller model capacity
        weight_decay=0.01,
        warmup_ratio=0.1,
        label_smoothing_factor=0.0,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        save_total_limit=2,
        disable_tqdm=False,
        fp16=False,
        log_level="error",
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
                early_stopping_patience=8,
                early_stopping_threshold=0.0005,
            ),
            CleanLogCallback(),
        ],
    )

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


# ── ONNX Export ────────────────────────────────────────────────────────────────

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
    print(f"\n  Model exported to: {ONNX_OUTPUT_PATH}")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  Knowledge Distillation: rbt6 → rbt3")
    print(f"  α={DISTILL_ALPHA}  T={DISTILL_TEMPERATURE}  MAX_LEN={MAX_LENGTH}")
    print("=" * 60)

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
