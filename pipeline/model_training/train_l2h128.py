"""
train_l2h128.py - Cross-Encoder with uer/chinese_roberta_L-2_H-128

Architecture: 2-layer Chinese RoBERTa (hidden=128) — ~5x smaller than rbt3
Goal: test whether a much smaller model can match rbt3 NDCG@5 on this task.

Loss = CE(label_smoothing=0.05) + RankNet(T=2.0)×1.5 + ListNet(T=2.0)
     + FGM adversarial perturbation
     (KD disabled: hidden_size mismatch with rbt6 teacher would require adapter)
"""
import sys
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]   = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]          = "1"

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
    PreTrainedModel,
    PreTrainedTokenizer,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()

from datasets import Dataset
import torch.nn.functional as F


# ── Configuration ──────────────────────────────────────────────────────────────

STUDENT_CHECKPOINT = "uer/chinese_roberta_L-2_H-128"

FOCAL_GAMMA     = 0.0
LABEL_SMOOTHING = 0.05
MAX_LENGTH      = 64

ONNX_OUTPUT_PATH = os.path.join(os.path.dirname(__file__),
    "../../frontend/models/custom_onnx_model_dir/my_custom_model_l2h128.onnx")
SAVED_MODEL_DIR  = os.path.join(os.path.dirname(__file__),
    "../../saved_models/l2h128_finetuned")


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
    neg_all     = [d for d in train_data if d["label"] == 0]
    neg_samples = random.sample(neg_all, min(len(pos_samples), len(neg_all)))

    train_data = pos_samples + neg_samples + hard_examples
    random.shuffle(train_data)
    print(f"  Final train: {len(train_data)} samples")

    return Dataset.from_list(train_data), Dataset.from_list(dev_data)


# ── Shared utilities ───────────────────────────────────────────────────────────
from .training_utils import FGM, compute_metrics, CleanLogCallback, CustomEarlyStoppingCallback


# ── Trainer: CE + RankNet + ListNet + FGM (no KD) ────────────────────────────

class RankTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels    = inputs.get("labels")
        weights   = inputs.get("sample_weight", None)
        relevance = inputs.get("relevance", labels.float())

        _EXTRA = {"sample_weight", "relevance"}
        model_inputs = {k: v for k, v in inputs.items() if k not in _EXTRA}

        outputs = model(**model_inputs)
        logits  = outputs.get("logits")  # [B, 2]

        T_task     = 2.0
        rel_logits = logits[:, 1] / T_task

        # 1. Focal CE
        with torch.no_grad():
            p_t = torch.exp(-F.cross_entropy(logits.detach(), labels, reduction="none"))
        focal_weight = (1.0 - p_t) ** FOCAL_GAMMA
        ce_loss = F.cross_entropy(logits, labels, reduction="none",
                                  label_smoothing=LABEL_SMOOTHING)
        focal_ce = focal_weight * ce_loss

        pp = torch.where(labels == 0,
                         torch.tensor(1.5, device=labels.device),
                         torch.ones(labels.shape, device=labels.device))

        if weights is not None:
            task_loss = (focal_ce * weights * pp).mean()
        else:
            task_loss = (focal_ce * pp).mean()

        # 2. RankNet
        s_i = rel_logits.unsqueeze(1)
        s_j = rel_logits.unsqueeze(0)
        r_i = relevance.unsqueeze(1)
        r_j = relevance.unsqueeze(0)
        mask = (r_i > r_j).float()
        if mask.sum() > 0:
            ranknet   = F.softplus(-(s_i - s_j)) * mask
            task_loss = task_loss + (ranknet.sum() / mask.sum()) * 1.5

        # 3. ListNet
        target_dist = torch.softmax(relevance, dim=0)
        pred_dist   = torch.log_softmax(rel_logits, dim=0)
        task_loss   = task_loss + (-torch.sum(target_dist * pred_dist)) * 1.0

        return (task_loss, outputs) if return_outputs else task_loss

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)
        self.accelerator.backward(loss)

        fgm = FGM(model)
        fgm.attack()
        try:
            loss_adv = self.compute_loss(model, inputs)
            self.accelerator.backward(loss_adv)
        finally:
            fgm.restore()

        return loss.detach()


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenize_datasets(
    train_dataset: Dataset,
    eval_dataset:  Dataset,
    tokenizer:     PreTrainedTokenizer,
) -> Tuple[Dataset, Dataset]:
    print("\n[Step 2] Tokenizing sentence pairs...")
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


# ── Train ──────────────────────────────────────────────────────────────────────

def train_model(
    train_dataset: Dataset,
    eval_dataset:  Dataset,
) -> Tuple[Trainer, PreTrainedModel]:

    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n[Train] Device: CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"\n[Train] Device: MPS (Apple Silicon)")
    else:
        device = "cpu"
        print(f"\n[Train] Device: CPU (no GPU available)")

    print(f"[Train] Student: {STUDENT_CHECKPOINT}")
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_CHECKPOINT,
        num_labels=2,
        id2label={0: "NOT_MATCH", 1: "MATCH"},
        label2id={"NOT_MATCH": 0, "MATCH": 1},
        hidden_dropout_prob=0.15,
        attention_probs_dropout_prob=0.15,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__),
                                "../../saved_models/l2h128_output"),
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=15,
        weight_decay=0.01,
        warmup_ratio=0.08,
        label_smoothing_factor=0.0,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        disable_tqdm=False,
        fp16=(device == "cuda"),
        use_mps_device=(device == "mps"),
        log_level="error",
        max_grad_norm=1.0,
    )

    trainer = RankTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            CustomEarlyStoppingCallback(
                early_stopping_patience=6,
                early_stopping_threshold=0.001,
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

    from transformers import AutoModelForSequenceClassification as AMSC
    model = AMSC.from_pretrained(SAVED_MODEL_DIR, attn_implementation="eager")
    model.to("cpu")
    model.eval()

    import logging as _logging
    _logging.getLogger("torch.onnx").setLevel(_logging.ERROR)
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

    import shutil
    frontend_dir = os.path.dirname(ONNX_OUTPUT_PATH)
    for fname in ("tokenizer.json", "tokenizer_config.json", "config.json",
                  "special_tokens_map.json", "vocab.txt"):
        src = os.path.join(SAVED_MODEL_DIR, fname)
        dst = os.path.join(frontend_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Synced {fname} -> frontend model dir")

    # INT8 Dynamic Quantization
    print("\n[Quantize] Applying INT8 dynamic quantization...")
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process

        quant_path = ONNX_OUTPUT_PATH.replace(".onnx", "_quant.onnx")

        fp32_model = onnx.load(ONNX_OUTPUT_PATH)
        for _ in range(len(fp32_model.graph.value_info)):
            fp32_model.graph.value_info.pop()
        onnx.save(fp32_model, ONNX_OUTPUT_PATH)

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
            per_channel=True,
            reduce_range=True,
            extra_options={"MatMulConstBOnly": True},
        )

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
    print("=" * 60)
    print(f"  L2H128: CE + RankNet + ListNet + FGM  (KD disabled)")
    print(f"  Student: {STUDENT_CHECKPOINT}")
    print(f"  Target:  {ONNX_OUTPUT_PATH}")
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
