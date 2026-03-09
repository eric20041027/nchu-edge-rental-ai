"""
train_and_export_onnx.py — 訓練 Sentence-Pair Classification 推薦模型並匯出 ONNX

模型學習判斷 "使用者查詢" 與 "房屋描述" 是否匹配 (二分類)。
訓練資料由 generate_dataset.py 自動從 CSV 房源資料生成。
"""
import os
import json
import torch
import numpy as np
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# ==========================================
# Step 1: Load Data
# ==========================================
print("=" * 60)
print("[Step 1] Loading datasets...")

model_checkpoint = "clue/albert_chinese_tiny"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)

with open("recommendation_train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
print(f"  Train: {len(train_data)} samples")

with open("recommendation_dev.json", "r", encoding="utf-8") as f:
    dev_data = json.load(f)
print(f"  Dev:   {len(dev_data)} samples")

# 統計類別分布
train_pos = sum(1 for d in train_data if d["label"] == 1)
train_neg = len(train_data) - train_pos
print(f"  Train distribution: POS={train_pos}, NEG={train_neg}, ratio=1:{train_neg/train_pos:.1f}")

train_dataset_raw = Dataset.from_list(train_data)
eval_dataset_raw = Dataset.from_list(dev_data)

# ==========================================
# Step 2: Tokenize as Sentence Pairs
# ==========================================
print("\n[Step 2] Tokenizing sentence pairs...")

MAX_LENGTH = 128

def tokenize_function(examples):
    """Tokenize as [CLS] query [SEP] property [SEP]"""
    tokenized = tokenizer(
        examples["query"],
        examples["property"],
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
    )
    tokenized["labels"] = examples["label"]
    return tokenized

train_dataset = train_dataset_raw.map(tokenize_function, batched=True)
eval_dataset = eval_dataset_raw.map(tokenize_function, batched=True)

# ==========================================
# Step 3: Load Model & Train
# ==========================================
print("\n[Step 3] Loading model and starting fine-tuning...")

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label={0: "NOT_MATCH", 1: "MATCH"},
    label2id={"NOT_MATCH": 0, "MATCH": 1},
)

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    accuracy = (preds == labels).mean()

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

training_args = TrainingArguments(
    output_dir="./recommendation_model_output",
    # 每 100 步顯示一次驗證結果 (含 accuracy)
    eval_strategy="steps",
    eval_steps=100,
    learning_rate=3e-5,              # 稍微提高學習率
    per_device_train_batch_size=32,   # 加大 batch size
    per_device_eval_batch_size=32,
    num_train_epochs=8,               # 增加到 8 個 epoch
    weight_decay=0.01,
    warmup_ratio=0.1,                 # 前 10% 步數慢慢增加學習率
    label_smoothing_factor=0.1,       # 標籤平滑，防止過擬合
    logging_steps=50,                 # 每 50 步顯示 loss
    logging_first_step=True,          # 第一步就顯示
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
    # 早停：如果 3 次評估都沒改善就停止
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# ==========================================
# Step 4: Evaluate on Test Set
# ==========================================
print("\n[Step 4] Evaluating on test set...")

with open("recommendation_test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_dataset_raw = Dataset.from_list(test_data)
test_dataset = test_dataset_raw.map(tokenize_function, batched=True)
test_results = trainer.evaluate(test_dataset)

print(f"  Test Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"  Test Precision: {test_results['eval_precision']:.4f}")
print(f"  Test Recall:    {test_results['eval_recall']:.4f}")
print(f"  Test F1:        {test_results['eval_f1']:.4f}")

# ==========================================
# Step 5: Save & Export to ONNX
# ==========================================
print("\n[Step 5] Saving model and exporting to ONNX...")

save_path = "./my_trained_albert"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

onnx_output_path = "my_custom_model.onnx"

dummy_query = "預算五千套房"
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

torch.onnx.export(
    model,
    (
        inputs["input_ids"].to("cpu"),
        inputs["attention_mask"].to("cpu"),
        inputs["token_type_ids"].to("cpu"),
    ),
    onnx_output_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
)

print(f"\n  Model exported to: {onnx_output_path}")
print("=" * 60)
print("Training complete! Next steps:")
print("  1. Run: python3 precompute_embeddings.py")
print("  2. Copy model files to custom_onnx_model_dir/")
print("  3. Refresh browser to test")
