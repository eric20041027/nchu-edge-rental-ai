import os
import torch
from transformers import BertTokenizerFast, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset

# ==========================================
# 步驟 1: 準備您的訓練資料 (線下建模 Data Prep)
# ==========================================
print("[Step 1] Initializing Tokenizer and Dummy Data...")
model_checkpoint = "clue/albert_chinese_tiny" # 使用輕量化中文 ALBERT 作為基底
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)

import json

# 定義標籤對應表 (B-Target: 特徵開頭, I-Target: 特徵內部, O: 無關字元)
label_list = ["O", "B-Target", "I-Target"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)} # Keep id2label for model config

# 讀取訓練集與驗證集
with open("train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
print(f"✅ 成功載入 {len(train_data)} 筆『訓練資料』 (train.json)！")

with open("test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
print(f"✅ 成功載入 {len(test_data)} 筆『驗證資料』 (test.json)！")

# 建立 Dataset
train_dataset_raw = Dataset.from_list(train_data)
eval_dataset_raw = Dataset.from_list(test_data)

# 將假資料轉換為 HuggingFace Dataset 格式 (Tokenization & Alignment)
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], is_split_into_words=True, padding="max_length", max_length=16, truncation=True)
    
    labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # -100 是 PyTorch 忽略計算 loss 的預設值
            else:
                label_ids.append(label_to_id[tags[word_idx]])
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = train_dataset_raw.map(tokenize_and_align_labels, batched=True)
eval_dataset = eval_dataset_raw.map(tokenize_and_align_labels, batched=True)

# ==========================================
# 步驟 2: 載入原始模型並進行微調 (Fine-Tuning)
# ==========================================
print("\n🧠 [Step 2] Loading Model and Starting Fine-Tuning...")
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, 
    num_labels=len(label_list), 
    id2label=id2label,
    label2id=label_to_id
)

# 定義簡單的準確率計算公式 (免額外套件版本)
import numpy as np
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 忽略 -100 (padding/特殊字元) 的標籤
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 計算簡單的 Token 級別準確率
    correct = sum(p == l for pred, lbl in zip(true_predictions, true_labels) for p, l in zip(pred, lbl))
    total = sum(len(lbl) for lbl in true_labels)
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy}

# 設定訓練參數
training_args = TrainingArguments(
    output_dir="./custom_model_output",
    eval_strategy="epoch",  # 每個 epoch 結束後進行驗證
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,         # 10k 資料量大，先設定跑 3 次
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True, # 訓練完自動載入表現最好的一次
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # 加入驗證集
    compute_metrics=compute_metrics # 加入評估公式
)

# 開始訓練 (這就是所謂的線下建模！)
trainer.train()

# ==========================================
# 步驟 3: 模型輕量化匯出 (ONNX Export)
# ==========================================
print("[Step 3] Exporting Fine-Tuned Model to ONNX format...")
# 先將微調好的模型存下來
save_path = "./my_trained_albert"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

onnx_output_path = "my_custom_model.onnx"

# 準備一個假的輸入張量給 ONNX 描繪神經網路計算圖的形狀 (Shape)
dummy_text = "預算五千套房"
inputs = tokenizer(dummy_text, return_tensors="pt", max_length=16, padding="max_length", truncation=True)

# 強制將模型與輸入張量移至 CPU，避免 Mac MPS 裝置在 ONNX 轉換時發生 device mismatch
model.to("cpu")

# 使用 PyTorch 內建的 ONNX 轉換工具
torch.onnx.export(
    model, 
    (inputs["input_ids"].to("cpu"), inputs["attention_mask"].to("cpu"), inputs["token_type_ids"].to("cpu")), 
    onnx_output_path, 
    export_params=True,
    opset_version=18, 
    do_constant_folding=True, # 常數折疊，最佳化體積
    input_names=['input_ids', 'attention_mask', 'token_type_ids'], # 這裡的名字必須對應前端 inference.js 的 key!
    output_names=['logits'],
    dynamic_axes={ # 允許前端輸入不同長度的句子
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }
)

print(f"模型已經匯出至 {onnx_output_path}")
