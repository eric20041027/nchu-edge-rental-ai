import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset

# ==========================================
# 步驟 1: 準備您的訓練資料 (線下建模 Data Prep)
# ==========================================
print("🚀 [Step 1] Initializing Tokenizer and Dummy Data...")
model_checkpoint = "clue/albert_chinese_tiny" # 使用輕量化中文 ALBERT 作為基底
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 定義標籤對應表 (B-Target: 特徵開頭, I-Target: 特徵內部, O: 無關字元)
label_list = ["O", "B-Target", "I-Target"]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# 為了示範，我們建立三筆簡單的微調假資料 (實際專題中這裡會是一個大型 JSON 或 CSV)
# 句子1: "預算六千" -> O, O, B, I
# 句子2: "獨洗套房" -> B, I, O, O
dummy_data = [
    {"tokens": ["預", "算", "六", "千"], "ner_tags": [0, 0, 1, 2]},
    {"tokens": ["獨", "洗", "套", "房"], "ner_tags": [1, 2, 0, 0]},
    {"tokens": ["近", "正", "門", "好"], "ner_tags": [1, 2, 2, 0]},
]

# 將假資料轉換為 HuggingFace Dataset 格式
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding="max_length", max_length=16, truncation=True)
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # -100 是 PyTorch 忽略計算 loss 的預設值
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

dataset = Dataset.from_list(dummy_data)
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# ==========================================
# 步驟 2: 載入原始模型並進行微調 (Fine-Tuning)
# ==========================================
print("🧠 [Step 2] Loading Model and Starting Fine-Tuning...")
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, 
    num_labels=len(label_list), 
    id2label=id2label, 
    label2id=label2id
)

# 設定訓練參數
training_args = TrainingArguments(
    output_dir="./custom_model_output",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3, # 為了示範只跑 3 個 Epoch
    weight_decay=0.01,
    logging_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# 開始訓練 (這就是所謂的線下建模！)
trainer.train()

# ==========================================
# 步驟 3: 模型輕量化匯出 (ONNX Export)
# ==========================================
print("📉 [Step 3] Exporting Fine-Tuned Model to ONNX format...")
# 先將微調好的模型存下來
save_path = "./my_trained_albert"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

onnx_output_path = "my_custom_model.onnx"

# 準備一個假的輸入張量給 ONNX 描繪神經網路計算圖的形狀 (Shape)
dummy_text = "預算五千套房"
inputs = tokenizer(dummy_text, return_tensors="pt", max_length=16, padding="max_length", truncation=True)

# 使用 PyTorch 內建的 ONNX 轉換工具
torch.onnx.export(
    model, 
    (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]), 
    onnx_output_path, 
    export_params=True,
    opset_version=14, 
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

print(f"✅ 大功告成！專屬您的模型已經匯出至 {onnx_output_path}，體積僅 16MB！")
