"""
train_and_export_onnx.py - Train Sentence-Pair Classification model and export to ONNX.
Fine-tunes ALBERT on query-property pairs for binary matching (MATCH/NOT_MATCH).
Training data is synthesized by generate_dataset.py.
"""
import os
import json
import random
import torch
import numpy as np
from typing import Tuple, Dict, Any, List
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from datasets import Dataset

# Global Configurations
MODEL_CHECKPOINT = "clue/albert_chinese_tiny"
MAX_LENGTH = 128
ONNX_OUTPUT_PATH = "my_custom_model.onnx"
SAVED_MODEL_DIR = "./my_trained_albert"

def load_and_balance_data(train_path: str, dev_path: str) -> Tuple[Dataset, Dataset]:
    """Loads JSON datasets and balances the training classes."""
    print("=" * 60)
    print("[Step 1] Loading and balancing datasets...")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"  Raw Train: {len(train_data)} samples")

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    print(f"  Raw Dev:   {len(dev_data)} samples")

    random.seed(42)
    pos_samples = [d for d in train_data if d["label"] == 1]
    neg_samples = [d for d in train_data if d["label"] == 0]
    print(f"  Original distribution: POS={len(pos_samples)}, NEG={len(neg_samples)}")

    if len(neg_samples) > len(pos_samples):
        neg_samples = random.sample(neg_samples, len(pos_samples))
        print(f"  Balanced NEG down to {len(neg_samples)}")
    elif len(pos_samples) > len(neg_samples):
        pos_samples = random.sample(pos_samples, len(neg_samples))
        print(f"  Balanced POS down to {len(pos_samples)}")

    train_data = pos_samples + neg_samples
    random.shuffle(train_data)
    print(f"  Final balanced train set: {len(train_data)} samples")

    return Dataset.from_list(train_data), Dataset.from_list(dev_data)


def tokenize_datasets(
    train_dataset: Dataset, 
    eval_dataset: Dataset, 
    tokenizer: PreTrainedTokenizer
) -> Tuple[Dataset, Dataset]:
    """Tokenizes datasets as sentence pairs [CLS] query [SEP] property [SEP]."""
    print("\n[Step 2] Tokenizing sentence pairs...")

    def tokenize_function(examples: Dict[str, list]) -> Dict[str, list]:
        tokenized = tokenizer(
            examples["query"],
            examples["property"],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True)
    
    return train_tokenized, eval_tokenized


def compute_metrics(p: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Computes binary classification metrics for the Trainer."""
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    
    accuracy = (preds == labels).mean()
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train_model(train_dataset: Dataset, eval_dataset: Dataset) -> Tuple[Trainer, PreTrainedModel]:
    """Initializes model, configures Trainer, and fine-tunes ALBERT."""
    print("\n[Step 3] Loading model and starting fine-tuning...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "NOT_MATCH", 1: "MATCH"},
        label2id={"NOT_MATCH": 0, "MATCH": 1},
    )

    training_args = TrainingArguments(
        output_dir="./recommendation_model_output",
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=5e-5,               
        per_device_train_batch_size=32,   
        per_device_eval_batch_size=32,
        num_train_epochs=8,               
        weight_decay=0.01,
        warmup_ratio=0.1,                 
        label_smoothing_factor=0.0,       
        logging_steps=50,                 
        logging_first_step=True,          
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
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
    return trainer, model


def evaluate_on_test(trainer: Trainer, tokenizer: PreTrainedTokenizer, test_path: str):
    """Evaluates the trained model against a holdout test dataset."""
    print("\n[Step 4] Evaluating on test set...")
    
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_dataset_raw = Dataset.from_list(test_data)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["query"], examples["property"],
            padding="max_length", max_length=MAX_LENGTH, truncation=True,
        )
        tokenized["labels"] = examples["label"]
        return tokenized
        
    test_dataset = test_dataset_raw.map(tokenize_function, batched=True)
    test_results = trainer.evaluate(test_dataset)

    print(f"  Test Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"  Test Precision: {test_results['eval_precision']:.4f}")
    print(f"  Test Recall:    {test_results['eval_recall']:.4f}")
    print(f"  Test F1:        {test_results['eval_f1']:.4f}")


def export_to_onnx(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Saves PyTorch model and tokenizers, then exports architecture to ONNX format."""
    print("\n[Step 5] Saving model and exporting to ONNX...")

    model.save_pretrained(SAVED_MODEL_DIR)
    tokenizer.save_pretrained(SAVED_MODEL_DIR)

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
        ONNX_OUTPUT_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "num_labels"},
        },
    )
    print(f"\n  Model exported to: {ONNX_OUTPUT_PATH}")
    print("=" * 60)


def main():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    
    train_dataset_raw, eval_dataset_raw = load_and_balance_data(
        "recommendation_train.json", "recommendation_dev.json"
    )
    
    train_dataset, eval_dataset = tokenize_datasets(train_dataset_raw, eval_dataset_raw, tokenizer)
    
    trainer, model = train_model(train_dataset, eval_dataset)
    
    evaluate_on_test(trainer, tokenizer, "recommendation_test.json")
    export_to_onnx(model, tokenizer)


if __name__ == "__main__":
    main()
