"""
train_and_export_onnx.py - Train Sentence-Pair Classification model and export to ONNX.
Fine-tunes hfl/rbt3 (3-layer Chinese RoBERTa) on query-property pairs for binary matching.
Training data is synthesized by generate_dataset.py.
"""
import os
import json
import random
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

from typing import Tuple, Dict, Any, List
from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    PrinterCallback,
    PreTrainedModel,


    PreTrainedTokenizer,
    logging
)
logging.set_verbosity_error() # Suppress "Some weights were not initialized" and config logs

from datasets import Dataset
from torch import nn


# Global Configurations
MODEL_CHECKPOINT = "hfl/rbt3"   # Upgraded: 3-layer Chinese RoBERTa (better semantic understanding)
MAX_LENGTH = 64                   # Aligned with inference.js MAX_LENGTH for consistency
ONNX_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../../frontend/models/custom_onnx_model_dir/my_custom_model.onnx")
SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../saved_models/rbt3_finetuned")

def load_and_balance_data(train_path: str, dev_path: str) -> Tuple[Dataset, Dataset]:
    """Loads JSON datasets and balances the training classes."""
    print("[Load Data] Loading and balancing datasets...")


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


class WeightedTrainer(Trainer):
    """Custom Trainer to support sample weighting based on graded relevance."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        weights = inputs.pop("sample_weight", None)
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if weights is not None:
            # CrossEntropyLoss with reduction='none' to apply per-sample weights
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            loss = (loss * weights).mean()
        else:
            loss = outputs.get("loss")
            
        return (loss, outputs) if return_outputs else loss



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
        
        # Map relevance (0-3) to sample weights
        # Perfect Match (3) = 2.0, Good (2) = 1.5, Partial (1) = 1.0, Neg (0) = 1.0
        rel_map = {3: 2.0, 2: 1.5, 1: 1.0, 0: 1.0}
        tokenized["sample_weight"] = [float(rel_map.get(r, 1.0)) for r in examples["relevance"]]
        
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

    return {
        "accuracy": round(float(accuracy), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3)
    }



class CleanLogCallback(TrainerCallback):
    """Custom callback to print training logs in a concise, readable format."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        
        # We only care about training logs (which have 'loss') or evaluation logs (which have 'eval_loss')
        if "loss" in logs:
            lr = logs.get("learning_rate", 0)
            loss = logs.get("loss", 0)
            epoch = logs.get("epoch", 0)
            print(f"  Epoch {epoch:>5.2f} | Loss: {loss:>8.5f} | LR: {lr:>9.2e}")
        
        elif "eval_loss" in logs:
            e_loss = logs.get("eval_loss", 0)
            e_acc = logs.get("eval_accuracy", 0)
            e_f1 = logs.get("eval_f1", 0)
            print("-" * 55)
            print(f"  VALIDATION | Loss: {e_loss:>8.5f} | Acc: {e_acc:>8.5f} | F1: {e_f1:>8.5f}")
            print("-" * 55)


def train_model(train_dataset: Dataset, eval_dataset: Dataset) -> Tuple[Trainer, PreTrainedModel]:

    """Initializes model, configures Trainer, and fine-tunes ALBERT."""
    print("\n[Train] Starting fine-tuning (RBT3)...")


    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "NOT_MATCH", 1: "MATCH"},
        label2id={"NOT_MATCH": 0, "MATCH": 1},
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), "../../saved_models/recommendation_model_output"),
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=3e-5,               # Lower LR for RoBERTa (less aggressive updates)
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=12,              # More epochs; EarlyStopping will prevent overfit
        weight_decay=0.01,
        warmup_ratio=0.1,
        label_smoothing_factor=0.1,       # Prevent over-confidence on hard negatives
        logging_steps=100,                # Less frequent logging for cleaner output
        logging_first_step=False,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=3,
        disable_tqdm=False,               # 恢復進度條
        log_level="error",                # 隱藏內部日誌
    )


    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001),
            CleanLogCallback()
        ]
    )

    trainer.train()
    return trainer, model


def evaluate_on_test(trainer: Trainer, tokenizer: PreTrainedTokenizer, test_path: str):
    """Evaluates the trained model against a holdout test dataset."""
    print("\n[Evaluate] Testing on holdout set...")
    
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
        
    test_dataset = test_dataset_raw.map(tokenize_function, batched=True, remove_columns=test_dataset_raw.column_names)
    results = trainer.evaluate(test_dataset)

    print(f"  Accuracy:  {results.get('eval_accuracy', 0):.5f}")
    print(f"  Precision: {results.get('eval_precision', 0):.5f}")
    print(f"  Recall:    {results.get('eval_recall', 0):.5f}")
    print(f"  F1 Score:  {results.get('eval_f1', 0):.5f}")



def export_to_onnx(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """Saves PyTorch model and tokenizers, then exports architecture to ONNX format."""
    print("\n[Export] Saving model and exporting to ONNX...")


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

    # Suppress redundant torch.onnx internal logs
    import logging
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)

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
        os.path.join(os.path.dirname(__file__), "../../data/processed/recommendation_train.json"), 
        os.path.join(os.path.dirname(__file__), "../../data/processed/recommendation_dev.json")
    )
    
    train_dataset, eval_dataset = tokenize_datasets(train_dataset_raw, eval_dataset_raw, tokenizer)
    
    trainer, model = train_model(train_dataset, eval_dataset)
    
    evaluate_on_test(trainer, tokenizer, os.path.join(os.path.dirname(__file__), "../../data/processed/recommendation_test.json"))
    export_to_onnx(model, tokenizer)


if __name__ == "__main__":
    main()
