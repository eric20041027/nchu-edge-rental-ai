"""NER trainer — fine-tunes bert-base-chinese for LOC/BGT/FEAT token classification."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification,
)

from .config import NERConfig

logger = logging.getLogger(__name__)


class NERDataset(Dataset):
    """Token-classification dataset.

    Each sample in the JSON file:
        {"tokens": ["我", "想", "找", "南區"], "labels": ["O","O","O","B-LOC"]}
    """

    def __init__(self, data: list[dict], tokenizer, config: NERConfig):
        self.samples = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        labels = sample["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
        )

        word_ids = encoding.word_ids()
        label_ids = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word_id:
                label_ids.append(self.config.label2id.get(labels[wid], 0))
            else:
                # Use I- label for continuation sub-tokens
                raw = labels[wid]
                if raw.startswith("B-"):
                    label_ids.append(self.config.label2id.get("I-" + raw[2:], 0))
                else:
                    label_ids.append(self.config.label2id.get(raw, 0))
            prev_word_id = wid

        encoding["labels"] = label_ids
        return {k: torch.tensor(v) for k, v in encoding.items()}


def _compute_metrics(eval_pred):
    from seqeval.metrics import f1_score, accuracy_score
    from .config import ID2LABEL

    logits, labels = eval_pred
    preds = logits.argmax(-1)

    true_seqs, pred_seqs = [], []
    for pred_row, label_row in zip(preds, labels):
        true_seq, pred_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            true_seq.append(ID2LABEL.get(l, "O"))
            pred_seq.append(ID2LABEL.get(p, "O"))
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)

    return {
        "f1":       f1_score(true_seqs, pred_seqs),
        "accuracy": accuracy_score(true_seqs, pred_seqs),
    }


class NERTrainer:
    """Fine-tunes bert-base-chinese for 3-class NER (LOC, BGT, FEAT)."""

    def __init__(self, config: NERConfig | None = None):
        self.config = config or NERConfig()
        self.tokenizer = None
        self.model = None

    def _load_json(self, path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"NER data not found: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def train(self) -> dict[str, Any]:
        cfg = self.config
        logger.info("Loading tokenizer: %s", cfg.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

        logger.info("Loading model: %s", cfg.base_model)
        self.model = BertForTokenClassification.from_pretrained(
            cfg.base_model,
            num_labels=cfg.num_labels,
            id2label=cfg.id2label,
            label2id=cfg.label2id,
        )

        train_data = self._load_json(cfg.train_data_path)
        val_data   = self._load_json(cfg.val_data_path)
        train_ds   = NERDataset(train_data, self.tokenizer, cfg)
        val_ds     = NERDataset(val_data,   self.tokenizer, cfg)

        args = TrainingArguments(
            output_dir=str(cfg.output_dir),
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size * 2,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=20,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stop_patience)],
        )

        logger.info("Training NER model...")
        trainer.train()

        metrics = trainer.evaluate()
        logger.info("Eval metrics: %s", metrics)

        f1 = metrics.get("eval_f1", 0)
        acc = metrics.get("eval_accuracy", 0)
        if f1 >= cfg.target_f1 and acc >= cfg.target_accuracy:
            logger.info("✅ NER targets met: F1=%.3f  Acc=%.3f", f1, acc)
        else:
            logger.warning("⚠️  NER targets NOT met: F1=%.3f (target %.3f)  Acc=%.3f (target %.3f)",
                           f1, cfg.target_f1, acc, cfg.target_accuracy)

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(cfg.output_dir))
        self.tokenizer.save_pretrained(str(cfg.output_dir))
        return metrics
