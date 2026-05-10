"""NER predictor — runs inference with the fine-tuned model."""
from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, BertForTokenClassification

from .config import NERConfig, ID2LABEL


@dataclass
class NEREntity:
    text: str
    label: str   # LOC | BGT | FEAT
    start: int
    end: int


class NERPredictor:
    """Loads a saved NER model and extracts entities from query text."""

    def __init__(self, model_dir: str | Path | None = None, config: NERConfig | None = None):
        self.config = config or NERConfig()
        self.model_dir = Path(model_dir) if model_dir else self.config.output_dir
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(f"NER model not found at {self.model_dir}. Run NERTrainer first.")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = BertForTokenClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        self._loaded = True

    def predict(self, text: str) -> list[NEREntity]:
        """Extract LOC, BGT, FEAT entities from query text."""
        self.load()
        tokens = list(text)   # character-level tokenisation for Chinese
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.config.max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encoding).logits
        preds = logits.argmax(-1).squeeze().tolist()

        # Map predictions back to characters
        word_ids = encoding.word_ids()
        char_labels: list[str] = []
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid < len(tokens):
                char_labels.append(ID2LABEL.get(preds[i], "O"))

        # Merge BIO spans into entities
        return self._bio_to_entities(tokens, char_labels)

    def extract_structured(self, text: str) -> dict:
        """Return structured dict {location, budget, features} from query."""
        entities = self.predict(text)
        result: dict = {"location": [], "budget": [], "features": []}
        for ent in entities:
            if ent.label == "LOC":
                result["location"].append(ent.text)
            elif ent.label == "BGT":
                result["budget"].append(ent.text)
            elif ent.label == "FEAT":
                result["features"].append(ent.text)
        return result

    # ------------------------------------------------------------------ #
    def _bio_to_entities(self, tokens: list[str], labels: list[str]) -> list[NEREntity]:
        entities: list[NEREntity] = []
        i, n = 0, len(labels)
        while i < n:
            if labels[i].startswith("B-"):
                entity_type = labels[i][2:]
                start = i
                j = i + 1
                while j < n and labels[j] == f"I-{entity_type}":
                    j += 1
                span = "".join(tokens[start:j])
                entities.append(NEREntity(text=span, label=entity_type, start=start, end=j))
                i = j
            else:
                i += 1
        return entities
