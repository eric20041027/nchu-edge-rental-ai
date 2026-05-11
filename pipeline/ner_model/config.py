"""NER model configuration."""
from __future__ import annotations
import os
from pathlib import Path

LABEL2ID = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-BGT": 3, "I-BGT": 4, "B-FEAT": 5, "I-FEAT": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class NERConfig:
    def __init__(self):
        base = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent))

        self.base_model      = os.environ.get("NER_BASE_MODEL", "hfl/rbt3")
        self.num_labels      = len(LABEL2ID)
        self.label2id        = LABEL2ID
        self.id2label        = ID2LABEL

        self.train_data_path = Path(os.environ.get("NER_TRAIN_DATA", base / "data/ner/train.json"))
        self.val_data_path   = Path(os.environ.get("NER_VAL_DATA",   base / "data/ner/dev.json"))
        self.output_dir      = Path(os.environ.get("NER_OUTPUT_DIR", base / "saved_models/ner_model"))
        self.onnx_output_dir = Path(os.environ.get("NER_ONNX_DIR",   base / "frontend/models/ner_model_dir"))

        self.max_length       = int(os.environ.get("NER_MAX_LENGTH", "64"))
        self.learning_rate    = float(os.environ.get("NER_LR", "3e-5"))
        self.batch_size       = int(os.environ.get("NER_BATCH", "32"))
        self.num_epochs       = int(os.environ.get("NER_EPOCHS", "10"))
        self.warmup_steps     = int(os.environ.get("NER_WARMUP", "100"))
        self.early_stop_patience = int(os.environ.get("NER_PATIENCE", "3"))

        # README targets: F1 ≥ 0.958, Accuracy ≥ 0.972
        self.target_f1       = 0.958
        self.target_accuracy = 0.972
