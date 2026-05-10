import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os
import random
import sys

# Configurations
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "frontend/models/custom_onnx_model_dir/my_custom_model.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "frontend/models/custom_onnx_model_dir")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data/processed/recommendation_test.json")

def main():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Test data not found at {TEST_DATA_PATH}")
        return

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    session = ort.InferenceSession(MODEL_PATH)

    print(f"Total test samples: {len(test_data)}")
    sample_size = 100
    sample_test = random.sample(test_data, sample_size)

    test_queries = [d["query"] for d in sample_test]
    test_props = [d["property"] for d in sample_test]
    test_labels = np.array([d["label"] for d in sample_test])
    
    all_probs = []
    for i in range(len(test_queries)):
        inputs = tokenizer(
            test_queries[i], test_props[i],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="np"
        )
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64) if "token_type_ids" in inputs else np.zeros_like(inputs["input_ids"], dtype=np.int64)
        }
        logits = session.run(["logits"], onnx_inputs)[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        all_probs.append(probs[0, 1])

    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)
    acc = (preds == test_labels).mean()
    print(f"Quick Accuracy (n=100): {acc:.4f}")

if __name__ == "__main__":
    main()
