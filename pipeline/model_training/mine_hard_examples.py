
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os
import pandas as pd
from typing import List, Dict, Any

# Configurations
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../../frontend/models/custom_onnx_model_dir/my_custom_model.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "../../frontend/models/custom_onnx_model_dir")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "../../data/processed/recommendation_train.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "../../data/processed/hard_examples.json")

def run_onnx_batch(session, tokenizer, queries, properties, batch_size=1):
    all_probs = []
    input_names = [input.name for input in session.get_inputs()]
    
    for i in range(0, len(queries), batch_size):
        batch_q = queries[i : i + batch_size]
        batch_p = properties[i : i + batch_size]
        
        inputs = tokenizer(batch_q, batch_p, 
                          padding='max_length', truncation=True, max_length=128, return_tensors="np")
        
        onnx_inputs = {
            input_names[0]: inputs["input_ids"].astype(np.int64),
            input_names[1]: inputs["attention_mask"].astype(np.int64),
            input_names[2]: inputs["token_type_ids"].astype(np.int64)
        }
            
        logits = session.run(["logits"], onnx_inputs)[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        all_probs.extend(probs[:, 1].tolist())
        
        if (i // batch_size) % 50 == 0:
            print(f"  Processed {i}/{len(queries)} samples...")
        
    return np.array(all_probs)

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train the model first.")
        return

    print("Starting Hard Example Mining...")
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    session = ort.InferenceSession(MODEL_PATH)

    queries = [d["query"] for d in train_data]
    props = [d["property"] for d in train_data]
    true_labels = [d["label"] for d in train_data] # Binary labels (0 or 1)
    true_rels = [d.get("relevance", 0) for d in train_data] # Graded labels (0-3)

    print(f"Predicting scores for {len(train_data)} samples...")
    # Batch process to avoid memory issues (though currently 1 by 1 in run_onnx_batch)
    scores = run_onnx_batch(session, tokenizer, queries, props)

    hard_examples = []
    
    for i in range(len(train_data)):
        score = scores[i]
        rel = true_rels[i]
        
        # Case A: False Positives (Model likes it, but it's irrelevant)
        if score > 0.7 and rel <= 1:
            hard_examples.append({
                "query": queries[i],
                "property": props[i],
                "label": 0,
                "relevance": rel,
                "model_score": float(score),
                "error_type": "false_positive"
            })
            
        # Case B: False Negatives (Model hates it, but it's perfect)
        elif score < 0.3 and rel >= 2:
            hard_examples.append({
                "query": queries[i],
                "property": props[i],
                "label": 1,
                "relevance": rel,
                "model_score": float(score),
                "error_type": "false_negative"
            })

    print(f"Found {len(hard_examples)} hard examples.")
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(hard_examples, f, ensure_ascii=False, indent=2)
    
    print(f"Hard examples saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
