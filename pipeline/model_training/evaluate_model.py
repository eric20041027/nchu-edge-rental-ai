"""
evaluate_model.py
Optimized Evaluation: Simulates the real-world pipeline (Pre-filter -> AI Re-rank).
Calculates Binary metrics (Acc, F1) and Ranking metrics (NDCG, MRR) using a sampled candidate pool.
"""
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os
import random
import sys

# Configurations
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../../frontend/models/custom_onnx_model_dir/my_custom_model.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "../../frontend/models/custom_onnx_model_dir")
TEST_DATA_PATH = os.path.join(BASE_DIR, "../../data/processed/recommendation_test.json")
PROPERTY_TEXTS_PATH = os.path.join(BASE_DIR, "../../data/processed/property_texts.json")
BATCH_SIZE = 1 # Static batch size for this ONNX model

# Import compatibility logic
sys.path.append(os.path.join(BASE_DIR, "../../"))
from pipeline.data_prep.generate_dataset import load_properties, is_compatible

def load_data():
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(PROPERTY_TEXTS_PATH, "r", encoding="utf-8") as f:
        property_texts = json.load(f)
    return test_data, property_texts

def run_onnx_batch(session, tokenizer, queries, properties):
    all_probs = []
    input_names = [input.name for input in session.get_inputs()]
    
    for i in range(len(queries)):
        inputs = tokenizer(
            queries[i], properties[i],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="np"
        )
        onnx_inputs = {name: inputs[name].astype(np.int64) for name in input_names if name in inputs}
        if "token_type_ids" in input_names and "token_type_ids" not in inputs:
            onnx_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)
            
        logits = session.run(["logits"], onnx_inputs)[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        all_probs.append(probs[0, 1])
        
    return np.array(all_probs)

def main():
    test_data, property_texts = load_data()
    original_properties = load_properties()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    session = ort.InferenceSession(MODEL_PATH)

    # Part 1: Binary Classification (sampled 200 for speed)
    print("[Part 1] Binary Classification Metrics (Sampled 200)...")
    sample_test = random.sample(test_data, 200)
    test_queries = [d["query"] for d in sample_test]
    test_props = [d["property"] for d in sample_test]
    test_labels = np.array([d["label"] for d in sample_test])
    
    probs = run_onnx_batch(session, tokenizer, test_queries, test_props)
    preds = (probs >= 0.5).astype(int)
    
    acc = (preds == test_labels).mean()
    tp = ((preds == 1) & (test_labels == 1)).sum()
    fp = ((preds == 1) & (test_labels == 0)).sum()
    fn = ((preds == 0) & (test_labels == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Part 2: Ranking (Simulate Top-30 Pipeline)
    print("\n[Part 2] Ranking Metrics (Simulating Top-30 Re-ranking)...")
    pos_queries = list(set([d["query"] for d in test_data if d["label"] == 1]))
    random.seed(42)
    eval_queries = random.sample(pos_queries, 50) # Sample 50 queries for speed
    
    ndcg_list, mrr_list = [], []
    sat_at_3, sat_at_5 = 0, 0
    
    for i, query in enumerate(eval_queries):
        if i % 10 == 0: print(f"  Processing query {i+1}/50...")
        
        # Step A: Pre-filter (Plausible candidates)
        # In reality, this is rule-based. Here we simulate by taking 
        # all compatible properties + some random ones to reach 30.
        compatible_indices = [idx for idx, p in enumerate(original_properties) if is_compatible(query, p)]
        
        if len(compatible_indices) > 30:
            candidate_indices = random.sample(compatible_indices, 30)
        else:
            other_indices = list(set(range(len(original_properties))) - set(compatible_indices))
            candidate_indices = compatible_indices + random.sample(other_indices, 30 - len(compatible_indices))
        
        # Step B: AI Re-rank
        query_batch = [query] * len(candidate_indices)
        prop_batch = [property_texts[idx] for idx in candidate_indices]
        scores = run_onnx_batch(session, tokenizer, query_batch, prop_batch)
        
        # Step C: Evaluate Ranking
        ranked_indices = np.argsort(scores)[::-1]
        relevance = [1 if idx in compatible_indices else 0 for idx in [candidate_indices[r] for r in ranked_indices]]
        
        # NDCG@5
        dcg = sum([rel / np.log2(rank + 2) for rank, rel in enumerate(relevance[:5])])
        idcg = sum([rel / np.log2(rank + 2) for rank, rel in enumerate(sorted(relevance, reverse=True)[:5])])
        ndcg_list.append(dcg / idcg if idcg > 0 else 0)
        
        # MRR
        for rank, rel in enumerate(relevance):
            if rel > 0:
                mrr_list.append(1 / (rank + 1))
                break
        else: mrr_list.append(0)
        
        sat_at_3 += (sum(relevance[:3]) / 3)
        sat_at_5 += (sum(relevance[:5]) / 5)

    print("-" * 50)
    print(f"Ranking Results (Sampled 50 Queries, Top-30 Pool):")
    print(f"  Mean NDCG @ 5:         {np.mean(ndcg_list):.4f}")
    print(f"  Mean Reciprocal Rank:  {np.mean(mrr_list):.4f}")
    print(f"  Avg Satisfaction @ 3:  {sat_at_3/50:.4f}")
    print(f"  Avg Satisfaction @ 5:  {sat_at_5/50:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
