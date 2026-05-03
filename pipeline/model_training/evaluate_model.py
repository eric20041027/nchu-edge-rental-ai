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
sys.path.append(os.path.join(BASE_DIR, "../../"))
from pipeline.data_prep.generate_dataset import load_properties, is_compatible, compute_relevance_score


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

    # Phase 1: Binary Classification
    print("[Phase 1] Binary Classification Metrics (n=1000)")
    sample_size = min(len(test_data), 1000)
    sample_test = random.sample(test_data, sample_size)

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
    
    print(f"  Accuracy: {acc:.5f} | F1: {f1:.5f}")

    # Phase 2: Ranking Pipeline
    print("\n[Phase 2] Ranking Metrics (Top-30 Re-ranking Simulation)")

    pos_queries = list(set([d["query"] for d in test_data if d["label"] == 1]))
    random.seed(42)
    num_eval_queries = min(len(pos_queries), 200)
    eval_queries = random.sample(pos_queries, num_eval_queries) # Increased to 200 queries
    
    ndcg_list, mrr_list = [], []
    ndcg_graded_list = []
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    sat_at_3, sat_at_5 = 0, 0
    
    for i, query in enumerate(eval_queries):
        if i % 20 == 0: print(f"  Evaluating query {i+1}/{num_eval_queries}...")

        
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
        
        # Binary relevance (for legacy NDCG/MRR)
        relevance_bin = [1 if idx in compatible_indices else 0 for idx in [candidate_indices[r] for r in ranked_indices]]
        
        # Graded relevance (0-3) using the official scoring logic
        relevance_graded = []
        for r in ranked_indices:
            idx = candidate_indices[r]
            rel = compute_relevance_score(query, original_properties[idx])
            relevance_graded.append(rel)
            label_counts[rel] += 1

        
        # Binary NDCG@5 (Old method)
        dcg_bin = sum([rel / np.log2(rank + 2) for rank, rel in enumerate(relevance_bin[:5])])
        idcg_bin = sum([rel / np.log2(rank + 2) for rank, rel in enumerate(sorted(relevance_bin, reverse=True)[:5])])
        ndcg_list.append(dcg_bin / idcg_bin if idcg_bin > 0 else 0)
        
        # Graded NDCG@5 (New Exponential method: (2^rel - 1) / log2(rank+2))
        def get_graded_dcg(rel_list):
            return sum([(2**rel - 1) / np.log2(rank + 2) for rank, rel in enumerate(rel_list)])
            
        dcg_graded = get_graded_dcg(relevance_graded[:5])
        idcg_graded = get_graded_dcg(sorted(relevance_graded, reverse=True)[:5])
        ndcg_graded_list.append(dcg_graded / idcg_graded if idcg_graded > 0 else 0)
        
        # MRR
        for rank, rel in enumerate(relevance_bin):
            if rel > 0:
                mrr_list.append(1 / (rank + 1))
                break
        else: mrr_list.append(0)
        
        sat_at_3 += (sum(relevance_bin[:3]) / 3)
        sat_at_5 += (sum(relevance_bin[:5]) / 5)


    print("-" * 40)
    print("Final Ranking Report:")
    print(f"  Binary NDCG @ 5:   {np.mean(ndcg_list):.5f}")
    print(f"  Graded NDCG @ 5:   {np.mean(ndcg_graded_list):.5f}")
    print(f"  Mean MRR:          {np.mean(mrr_list):.5f}")
    print(f"  Avg Satisfaction:  {(sat_at_3 + sat_at_5)/(2 * num_eval_queries):.5f}")
    
    total_labels = sum(label_counts.values())
    print("-" * 40)
    print("Label Distribution:")
    print(f"  Perfect (3): {label_counts[3]/total_labels*100:5.1f}%")
    print(f"  Good (2):    {label_counts[2]/total_labels*100:5.1f}%")
    print(f"  Partial (1): {label_counts[1]/total_labels*100:5.1f}%")
    print(f"  None (0):    {label_counts[0]/total_labels*100:5.1f}%")
    print("-" * 40)



if __name__ == "__main__":
    main()
