import os
import json
import torch
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizerFast
from typing import List, Dict, Any
import tqdm

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROPERTY_DATA = os.path.join(BASE_DIR, "frontend/assets/property_data.json")
LLM_QUERIES = os.path.join(BASE_DIR, "data/raw/llm_queries.json")
MODEL_PATH = os.path.join(BASE_DIR, "frontend/models/custom_onnx_model_dir/my_custom_model_quant.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "frontend/models/custom_onnx_model_dir")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/mined_hard_negatives.json")

def load_resources():
    with open(PROPERTY_DATA, 'r', encoding='utf-8') as f:
        properties = json.load(f)
    with open(LLM_QUERIES, 'r', encoding='utf-8') as f:
        llm_samples = json.load(f)
    
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return properties, llm_samples, tokenizer, session

def score_batch(tokenizer, session, query: str, prop_texts: List[str]):
    """Scores a query against a batch of properties."""
    inputs = tokenizer(
        [query] * len(prop_texts),
        prop_texts,
        return_tensors="np",
        max_length=64,
        padding="max_length",
        truncation=True
    )
    
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs["token_type_ids"].astype(np.int64),
    }
    
    logits = session.run(None, ort_inputs)[0]
    # Softmax on logits
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs[:, 1] # Return probabilities for label 1 (relevant)

def is_hard_conflict(query: str, prop: Dict[str, Any]) -> bool:
    """Checks if a query and property have a fundamental hard conflict."""
    p_text = (prop['text'] + " " + " ".join(prop.get('notes', []))).lower()
    q = query.lower()
    
    # 1. Gender Conflict
    if "限女" in p_text and ("限男" in q or "男生" in q): return True
    if "限男" in p_text and ("限女" in q or "女生" in q): return True
    
    # 2. Budget Conflict (Explicit)
    import re
    # Extract numbers followed by price-related keywords
    budget_match = re.search(r"(\d+)(?:元|k)?(?:以下|以內|內|左右)", q)
    if budget_match:
        val = int(budget_match.group(1))
        # Handle "k" notation
        if "k" in budget_match.group(0).lower() and val < 100: val *= 1000
        
        rent = prop.get("rent", 0)
        if rent > val * 1.2: return True # Allow 20% buffer for negotiation/management fee
        
    # 3. Pet Conflict
    if ("可養寵" in q or "可寵" in q or "養貓" in q or "養狗" in q) and ("禁寵" in p_text or "不可養寵" in p_text): return True
    if ("禁寵" in q or "不可寵" in q) and ("可寵" in p_text or "寵物友善" in p_text): 
        # This is less of a "hard" conflict (user might accept a pet-friendly place if they don't have pets)
        # but for negative mining, we focus on the direction where the property lacks a MUST-HAVE.
        pass

    # 4. Rooftop Conflict
    if ("不找頂加" in q or "不要頂加" in q or "非頂加" in q) and ("頂加" in p_text or "頂樓加蓋" in p_text): return True
    
    # 5. Smoking Conflict
    if ("禁菸" in q or "不抽菸" in q) and ("可菸" in p_text): return True
    
    # 6. Specific Amenities (MUST HAVES)
    must_haves = {
        "電梯": ["電梯"],
        "陽台": ["陽台", "露台"],
        "車位": ["車位", "停車"],
        "獨洗": ["獨洗", "個人洗衣機", "自用洗衣機"],
        "開伙": ["開伙", "廚房", "瓦斯爐"],
        "管理員": ["管理員", "子母車", "收包裹"]
    }
    for key, synonyms in must_haves.items():
        if key in q:
            if not any(syn in p_text for syn in synonyms):
                return True # Property lacks a specifically requested amenity

    return False

def main():
    print("=" * 60)
    print("Starting Optimized Active Hard Negative Mining...")
    
    properties, llm_samples, tokenizer, session = load_resources()
    unique_queries = list(set([s['query'] for s in llm_samples]))
    # Shuffle to get diverse samples
    import random
    random.shuffle(unique_queries)
    
    selected_queries = unique_queries[:15] 
    print(f"  Loaded {len(properties)} properties and {len(unique_queries)} queries. Sampling {len(selected_queries)}.")
    
    mined_samples = []
    batch_size = 128 
    
    max_score_found = 0
    for q in tqdm.tqdm(selected_queries, desc="Mining Queries"):
        sampled_props = random.sample(properties, min(300, len(properties)))
        batch_texts = [p['text'] for p in sampled_props]
        
        for j in range(0, len(batch_texts), batch_size):
            sub_texts = batch_texts[j : j + batch_size]
            sub_props = sampled_props[j : j + batch_size]
            
            scores = score_batch(tokenizer, session, q, sub_texts)
            max_score_found = max(max_score_found, np.max(scores))
            
            for idx, score in enumerate(scores):
                prop = sub_props[idx]
                if score > 0.45 and is_hard_conflict(q, prop): # Lowered threshold
                    mined_samples.append({
                        "query": q,
                        "property_text": prop['text'],
                        "ai_score": float(score),
                        "relevance": 0,
                        "is_hard": True
                    })

    print(f"\n  Max AI score encountered: {max_score_found:.4f}")
    print(f"  Successfully mined {len(mined_samples)} hard negative candidates.")
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(mined_samples, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
