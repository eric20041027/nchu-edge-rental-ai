import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os
import sys

# Configurations
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "frontend/models/custom_onnx_model_dir/my_custom_model.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "frontend/models/custom_onnx_model_dir")
PROPERTY_DATA_PATH = os.path.join(BASE_DIR, "frontend/assets/property_data.json")

def load_properties():
    with open(PROPERTY_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_stress_test():
    print("=" * 50)
    print("AI Rental Recommendation - SEMANTIC STRESS TEST")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model not found. Please run training first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    session = ort.InferenceSession(MODEL_PATH)
    properties = load_properties()
    
    # 1. Test Cases: (Query, Expected Match Key Feature, Expected Rejection Key Feature)
    stress_cases = [
        {"id": "TC-01", "query": "我是個懶人，不想爬樓梯，搬東西好累", "focus": "電梯需求 (Elevator)", "pass_desc": "優先推薦有電梯的房源"},
        {"id": "TC-02", "query": "衣服很難乾，急尋可以曬衣服的地方", "focus": "通風/陽台 (Balcony)", "pass_desc": "優先推薦有陽台或曬衣空間的房源"},
        {"id": "TC-03", "query": "夏天怕熱，需要吹冷氣", "focus": "冷氣需求 (AC)", "pass_desc": "優先推薦有冷氣的房源"},
        {"id": "TC-04", "query": "平常晚上都要打報告，上網方便很重要", "focus": "網路需求 (Internet)", "pass_desc": "優先推薦有寬頻網路的房源"},
        {"id": "TC-05", "query": "＃求租 ＃獨立套房 ＃獨洗獨曬 [人數] 1人，女生", "focus": "FB真實貼文：獨洗獨曬", "pass_desc": "必須有洗衣機且有陽台/曬衣場"},
        {"id": "TC-06", "query": "#求租 #可貓 求租 【地點】需近中興&中山醫", "focus": "FB真實貼文：寵物友善", "pass_desc": "必須標註為可養寵物"},
        {"id": "TC-07", "query": "預算：平均4000-6000 水電費：台水台電計費為佳", "focus": "FB真實貼文：台水台電", "pass_desc": "預算符合且標註為台電計費"},
        {"id": "TC-08", "query": "【需求】：有對外窗 【謝絕】：凶宅、漏水、潮濕、壁癌、房間無對外窗、頂樓加蓋", "focus": "FB真實貼文：嚴格排除", "pass_desc": "拒絕頂加、無窗、漏水房源"},
        {"id": "TC-09", "query": "【加分項】：獨洗曬 【優先選擇】：台水電、可貓", "focus": "FB真實貼文：複合意圖", "pass_desc": "高分房源需盡可能滿足多項加分項"},
        {"id": "TC-10", "query": "【需求】：獨洗曬、有對外窗、可開火、好停機車", "focus": "FB真實貼文：生活機能", "pass_desc": "優先滿足獨洗與開火需求"},
        {"id": "TC-11", "query": "想省伙食費，平常喜歡自己煮點東西吃", "focus": "生活型態：自炊推測", "pass_desc": "優先推薦有開火/廚房設備的房源"},
        {"id": "TC-12", "query": "我是外送族，回家只想休息不想出門", "focus": "生活型態：居家族推測", "pass_desc": "優先推薦有電梯、子母車、飲水機的房源"}
    ]

    for case in stress_cases:
        print(f"\n[Testing {case['id']}] Focus: {case['focus']}")
        print(f"Query: {case['query']}")
        
        # Select 50 candidates (mixture of hits and misses)
        candidates = random_sample_candidates(case['query'], properties)
        
        # Run inference
        scores = []
        for p in candidates:
            score = get_score(session, tokenizer, case['query'], p['text'])
            scores.append((score, p))
            
        # Sort by AI Score
        scores.sort(key=lambda x: x[0], reverse=True)
        
        top_3 = scores[:3]
        print(f"  Top 3 Results:")
        for i, (s, p) in enumerate(top_3):
            # Encode/Decode to clean up characters that can't be displayed in CP950
            clean_text = p['text'][:100].encode('cp950', 'ignore').decode('cp950')
            print(f"    {i+1}. [{s:.4f}] {clean_text}...")
            
        # Evaluation Logic
        passed = evaluate_pass(case['id'], case['query'], top_3)
        if passed:
            print(f"  [PASS] Result: PASSED - {case['pass_desc']}")
        else:
            print(f"  [FAIL] Result: FAILED - Failed to strictly satisfy constraints")

def get_score(session, tokenizer, query, prop_text):
    inputs = tokenizer(
        query, prop_text,
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
    return float(probs[0, 1])

def random_sample_candidates(query, properties):
    """Sample candidates but ensure some potential matches are included to test ranking."""
    import random
    # Clean query to extract keywords
    clean_q = query.replace("#", " ").replace("【", " ").replace("】", " ").replace("：", " ")
    kws = [k for k in clean_q.split() if len(k) > 1]
    
    potential_hits = [p for p in properties if any(k in p['text'] for k in kws)]
    
    # Take up to 15 potential hits + random noise to make 50
    hits = random.sample(potential_hits, min(15, len(potential_hits)))
    noise_pool = [p for p in properties if p not in hits]
    noise = random.sample(noise_pool, min(50 - len(hits), len(noise_pool)))
    
    final = hits + noise
    random.shuffle(final)
    return final

def evaluate_pass(case_id, query, top_results):
    """Hard-coded evaluation rules for specific test cases."""
    # We only check the TOP 1 result for strict compliance
    score, prop = top_results[0]
    
    # Extract ALL raw features for accurate judgment
    furniture = prop.get('furniture', [])
    notes = prop.get('notes', [])
    all_raw_text = " ".join(furniture) + " " + " ".join(notes) + " " + prop.get('text', "") + " " + prop.get('building_type', "")
    
    if case_id == "TC-01": # Elevator
        if not any(k in all_raw_text for k in ["電梯", "大樓", "華廈"]): return False
    if case_id == "TC-02": # Balcony
        if not any(k in all_raw_text for k in ["陽台", "曬衣", "晾衣", "窗"]): return False
    if case_id == "TC-03": # AC
        if "冷氣" not in all_raw_text: return False
    if case_id == "TC-05": # 獨洗獨曬
        has_laundry = any(k in all_raw_text for k in ["洗衣機", "獨洗"])
        has_drying = any(k in all_raw_text for k in ["陽台", "曬衣", "晾衣", "對外窗"])
        if not (has_laundry and has_drying): return False
    if case_id == "TC-06": # Pet
        if "禁寵" in all_raw_text: return False
        if not any(k in all_raw_text for k in ["可寵", "養寵", "可貓", "可狗"]): return False
    if case_id == "TC-07": # 台水電 + Budget
        rent = prop.get('rent', 99999)
        if rent > 7000: return False 
        if not any(k in all_raw_text for k in ["台電", "台水", "帳單"]): return False
    if case_id == "TC-08": # Exclusions
        if any(k in all_raw_text for k in ["頂加", "頂樓加蓋", "無窗", "漏水"]): return False
    if case_id == "TC-10": # Functional
        if not any(k in all_raw_text for k in ["開火", "瓦斯", "廚房"]): return False
    if case_id == "TC-11": # Cooking Inference
        if not any(k in all_raw_text for k in ["開火", "瓦斯", "廚房"]): return False
    if case_id == "TC-12": # Homebody Inference
        if not all(any(k in all_raw_text for k in group) for group in [["電梯"], ["子母車", "垃圾"], ["飲水機"]]): return False
            
    return True

if __name__ == "__main__":
    run_stress_test()
