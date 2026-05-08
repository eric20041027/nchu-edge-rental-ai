import os
import json
import json
import onnxruntime as ort
from transformers import BertTokenizerFast
import numpy as np

# --- Configuration ---
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../frontend/assets/property_data.json")
MODEL_PATH = os.path.join(BASE_DIR, "../frontend/models/custom_onnx_model_dir/my_custom_model_quant.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "../frontend/models/custom_onnx_model_dir")

def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def init_ai():
    print("[System] Loading AI Model and Tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return tokenizer, session

def score_pair_ai(tokenizer, session, query, property_text):
    inputs = tokenizer(
        query, property_text,
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
    
    logits = session.run(None, ort_inputs)[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    raw_prob = probs[1]
    
    # [Normalization] Match browser logic: (raw - 0.01) / 0.89
    normalized = max(0, min(1.0, (raw_prob - 0.01) / 0.89))
    return normalized, raw_prob

def explain_match(query, prop):
    reasons = []
    p_text = prop['text'] + prop.get('furniture', '') + " ".join(prop.get('notes', []))
    q = query.lower()

    # 1. User-Specified Matches (Priority)
    if '陽台' in q and ('陽台' in p_text or prop.get('has_balcony')): reasons.append("有陽台")
    if '電視' in q and '電視' in p_text: reasons.append("有電視")
    if '冰箱' in q and '冰箱' in p_text: reasons.append("有冰箱")
    if '洗衣機' in q and '洗衣機' in p_text: reasons.append("有洗衣機")
    if '電費' in q or '台電' in q:
        if '台水台電' in p_text or '獨立電錶' in p_text: reasons.append("台電計費")

    # 2. General Highlights (Space permitting)
    highlights = [("子母車", "免追垃圾車"), ("飲水機", "有飲水機"), ("電梯", "有電梯"), ("獨立洗衣機", "個人洗衣機")]
    for key, label in highlights:
        if len(reasons) < 3 and key in p_text:
            if label not in reasons: reasons.append(label)
    
    return reasons[:3]

def main():
    properties = load_data()
    tokenizer, session = init_ai()
    
    print("\n" + "="*50)
    print("  NCHU AI Rental Recommendation - CLI TESTER (V2 - Synced)")
    print("="*50)

    while True:
        query = input("\n請輸入搜尋需求 (或輸入 'exit' 退出): ").strip()
        if query.lower() == 'exit': break
        if not query: continue

        print(f"正在分析: '{query}'...")
        
        scored = []
        for p in properties[:100]: # Score top 100 for better analysis
            ai_score, raw_prob = score_pair_ai(tokenizer, session, query, p['text'])
            pText = (p['text'] + p.get('furniture', '') + " ".join(p.get('notes', []))).lower()
            
            # --- Better Rule Based Match (RMS) with Intent ---
            match_count = 0
            total_req = 0
            query_kws = query.lower().split() # Simplified for CLI

            for kw in query_kws:
                if len(kw) < 2: continue
                total_req += 1
                is_match = kw in pText
                
                # Intent mapping
                if '垃圾' in kw or '追車' in kw:
                    is_match = any(x in pText for x in ['子母車', '代收垃圾', '垃圾處理'])
                elif '樓梯' in kw or '電梯' in kw:
                    is_match = any(x in pText for x in ['電梯', '華廈', '大樓'])
                elif '電' in kw or '省' in kw:
                    is_match = any(x in pText for x in ['台電', '獨立電錶', '台水台電'])
                
                if is_match: match_count += 1

            rms = (match_count / total_req) if total_req > 0 else 1.0
            
            # Final Score Logic (35% Rules + 65% AI)
            final_score = (rms * 35) + (ai_score * 65)
            
            # Special boost for perfect rule matches
            if rms == 1.0 and final_score < 80:
                final_score = 80 + (ai_score * 15)

            scored.append({
                "prop": p,
                "score": final_score,
                "ai_confidence": ai_score,
                "raw_prob": raw_prob,
                "rms": rms,
                "reasons": explain_match(query, p)
            })
            
        scored.sort(key=lambda x: x['score'], reverse=True)
        
        print("\n--- 推薦結果 (Top 3) ---")
        for i, res in enumerate(scored[:3]):
            p = res['prop']
            print(f"{i+1}. [{p['room_type']}] {p['address']}")
            print(f"   [評分組成] 總分: {res['score']:.1f}% | 規則匹配: {res['rms']*100:.0f}% | AI 信心度: {res['ai_confidence']*100:.1f}%")
            print(f"   [原始數據] AI 原始機率: {res['raw_prob']:.4f}")
            if res['reasons']:
                print(f"   命中亮點: {' / '.join(res['reasons'])}")
            print("-" * 30)

if __name__ == "__main__":
    main()
