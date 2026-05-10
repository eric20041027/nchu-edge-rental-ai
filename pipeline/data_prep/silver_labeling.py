import os
import json
import random
from google import genai
from google.genai import types
from dotenv import load_dotenv
import tqdm

# --- Configuration ---
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROPERTY_DATA = os.path.join(BASE_DIR, "frontend/assets/property_data.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/silver_labeled_queries.json")

def load_properties():
    with open(PROPERTY_DATA, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_queries_and_labels(properties, num_samples_per_prop=6):
    samples = []
    
    for i, prop in enumerate(tqdm.tqdm(properties, desc="Generating Silver Labels (All Props)")):
        p_text = f"房源資訊: {prop['text']}\n租金: {prop['rent']}元\n備註: {' '.join(prop.get('notes', []))}"
        
        prompt = f"""
        你是一位專業的租屋大數據標記專家。請針對以下房源，生成 {num_samples_per_prop} 組「極具挑戰性」且「口語化」的用戶搜尋查詢，並給出精確的相關性評分 (0-3)。
        
        {p_text}
        
        評分標準:
        3 (Perfect): 查詢精確描述了該房源的所有關鍵亮點（地點、預算、特定稀有設施如獨洗、台電等）。
        2 (Good): 查詢與房源大致契合，但房源在某些次要維度（如樓層或稍微偏離的核心區域）僅是「還不錯」。
        1 (Partial): 查詢與房源只有一點點交集（例如預算對但地點不對，或設施完全不符但價格極低）。
        0 (None): 查詢的需求與房源完全衝突（例如限女 vs 男生，或禁寵 vs 要養貓）。
        
        請確保生成多樣化的查詢：包含「預算導向」、「設施導向」、「生活風格導向」以及「帶有負面排除條件的導向」。
        
        請輸出為 JSON 陣列，包含 'query', 'relevance' 欄位。
        """
        
        try:
            response = client.models.generate_content(
                model="models/gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            results = json.loads(response.text)
            for r in results:
                rel = int(r['relevance'])
                samples.append({
                    "query": r['query'],
                    "property": prop['text'],
                    "relevance": rel,
                    "label": 1 if rel >= 2 else 0,
                    "is_hard": True if rel == 0 or rel == 3 else False
                })
                
            # Periodic Save every 10 properties
            if (i + 1) % 10 == 0:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)
                    
        except Exception as e:
            continue
            
    return samples

def main():
    properties = load_properties()
    print(f"Starting MASSIVE Silver Labeling for {len(properties)} properties...")
    
    # Process all properties to reach ~5000 samples (969 * 6 approx 5800)
    silver_samples = generate_queries_and_labels(properties, num_samples_per_prop=6)
    
    # Load existing to avoid duplication if running multiple times
    all_samples = silver_samples
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                # Simple deduplication based on query-property pair
                seen = set((s['query'], s['property']) for s in silver_samples)
                for s in existing:
                    if (s['query'], s['property']) not in seen:
                        all_samples.append(s)
        except: pass
            
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved {len(all_samples)} silver labeled samples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
