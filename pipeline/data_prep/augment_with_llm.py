import os
import json
import time
from dotenv import load_dotenv
from google import genai

def generate_trap_batch(client, batch_id, count=100):
    """Calls Gemini to generate a batch of hard negative (trap) samples."""
    PROMPT = f"""
你是一個房地產數據專家。請幫我生成 {count} 組「租屋推薦系統」的困難負樣本 (Hard Negatives)。
這些樣本的特徵是：房源描述看起來與查詢 (Query) 非常匹配，但在關鍵約束上存在衝突。

請針對以下類別均勻生成：
1. 寵物衝突：Query 要求「可狗」，Property 描述「僅限貓」或「不可養寵物」。
2. 費用衝突：Query 要求「台水台電」，Property 描述「獨立電表一度5元」。
3. 地點衝突：Query 要求「近興大正門」，Property 描述「近興大南門/後門」。
4. 設備衝突：Query 要求「獨立洗脫烘」，Property 描述「公共洗衣房/投幣式洗衣機」。
5. 噪音衝突：Query 要求「淺眠怕吵/隔音好」，Property 描述「臨大馬路第一排/樓下是夜市」。

輸出格式必須是 JSON 陣列，每個元素包含：
{{
    "query": "使用者的口語化查詢",
    "property": "房源描述（要包含誘人條件但關鍵處衝突）",
    "relevance": 0,
    "label": 0,
    "category": "類別名稱"
}}

注意：
- 語言請使用繁體中文（台灣習慣用語）。
- 不要輸出任何說明文字，只要純 JSON 陣列。
"""
    try:
        response = client.models.generate_content(
            model='gemini-flash-latest',
            contents=PROMPT,
        )
        text = response.text.strip()
        # Basic cleanup for accidental markdown formatting
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        
        return json.loads(text.strip())
    except Exception as e:
        print(f"❌ Batch {batch_id} failed: {e}")
        return []

def main():
    load_dotenv()
    API_KEY = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=API_KEY)
    out_path = os.path.join(os.path.dirname(__file__), "../../data/raw/llm_queries.json")
    
    all_traps = []
    # Load existing if any
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                all_traps = json.load(f)
            print(f"📂 Loaded {len(all_traps)} existing samples.")
        except:
            print("⚠️ Failed to load existing file, starting fresh.")
    
    total_needed = 1000
    batch_size = 50 # Smaller batches for better success rate
    
    while len(all_traps) < total_needed:
        remaining = total_needed - len(all_traps)
        print(f"🤖 [Augment] {len(all_traps)}/{total_needed} collected. Generating next batch...")
        batch = generate_trap_batch(client, len(all_traps), min(batch_size, remaining))
        
        if batch:
            all_traps.extend(batch)
            print(f"✅ Success! Now have {len(all_traps)} total samples.")
            # Save progress every batch
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_traps, f, ensure_ascii=False, indent=2)
        else:
            print("⚠️ Batch failed or empty, retrying in 5 seconds...")
            time.sleep(5)
            
        time.sleep(2)
        
    print(f"🎉 MISSION COMPLETE! Total {len(all_traps)} trap samples saved to {out_path}")

if __name__ == "__main__":
    main()

