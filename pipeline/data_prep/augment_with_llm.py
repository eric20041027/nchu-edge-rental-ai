import os
import json
import time
from dotenv import load_dotenv
from google import genai

def generate_augmentation_batch(client, batch_id, count=50, mode="negative"):
    """
    mode="negative": Generates hard negative samples (conflict in key constraints).
    mode="positive": Generates colloquial semantic mapping samples (metaphors to features).
    """
    if mode == "negative":
        PROMPT = f"""
你是一個房地產數據專家。請幫我生成 {count} 組「租屋推薦系統」的困難負樣本 (Hard Negatives)。
這些樣本的特徵是：房源描述看起來與查詢 (Query) 非常匹配，但在關鍵約束上存在衝突。

請針對以下類別均勻生成：
1. 寵物衝突：Query 要求「可狗」，Property 描述「僅限貓」或「不可養寵物」。
2. 費用衝突：Query 要求「台水台電」，Property 描述「獨立電表一度5元」。
3. 地點衝突：Query 要求「近興大正門」，Property 描述「近興大南門/後門」。
4. 設備衝突：Query 要求「獨立洗脫烘」，Property 描述「公共洗衣房/投幣式洗衣機」。
5. 垃圾處理衝突：Query 要求「不追垃圾車/要有子母車」，Property 描述「需自行等候垃圾車」。

輸出格式必須是 JSON 陣列，每個元素包含：
{{
    "query": "使用者的口語化查詢",
    "property": "房源描述（要包含誘人條件但關鍵處衝突）",
    "relevance": 0,
    "label": 0,
    "category": "hard_negative"
}}
"""
    else:
        PROMPT = f"""
你是一個房地產數據專家。請幫我生成 {count} 組「租屋推薦系統」的正向語義映射樣本 (Positive Semantic Mapping)。
目標是讓模型學會理解口語化隱喻背後的真實特徵需求。

請針對以下「口語隱喻 → 核心特徵」均勻涵蓋所有類別進行生成：

【A. 煮飯 / 自炊 / 廚房需求】（本類別至少佔 20%）
- 「想在家煮飯」「想自己煮飯」「在家開伙」「喜歡下廚」「自炊」
  → 房源具備「可伙」「廚房」「流理台」「瓦斯爐」「電磁爐」「抽油煙機」「開火」
- 「省餐費」「不想外食」「不吃外食」「省伙食費」
  → 房源具備「可伙」「廚房」「流理台」
- 「天然瓦斯」「有瓦斯」「要有廚房」「要能煮飯」「煮飯」「開火」
  → 房源具備「天然瓦斯」「瓦斯爐」「可伙」「廚房」

【B. 女生安全 / 門禁】
- 「女生獨居」「獨居女」「女生住安全嗎」「怕危險」「治安」
  → 房源具備「管理員」「門禁」「監視器」「女性友善」「安全」「刷卡」
- 「晚歸」「作息晚」「夜貓子」
  → 房源具備「無門禁」「24小時」「自由進出」或「管理員 刷卡門禁」

【C. 拎包入住 / 家具家電】
- 「拎包入住」「不想買家具」「什麼都有」「家電齊全」
  → 房源具備「全配」「全家具」「全家電」「冰箱」「洗衣機」「床」
- 「要有冰箱」「要有書桌」「要有床」
  → 房源具備對應家具家電（「冰箱」「書桌書桌椅」「床架床墊」）
- 「空屋」「自己買家具」
  → 房源為「空屋」「自備家具」

【D. 衛浴獨立】
- 「不想共用廁所」「不想共廁」「個人衛浴」「獨立衛浴」
  → 房源具備「獨衛」「獨立衛浴」「套房」
- 「想泡澡」→ 房源具備「浴缸」「獨衛」
- 「要有熱水」→ 房源具備「熱水器」「天然瓦斯熱水器」

【E. 租期彈性】
- 「短租」「只租幾個月」「不確定租多久」「剛畢業」「工作不穩定」
  → 房源具備「短租」「彈性租期」「不限租期」「月租」

【F. 合租 / 室友】
- 「找室友」「想合租」「不想一個人住」
  → 房源為「雅房」「分租」「室友」「合租」
- 「一個人住」「不想跟人共用」
  → 房源為「獨立套房」「獨衛」「套房」

【G. 交通通勤】
- 「騎車上班」→「機車停車位」「停車」
- 「通勤族」「上班方便」→「近公車」「近捷運」「交通便利」
- 「沒有車」「不開車」→「近公車」「生活機能」「便利商店」

【H. 採光朝向】
- 「不要西曬」→「非西向」「東向」「北向」
- 「要有陽台」→「陽台」「曬衣」「採光」「通風」
- 「不要頂樓」→「非頂樓」「非頂加」

【I. 在家工作 / WFH】
- 「在家工作」「WFH」「遠距工作」「居家辦公」
  → 房源具備「網路」「寬頻」「書桌」「安靜」

【J. 預算暗示】
- 「學生」「剛出社會」「薪水不多」「不要太貴」「便宜」
  → 房源為「學生套房」「經濟實惠」「低租金」「實惠」

【K. 便利 / 懶人設施】
- 「不想追垃圾車」→「子母車」「垃圾代收」
- 「懶得爬樓梯」「膝蓋不好」→「電梯」「大樓」
- 「不想去自助洗」「不喜歡共用洗衣機」→「獨立洗衣機」

【L. 溫度 / 電費 / 寵物 / 安靜】
- 「怕熱」→「變頻冷氣」；「想省電費」→「台電計費」
- 「有貓」「想養狗」「有毛孩」→「可養貓」「可養狗」「寵物友善」
- 「讀書」「念書」「打報告」→「書桌」「安靜」「寬頻」
- 「潔癖」「愛乾淨」→「全新」「獨洗」「禁菸」

注意事項：
- query 要非常口語、符合台灣租屋族習慣用語
- property 要精確包含上述對應特徵關鍵字，讓模型看到明確的語意配對
- 同一類別請生成語氣各異的多個表達（避免重複樣板句）
- 語言請使用繁體中文（台灣習慣用語）
- 不要輸出任何說明文字，只要純 JSON 陣列

輸出格式必須是 JSON 陣列，每個元素包含：
{{
    "query": "非常口語、台灣慣用語的查詢",
    "property": "精確匹配該需求的房源描述（要強調對應的解決方案特徵）",
    "relevance": 3,
    "label": 1,
    "category": "semantic_positive"
}}
"""

    import re
    try:
        # Using a more standard model name version
        response = client.models.generate_content(
            model='gemini-flash-latest',
            contents=PROMPT,
        )
        text = response.text.strip()
        
        # Robust JSON extraction using regex
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            # Fallback for simple responses
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            return json.loads(text.strip())
            
    except Exception as e:
        print(f"❌ Batch {batch_id} failed: {e}")
        if 'text' in locals():
            print(f"DEBUG: Raw LLM Output (first 100 chars): {text[:100]}...")
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
        mode = "positive" if len(all_traps) % 100 < 50 else "negative"
        
        print(f"🤖 [Augment] {len(all_traps)}/{total_needed} collected. Mode: {mode}. Generating next batch...")
        batch = generate_augmentation_batch(client, len(all_traps), min(batch_size, remaining), mode=mode)
        
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

