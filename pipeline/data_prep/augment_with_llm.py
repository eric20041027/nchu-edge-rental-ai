import os
import json
from dotenv import load_dotenv
from google import genai

def main():
    # Load environment variables from .env file (safely ignored by git)
    load_dotenv()
    
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if not API_KEY:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        print("Please create a .env file and add: GEMINI_API_KEY='your_api_key_here'")
        exit(1)

    # We use the new google-genai SDK
    client = genai.Client(api_key=API_KEY)


    PROMPT = """
你現在是一個在台灣（台中中興大學附近）讀書的大學生。請幫我生成 200 個「在 FB 租屋社團、Dcard 或 PTT 找房」的真實貼文或留言。
這些句子必須非常口語化、帶有個人情緒、冗言贅字，以及各種隱含的租屋需求。

範例：
- "我真的不想再追垃圾車了，求求推薦有子母車的套房，預算6000內"
- "有沒有南區可養貓的房子，被之前的房東氣死，除了凶宅都可以"
- "淺眠怕吵，希望隔音好一點，不要在馬路旁邊，台電計費佳"
- "衣服曬房間都發霉，求獨洗獨曬，有大陽台更好！"

請直接輸出一個 JSON 陣列 (Array of Strings)，包含這 200 個字串。
注意：不要輸出任何其他說明文字或 markdown 標籤（如 ```json），只要純 JSON 陣列格式即可。
    """

    print("🤖 Calling Gemini API to generate synthetic human-like queries...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-flash-latest',
                contents=PROMPT,
            )
            text = response.text.strip()
            
            # Clean up formatting if it accidentally outputs markdown code blocks
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            queries = json.loads(text.strip())
            
            out_path = os.path.join(os.path.dirname(__file__), "../../data/raw/llm_queries.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(queries, f, ensure_ascii=False, indent=2)
                
            print(f"✅ Successfully generated {len(queries)} queries and saved to {out_path}")
            break # Break out of loop on success
            
        except json.JSONDecodeError:
            print("❌ Failed to parse JSON. Raw output from model:")
            print(text)
            break # Don't retry on parsing error, it's a format issue
        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed to generate response: {e}")
            if attempt < max_retries - 1:
                import time
                print("Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print("❌ All retries exhausted.")

if __name__ == "__main__":
    main()
