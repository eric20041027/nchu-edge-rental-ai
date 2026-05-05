"""
fb_crawler.py
A semi-automated scraper using Playwright to extract posts from a Facebook group.
Because Facebook requires login and has anti-bot measures, this script will:
1. Open a visible browser window.
2. Ask you to log in to your Facebook account manually.
3. Once logged in, press Enter in the terminal.
4. The script will navigate to the group, scroll down, and extract posts containing '求租'.
5. It saves the results to 'fb_queries.json'.
"""

import json
import time
import asyncio
import os
from playwright.async_api import async_playwright

GROUP_URL = "https://www.facebook.com/groups/NCHU110Fresh/"
TARGET_FILE = os.path.join(os.path.dirname(__file__), "../../data/raw/fb_queries.json")
SCROLL_COUNT = 100  # 往下捲動的次數

async def main():
    async with async_playwright() as p:
        print("Launching browser... (Please wait)")
        # 開啟有介面的瀏覽器，讓使用者可以手動登入
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        print(f"Navigating to {GROUP_URL}")
        await page.goto(GROUP_URL)

        print("\n" + "="*50)
        print("⚠️  請在彈出的瀏覽器中登入您的 Facebook 帳號。")
        print("⚠️  如果您已經登入，或者頁面已成功載入社團貼文，請繼續。")
        print("="*50 + "\n")
        
        # 暫停執行，等待使用者手動登入並在終端機按下 Enter
        input("👉 登入完成且看到社團貼文後，請在此按下 [Enter] 鍵繼續抓取...")

        print("\n開始捲動頁面抓取資料...")
        posts_data = set()
        
        for i in range(SCROLL_COUNT):
            print(f"Scrolling... ({i+1}/{SCROLL_COUNT})")
            # 捲動到底部
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(3000)  # 等待 3 秒讓 FB 載入新貼文
            
            # 點擊畫面上的「查看更多」或「顯示更多」以展開全文
            for expand_text in ["查看更多", "顯示更多"]:
                try:
                    btns = await page.get_by_text(expand_text).all()
                    for btn in btns:
                        if await btn.is_visible():
                            await btn.click(timeout=1000)
                            await page.wait_for_timeout(300)
                except Exception:
                    pass
            
            # 嘗試擷取畫面上的貼文文字
            # Facebook 經常改變 DOM 結構，這裡盡可能抓取可能包含內文的元素
            # 通常貼文內文會在 div[dir="auto"] 裡面
            elements = await page.query_selector_all('div[dir="auto"]')
            
            for el in elements:
                text = await el.inner_text()
                if text and len(text.strip()) > 10:
                    # 過濾出真的像是求租文的內容
                    if "求租" in text or "想找" in text or "找房" in text:
                        # 避免抓到留言、姓名或太短的片段
                        if len(text) > 30 and "留言" not in text[:10]:
                            posts_data.add(text.strip())
                            
            print(f"  目前已找到 {len(posts_data)} 筆潛在求租貼文。")

        print("\n抓取完成！正在儲存資料...")
        
        # 轉換為符合 generate_dataset.py 讀取的格式
        # 其實只要是一個字串陣列即可，或者存成 JSON object
        results = list(posts_data)
        
        with open(TARGET_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"🎉 成功儲存 {len(results)} 筆真實求租貼文至 {TARGET_FILE}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
