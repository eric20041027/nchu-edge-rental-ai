"""
crawler_ddroom.py
Scraper for dd-room.com (租租通) using Playwright.
Extracts structured JSON-LD data from property detail pages.
"""
import asyncio
import csv
import json
import os
import random
import re
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

TARGET_CSV = os.path.join(os.path.dirname(__file__), "../../data/raw/nchu_rental_info.csv")

# 區域: 8=台中市, 區域代碼: 69=南區, 82=大里區, 92=烏日區, 65=西區, 66=東區
# 區域: 8=台中市, 區域代碼: 69=南區, 82=大里區, 92=烏日區, 65=西區, 66=東區, 64=中區, 68=北區
TARGET_SECTIONS = [69, 82, 92, 65, 66, 64, 68]
MAX_PAGES = 100

CSV_COLUMNS = [
    "網址", "地址", "格局", "類型", "室內坪數", "租金",
    "空房間數", "押金", "安全標章", "樓層", "聯絡人", "電話",
    "家具設施", "租金包含", "另計費用", "安全管理", "消防逃生",
    "備註", "圖片網址", "距離(km)", "walk_mins", "scooter_mins",
]

def load_existing_urls(csv_path: str) -> set:
    if not os.path.exists(csv_path): return set()
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return {row.get("網址", "").strip() for row in csv.DictReader(f)}

def append_to_csv(rows: list, csv_path: str):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists: writer.writeheader()
        for row in rows: writer.writerow(row)
    print(f"  Appended {len(rows)} new rows to {csv_path}")

async def get_detail_info(page, url: str) -> dict:
    try:
        await page.goto(url, wait_until='domcontentloaded')
        html = await page.content()
    except Exception as e:
        print(f"    Failed to load {url}")
        return {}

    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract JSON-LD
    script_tag = soup.find('script', type='application/ld+json')
    if not script_tag:
        return {}
    
    try:
        data = json.loads(script_tag.string)
    except Exception:
        return {}
    
    if isinstance(data, list):
        data = data[0] if data else {}

    # Map JSON-LD to CSV Schema
    addr = ""
    size = ""
    floor = ""
    room_type = "套房"
    furniture = ""
    
    for prop in data.get("additionalProperty", []):
        if prop.get("name") == "地址": addr = prop.get("value", "")
        elif prop.get("name") == "坪數": size = str(prop.get("value", "")).replace(" 坪", "")
        elif prop.get("name") == "樓層": floor = str(prop.get("value", ""))
        elif prop.get("name") == "房數": 
            val = str(prop.get("value", ""))
            room_type = f"{val}房" if val and val.isdigit() else "套房"
        elif prop.get("name") == "家具": furniture = str(prop.get("value", "")).replace(", ", "/")

    if not addr and "address" in data:
        a = data["address"]
        addr = f"{a.get('addressLocality','')}{a.get('addressRegion','')}{a.get('streetAddress','')}"
    
    price_val = ""
    if "offers" in data:
        price_val = f"{data['offers'].get('price', '')} 元/月"

    desc = data.get("description", "")
    notes = []
    if "禁寵" in desc or "不可養寵物" in desc: notes.append("禁養寵物")
    if "限女" in desc: notes.append("限女")
    if "限男" in desc: notes.append("限男")
    if "頂加" in desc or "頂樓加蓋" in desc: notes.append("頂樓加蓋")

    # Extract extra rich condition tags from the page DOM
    tags = soup.select('span.rounded-md.bg-gray-100')
    if not tags:
        tags = soup.select('div.flex.text-base.line-clamp-1 span')
        
    for t in tags:
        tag_text = t.text.strip()
        if tag_text and tag_text not in notes:
            notes.append(tag_text)

    # Extract contact name and phone from JSON-LD (author/provider) or DOM
    contact_name = ""
    contact_phone = ""

    # Try JSON-LD author field
    for key in ("author", "provider", "seller"):
        val = data.get(key, {})
        if isinstance(val, dict):
            contact_name = contact_name or val.get("name", "")
            contact_phone = contact_phone or val.get("telephone", "")

    # Try additionalProperty
    for prop in data.get("additionalProperty", []):
        name = prop.get("name", "")
        value = str(prop.get("value", ""))
        if "聯絡" in name or "姓名" in name or "房東" in name:
            contact_name = contact_name or value
        elif "電話" in name or "手機" in name or "phone" in name.lower():
            contact_phone = contact_phone or value

    # Try DOM as fallback — look for visible phone patterns
    if not contact_phone:
        phone_pattern = re.compile(r'09\d{2}[-\s]?\d{3}[-\s]?\d{3}|0\d{1,2}-\d{6,8}')
        text_nodes = soup.get_text()
        matches = phone_pattern.findall(text_nodes)
        if matches:
            contact_phone = matches[0]

    image_url = data.get("image", "")

    return {
        "網址": url, "地址": addr, "格局": room_type, "類型": "大樓",
        "室內坪數": size, "租金": price_val, "空房間數": "",
        "押金": "", "安全標章": "", "樓層": floor,
        "聯絡人": contact_name, "電話": contact_phone, "家具設施": furniture,
        "租金包含": "", "另計費用": "", "安全管理": "",
        "消防逃生": "", "備註": "/".join(notes), "圖片網址": image_url,
        "距離(km)": "", "walk_mins": "", "scooter_mins": "",
    }


async def main():
    print("=" * 60)
    print("DD-Room Rental Crawler (Playwright JSON-LD Extraction)")
    print("=" * 60)

    existing_urls = load_existing_urls(TARGET_CSV)
    print(f"  Existing properties in CSV: {len(existing_urls)}")
    all_new_rows = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        for section in TARGET_SECTIONS:
            print(f"\nSearching section={section} ...")
            
            for page_num in range(MAX_PAGES):
                search_url = f"https://dd-room.com/search?region_id=8&section_ids={section}&page={page_num + 1}"
                print(f"  Fetching Search Page {page_num + 1}: {search_url}")
                
                try:
                    await page.goto(search_url, wait_until='networkidle')
                    await asyncio.sleep(1)
                except Exception:
                    continue
                
                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')
                
                links = set([a['href'] for a in soup.select('a[href^=\"/object/\"]')])
                urls_to_visit = []
                for href in links:
                    full_url = "https://dd-room.com" + href
                    if full_url not in existing_urls:
                        urls_to_visit.append(full_url)
                
                print(f"    Found {len(urls_to_visit)} new URLs for this page.")
                
                if not links:
                    # If there are no listings on this page at all, we reached the end for this section.
                    print("    No listings found. Moving to next section.")
                    break
                
                for i, url in enumerate(urls_to_visit):
                    print(f"    [{i+1}/{len(urls_to_visit)}] Extracting {url} ...")
                    detail_page = await context.new_page()
                    row = await get_detail_info(detail_page, url)
                    if row and row.get("地址"):
                        append_to_csv([row], TARGET_CSV)
                        all_new_rows.append(row)
                        existing_urls.add(url)
                    await detail_page.close()
                    await asyncio.sleep(random.uniform(0.3, 0.8))

    await browser.close()

    if all_new_rows:
        print(f"\nDone! Added total {len(all_new_rows)} new properties.")
    else:
        print("\nNo new properties found.")


if __name__ == "__main__":
    asyncio.run(main())
