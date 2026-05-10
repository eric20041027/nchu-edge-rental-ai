import asyncio
import csv
import json
import os
import re
import sys
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

sys.stdout.reconfigure(encoding='utf-8')

TARGET_CSV = os.path.join(os.path.dirname(__file__), "../../data/raw/nchu_rental_info.csv")
BASE_URL = "https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/index.php"
DETAIL_URL_BASE = "https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/detail.php?rid="

CSV_COLUMNS = [
    "網址", "地址", "類型", "室內坪數", "租金", "押金", "樓層", "電話", 
    "家具設施", "另計費用", "水費", "電費", "租屋補助", "特色", "最短租期", 
    "圖片網址", "距離(km)", "walk_mins", "scooter_mins"
]

FURNITURE_DB = ["床", "桌子", "椅子", "沙發", "衣櫃", "鞋櫃", "櫃子", "排油煙機", "瓦斯爐", "電磁爐", "流理台", "電視", "第四台", "電視盒", "冰箱", "洗衣機", "冷氣", "網路", "熱水器", "天然瓦斯", "住警器", "飲水機", "電梯", "陽台"]

def log(msg):
    print(msg, flush=True)

def clean_numerical(text):
    if not text: return ""
    text = str(text)
    # Remove spaces and normalize
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = re.sub(r'(\d)\s+元', r'\1元', text)
    return text.strip()

def append_to_csv(rows: list, csv_path: str):
    file_exists = os.path.exists(csv_path)
    write_header = not file_exists or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        if write_header:
            writer.writerow(CSV_COLUMNS)
        for row in rows:
            clean_row = []
            for col in CSV_COLUMNS:
                val = str(row.get(col, "")).replace("\n", " ").replace("\r", " ").replace(",", " ")
                if col in ["租金", "押金", "另計費用"]:
                    val = clean_numerical(val)
                clean_row.append(val)
            if len(clean_row) == 19:
                writer.writerow(clean_row)

async def get_nchu_detail(page, rid: str) -> dict:
    url = f"{DETAIL_URL_BASE}{rid}"
    try:
        await page.goto(url, wait_until='networkidle', timeout=30000)
        html = await page.content()
    except: return {}

    soup = BeautifulSoup(html, 'html.parser')
    res = {col: "" for col in CSV_COLUMNS}
    res["網址"] = url

    # Main data table
    main_table = soup.find("table", class_="table-striped")
    data_map = {}
    if main_table:
        tds = main_table.find_all("td")
        for i in range(len(tds)):
            text = tds[i].get_text(strip=True)
            if text in ["地址", "格局", "類型", "室內坪數", "租金", "押金", "樓層", "電話", "聯絡人"]:
                if i + 1 < len(tds):
                    data_map[text] = tds[i+1].get_text(strip=True)

    res["地址"] = data_map.get("地址", "")
    layout = data_map.get("格局", "")
    btype = data_map.get("類型", "")
    res["類型"] = f"{layout} {btype}".strip()
    res["室內坪數"] = data_map.get("室內坪數", "").replace("坪", "").strip()
    res["租金"] = data_map.get("租金", "")
    res["押金"] = data_map.get("押金", "")
    res["樓層"] = data_map.get("樓層", "")
    res["電話"] = data_map.get("電話", "")
    
    # Secondary tables (Amenities, Includes, Extras)
    extra_text = ""
    for table in soup.find_all("table"):
        header_td = table.find("td", style=re.compile(r"background-color:#F555A8"))
        if header_td:
            label = header_td.get_text(strip=True)
            val_td = table.find_all("td")[1] if len(table.find_all("td")) > 1 else None
            if val_td:
                val = val_td.get_text(strip=True).replace("/", " / ")
                if label == "家具設施":
                    items = [it.strip() for it in val.split("/") if it.strip()]
                    res["家具設施"] = "/".join(items)
                elif label == "另計費用":
                    res["另計費用"] = val
                    extra_text += " " + val
                elif label == "備註":
                    res["特色"] = val
                    extra_text += " " + val
    
    # Improved Electric Fee detection: look for per unit price
    # Usually format is like "電費5元" or "每度5元"
    all_text = soup.get_text()
    e_match = re.search(r"(?:電|度).*?(\d\.?\d*)\s*元", extra_text + " " + all_text)
    if e_match:
        val = e_match.group(1)
        if float(val) < 20: # Sanity check for per unit price
            res["電費"] = f"每度{val}元"

    img = soup.find("img", src=re.compile(r"upload"))
    if img:
        src = img.get("src")
        if src.startswith("http"):
            res["圖片網址"] = src
        else:
            if src.startswith("./"): src = src[2:]
            res["圖片網址"] = "https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/" + src

    return res

async def main():
    log("Starting NCHU Rental Crawler...")
    processed_rids = set()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Go to the search page to get RIDs
        await page.goto(f"{BASE_URL}?tpl=0", wait_until='networkidle')
        
        # Find all detail links (detail.php?rid=...)
        rids = []
        for p_idx in range(10): # Process up to 10 pages
            await page.goto(f"{BASE_URL}?start={p_idx * 20}&tpl=0", wait_until='networkidle')
            elements = await page.query_selector_all('a[href*="rid="]')
            page_rids = []
            for el in elements:
                href = await el.get_attribute("href")
                m = re.search(r"rid=(\d+)", href)
                if m: page_rids.append(m.group(1))
            
            if not page_rids: break
            rids.extend(page_rids)
            log(f"  Found {len(set(page_rids))} properties on page {p_idx + 1}")
            await asyncio.sleep(1)
        
        unique_rids = sorted(list(set(rids)))
        log(f"Found {len(unique_rids)} unique NCHU properties.")
        
        for rid in unique_rids:
            if rid in processed_rids: continue
            processed_rids.add(rid)
            dp = await context.new_page()
            res = await get_nchu_detail(dp, rid)
            if res:
                append_to_csv([res], TARGET_CSV)
                log(f"  Processed RID {rid}: {res.get('地址')}")
            await dp.close()
            await asyncio.sleep(1)
            
        await browser.close()
    log("NCHU Crawler Done.")

if __name__ == "__main__":
    asyncio.run(main())
