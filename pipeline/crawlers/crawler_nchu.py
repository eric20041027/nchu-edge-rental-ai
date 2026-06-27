import asyncio
import csv
import json
import os
import re
import sys
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from .shared import CSV_COLUMNS, FEATURES_DB, FURNITURE_DB, TARGET_CSV, append_to_csv, log

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/index.php"
DETAIL_URL_BASE = "https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/detail.php?rid="
# 註:nchu 的 FEATURES_DB 原與 ddroom 相同(現由 shared 提供);NCHU 把這些放在
# 家具設施/安全管理/消防逃生/租金包含/備註 次表,故下方 derive_nchu_features 推導。

# Keyword -> canonical feature. Scanned over the union of the captured NCHU
# tables. Keeps the front-end vocabulary aligned across both crawlers.
NCHU_FEATURE_RULES = [
    (["飲水機"], "飲水機"),
    (["陽台", "曬衣"], "有陽台"),
    (["電梯"], "有電梯"),
    (["對外窗", "鋁窗", "鐵鋁門 / 窗", "鐵鋁門/窗"], "對外窗"),
    (["氣密窗"], "氣密窗"),
    (["監視", "防盜", "辨識器", "門禁", "感應", "保全"], "保全設施"),
    (["子母車", "垃圾代收", "垃圾處理"], "垃圾代收"),
    (["包裹", "管理員", "管理室"], "包裹代收"),
    (["定期清潔", "清潔服務"], "定期清潔"),
    (["免仲介", "免服務費"], "免仲介費"),
    (["可報稅", "可申報"], "可報稅"),
    (["可入籍", "可遷入戶籍", "可設籍"], "可入籍"),
    (["瓦斯", "電磁爐", "流理台", "排油煙"], "可開伙"),
]


def derive_nchu_features(blob: str) -> list:
    """Map free text from NCHU secondary tables to canonical feature labels."""
    found = []
    for keywords, label in NCHU_FEATURE_RULES:
        if label not in found and any(kw in blob for kw in keywords):
            found.append(label)
    return found


def clean_numerical(text):
    if not text: return ""
    text = str(text)
    # Remove spaces and normalize
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = re.sub(r'(\d)\s+元', r'\1元', text)
    return text.strip()


def _nchu_clean_cell(value: str, column: str) -> str:
    """NCHU 逐欄清理(與 591/ddroom 不同:只去換行/逗號,金額欄做 clean_numerical;
    不去控制字元、不抽電話)。傳給 shared.append_to_csv 保留原行為。"""
    val = str(value).replace("\n", " ").replace("\r", " ").replace(",", " ")
    if column in ["租金", "押金", "另計費用"]:
        val = clean_numerical(val)
    return val


def load_existing_rids(csv_path: str) -> set:
    """讀主檔已存在的興大房源 RID,供重跑去重(比照 591/ddroom 的 existing-URL)。

    原本只 in-run 去重 → 重跑會把現有 145 筆興大全部重複 append。讀現有 CSV 的
    detail.php?rid=<RID> 抽 RID,跳過已抓過的。
    """
    rids: set = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                m = re.search(r"rid=(\d+)", row.get("網址", ""))
                if m:
                    rids.add(m.group(1))
    return rids

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
    
    # Secondary tables. All use a #F555A8 pink header cell + a value cell.
    # Previously only 家具設施 / 另計費用 / 備註 were captured, dropping the
    # 租金包含 / 安全管理 / 消防逃生 tables that NCHU pages actually provide —
    # which is why has_window / has_subsidy / service_level collapsed for NCHU.
    extra_text = ""
    sections = {}  # label -> raw value text, for feature derivation
    for table in soup.find_all("table"):
        header_td = table.find("td", style=re.compile(r"background-color:#F555A8"))
        if header_td:
            label = header_td.get_text(strip=True)
            val_td = table.find_all("td")[1] if len(table.find_all("td")) > 1 else None
            if val_td:
                val = val_td.get_text(strip=True).replace("/", " / ")
                sections[label] = val
                if label == "家具設施":
                    items = [it.strip() for it in val.split("/") if it.strip()]
                    res["家具設施"] = "/".join(items)
                elif label == "另計費用":
                    res["另計費用"] = val
                    extra_text += " " + val
                elif label == "備註":
                    res["特色"] = val
                    extra_text += " " + val
                elif label in ("安全管理", "消防逃生", "租金包含"):
                    extra_text += " " + val

    # Rent inclusions -> water/subsidy columns so downstream "含水"/補助 match.
    rent_includes = sections.get("租金包含", "")
    if "水費" in rent_includes or "含水" in rent_includes:
        res["水費"] = "含水費"
    if any(kw in rent_includes for kw in ["租屋補助", "租金補貼", "租補"]):
        res["租屋補助"] = "可申請租屋補助"

    # Derive canonical feature labels from every captured table + furniture,
    # mirroring crawler_ddroom's 特色 chips. Merge with the 備註-derived 特色
    # text rather than overwriting (备注 carries 禁養寵物/限女性 hard-filter signal).
    feature_blob = " ".join([
        res.get("家具設施", ""), res.get("特色", ""),
        sections.get("安全管理", ""), sections.get("消防逃生", ""),
        sections.get("租金包含", ""),
    ])
    derived = derive_nchu_features(feature_blob)
    if derived:
        existing = [it.strip() for it in res["特色"].split("/") if it.strip()]
        for d in derived:
            if d not in existing:
                existing.append(d)
        res["特色"] = " / ".join(existing)
    
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
    existing_rids = load_existing_rids(TARGET_CSV)
    log(f"  已有 {len(existing_rids)} 筆興大 RID(去重基準)")
    processed_rids = set(existing_rids)  # 已存在的視同已處理 → 跳過,不重複 append
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
                append_to_csv([res], TARGET_CSV, clean_cell=_nchu_clean_cell)
                log(f"  Processed RID {rid}: {res.get('地址')}")
            await dp.close()
            await asyncio.sleep(1)
            
        await browser.close()
    log("NCHU Crawler Done.")

if __name__ == "__main__":
    asyncio.run(main())
