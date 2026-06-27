"""
crawler_ddroom.py
Ultimate Robust Scraper for dd-room.com (PRODUCTION v11 - CLEAN PRICES)
- Fixed Prices: Removes spaces inside numerical strings (e.g. 10 000 -> 10000).
- Fixed Type: Combined Space and Building info.
- Fixed Phone: Multi-source scanning.
- Low CPU: High delays (6-10s) and single instance lock.
- No Layout: 20 columns, perfectly flat.
"""
import asyncio
import csv
import json
import os
import random
import re
import sys
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from .shared import (
    CSV_COLUMNS, FEATURES_DB, FURNITURE_DB, LOCK_FILE, TARGET_CSV,
    append_to_csv, log,
)
from .shared import polish as final_polish  # 保留呼叫點名稱

sys.stdout.reconfigure(encoding='utf-8')

TARGET_AREAS = ["南區", "西區", "東區", "大里區", "太平區"]
MAX_PAGES = 100

async def get_detail_info_ultimate(page, url: str) -> dict:
    try:
        await page.goto(url, wait_until='networkidle', timeout=35000)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 3)")
        await asyncio.sleep(1.5)
        html = await page.content()
        title = await page.title()
    except Exception: return {}

    soup = BeautifulSoup(html, 'html.parser')
    res = {col: "" for col in CSV_COLUMNS}
    res["網址"] = url
    full_text = soup.get_text(separator=" ", strip=True)
    
    script_tag = soup.find('script', type='application/ld+json')
    json_data = {}
    if script_tag:
        try:
            json_data = json.loads(script_tag.string)
            if isinstance(json_data, list): json_data = json_data[0]
        except: pass
    desc = json_data.get("description", "") or ""

    spec_grid = {}
    for p in soup.find_all("p", class_=re.compile(r"flex")):
        spans = p.find_all("span")
        if len(spans) >= 2:
            key = spans[0].get_text(strip=True)
            val = spans[1].get_text(strip=True)
            if key: spec_grid[key] = val

    space = spec_grid.get("空間", "")
    building = spec_grid.get("建物", spec_grid.get("建物類型", ""))
    res["類型"] = (space + " " + building).strip()
    if not res["類型"]:
        bc = soup.find("nav", {"aria-label": "breadcrumb"})
        if bc:
            for it in bc.find_all("li"):
                txt = it.get_text(strip=True)
                if any(k in txt for k in ["套房", "雅房", "整層住家"]):
                    res["類型"] = txt
                    break

    raw_addr = ""
    title_match = re.search(r"(台中市[^\s,，。'|]+)", title)
    if title_match: raw_addr = title_match.group(0)
    if not raw_addr and isinstance(json_data.get("address"), dict):
        raw_addr = json_data["address"].get("streetAddress", "")
    if not raw_addr: raw_addr = spec_grid.get("地址", "")
    if not raw_addr or "台中" not in raw_addr:
        m = re.search(r"(台中市|台中縣|臺中市|臺中縣)[^\s,，。']{5,40}", full_text)
        if m: raw_addr = m.group(0)
    res["地址"] = raw_addr
    if res["地址"]:
        res["地址"] = res["地址"].replace("臺中", "台中").replace("台中市台中市", "台中市")
        if "台中" not in res["地址"]: res["地址"] = "台中市" + res["地址"]

    res["室內坪數"] = (spec_grid.get("室內坪數", "") or spec_grid.get("坪數", "")).replace("坪", "")
    res["租金"] = spec_grid.get("租金", "") or (f"{json_data['offers'].get('price', '')} 元/月" if "offers" in json_data else "")
    res["押金"] = spec_grid.get("押金", "")
    res["樓層"] = spec_grid.get("樓層", "")
    res["最短租期"] = spec_grid.get("最短租期", "")
    res["另計費用"] = spec_grid.get("管理費", "")

    furn_found = set()
    feat_found = set()
    for el in soup.find_all(["div", "span", "li"]):
        txt = el.get_text(strip=True)
        if not txt or len(txt) > 20: continue
        is_furn = txt in FURNITURE_DB
        is_feat = any(k == txt for k in FEATURES_DB)
        if is_furn or is_feat:
            inactive = "opacity-40" in str(el.get("class", "")) or any("opacity-40" in str(p.get("class", "")) for p in el.parents) or el.find("strike")
            if not inactive:
                if is_furn: furn_found.add(txt)
                else: feat_found.add(txt)
    
    res["家具設施"] = "/".join(sorted(list(furn_found)))
    res["特色"] = "/".join(sorted(list(feat_found)))
    if any(k in res["特色"] or k in full_text for k in ["台水", "水費照台水"]): res["水費"] = "水費照台水"
    if any(k in res["特色"] or k in full_text for k in ["台電", "電費照台電"]): res["電費"] = "電費照台電"
    if any(k in res["特色"] or k in full_text for k in ["租補", "租金補貼", "可申請補助"]): res["租屋補助"] = "可申請租屋補助"

    res["區域"] = ""
    bc = soup.find("nav", {"aria-label": "breadcrumb"})
    if bc:
        for it in bc.find_all("li"):
            txt = it.get_text(strip=True)
            if any(k in txt for k in ["區", "市", "鎮", "鄉"]):
                if "台中" not in txt and any(d in txt for d in TARGET_AREAS):
                    res["區域"] = txt
                    break

    res["電話"] = json_data.get("author", {}).get("telephone", "")
    if not res["電話"]:
        phone_m = re.search(r'(09\d{2}-?\d{3}-?\d{3}|0\d{1,2}-?\d{6,8})', desc + " " + full_text)
        if phone_m: res["電話"] = phone_m.group(1)
    res["圖片網址"] = json_data.get("image", "")

    return res

CONCURRENCY = 3   # parallel detail pages

async def fetch_one(context, url: str, csv_lock: asyncio.Lock) -> None:
    dp = await context.new_page()
    try:
        res = await get_detail_info_ultimate(dp, url)
        if res and (res.get("地址") or res.get("租金")):
            addr = res.get("地址", "")
            district_bc = res.get("區域", "")
            m = re.search(r"(南區|西區|東區|大里區|太平區|西屯區|北屯區|南屯區|北區|中區|豐原區|潭子區|龍井區|沙鹿區|清水區|大雅區|神岡區|烏日區|霧峰區)", addr)
            addr_dist = m.group(1) if m else ""
            is_target = (addr_dist in TARGET_AREAS) or (district_bc in TARGET_AREAS)
            if is_target:
                async with csv_lock:
                    append_to_csv([res], TARGET_CSV)
            else:
                log(f"  Skipping (Not target area): {addr_dist or 'Unknown'} in {addr}")
    finally:
        await dp.close()


async def main():
    if os.path.exists(LOCK_FILE):
        log("Crawler already running. Exiting.")
        return
    with open(LOCK_FILE, "w") as f: f.write("locked")

    try:
        log("=" * 60)
        log("DD-Room Scraper (v12 - FAST, 3x parallel)")
        log("=" * 60)
        async with async_playwright() as p:
            processed_urls = set()
            if os.path.exists(TARGET_CSV):
                with open(TARGET_CSV, "r", encoding="utf-8-sig") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if row: processed_urls.add(row[0])
            log(f"Resuming crawler: {len(processed_urls)} already processed.")

            MAX_PROPERTIES = 1000
            SEARCH_AREAS = TARGET_AREAS + ["西屯區", "北屯區", "南屯區", "北區"]
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            csv_lock = asyncio.Lock()
            list_page = await context.new_page()

            for area in SEARCH_AREAS:
                for page_num in range(MAX_PAGES):
                    search_url = f"https://dd-room.com/search?city=臺中市&area={area}&page={page_num + 1}"
                    log(f"District {area} Page {page_num + 1}")
                    try:
                        await list_page.goto(search_url, wait_until='networkidle', timeout=60000)
                        await list_page.wait_for_selector('a[href^="/object/"]', timeout=35000)
                        elements = await list_page.query_selector_all('a[href^="/object/"]')
                        links = []
                        for el in elements:
                            href = await el.get_attribute("href")
                            if href: links.append(f"https://dd-room.com{href}")
                        urls = [u for u in sorted(set(links)) if u not in processed_urls]

                        if not urls and links:
                            log(f"  All properties on this page already processed. Moving on.")
                            break

                        quota = MAX_PROPERTIES - len(processed_urls)
                        urls = urls[:quota]
                        log(f"  New Properties: {len(urls)}")

                        for u in urls:
                            processed_urls.add(u)

                        # Process in parallel batches of CONCURRENCY
                        for i in range(0, len(urls), CONCURRENCY):
                            batch = urls[i:i + CONCURRENCY]
                            await asyncio.gather(*[
                                fetch_one(context, u, csv_lock) for u in batch
                            ])
                            await asyncio.sleep(random.uniform(0.3, 0.6))

                        if len(processed_urls) >= MAX_PROPERTIES:
                            log(f"Reached {MAX_PROPERTIES} properties. Stopping.")
                            break

                        await asyncio.sleep(0.5)
                    except Exception as e:
                        log(f"  ERROR: {e}")
                        break

                if len(processed_urls) >= MAX_PROPERTIES:
                    break

    finally:
        if os.path.exists(LOCK_FILE): os.remove(LOCK_FILE)
        scratch_dir = os.path.join(os.path.dirname(__file__), "../../scratch")
        if os.path.exists(scratch_dir):
            import shutil
            try:
                shutil.rmtree(scratch_dir)
            except Exception:
                pass
    log(f"Done. Total unique properties: {len(processed_urls)}")

if __name__ == "__main__":
    asyncio.run(main())
