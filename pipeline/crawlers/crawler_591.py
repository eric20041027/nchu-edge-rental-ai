"""
crawler_591.py
DrissionPage version based on ceshine/591scraper.
Extracts data directly from the DOM to bypass strict API anti-bot protections.
"""
import csv
import os
import random
import re
import time

from DrissionPage import ChromiumPage, ChromiumOptions

# ============================================================
# Configuration
# ============================================================
TARGET_CSV = os.path.join(os.path.dirname(__file__), "../../data/raw/nchu_rental_info.csv")

# section IDs near NCHU: 南區=69, 大里區=82, 烏日區=92, 西區=65, 東區=66
TARGET_SECTIONS = [69, 82, 92, 65, 66]
ROOM_KINDS = [2, 3]    # 2=整套, 3=分租套房
MAX_PAGES = 3          # 每個地區最多爬幾頁

CSV_COLUMNS = [
    "網址", "地址", "格局", "類型", "室內坪數", "租金",
    "空房間數", "押金", "安全標章", "樓層", "聯絡人", "電話",
    "家具設施", "租金包含", "另計費用", "安全管理", "消防逃生",
    "備註", "圖片網址", "距離(km)", "walk_mins", "scooter_mins",
]

# ============================================================
# Helpers
# ============================================================
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

def create_browser(headless: bool = False) -> ChromiumPage:
    opts = ChromiumOptions()
    if headless:
        opts.headless()
    opts.set_argument("--disable-blink-features=AutomationControlled")
    opts.set_argument("--no-first-run")
    opts.set_argument("--no-default-browser-check")
    return ChromiumPage(opts)

def parse_price(price_str: str) -> str:
    m = re.search(r"(\d[\d,]*)", price_str)
    if m:
        return f"{m.group(1).replace(',', '')} 元/月"
    return ""

# ============================================================
# Main Crawler Logic
# ============================================================
def get_detail_info(page: ChromiumPage, url: str) -> dict:
    """Visits a detail page and extracts info from the DOM."""
    try:
        page.get(url)
        page.wait.eles_loaded("css:div.title", timeout=10)
    except Exception as e:
        print(f"    Failed to load detail page: {url}")
        return {}

    title_el = page.ele("css:div.title", timeout=2)
    if title_el and "不存在" in (title_el.text or ""):
        return {}

    time.sleep(random.uniform(1.0, 2.5))

    # 地址
    addr_el = page.ele("css:div.address div", timeout=2)
    addr = addr_el.text.strip() if addr_el else ""

    # 租金
    price_el = page.ele("css:div.house-price", timeout=2)
    rent_str = parse_price(price_el.text if price_el else "")

    # 格局、坪數、樓層、型態 (These are usually in base-info class)
    room_type = "套房"
    size = ""
    floor_str = ""
    building_type = "大樓"
    
    base_infos = page.eles("css:.base-info .info-item")
    for info in base_infos:
        text = info.text or ""
        if "坪" in text: size = text.split("\n")[0].strip()
        if "樓層" in text: floor_str = text.split("\n")[0].strip()
        if "型態" in text: building_type = text.split("\n")[0].strip()
        if "格局" in text: room_type = text.split("\n")[0].strip()

    if room_type == "格局": room_type = "套房"

    # 租金含、車位費、管理費
    included = []
    for label_name in ("租金含", "管理費"):
        label_el = page.ele(f"text={label_name}", timeout=1)
        if label_el:
            parent = label_el.parent()
            text_el = parent.ele("css:div.text", timeout=1) if parent else None
            if text_el and text_el.text.strip():
                if label_name == "租金含":
                    inc_text = text_el.text.strip()
                    if "水" in inc_text: included.append("水費")
                    if "電" in inc_text: included.append("電費")
                    if "網" in inc_text: included.append("網路費")
                elif label_name == "管理費":
                    included.append("管理費")

    # 提供設備
    facility_el = page.ele("css:div.service-facility", timeout=2)
    furniture_list = []
    if facility_el:
        items = facility_el.eles("css:dl:not(.del) dd")
        furniture_list = [item.text.strip() for item in items if item.text]

    # 備注與規定
    notes_list = []
    service_el = page.ele("css:section.service", timeout=2)
    if service_el:
        srv_text = service_el.text or ""
        if "不可養寵物" in srv_text: notes_list.append("禁養寵物")
        elif "可養寵物" not in srv_text: pass
        if "限女" in srv_text: notes_list.append("限女")
        if "限男" in srv_text: notes_list.append("限男")

    desc_el = page.ele("css:div.house-condition-content", timeout=2)
    if desc_el:
        desc_text = desc_el.text or ""
        if "頂加" in desc_text or "頂樓加蓋" in desc_text: notes_list.append("頂樓加蓋")
        if "禁寵" in desc_text: notes_list.append("禁養寵物")

    # 圖片
    img_el = page.ele("css:.carousel-list img", timeout=2)
    img_url = img_el.attr("src") if img_el else ""

    return {
        "網址": url, "地址": addr, "格局": room_type, "類型": building_type,
        "室內坪數": size, "租金": rent_str, "空房間數": "",
        "押金": "", "安全標章": "", "樓層": floor_str,
        "聯絡人": "", "電話": "", "家具設施": "/".join(furniture_list),
        "租金包含": "/".join(included), "另計費用": "", "安全管理": "",
        "消防逃生": "", "備註": "/".join(list(set(notes_list))), "圖片網址": img_url,
        "距離(km)": "", "walk_mins": "", "scooter_mins": "",
    }


def main():
    print("=" * 60)
    print("591 Rental Crawler (DrissionPage DOM Extraction)")
    print("=" * 60)

    existing_urls = load_existing_urls(TARGET_CSV)
    print(f"  Existing properties in CSV: {len(existing_urls)}")
    all_new_rows = []

    print("Launching Chromium browser...")
    page = create_browser(headless=False)

    for section in TARGET_SECTIONS:
        for kind in ROOM_KINDS:
            search_url = f"https://rent.591.com.tw/?region=8&section={section}&kind={kind}"
            print(f"\nSearching section={section}, kind={kind}: {search_url}")
            
            try:
                page.get(search_url)
                page.wait.eles_loaded("css:.item-info-title a", timeout=10)
            except Exception as e:
                print(f"  Failed to load search page or no results: {e}")
                continue

            time.sleep(random.uniform(2.0, 4.0))

            for page_num in range(MAX_PAGES):
                print(f"  Processing Search Page {page_num + 1}...")
                
                # 擷取連結
                links = page.eles("css:.item-info-title a")
                urls_to_visit = []
                for link in links:
                    href = link.attr("href") or ""
                    if href.startswith("//"): href = "https:" + href
                    elif href.startswith("/"): href = "https://rent.591.com.tw" + href
                    if "rent-detail" in href and href not in existing_urls:
                        urls_to_visit.append(href)

                # 訪問每個詳細頁面
                for i, url in enumerate(urls_to_visit):
                    print(f"    [{i+1}/{len(urls_to_visit)}] Extracting {url} ...")
                    # 開新分頁以保留搜尋頁
                    tab_id = page.new_tab(url)
                    detail_page = page.get_tab(tab_id)
                    
                    row = get_detail_info(detail_page, url)
                    if row and row.get("地址"):
                        all_new_rows.append(row)
                        existing_urls.add(url)
                    
                    detail_page.close()
                    time.sleep(random.uniform(1.5, 3.5))

                if page_num == MAX_PAGES - 1: break

                # 下一頁
                next_page_btn = page.ele("text=下一頁", timeout=3)
                if not next_page_btn: break
                
                next_class = next_page_btn.attr("class") or ""
                if "disabled" in next_class: break

                try:
                    next_page_btn.click()
                    time.sleep(random.uniform(2.0, 4.0))
                    page.wait.eles_loaded("css:.item-info-title a", timeout=10)
                except Exception:
                    break

    page.quit()

    if all_new_rows:
        append_to_csv(all_new_rows, TARGET_CSV)
        print(f"\nDone! Added {len(all_new_rows)} new properties.")
    else:
        print("\nNo new properties found.")


if __name__ == "__main__":
    main()
