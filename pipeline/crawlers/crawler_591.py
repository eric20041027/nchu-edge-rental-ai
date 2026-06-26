"""
crawler_591.py
591 租屋網 crawler(台中 region=8,興大周邊)。

階段⑥-A 擴量:接第三個來源 591,餵進階段③管線的主 CSV(nchu_rental_info.csv)。
比照 crawler_ddroom.py 的 production 風格(獨立 script、自帶 CSV schema、Playwright、
lock + 禮貌延遲 + existing-URL 去重),抽取核心沿用已驗證的 poc_591.py:
  - 列表頁 → house IDs(/<6+位數字> 連結)
  - 詳情頁 → 地址(.address)、座標(window.__NUXT__ 的 lat/lng)、租金、風控偵測

591 強項:座標 100% 可得(PoC 驗證)。CSV schema 沿用 ddroom 19 欄、不擴欄,
座標暫存進「備註」(地圖標點是另案,不綁進擴量)。

用法:
    python -m pipeline.crawlers.crawler_591 [--limit N] [--max-pages N] [--headed]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import sys
from pathlib import Path

from playwright.async_api import async_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_CSV = str(_REPO_ROOT / "data/raw/nchu_rental_info.csv")
LOCK_FILE = str(Path(__file__).parent / "crawler.lock")

# 興大周邊(中興大學在南區),沿用 ddroom 的目標行政區。
TARGET_AREAS = ["南區", "西區", "東區", "大里區", "太平區"]
REGION_TAICHUNG = 8
DEFAULT_MAX_PAGES = 5
REQUEST_DELAY_S = 2.5  # 禮貌爬,比 PoC 略保守

_LIST_URL = "https://rent.591.com.tw/list?region={region}&keywords=%E8%88%88%E5%A4%A7&page={page}"
_DETAIL_URL = "https://rent.591.com.tw/{house_id}"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# 與 crawler_ddroom.py 同一份 19 欄 schema(管線讀 nchu_rental_info.csv)。
CSV_COLUMNS = [
    "網址", "地址", "類型", "室內坪數", "租金", "押金", "樓層", "電話",
    "家具設施", "另計費用", "水費", "電費", "租屋補助", "特色", "最短租期",
    "圖片網址", "距離(km)", "walk_mins", "scooter_mins",
]

FURNITURE_DB = ["床", "桌子", "椅子", "沙發", "衣櫃", "鞋櫃", "櫃子", "排油煙機", "瓦斯爐",
                "電磁爐", "流理台", "電視", "第四台", "冰箱", "洗衣機", "冷氣", "網路",
                "熱水器", "天然瓦斯", "住警器", "飲水機", "電梯", "陽台"]
FEATURES_DB = ["可養貓", "可養狗", "可養其他寵物", "對外窗", "有電梯", "水泥隔間", "保全設施",
               "垃圾代收", "包裹代收", "定期清潔", "免仲介費", "可報稅", "租金補貼",
               "氣密窗", "有陽台", "可開伙", "可入籍"]

_DOOR_NUM_RE = re.compile(r"\d+號|\d+-\d+號|\d+之\d+號")
_BLOCK_MARKERS = ["請輸入驗證碼", "captcha", "請稍後再試", "異常流量", "403 forbidden", "access denied"]


def log(msg: str) -> None:
    print(msg, flush=True)


def polish(text: str, is_phone: bool = False) -> str:
    """比照 ddroom final_polish:去控制字元、攤平空白、金額去空格、電話抽號碼。"""
    if not text:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ").replace(",", " ").replace('"', " ")
    text = "".join(c for c in text if ord(c) >= 32 and ord(c) != 127)
    if is_phone:
        if any(k in text for k in ["註冊", "登入", "聯絡"]):
            m = re.search(r"(09\d{2}-?\d{3}-?\d{3}|0\d{1,2}-?\d{6,8})", text)
            return m.group(1) if m else ""
        return "".join(c for c in text if c.isdigit() or c == "-")
    if any(c.isdigit() for c in text) and any(k in text for k in ["元", "月", "天"]):
        text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
        text = re.sub(r"(\d)\s+元", r"\1元", text)
    return " ".join(text.split()).strip()


def looks_blocked(html: str) -> bool:
    low = html.lower()
    return any(m.lower() in low for m in _BLOCK_MARKERS)


# 嚴格台灣電話,避免黏到物件 ID/追蹤碼等長數字串:
#   手機 = 09 + 8 碼(共 10 碼,可選 dash 分隔)
#   市話 = 區碼(0 開頭 2~3 碼)+ "-" + 7~8 碼(市話必須有 dash,否則無法與長數字串區分)
# (?<!\d)...(?!\d) 邊界 + 市話強制 dash → 11 碼無分隔的雜訊串(如 01543446671)不命中。
_PHONE_RE = re.compile(
    r"(?<!\d)(09\d{2}-?\d{3}-?\d{3}|0\d{1,2}-\d{6,8})(?!\d)"
)

# 591 房東真號需點擊揭露(JS),raw HTML 只剩平台客服總機 → 黑名單,命中視同無號。
# (live 實測 5/5 都抓到 02-55722000 = 591 客服,灌進去會誤導使用者。)
_PLATFORM_PHONES = {"02-55722000", "0255722000"}


def _extract_phone(html: str) -> str:
    """從 HTML 抽合法電話;591 房東號需點擊揭露,抓不到/抓到平台號就留空(不寧濫勿缺)。"""
    m = _PHONE_RE.search(html)
    if not m:
        return ""
    phone = m.group(1)
    return "" if phone.replace("-", "") in {p.replace("-", "") for p in _PLATFORM_PHONES} else phone


def load_existing_urls(csv_path: str) -> set[str]:
    """讀已抓 URL 去重(比照 ddroom)。"""
    urls: set[str] = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("網址"):
                    urls.add(row["網址"].strip())
    return urls


def append_to_csv(rows: list[dict], csv_path: str) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        if write_header:
            writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([polish(row.get(col, ""), is_phone=(col == "電話")) for col in CSV_COLUMNS])


async def extract_house_ids(page) -> list[str]:
    """列表頁抓物件 ID(/<6+位數字> 連結)。沿用 PoC 邏輯。"""
    ids: list[str] = []
    for a in await page.query_selector_all("a[href]"):
        href = (await a.get_attribute("href")) or ""
        m = re.search(r"rent\.591\.com\.tw/(\d{6,})", href) or re.match(r"^/(\d{6,})$", href)
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


async def extract_coords(page) -> tuple[str, str]:
    """從 window.__NUXT__ 取 lat/lng(591 強項,PoC 驗證 100% 可得)。

    ponytail: 目前未被 get_detail_info 呼叫(CSV schema 無座標欄)。刻意保留 —
    地圖標點升級時(擴 schema)直接接上,不必重寫已驗證的 __NUXT__ 抽取。
    """
    try:
        lat = await page.evaluate(
            "() => { const m = JSON.stringify(window.__NUXT__||{})"
            ".match(/\"lat\"\\s*:\\s*\"?(2[0-9]\\.\\d{4,})\"?/); return m?m[1]:null; }"
        )
        lng = await page.evaluate(
            "() => { const m = JSON.stringify(window.__NUXT__||{})"
            ".match(/\"lng\"\\s*:\\s*\"?(1[0-2][0-9]\\.\\d{4,})\"?/); return m?m[1]:null; }"
        )
        if lat and lng:
            return str(lat), str(lng)
    except Exception:  # noqa: BLE001  座標取不到不致命
        pass
    return "", ""


async def extract_address(page) -> str:
    el = await page.query_selector(".address")
    if el:
        txt = (await el.inner_text()).strip()
        first = re.sub(r"^地址[:：]\s*", "", txt.split("\n")[0]).strip()
        if first:
            return first if "市" in first else f"台中市{first}"
    return ""


async def get_detail_info(page, url: str) -> dict:
    """解析 591 詳情頁 → ddroom schema dict。"""
    res = {col: "" for col in CSV_COLUMNS}
    res["網址"] = url
    try:
        resp = await page.goto(url, wait_until="networkidle", timeout=35000)
        await page.wait_for_timeout(1500)  # 等 Nuxt 注入 __NUXT__
        html = await page.content()
        if (resp and resp.status >= 400) or looks_blocked(html):
            return {}
        res["地址"] = await extract_address(page)
        m = re.search(r"(\d[\d,]{2,})\s*元", html)
        res["租金"] = m.group(0) if m else ""
        # 家具/特色:命中詞表即收(比照 ddroom 的 DB 比對法)
        res["家具設施"] = "/".join(w for w in FURNITURE_DB if w in html)
        res["特色"] = "/".join(w for w in FEATURES_DB if w in html)
        res["電話"] = _extract_phone(html)
        # ponytail: 不抽座標 — ddroom CSV schema 無 lat/lng 欄,座標目前無處可去。
        # 591 座標 100% 可得(PoC 已驗)是地圖標點的好料,但那是另案(擴 schema +
        # 前端地圖),不綁進本次擴量。要做時用 extract_coords() + 加欄。
    except Exception as exc:  # noqa: BLE001  單筆失敗不中斷整輪
        log(f"  detail error {url}: {type(exc).__name__}")
        return {}
    return res if (res.get("地址") or res.get("租金")) else {}


async def main() -> None:
    ap = argparse.ArgumentParser(description="591 台中興大周邊 crawler")
    ap.add_argument("--limit", type=int, default=0, help="本次最多新增筆數(0=不限)")
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    if os.path.exists(LOCK_FILE):
        log("⚠ crawler.lock 存在,可能有其他 crawler 在跑。移除後重試。")
        return
    with open(LOCK_FILE, "w") as f:
        f.write("locked")

    seen = load_existing_urls(TARGET_CSV)
    log(f"已有 {len(seen)} 筆 URL(去重基準)")
    added = 0
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=not args.headed)
            ctx = await browser.new_context(user_agent=_UA, locale="zh-TW",
                                            viewport={"width": 1280, "height": 900})
            list_page = await ctx.new_page()
            collected_ids: list[str] = []
            for page_num in range(1, args.max_pages + 1):
                url = _LIST_URL.format(region=REGION_TAICHUNG, page=page_num)
                log(f"列表頁 {page_num}: {url}")
                try:
                    await list_page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    await list_page.wait_for_timeout(2500)
                except Exception as exc:  # noqa: BLE001
                    log(f"  列表頁失敗: {type(exc).__name__}")
                    continue
                if looks_blocked(await list_page.content()):
                    log("  ⚠ 列表頁疑似風控,停止。")
                    break
                ids = await extract_house_ids(list_page)
                new_ids = [i for i in ids if _DETAIL_URL.format(house_id=i) not in seen]
                collected_ids.extend(i for i in new_ids if i not in collected_ids)
                log(f"  本頁 {len(ids)} 筆,去重後新增 {len(new_ids)}")

            detail_page = await ctx.new_page()
            for i, hid in enumerate(collected_ids, 1):
                if args.limit and added >= args.limit:
                    break
                durl = _DETAIL_URL.format(house_id=hid)
                res = await get_detail_info(detail_page, durl)
                if res:
                    append_to_csv([res], TARGET_CSV)
                    seen.add(durl)
                    added += 1
                    log(f"  [{i}/{len(collected_ids)}] ✓ {res.get('地址', '')[:30]}")
                await asyncio.sleep(REQUEST_DELAY_S)
            await browser.close()
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    log(f"\n完成。新增 {added} 筆 → {TARGET_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
