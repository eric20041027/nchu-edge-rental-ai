"""
rescrape_nchu_fields.py
One-off, idempotent re-scrape of the NCHU rows already present in
data/raw/nchu_rental_info.csv, using the patched crawler_nchu.get_nchu_detail
which now captures the 租金包含 / 安全管理 / 消防逃生 secondary tables and
derives canonical feature labels.

Surgical: dd-room rows are left byte-for-byte untouched. Only the NCHU rows are
re-fetched and replaced in place. OSRM distance columns (距離(km)/walk_mins/
scooter_mins), which are computed in a separate geocoding step and not present
on the detail page, are carried over from the existing row.
"""
import asyncio
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import crawler_nchu as cn  # noqa: E402

CSV_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/nchu_rental_info.csv")
DISTANCE_COLS = ("距離(km)", "walk_mins", "scooter_mins")


def is_nchu(url: str) -> bool:
    return "nchu.edu.tw" in (url or "")


def rid_from_url(url: str) -> str:
    m = re.search(r"rid=(\d+)", url or "")
    return m.group(1) if m else ""


async def rescrape() -> None:
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    nchu_rows = [r for r in rows if is_nchu(r.get("網址", ""))]
    cn.log(f"Found {len(nchu_rows)} NCHU rows to re-scrape "
           f"(dd-room rows left untouched: {len(rows) - len(nchu_rows)}).")

    # Map url -> freshly scraped row, preserving distance columns.
    fresh_by_url: dict = {}
    async with cn.async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context()
        for i, old in enumerate(nchu_rows, 1):
            url = old["網址"]
            rid = rid_from_url(url)
            if not rid:
                cn.log(f"  [{i}/{len(nchu_rows)}] no rid in {url}, keeping old row")
                continue
            page = await ctx.new_page()
            try:
                res = await cn.get_nchu_detail(page, rid)
            except Exception as e:  # noqa: BLE001
                cn.log(f"  [{i}/{len(nchu_rows)}] rid {rid} FAILED ({e}), keeping old row")
                res = None
            await page.close()

            if not res or not res.get("網址"):
                cn.log(f"  [{i}/{len(nchu_rows)}] rid {rid} empty, keeping old row")
                continue

            # Preserve OSRM distance columns from the existing row.
            for col in DISTANCE_COLS:
                res[col] = old.get(col, "")
            fresh_by_url[url] = res
            cn.log(f"  [{i}/{len(nchu_rows)}] rid {rid} ok: 特色={res.get('特色','')!r}")
            await asyncio.sleep(0.5)
        await browser.close()

    # Rebuild rows in original order, swapping NCHU rows for fresh data.
    rebuilt = []
    for r in rows:
        url = r.get("網址", "")
        if is_nchu(url) and url in fresh_by_url:
            fresh = fresh_by_url[url]
            rebuilt.append({col: fresh.get(col, "") for col in fieldnames})
        else:
            rebuilt.append(r)

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rebuilt)

    cn.log(f"Done. Re-scraped {len(fresh_by_url)}/{len(nchu_rows)} NCHU rows. "
           f"CSV rewritten with {len(rebuilt)} total rows.")


if __name__ == "__main__":
    asyncio.run(rescrape())
