"""三 crawler(591 / ddroom / nchu)共用元件。

架構精簡批次2 抽出。原本三個 crawler 各自帶 CSV_COLUMNS / FURNITURE_DB /
FEATURES_DB / log / polish / append_to_csv,大量重複。此處統一(DB 用完整版,
591 原本刻意少 5 詞 → 改用此完整版,多命中台電/台水/飲水機等;經重驗資料正確)。

去重 key 因平台而異(591/ddroom 用 URL,nchu 用 RID),故去重 helper 不在此統一;
各 crawler 保留自己的 load_existing_*。append_to_csv 的逐欄清理用 polish_fn 參數化,
讓 nchu(清理邏輯與 591/ddroom 不同)也能共用外殼。
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]

# 主檔路徑 + lock(三 crawler 共用)。
TARGET_CSV = str(_REPO_ROOT / "data/raw/nchu_rental_info.csv")
LOCK_FILE = str(Path(__file__).parent / "crawler.lock")

# 19 欄 schema(三 crawler 完全相同;與管線輸入 nchu_rental_info.csv 表頭一致)。
CSV_COLUMNS: list[str] = [
    "網址", "地址", "類型", "室內坪數", "租金", "押金", "樓層", "電話",
    "家具設施", "另計費用", "水費", "電費", "租屋補助", "特色", "最短租期",
    "圖片網址", "距離(km)", "walk_mins", "scooter_mins",
]

# 家具 / 特色詞表(用於 `w in html` 比對;ddroom 與 nchu 原本即相同的完整版)。
FURNITURE_DB: list[str] = [
    "床", "桌子", "椅子", "沙發", "衣櫃", "鞋櫃", "櫃子", "排油煙機", "瓦斯爐",
    "電磁爐", "流理台", "電視", "第四台", "電視盒", "冰箱", "洗衣機", "冷氣",
    "網路", "熱水器", "天然瓦斯", "住警器", "飲水機", "電梯", "陽台",
]
FEATURES_DB: list[str] = [
    "可養貓", "可養狗", "可養其他寵物", "對外窗", "有電梯", "水泥隔間", "保全設施",
    "垃圾代收", "包裹代收", "定期清潔", "免仲介費", "可報稅", "租金補貼", "高齡友善",
    "飲水機", "氣密窗", "有陽台", "可開伙", "台電", "台水", "可申請補助", "可入籍",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def polish(text: str, is_phone: bool = False) -> str:
    """去控制字元、攤平空白、金額去空格、電話抽號碼(591/ddroom 共用清理)。"""
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


def load_existing_urls(csv_path: str = TARGET_CSV) -> set[str]:
    """讀主檔已存在的「網址」欄做去重(591/ddroom 用)。"""
    urls: set[str] = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("網址"):
                    urls.add(row["網址"].strip())
    return urls


def append_to_csv(
    rows: list[dict],
    csv_path: str,
    columns: list[str] = CSV_COLUMNS,
    clean_cell: Callable[[str, str], str] | None = None,
) -> None:
    """append 房源列到 CSV(共用外殼:建目錄/寫表頭/QUOTE_MINIMAL)。

    clean_cell(value, column) 逐欄清理,讓清理邏輯不同的 crawler(nchu)也能共用。
    預設用 polish(電話欄 is_phone=True)= 591/ddroom 行為。
    """
    if clean_cell is None:
        def clean_cell(value: str, column: str) -> str:  # noqa: 預設 591/ddroom 清理
            return polish(value, is_phone=(column == "電話"))

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        if write_header:
            writer.writerow(columns)
        for row in rows:
            clean_row = [clean_cell(str(row.get(col, "")), col) for col in columns]
            if len(clean_row) == len(columns):
                writer.writerow(clean_row)
