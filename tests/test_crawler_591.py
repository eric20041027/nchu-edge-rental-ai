"""階段⑥-A:crawler_591 純邏輯 + schema 守門測試。

async/Playwright 抓取無法離線測;這裡守住確定性部分:
  - polish / looks_blocked / CSV 寫出
  - **CSV schema 與管線輸入(nchu_rental_info.csv 的 ddroom 19 欄)逐欄一致** —
    這是最關鍵的回歸守門:schema 漂移 = 591 列灌進管線時欄位錯位。
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

# playwright 是 crawler_591 的 module-level import;CI 輕量 job 無 → 整檔 skip。
pytest.importorskip("playwright")

from pipeline.crawlers import crawler_591 as c  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PIPELINE_CSV = REPO / "data" / "raw" / "nchu_rental_info.csv"


def test_polish_amounts_and_phone():
    assert c.polish("10 000 元") == "10000元"
    assert c.polish("註冊後查看 0912-345-678", is_phone=True) == "0912-345-678"
    assert c.polish("  a\n\tb  ") == "a b"
    assert c.polish("") == ""


def test_looks_blocked():
    assert c.looks_blocked("頁面含 請輸入驗證碼 字樣") is True
    assert c.looks_blocked("403 Forbidden") is True
    assert c.looks_blocked("正常租屋詳情") is False


def test_extract_phone_rejects_garbage():
    """live run 曾把 11 碼雜訊串 01543446671 當電話 → 回歸守門。"""
    assert c._extract_phone("...id=01543446671...") == ""   # 11 碼無 dash:雜訊
    assert c._extract_phone("x09123456789999") == ""        # 黏進長數字串
    # 合法號碼仍抽得到
    assert c._extract_phone("聯絡 0912-345-678") == "0912-345-678"
    assert c._extract_phone("0912345678") == "0912345678"
    assert c._extract_phone("04-22850000 分機") == "04-22850000"


def test_csv_schema_matches_pipeline_input():
    """591 輸出 schema 必須與管線讀的 nchu_rental_info.csv 表頭逐欄一致。"""
    assert len(c.CSV_COLUMNS) == 19
    assert c.CSV_COLUMNS[0] == "網址"
    if PIPELINE_CSV.exists():
        with open(PIPELINE_CSV, encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
        assert header == c.CSV_COLUMNS, f"schema 漂移:CSV={header} crawler={c.CSV_COLUMNS}"


def test_append_to_csv_roundtrip(tmp_path):
    out = tmp_path / "out.csv"
    row = {col: "" for col in c.CSV_COLUMNS}
    row["網址"] = "https://rent.591.com.tw/123456"
    row["地址"] = "台中市南區興大路"
    row["租金"] = "8 000 元"  # polish 應去空格
    c.append_to_csv([row], str(out))
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["網址"] == row["網址"]
    assert rows[0]["租金"] == "8000元"
    # 再 append 一筆不應重寫表頭
    c.append_to_csv([row], str(out))
    with open(out, encoding="utf-8-sig") as f:
        assert sum(1 for _ in f) == 3  # header + 2 rows


def test_load_existing_urls(tmp_path):
    out = tmp_path / "out.csv"
    row = {col: "" for col in c.CSV_COLUMNS}
    row["網址"] = "https://rent.591.com.tw/999999"
    c.append_to_csv([row], str(out))
    assert c.load_existing_urls(str(out)) == {"https://rent.591.com.tw/999999"}
    assert c.load_existing_urls(str(tmp_path / "missing.csv")) == set()
