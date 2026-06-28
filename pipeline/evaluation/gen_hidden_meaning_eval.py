"""隱藏含義評估集 — 測 bi-encoder 能否抓 query 背後的隱藏含義。

需求:使用者輸入口語/隱藏含義(不直接講特徵),希望 TOP30 裡都是符合該隱藏
含義對應條件的房源。

設計三鐵則(都是踩過坑後的正解):
  1. query 純隱藏含義 —— 只說「不想每天提水上樓」,不說「飲水機」。測模型自己連到。
  2. 只收【有區辨力】的條件 —— GT base rate 在 3%~60%。排除冷氣(97%)/陽台(95%)/
     電梯(88%)這種同質特徵:95% 房源都有 → 隨便撈 30 個 precision 都接近 1,指標失效。
  3. 指標按桶自動切換,防 Recall 虛降:
     - GT 集 > 30(大桶):用 Precision@30 —— 「TOP30 有幾成符合」。符合的有 100 間、
       TOP30 全中 → P@30=1.0,不會因分母大而虛降。
     - GT 集 ≤ 30(小桶):用 Recall@30 —— 「該召回的漏沒漏」。位子裝得下全部,
       Recall 才有意義(此時 Precision 反因位子用不完而虛降)。

GT 全部由 property_data 客觀欄位/關鍵字算 → 可解釋、非合成標註、非訓練同源。

產物:tests/fixtures/hidden_meaning_eval.json
用法:python pipeline/evaluation/gen_hidden_meaning_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PD = json.loads((ROOT / "frontend/assets/property_data.json").read_text(encoding="utf-8"))

K = 30
BUCKET_SWITCH = K  # GT > K → precision(防虛降);GT ≤ K → recall


# --- objective predicates(GT 由客觀欄位/關鍵字算)---
def flag(f):   return lambda p: p.get(f) is True
def cheap(t):  return lambda p: 0 < p.get("rent", 0) <= t
def near(km):  return lambda p: 0 < p.get("distance", 0) <= km
def kw(*ks):
    def f(p):
        blob = (str(p.get("ce_text", "")) + " " + str(p.get("furniture", "")) + " "
                + " ".join(p.get("notes", []) if isinstance(p.get("notes"), list) else []))
        return any(k in blob for k in ks)
    return f


def _idxs(*preds) -> list[int]:
    return [i for i, p in enumerate(PD) if all(pred(p) for pred in preds)]


# (隱藏含義 query, [predicates], 隱藏含義說明) — query 不含特徵詞本身。
# 只收有區辨力條件(base rate 3~60%,見鐵則 2)。
EVAL = [
    # 飲水機 18% —— 「提水/買水/喝水」不講飲水機
    ("不想每天提水上樓",        [flag("water_dispenser")], "→ 飲水機"),
    ("懶得一直出門買水",        [flag("water_dispenser")], "→ 飲水機"),
    # 車位 9% —— 「機車/停車」
    ("我有機車要停",            [flag("has_parking")], "→ 車位"),
    ("騎車通勤要有地方放車",    [flag("has_parking")], "→ 車位"),
    # 曬衣場 7% —— 「曬棉被/曬衣服/太陽」
    ("想把棉被拿出去曬太陽",    [kw("曬衣", "曬衣場")], "→ 曬衣場"),
    # 含水 4% —— 「水費」
    ("不想另外付水費",          [kw("含水")], "→ 含水租金"),
    # 台電獨立電錶 38% —— 「電費被收貴/夏天電費」
    ("夏天怕被收很貴的電費",    [flag("is_taipower")], "→ 台電獨立電錶"),
    # 可報稅/補助 42% —— 「報稅/扣抵/補助」
    ("租金想拿來報稅扣抵",      [kw("可報稅")], "→ 可報稅"),
    # 距離 <1km 15% —— 「走路到校/超近」
    ("住超近走路就到學校",      [near(1.0)], "→ 距校 <1km"),
    # 預算 <6000 15% —— 「窮/省/便宜」
    ("學生黨沒什麼錢想省一點",  [cheap(6000)], "→ 月租 <6000"),
    # 複合:近 + 便宜(小桶交集)
    ("想找又近又便宜的",        [near(1.5), cheap(7000)], "→ <1.5km 且 <7000"),
    # 複合:機車位 + 便宜
    ("有機車又想省錢",          [flag("has_parking"), cheap(8000)], "→ 車位 且 <8000"),

    # ── 窄複合(GT ≤ 30 小桶 → recall:測漏不漏)──
    # 974 規模下單一條件幾乎都 >30,要多條件交集才壓得進小桶。
    ("想找超近又超便宜的",      [near(0.7), cheap(6000)], "→ <0.7km 且 <6000"),
    ("走路一下下就到的",        [near(0.5)], "→ 距校 <0.5km"),
    ("超近又便宜還要有飲水機",  [near(1.0), cheap(6000), flag("water_dispenser")],
                                "→ <1km 且 <6000 且 飲水機"),
]


def main() -> None:
    queries = []
    print("隱藏含義評估集 — 每筆指標 + GT 桶大小")
    print("-" * 70)
    for q, preds, meaning in EVAL:
        idxs = _idxs(*preds)
        n = len(idxs)
        metric = "precision" if n > BUCKET_SWITCH else "recall"
        note = ""
        if n == 0:
            note = "  ⚠ GT 空(條件無房源符合)"
        elif metric == "precision":
            note = f"  (GT={n}>{K} → 防 Recall 虛降)"
        print(f"  [{metric:9s}] {n:>3} 間  {q:<16}{meaning}{note}")
        queries.append({"query": q, "metric": metric, "n_relevant": n,
                        "hidden_meaning": meaning, "relevant_idxs": idxs})

    obj = {
        "meta": {
            "purpose": "隱藏含義評估集:口語 query 抓背後條件,TOP30 命中該條件房源",
            "metric_rule": f"GT>{K}→Precision@{K}(防虛降);GT≤{K}→Recall@{K}",
            "discriminative_rule": "只收 base rate 3~60% 的條件(排除冷氣/陽台等同質特徵)",
            "gt_method": "property_data 客觀欄位/關鍵字算,可解釋、非合成、非訓練同源",
            "k": K,
            "property_count": len(PD),
            "n_queries": len(queries),
        },
        "queries": queries,
    }
    out = ROOT / "tests/fixtures/hidden_meaning_eval.json"
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    nr = sum(1 for q in queries if q["metric"] == "recall")
    npr = sum(1 for q in queries if q["metric"] == "precision")
    empty = sum(1 for q in queries if q["n_relevant"] == 0)
    print("-" * 70)
    print(f"共 {len(queries)} 筆(recall {nr} / precision {npr}"
          + (f" / ⚠空 {empty}" if empty else "") + f")→ {out.name}")


if __name__ == "__main__":
    main()
