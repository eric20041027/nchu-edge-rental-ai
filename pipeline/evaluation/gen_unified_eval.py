"""階段④ 統一評估集 — 終結三評估集互相矛盾。

問題:之前三個評估集(階段① A/B、真 GT、holdout)用不同 GT 定義 + 不同評分,
數字無法並比 → 「客觀指標↑但 holdout↓」的假矛盾。根因是**指標選錯**:
  - 設施類單訴求(balcony 656/elevator 596 間)是大桶 → Recall@K 數學上限被壓失真。
  - 距離/價格/複合(小桶)→ Recall@K 才有意義。

解法:統一評估集,每筆標對的指標:
  - metric="recall":小桶 GT(距離/價格/複合交集),算 Recall@K(召回率)。
  - metric="precision":大桶 GT(設施類),算 Precision@K(TOP K 命中該特徵的比例
    = holdout 手動判的「TOP5 純度」正式化)。

query 混合單訴求 + 複合(使用者兩種都打)。GT 全 property_data 客觀算。零標點。

產物:tests/fixtures/unified_eval.json
用法:python tests/gen_unified_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PD = json.loads((ROOT / "frontend/assets/property_data.json").read_text(encoding="utf-8"))


def walk(t):  return lambda p: isinstance(p.get("walk_mins"), (int, float)) and p["walk_mins"] <= t
def scoot(t): return lambda p: isinstance(p.get("scooter_mins"), (int, float)) and p["scooter_mins"] <= t
def cheap(t): return lambda p: 0 < p.get("rent", 0) <= t
def flag(f):  return lambda p: p.get(f) is True
def bt(k):    return lambda p: k in str(p.get("building_type", ""))
def quiet():  return lambda p: p.get("geo_tier") == "quiet"
def txt(k):   return lambda p: k in (str(p.get("features", "")) + str(p.get("text", "")) + str(p.get("ce_text", "")))


def _idxs(*preds) -> list[int]:
    return [p["idx"] for p in PD if all(f(p) for f in preds)]


# (query, [predicates], metric) — metric 決定算 recall(小桶)或 precision(大桶設施).
# 單訴求 + 複合混合;模擬真實使用者兩種打法。
EVAL = [
    # ── 單訴求 距離/價格(小桶 → recall)──
    ("走路五分鐘就到學校", [walk(5)], "recall"),
    ("騎車三分鐘到校", [scoot(3)], "recall"),
    ("找最便宜六千以內的", [cheap(6000)], "precision"),   # 124 間大桶 → TOP K 便宜純度
    ("要有停車位停機車", [flag("has_parking")], "precision"),  # 88 間 → TOP K 停車純度
    # ── 單訴求 設施類(大桶 → precision:TOP K 純度)──
    ("想曬棉被要有陽台", [flag("has_balcony")], "precision"),
    ("不想爬樓梯要電梯", [flag("has_elevator")], "precision"),
    ("白天房間不要黑漆漆要有窗", [flag("has_window")], "precision"),
    ("安靜好唸書的地段", [quiet()], "precision"),
    ("喜歡透天厝", [bt("透天")], "precision"),
    ("怕熱一定要冷氣", [txt("冷氣")], "precision"),
    # ── 複合多訴求(小桶交集 → recall)──
    ("走路五分內又便宜八千", [walk(5), cheap(8000)], "recall"),
    ("走路七分有停車位又平價", [walk(7), flag("has_parking"), cheap(8000)], "recall"),
    ("騎車三分電梯又要便宜", [scoot(3), flag("has_elevator"), cheap(7000)], "recall"),
    ("走路五分的透天厝", [walk(5), bt("透天")], "recall"),
]


def main() -> None:
    queries = []
    print("統一評估集 — 每筆指標 + GT 桶大小")
    print("-" * 64)
    for q, preds, metric in EVAL:
        idxs = _idxs(*preds)
        n = len(idxs)
        # recall 類要小桶(可解讀);precision 類本就大桶(算 TOP K 純度,桶大小不影響)。
        warn = ""
        if metric == "recall" and not (1 <= n <= 25):
            warn = " ⚠recall桶過大/空"
        print(f"  [{metric:9s}] {n:>3} 間  {q}{warn}")
        queries.append({"query": q, "metric": metric, "n_relevant": n,
                        "relevant_idxs": idxs})

    obj = {
        "meta": {"created": "2026-06-24",
                 "purpose": "統一評估集:單訴求+複合混合,每筆標對的指標終結假矛盾",
                 "metric_rule": "小桶(距離/價格/複合)→recall;大桶設施→precision(TOP K 純度)",
                 "gt_method": "property_data 欄位客觀算",
                 "query_source": "手寫(單訴求+複合),非訓練同源"},
        "queries": queries,
    }
    out = ROOT / "tests/fixtures/unified_eval.json"
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    nr = sum(1 for q in queries if q["metric"] == "recall")
    npr = sum(1 for q in queries if q["metric"] == "precision")
    print("-" * 64)
    print(f"共 {len(queries)} 筆(recall {nr} / precision {npr})→ {out.name}")


if __name__ == "__main__":
    main()
