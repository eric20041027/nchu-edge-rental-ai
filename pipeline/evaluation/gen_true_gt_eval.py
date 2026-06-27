"""階段④ 真 GT 評估集生成器 — 客觀衡量 bi-encoder 召回效果。

為何需要:現有 generalization_eval 用「大桶 GT」(balcony=656間)→ Recall@30
數學上限被壓到失真;且評估 query 與訓練 query 同源 → selection bias。

本評估集兩點更客觀:
  1. 小桶 GT:用「多條件交集」(距離骨幹 + 1-2 設施)讓 relevant 收斂到
     1-20 間 → Recall@K 回到可解讀範圍(階段① 同精神,小桶)。
  2. query 非同源:GT 由 property_data 欄位/OSRM 客觀算,query 是手寫的
     自然口語(刻意與訓練 query 不同說法),破 selection bias。

排序品質(NDCG)非 bi-encoder 職責(那是 CE 精排),故本評估集只算 Recall@K。

產物:tests/fixtures/true_gt_eval.json(schema 對齊 eval harness:
  query / bucket / n_relevant / relevant_idxs / note)。

用法:
    python tests/gen_true_gt_eval.py          # 生成 + 印 GT 桶大小自檢
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PD = json.loads((ROOT / "frontend/assets/property_data.json").read_text(encoding="utf-8"))

# ── GT predicate 工廠(全部 property_data 欄位客觀算)──────────────────────
def walk(t):    return lambda p: isinstance(p.get("walk_mins"), (int, float)) and p["walk_mins"] <= t
def scoot(t):   return lambda p: isinstance(p.get("scooter_mins"), (int, float)) and p["scooter_mins"] <= t
def cheap(t):   return lambda p: 0 < p.get("rent", 0) <= t
def flag(f):    return lambda p: p.get(f) is True
def bt(k):      return lambda p: k in str(p.get("building_type", ""))
def quiet():    return lambda p: p.get("geo_tier") == "quiet"


def _inter(preds) -> list[int]:
    return [p["idx"] for p in PD if all(f(p) for f in preds)]


# ── 評估 query:自然口語(非訓練同源說法)+ 多條件交集 GT(距離骨幹收斂小桶)──
# 每筆 = (query 自然說法, [GT 條件 predicates], note)。零標點對齊 ce_text 分布。
EVAL = [
    ("走路五分鐘到學校又要便宜八千內", [walk(5), cheap(8000)], "近+平價"),
    ("走路七分內有車位停機車", [walk(7), flag("has_parking")], "近+停車"),
    ("騎車三分鐘到校預算七千", [scoot(3), cheap(7000)], "騎車近+平價"),
    ("走路三分鐘超近的房", [walk(3)], "極近"),
    ("走路五分內想找最便宜", [walk(5), cheap(6000)], "近+很便宜"),
    ("騎車三分內要有電梯又便宜七千", [scoot(3), flag("has_elevator"), cheap(7000)], "騎車近+電梯+平價"),
    ("走路七分內透天又要便宜六千", [walk(7), bt("透天"), cheap(6000)], "近+透天+很平價"),
    ("走路七分有陽台可以曬衣服七千內", [walk(7), flag("has_balcony"), cheap(7000)], "近+陽台+平價"),
    ("騎車五分內安靜好唸書八千內", [scoot(5), quiet(), cheap(8000)], "騎車近+安靜+平價"),
    ("走路五分內有電梯不想爬樓", [walk(5), flag("has_elevator"), cheap(10000)], "近+電梯+中價"),
    ("走路七分有停車位又平價八千", [walk(7), flag("has_parking"), cheap(8000)], "近+停車+平價"),
    ("騎車三分內的透天厝", [scoot(3), bt("透天")], "騎車很近+透天"),
]


def main() -> None:
    queries = []
    print("真 GT 評估集 — GT 桶大小自檢(目標 1-20 間,Recall@K 可解讀)")
    print("-" * 60)
    for q, preds, note in EVAL:
        idxs = _inter(preds)
        n = len(idxs)
        mark = "✓" if 1 <= n <= 20 else ("⚠空" if n == 0 else "⚠大")
        print(f"  [{mark}] {n:>3} 間  {q}")
        queries.append({"query": q, "bucket": "true_gt", "n_relevant": n,
                        "relevant_idxs": idxs, "note": note})

    sizes = [x["n_relevant"] for x in queries]
    empty = [x for x in queries if x["n_relevant"] == 0]
    big = [x for x in queries if x["n_relevant"] > 20]
    print("-" * 60)
    print(f"共 {len(queries)} 筆 | GT 桶 min={min(sizes)} max={max(sizes)} "
          f"中位={sorted(sizes)[len(sizes)//2]} | 空={len(empty)} 過大={len(big)}")

    obj = {
        "meta": {"created": "2026-06-23",
                 "purpose": "客觀衡量 bi-encoder 召回:小桶 GT(多條件交集)+ 非同源 query",
                 "gt_method": "property_data 欄位/OSRM 客觀交集算",
                 "query_source": "手寫自然口語(刻意與訓練 query 不同源,破 selection bias)",
                 "metric": "Recall@K only(排序非 bi-encoder 職責)"},
        "queries": queries,
    }
    out = ROOT / "tests/fixtures/true_gt_eval.json"
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"寫入 {out.name}")


if __name__ == "__main__":
    main()
