"""stage4 第三輪 Task2 — 生成設施隱喻訓練 pair。

只補【有區辨力設施類】標靶(base rate 3~60%),排除 95% 同質特徵(怕熱→陽台)
與數值距離/預算(那些留結構化)。修 stage4 兩個死因:
  1. 正樣本 property = 真實房源 text(召回時模型看到的格式),非合成富化句。
  2. query 純隱藏含義(不直接講特徵詞),測模型抓背後條件。

每標靶:口語 query 模板 × 隨機抽的真實正樣本房源 → 正 pair;
        再配不命中該特徵的相似真實房源 → 硬負例(is_hard)。
量小(每標靶 ~per_target),避免新塌縮。確定性(SEED 可重現)。

用法:
    python pipeline/data_prep/gen_facility_intent_data.py             # 預設
    python pipeline/data_prep/gen_facility_intent_data.py --per 8     # 每標靶每模板正樣本數
產物:data/processed/facility_intent_train.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PD = json.loads((ROOT / "frontend/assets/property_data.json").read_text(encoding="utf-8"))
OUT = ROOT / "data/processed/facility_intent_train.json"
SEED = 42


def _blob(p: dict) -> str:
    notes = p.get("notes", [])
    return (str(p.get("ce_text", "")) + " " + str(p.get("furniture", "")) + " "
            + " ".join(notes if isinstance(notes, list) else []))


def _prop_text(p: dict) -> str:
    """房源 text 去掉「距離Xkm」段,對齊現役訓練 property 格式。"""
    return re.sub(r"\s*距離[\d.]+km", "", str(p.get("text", ""))).strip()


# 標靶:predicate(命中=正) + 多套口語/隱喻 query(不直接講特徵詞)
TARGETS = [
    ("water_dispenser", lambda p: p.get("water_dispenser") is True,
     ["不想每天提水上樓", "懶得一直出門買水", "想直接裝水喝", "不想扛整箱水回家"]),
    ("has_parking", lambda p: p.get("has_parking") is True,
     ["我有機車要停", "騎車通勤要有地方放車", "車子不知道停哪", "要能停我的摩托車"]),
    # 註:is_taipower(台電獨立電錶)排除 — 該欄位是結構化布林,但房源 text/ce_text
    # 完全不寫「台電/電錶」(線索率 0%)。模型無文字線索可學 → 本質該靠結構化過濾,
    # 非 bi-encoder。這正是「除非本質上該靠結構化的條件」的實例。
    ("曬衣場", lambda p: "曬衣" in _blob(p),
     ["想把棉被拿出去曬太陽", "衣服想曬到太陽", "想曬被子殺菌"]),
    ("可報稅", lambda p: "可報稅" in _blob(p),
     ["租金想拿來報稅扣抵", "想申請租屋補助", "需要能報稅的房東"]),
    ("含水", lambda p: "含水" in _blob(p),
     ["不想另外付水費", "水費想包在房租裡", "不想分開算水錢"]),
    ("第四台", lambda p: "第四台" in _blob(p),
     ["想在房間看電視", "要有第四台可以看", "想追劇看有線台"]),
    ("沙發", lambda p: "沙發" in _blob(p),
     ["想要有地方坐著放鬆", "客廳想擺個沙發", "要有舒服的座位"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=8,
                    help="每標靶每模板的正樣本房源數")
    args = ap.parse_args()
    rng = random.Random(SEED)

    out: list[dict] = []
    print("生成設施隱喻訓練 pair(正樣本=真實房源 text)")
    print("-" * 64)
    for field, pred, queries in TARGETS:
        pos_pool = [p for p in PD if pred(p)]
        neg_pool = [p for p in PD if not pred(p)]
        n_pos = n_neg = 0
        for q in queries:
            k = min(args.per, len(pos_pool))
            for p in rng.sample(pos_pool, k):
                out.append({"query": q, "property": _prop_text(p),
                            "label": 1, "relevance": 1, "is_hard": False})
                n_pos += 1
            # 每模板配 2 個硬負例(不命中該特徵的相似真實房源)
            for p in rng.sample(neg_pool, min(2, len(neg_pool))):
                out.append({"query": q, "property": _prop_text(p),
                            "label": 0, "relevance": 0, "is_hard": True})
                n_neg += 1
        print(f"  {field:16} {len(queries)} 模板 → 正 {n_pos} / 硬負 {n_neg}")

    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    pos = sum(1 for x in out if x["label"] == 1)
    print("-" * 64)
    print(f"共 {len(out)} pair(正 {pos} / 硬負 {len(out)-pos})→ {OUT.name}")


if __name__ == "__main__":
    main()
