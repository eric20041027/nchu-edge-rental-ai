"""做法 B:用 extension map 的口語 key 當「額外正樣本 query」增強 CE 訓練資料。

動機(見 docs/extension_map_training_coverage.md + ce_query_expansion 記憶):
  extension map 目前只在**推論時**用 semanticExpandQuery 拼接擴展詞餵 CE,屬
  train/inference 分佈偏移(CE 訓練時沒見過這些口語格式)。本腳本把擴展表的
  口語 key **當原樣 query 加進訓練資料(不拼擴展詞)**,讓 CE 從口語變體**內化**
  語意,而非依賴推論時規則拼接。

  關鍵設計 = 做法 B 而非做法 A:
    - 做法 A(不採用):query 拼擴展詞進訓練 → CE 只是背擴展表,沒提升理解。
    - 做法 B(本腳本):口語 key 原樣當 query → CE 學「怕熱↔冷氣房源」的語意關聯。

對齊原訓練資料以避免格式/標籤漂移:
  - property 文字用 generate_dataset.property_to_text(完全同格式)。
  - relevance 用 generate_dataset.compute_relevance_score(同一套 0-3 標籤邏輯)。
  - 物件級隔離:只配對 TRAIN split 的房源(不碰 dev/test 房源,防洩漏)。
  - 同時生正樣本(rel>=2 的口語×房源)+ 困難負樣本(語意衝突),維持類別平衡。

用法:
    python pipeline/data_prep/augment_with_expansion_map.py            # 預覽統計+範例
    python pipeline/data_prep/augment_with_expansion_map.py --write    # 寫出增強訓練檔

輸出(--write):data/processed/recommendation_train_augmented.json
  = 原 recommendation_train.json + 新增的擴展口語樣本。Colab 訓練時把訓練輸入
  指向此檔即可做 baseline vs 增強版 A/B。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline" / "data_prep"))

import generate_dataset as G  # noqa: E402  復用 property_to_text / compute_relevance_score

# 原始 property_to_text 參照(在任何 monkeypatch 之前綁定),供富化版內部呼叫避免遞迴。
_ORIG_PROPERTY_TO_TEXT = G.property_to_text

RULES = ROOT / "data" / "semantic_rules.json"
TRAIN = ROOT / "data" / "processed" / "recommendation_train.json"
OUT = ROOT / "data" / "processed" / "recommendation_train_augmented.json"

# 距離類(走路/騎車到學校)CE 不該學,走 OSRM —— 排除(見覆蓋差距分析)。
DISTANCE_FEATURES = {"走路10分", "騎車10分"}

MIN_POS_RELEVANCE = 2   # 只收 rel>=2 的口語×房源當正樣本(高品質配對)
MAX_POS_PER_KEY = 6     # 每個口語 key 最多配幾個正樣本房源(避免單一意圖灌爆)
NEG_PER_KEY = 3         # 每個口語 key 配幾個困難負樣本

# production 同義詞橋(可開伙→廚房 等),從單一來源載入,避免「指向特徵」與房源
# 實際用詞不對齊導致漏配(見診斷:房源寫「可開伙」但 compute_relevance 查「開火」)。
_rules_json = json.loads(RULES.read_text(encoding="utf-8"))
PROP_SYNONYMS: dict[str, list[str]] = _rules_json.get("property_synonyms", {})


def load_expansion_keys() -> dict[str, list[str]]:
    """口語 key → 指向特徵列表;排除距離類。"""
    return {k: feats for k, feats in _rules_json["rules"].items()
            if not (set(feats) & DISTANCE_FEATURES)}


def prop_full_text(prop: dict) -> str:
    """房源完整可比對文字(含 notes/furniture)——對齊 production buildPropText 概念。
    注意:這只用於『判定正樣本』;存進訓練樣本的 property 仍用 property_to_text
    以保持與原訓練資料格式一致。"""
    parts = [_ORIG_PROPERTY_TO_TEXT(prop)]
    parts.extend(prop.get("notes", []) or [])
    parts.extend(prop.get("furniture", []) or [])
    return " ".join(str(p) for p in parts)


def property_to_text_enriched(prop: dict) -> str:
    """富化版 property 文字:在 property_to_text 基礎上,補上『完整 furniture + 全部
    notes 特徵詞』,讓 CE 看得到 notes-only 特徵(隔音/門禁/採光/車位/可開伙…)。

    對齊 production buildPropText 概念(前端 inference.js):房源所有結構化特徵都納入
    可比對文字,而非只取 furniture 前 5 項、丟掉 notes。

    ⚠️ 用此函式重生訓練資料 → 必須用同樣富化文字推論(前端 scorePair 改餵富化文字),
    否則訓練/推論不一致 → 重蹈 docs/ce_text_layer_decision.md 的 OOD NO-GO。

    去重保序,避免同詞重複膨脹序列長度。"""
    base = _ORIG_PROPERTY_TO_TEXT(prop)   # 用原始參照,避免 monkeypatch 後自我遞迴
    seen, extra = set(base.split()), []
    for f in (prop.get("furniture", []) or []):
        short = str(f).replace("（電）", "").replace("機車停車位", "機車位").replace("書桌椅", "書桌")
        if short and short not in seen:
            seen.add(short); extra.append(short)
    for note in (prop.get("notes", []) or []):
        note = str(note)
        if note and note not in seen:
            seen.add(note); extra.append(note)
    return base + (" " + " ".join(extra) if extra else "")


def prop_has_any_feature(full_text: str, features: list[str]) -> bool:
    """房源是否含任一指向特徵(含同義詞橋接):直接命中,或任一同義詞命中。
    例:指向特徵「廚房」可透過 PROP_SYNONYMS['廚房']=['可開伙','流理台'] 對上房源的「可開伙」。"""
    for f in features:
        if f in full_text:
            return True
        for syn in PROP_SYNONYMS.get(f, []):
            if syn in full_text:
                return True
    return False


def rebuild_enriched_datasets() -> None:
    """用富化房源文字重生整份 train/dev/test。

    做法:monkeypatch generate_dataset.property_to_text → property_to_text_enriched,
    再呼叫 G.main()。其內部所有 prop_texts 改用富化文字,但切分/負樣本/標籤邏輯
    完全不變(零重寫)。改寫輸出路徑為 *_enriched.json,不污染 baseline。

    產出:recommendation_{train,dev,test}_enriched.json + property_texts_enriched.json
    ⚠️ 用這些訓練 → 推論必須用富化文字 + MAX_LENGTH=128(見 property_to_text_enriched 警告)。
    """
    import os
    orig_p2t = G.property_to_text
    orig_join = os.path.join
    proc = str(ROOT / "data" / "processed")

    def enriched_join(*a):
        path = orig_join(*a)
        # 把 main() 寫死的三個輸出檔名改 *_enriched(只攔輸出,不動輸入讀取)
        for name in ("recommendation_train.json", "recommendation_dev.json",
                     "recommendation_test.json", "property_texts.json"):
            if path.endswith(name):
                return path[: -len(".json")] + "_enriched.json"
        return path

    G.property_to_text = property_to_text_enriched
    G.os.path.join = enriched_join
    try:
        print("=== 用富化房源文字重生 train/dev/test (→ *_enriched.json) ===")
        G.main()
    finally:
        G.property_to_text = orig_p2t
        G.os.path.join = orig_join
    print(f"\n✅ 富化資料集已生成於 {proc}/recommendation_*_enriched.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="寫出 query 增強訓練檔")
    ap.add_argument("--rebuild-enriched", action="store_true",
                    help="用富化房源文字重生 train/dev/test (MAX_LENGTH=128 實驗用)")
    ap.add_argument("--enriched", action="store_true",
                    help="搭配 --write:增強樣本的 property 也用富化文字 + append 進 enriched train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    if args.rebuild_enriched:
        rebuild_enriched_datasets()
        if not args.write:
            return 0

    keys = load_expansion_keys()
    properties = G.load_properties()
    # --enriched:存進樣本的 property 用富化文字(notes 特徵可見),判定也基於它
    #            → 安靜/門禁/採光類得以保留(不再被「線索不可見」丟棄)。
    p2t = property_to_text_enriched if args.enriched else G.property_to_text
    prop_texts = [p2t(p) for p in properties]                     # 存進樣本用
    full_texts = [prop_full_text(p) for p in properties]          # 負樣本判定用(含全 notes)

    # 物件級隔離:重現 generate_dataset 的 80/10/10 切分,只取 train 物件。
    # 用同一 seed 確保切分與原訓練一致(否則會污染 dev/test 房源)。
    idx = list(range(len(properties)))
    random.seed(args.seed)
    random.shuffle(idx)
    n = len(idx)
    train_prop_idx = set(idx[: int(n * 0.8)])
    random.seed(args.seed)  # reset 供後續抽樣

    pos_samples, neg_samples = [], []
    per_key_stats = []

    for key, feats in keys.items():
        # 正樣本:條件加嚴 —— 特徵線索必須在「存進 CE 的 property_to_text」裡(橋接後)
        # 可見,CE 才學得到真訊號。只在 notes/furniture 而 property_to_text 看不到的
        # 配對(實測佔 72%)會教 CE 雜訊(query 配一個看不出該特徵的房源),故丟棄。
        # relevance 取 compute_relevance_score 與「線索可見→good match」較大值。
        pos_cands = []
        for i in train_prop_idx:
            if not prop_has_any_feature(prop_texts[i], feats):   # 注意:查 prop_texts(存進CE的)
                continue
            rel_fn = G.compute_relevance_score(key, properties[i])
            rel = max(rel_fn, MIN_POS_RELEVANCE)   # 線索可見且具備特徵 → 至少 good match
            pos_cands.append((i, rel))
        # rel 高優先,取前 N
        pos_cands.sort(key=lambda x: -x[1])
        chosen_pos = pos_cands[:MAX_POS_PER_KEY]
        for i, rel in chosen_pos:
            pos_samples.append({
                "query": key, "property": prop_texts[i],
                "label": 1, "relevance": rel, "is_hard": False,
                "source": "expansion_aug",
            })

        # 困難負樣本:train 房源中(橋接後仍)「不含任一指向特徵」者(語意不符)中隨機取
        neg_cands = [i for i in train_prop_idx
                     if not prop_has_any_feature(full_texts[i], feats)]
        random.shuffle(neg_cands)
        for i in neg_cands[:NEG_PER_KEY]:
            neg_samples.append({
                "query": key, "property": prop_texts[i],
                "label": 0, "relevance": 0, "is_hard": True,
                "source": "expansion_aug",
            })

        per_key_stats.append((key, len(chosen_pos), min(NEG_PER_KEY, len(neg_cands))))

    aug = pos_samples + neg_samples
    random.shuffle(aug)

    # ── 報告 ──
    print(f"=== 做法 B 資料增強:extension map 口語 key → 正樣本 query ===")
    print(f"擴展口語 key(排除距離類): {len(keys)}")
    print(f"train 物件數(物件級隔離): {len(train_prop_idx)}")
    print(f"\n新增樣本: {len(aug)}  (正 {len(pos_samples)} / 負 {len(neg_samples)})")

    # 覆蓋:有幾個 key 完全配不到正樣本(房源沒這特徵 → 無法增強)
    no_pos = [k for k, p, _ in per_key_stats if p == 0]
    print(f"配不到正樣本的 key: {len(no_pos)}  {no_pos[:8]}{'...' if len(no_pos) > 8 else ''}")

    print(f"\n=== 範例增強正樣本(口語 query 原樣,未拼擴展詞)===")
    for s in pos_samples[:6]:
        print(f"  「{s['query']}」 rel={s['relevance']} → {s['property'][:46]}")

    if args.write:
        # --enriched:append 進富化 train(房源文字富化版),否則進原始 baseline train。
        train_in = (ROOT / "data" / "processed" / "recommendation_train_enriched.json"
                    if args.enriched else TRAIN)
        out_path = (ROOT / "data" / "processed" / "recommendation_train_enriched_augmented.json"
                    if args.enriched else OUT)
        if not train_in.exists():
            print(f"\n⚠️ 找不到 {train_in.name} — 請先跑 --rebuild-enriched 生成富化 train")
            return 1
        base = json.loads(train_in.read_text(encoding="utf-8"))
        merged = base + aug
        random.shuffle(merged)
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ 已寫出 {out_path.name}: {len(base)} 原 + {len(aug)} 增強 = {len(merged)}")
    else:
        print(f"\n(預覽模式 — 加 --write 才寫出增強訓練檔)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
