"""P2 token audit — find semantic-expansion tokens with ZERO end-to-end backing.

Unlike the raw-blob recompute in UNVERIFIABLE_TOKENS_AUDIT.md, this replicates
the ACTUAL frontend matching path (inference.js):
  - buildPropText: text + furniture + features + building_type + room_type
    + notes + other_fees + injected BOOL_FIELD_FEATURES words (when bool==true)
  - propHasFeature: direct substring OR any PROP_SYNONYMS bridge

A token is a DELETE candidate only if propHasFeature hits 0/704 properties
(i.e. no raw text AND no bool-field AND no synonym bridge can ever surface it).
Tokens that survive via a bridge (廚房→可開伙, 可寵→可養, 管理員→保全,
台電→is_taipower, 補助→has_subsidy, 陽台→has_balcony ...) are KEPT.

These mirrors MUST stay in lockstep with inference.js; if that file changes the
synonyms / bool map, update here too.

Usage:
    python pipeline/data_prep/audit_expansion_tokens.py            # full report
    python pipeline/data_prep/audit_expansion_tokens.py --max 5    # show tokens with <5 hits
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROPS = ROOT / "frontend" / "assets" / "property_data.json"
RULES = ROOT / "data" / "semantic_rules.json"

# --- mirrors of inference.js (keep in sync) ---
PROP_SYNONYMS = {
    "可寵": ["可養貓", "可養狗", "可養寵物", "可養其他寵物"],
    "寵物友善": ["可養貓", "可養狗", "可養寵物"],
    "廚房": ["可開伙", "流理台"], "開火": ["可開伙", "瓦斯", "電磁爐"],
    "自炊": ["可開伙"], "可伙": ["可開伙"], "抽油煙機": ["排油煙"],
    "獨衛": ["獨立衛浴", "專用衛浴"], "獨立衛浴": ["獨衛"],
    "獨廁": ["獨立衛浴", "獨衛"], "變頻": ["冷氣"], "變頻冷氣": ["冷氣"],
    "吹冷氣": ["冷氣"], "全新": ["新裝潢", "新成屋"],
    "管理員": ["保全", "警衛"], "監視器": ["保全", "監視"], "門禁": ["保全", "刷卡"],
    "床架": ["床"], "床墊": ["床"], "書桌椅": ["桌子", "書桌", "椅子"],
    "天然瓦斯熱水器": ["熱水器", "瓦斯"], "電熱水器": ["熱水器"],
    "全配": ["家具", "家電"], "全家具": ["家具"], "全家電": ["家電"],
    "家具齊全": ["家具"], "子母車": ["垃圾"], "垃圾代收": ["垃圾"],
    "獨立洗衣機": ["洗衣機"], "獨洗": ["洗衣機"],
    # P2 rescue bridges
    "禁菸": ["無菸"], "採光": ["對外窗", "窗"], "通風": ["對外窗", "窗"],
    "安全": ["保全"], "刷卡": ["保全", "門禁"],
    "女性友善": ["限女", "女性"], "租補": ["租金補貼", "補貼"],
    "室友": ["雅房", "分租"], "合租": ["雅房", "分租"], "隔音": ["氣密窗", "氣密"],
    "台水": ["水費"], "帳單": ["台電", "台水", "電費"], "自繳": ["台電", "台水"],
    "標準電費": ["台電"],
}
BOOL_FIELD_FEATURES = {
    "has_elevator": "電梯", "has_window": "對外窗", "has_balcony": "陽台",
    "has_parking": "車位 停車場", "has_waste_disposal": "垃圾處理",
    "is_rooftop": "頂樓", "water_dispenser": "飲水機", "private_washer": "獨洗",
    "has_subsidy": "補助", "is_taipower": "台電",
}
# distance sentinels: matched via OSRM commute, not text — never delete-flag.
SENTINELS = {"走路10分", "騎車10分"}


def build_prop_text(prop: dict) -> str:
    parts = [prop.get("text") or ""]
    for f in ("furniture", "features", "building_type", "room_type"):
        v = prop.get(f)
        if v:
            parts.append(str(v).replace("/", " "))
    for f in ("notes", "other_fees"):
        v = prop.get(f)
        if isinstance(v, list):
            parts.append(" ".join(v))
    for bk, wd in BOOL_FIELD_FEATURES.items():
        if prop.get(bk) is True:
            parts.append(wd)
    eb = prop.get("electricity_billing")
    if eb and eb != "不明":
        parts.append(str(eb))
    return " ".join(parts)


def prop_has_feature(ptext: str, feature: str) -> bool:
    if feature in ptext:
        return True
    return any(s in ptext for s in PROP_SYNONYMS.get(feature, ()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10,
                    help="report tokens with hit count < max")
    args = ap.parse_args()

    props = json.loads(PROPS.read_text(encoding="utf-8"))
    rules = json.loads(RULES.read_text(encoding="utf-8"))["rules"]
    texts = [build_prop_text(p) for p in props]

    # token -> set of rule keys that emit it (for impact reporting)
    token_rules: dict[str, set[str]] = defaultdict(set)
    for key, toks in rules.items():
        for t in (toks if isinstance(toks, list) else str(toks).split()):
            if t:
                token_rules[t].add(key)

    rows = []
    for tok in sorted(token_rules):
        hits = sum(prop_has_feature(tx, tok) for tx in texts)
        bridged = bool(PROP_SYNONYMS.get(tok))
        rows.append((hits, tok, bridged, token_rules[tok]))

    n = len(props)
    dead = [r for r in rows if r[0] == 0 and r[1] not in SENTINELS]
    low = [r for r in rows if 0 < r[0] < args.max and r[1] not in SENTINELS]

    print(f"properties: {n} | unique tokens: {len(rows)}")
    print(f"\n===== DELETE CANDIDATES: 0 end-to-end backing ({len(dead)}) =====")
    print("(0 hits via text + bool-field + synonym bridge → pure model fiction)")
    for hits, tok, bridged, keys in dead:
        print(f"  {hits:4d}  {tok:8s}  (in {len(keys)} rule(s): {', '.join(sorted(keys))})")

    print(f"\n----- LOW backing 1..{args.max-1} (review, do NOT auto-delete) -----")
    for hits, tok, bridged, keys in sorted(low):
        b = " [BRIDGED]" if bridged else ""
        print(f"  {hits:4d}  {tok:8s}{b}  (in {len(keys)} rule(s))")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
