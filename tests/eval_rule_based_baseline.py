"""T0 baseline harness — OFFLINE measurement of the CURRENT rule-based recall.

Faithfully ports the JS "rule-based recall" stage from frontend/js/inference.js
to pure Python and measures its baseline on a held-out query set:

  - Recall@K (K=15 = current production .slice(0,15); K=30 = planned vector-recall K)
  - end-to-end-ranking NDCG@5 over graded relevance (mirrors ndcg_at_k in
    pipeline/data_prep/eval_ce_query_expansion.py)

This is the CP0 / T0 gate baseline for docs/spec/vector-retrieval-plan.md. The
metric functions and loaders are importable so the vector-recall A/B (T7,
tests/eval_vector_vs_rulebased.py) can reuse the exact same definitions.

NOTE on scope: this measures ONLY the rule-based RECALL stage (filterHardExclusions
-> calculateRuleBasedScore), NOT the downstream CE rerank. So "NDCG@5" here is the
NDCG of the rule-based recall ordering itself (graded relevance), which is the
fair like-for-like target the vector-recall stage will be A/B'd against at the
recall layer. The full end-to-end (recall + CE) NDCG is measured separately with
the CE ONNX harness; here we isolate the recall stage so vector-recall can be
swapped in 1:1.

=== The join problem (Path B) ===
recommendation_train.json's `property` field is an OLDER text blob of the listing
and does NOT exact-match property_data.json (only ~4/2000 exact). It is the SAME
listing with token drift (mainly the `距離X.XXkm` token). We fuzzy-join each
distinct blob to a property_data record by max token-set overlap and key the
ground-truth graded relevance by the joined property_data idx.

Pure stdlib (no numpy / onnxruntime / transformers) — rule-based recall is pure
logic, NDCG is implemented in pure Python.

Usage:
    python3 tests/eval_rule_based_baseline.py
    python3 tests/eval_rule_based_baseline.py --sample 200
"""
from __future__ import annotations

import argparse
import json
import math
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPERTIES = ROOT / "frontend" / "assets" / "property_data.json"
TRAIN = ROOT / "data" / "processed" / "recommendation_train.json"

# Recall stage K values. 15 = current production (.slice(0,15) in
# calculateRuleBasedScore). 30 = the planned vector-recall K (spec Resolved #3).
K_PROD = 15
K_VECTOR = 30
NDCG_K = 5

# Fuzzy-join acceptance thresholds (Path B). Tuned for high precision: the blobs
# differ from property_data.text essentially only by the 距離X.XXkm token, so a
# true match scores Jaccard >= 0.9 in practice; 0.6 / 0.8-overlap is a safe floor.
JOIN_MIN_JACCARD = 0.6
JOIN_MIN_OVERLAP_FRAC = 0.8  # overlap as fraction of the SHORTER token set

# Asset paths for the frontend first-load weight measurement (other T0 number).
RECALL_ASSETS = [
    ROOT / "frontend" / "models" / "custom_onnx_model_dir" / "my_custom_model_quant.onnx",
    ROOT / "frontend" / "models" / "ner_model_dir" / "ner_model_quant.onnx",
    ROOT / "frontend" / "assets" / "property_data.json",
]


# =====================================================================
# Ported JS constants (frontend/js/inference.js)
# =====================================================================

# COLLAPSED_BOOL_FIELDS (inference.js:59) — source-aware bool reliability.
COLLAPSED_BOOL_FIELDS = {
    "nchu": {"has_waste_disposal", "has_subsidy"},
    "dd": {"has_parking", "water_dispenser"},
}

# PROP_SYNONYMS (inference.js:743) — property-side synonym normalization.
PROP_SYNONYMS = {
    "可寵": ["可養貓", "可養狗", "可養寵物", "可養其他寵物"],
    "寵物友善": ["可養貓", "可養狗", "可養寵物"],
    "廚房": ["可開伙", "流理台"], "開火": ["可開伙", "瓦斯", "電磁爐"],
    "自炊": ["可開伙"], "可伙": ["可開伙"],
    "抽油煙機": ["排油煙"], "獨衛": ["獨立衛浴", "專用衛浴"],
    "獨立衛浴": ["獨衛"], "獨廁": ["獨立衛浴", "獨衛"],
    "變頻": ["冷氣"], "變頻冷氣": ["冷氣"], "吹冷氣": ["冷氣"],
    "全新": ["新裝潢", "新成屋"],
    "管理員": ["保全", "警衛"], "監視器": ["保全", "監視"], "門禁": ["保全", "刷卡"],
    "床架": ["床"], "床墊": ["床"], "書桌椅": ["桌子", "書桌", "椅子"],
    "天然瓦斯熱水器": ["熱水器", "瓦斯"], "電熱水器": ["熱水器"],
    "全配": ["家具", "家電"], "全家具": ["家具"], "全家電": ["家電"], "家具齊全": ["家具"],
    "子母車": ["垃圾"], "垃圾代收": ["垃圾"], "獨立洗衣機": ["洗衣機"], "獨洗": ["洗衣機"],
    "禁菸": ["無菸"],
    "採光": ["對外窗", "窗"], "通風": ["對外窗", "窗"],
    "安全": ["保全"], "刷卡": ["保全", "門禁"], "女性友善": ["限女", "女性"],
    "租補": ["租金補貼", "補貼"], "室友": ["雅房", "分租"], "合租": ["雅房", "分租"],
    "隔音": ["氣密窗", "氣密"],
    "台水": ["水費"], "帳單": ["台電", "台水", "電費"], "自繳": ["台電", "台水"],
    "標準電費": ["台電"],
}

# BOOL_FIELD_FEATURES (inference.js:763).
BOOL_FIELD_FEATURES = {
    "has_elevator": "電梯", "has_window": "對外窗", "has_balcony": "陽台",
    "has_parking": "車位 停車場", "has_waste_disposal": "垃圾處理", "is_rooftop": "頂樓",
    "water_dispenser": "飲水機", "private_washer": "獨洗", "has_subsidy": "補助",
    "is_taipower": "台電",
}

# expandQueryIntent intentMap (inference.js:884) — GENERATED semantic rules.
INTENT_MAP = {
    "想在家煮飯": "可伙 廚房 流理台 瓦斯爐 電磁爐 開火",
    "想自己煮飯": "可伙 廚房 流理台 瓦斯爐 開火",
    "在家開伙": "可伙 廚房 抽油煙機 流理台 瓦斯 開火 自炊 電磁爐 排油煙機",
    "想下廚": "可伙 廚房 抽油煙機 瓦斯爐",
    "要下廚": "可伙 廚房 抽油煙機 瓦斯爐",
    "喜歡下廚": "可伙 廚房 抽油煙機 瓦斯爐 流理台",
    "喜歡自己煮": "可伙 廚房 流理台 瓦斯爐",
    "自己煮": "廚房 瓦斯 開火 流理台 可伙 自炊 電磁爐 排油煙機",
    "自炊": "可伙 廚房 流理台 電磁爐 開火 瓦斯 自炊 排油煙機",
    "省伙食費": "廚房 瓦斯 開火 流理台",
    "省餐費": "可伙 廚房 流理台",
    "不想外食": "可伙 廚房 流理台 電磁爐",
    "不吃外食": "可伙 廚房 流理台 瓦斯爐",
    "可以煮東西": "可伙 廚房",
    "要能煮飯": "可伙 廚房 流理台 電磁爐",
    "煮飯": "可伙 廚房 流理台",
    "開火": "可伙 廚房 瓦斯爐 電磁爐",
    "要有廚房": "廚房 流理台 可伙",
    "有瓦斯": "天然瓦斯 瓦斯爐 可伙",
    "天然瓦斯": "天然瓦斯 瓦斯爐 可伙 廚房",
    "怕熱": "冷氣 變頻 吹冷氣 變頻冷氣",
    "夏天": "冷氣",
    "怕悶熱": "陽台 採光 通風 對外窗",
    "採光好": "採光 對外窗",
    "網美": "採光",
    "獨洗獨曬": "洗衣機 陽台 曬衣 獨洗",
    "有車": "車位 停車場",
    "開車": "車位 停車場",
    "可貓": "可寵 養寵 寵物友善 可養貓",
    "可狗": "可寵 養寵 寵物友善 可養狗",
    "有毛孩": "可寵 寵物",
    "台水電": "台電 台水 帳單 自繳",
    "省電費": "變頻 台電",
    "懶人": "電梯 子母車 垃圾處理 飲水機",
    "外送族": "管理員 飲水機 子母車",
    "不想出門": "管理員 飲水機 子母車",
    "不想追垃圾車": "子母車 垃圾處理 垃圾代收",
    "怕吵": "隔音 氣密窗 禁菸",
    "安靜": "隔音 氣密窗 禁菸",
    "晚歸": "門禁 管理員 安全 刷卡",
    "女生獨居": "管理員 門禁 監視器 女性友善 安全",
    "女生住": "管理員 門禁 監視器 安全",
    "獨居女": "管理員 門禁 監視器 女性友善",
    "女生安全": "管理員 門禁 監視器 安全",
    "怕危險": "管理員 門禁 監視器 安全",
    "治安": "管理員 門禁 監視器 安全",
    "拎包入住": "冰箱 洗衣機 床",
    "什麼都有": "冰箱 洗衣機",
    "家電齊全": "冰箱 洗衣機 冷氣",
    "要有冰箱": "冰箱",
    "要有書桌": "書桌 書桌椅",
    "要有床": "床架 床墊",
    "要有熱水": "熱水器 天然瓦斯熱水器 電熱水器",
    "找室友": "雅房 分租 室友 合租",
    "想合租": "雅房 分租 室友 合租",
    "不想一個人住": "雅房 分租 室友",
    "騎車上班": "機車停車位 停車",
    "不要西曬": "採光",
    "要有陽台": "陽台 曬衣 採光 通風",
    "在家工作": "網路 寬頻 書桌",
    "WFH": "網路 寬頻 書桌",
    "遠距工作": "網路 寬頻 書桌",
    "居家辦公": "網路 寬頻 書桌",
    "打報告": "寬頻 網路 書桌",
    "上網": "寬頻 網路",
    "念書": "書桌 書桌椅 寬頻",
    "讀書": "書桌 書桌椅 寬頻",
    "不想爬樓梯": "電梯 大樓 華廈",
    "搬東西": "電梯",
    "膝蓋不好": "電梯 大樓 華廈",
    "機車": "機車停車位",
    "高品質": "管理員 電梯",
    "不想去自助洗": "洗衣機 獨立洗衣機",
    "不想共用洗衣機": "洗衣機 獨立洗衣機",
    "養貓": "可養貓 寵物友善 可寵",
    "養狗": "可養狗 寵物友善 可寵",
    "台電": "台電 台水 標準電費",
    "獨立電表": "獨立電錶 台電",
    "不爬樓梯": "電梯 華廈 大樓",
    "不要爬樓梯": "電梯 華廈 大樓",
    "腿不好": "電梯 華廈 大樓",
    "在家煮": "廚房 瓦斯 開火 自炊 電磁爐 排油煙機 流理台",
    "想煮飯": "廚房 瓦斯 開火 自炊 電磁爐 排油煙機 流理台",
    "希望煮飯": "廚房 瓦斯 開火 自炊 電磁爐 排油煙機 流理台",
    "下班晚": "子母車 垃圾代收 門禁 管理員 安全",
    "省錢": "台電 台水 補助 租補",
    "生活便利": "興大路",
    "走路到學校": "走路10分",
    "走路去學校": "走路10分",
    "走路可以到": "走路10分",
    "走路就可以": "走路10分",
    "走路過去": "走路10分",
    "步行到學校": "走路10分",
    "步行去學校": "走路10分",
    "步行可以到": "走路10分",
    "騎車到學校": "騎車10分",
    "騎車去學校": "騎車10分",
    "騎車可以到": "騎車10分",
    "騎車就可以": "騎車10分",
    "騎車過去": "騎車10分",
    "騎機車到學校": "騎車10分",
    "騎機車去學校": "騎車10分",
}

NEGATORS = "不沒無非免勿"


# =====================================================================
# Ported JS functions (behaviour-faithful)
# =====================================================================

def prop_source(prop: dict) -> str:
    """propSource (inference.js:64)."""
    return "nchu" if "nchu" in (prop.get("url") or "") else "dd"


def bool_field_state(prop: dict, field: str) -> str:
    """boolFieldState (inference.js:72) — tri-state read of a bool field."""
    if prop.get(field) is True:
        return "yes"
    if field in COLLAPSED_BOOL_FIELDS.get(prop_source(prop), set()):
        return "unknown"
    return "no"


def expand_query_intent(query: str) -> str:
    """expandQueryIntent (inference.js:882). Negation guard included.

    JS approximation note: the JS bi-encoder fallback (encoderFallbackExpand) is
    behind ENCODER_FALLBACK_ENABLED = false (inference.js:38) and is a no-op in
    production, so we faithfully omit it (it returns '' in the shipped frontend).
    """
    expanded = query
    for intent, expansion in INTENT_MAP.items():
        frm = 0
        while (idx := query.find(intent, frm)) != -1:
            # ''.includes is always true in JS; idx===0 is explicitly NOT negated.
            negated = idx > 0 and query[idx - 1] in NEGATORS
            if not negated:
                expanded += " " + expansion
                break
            frm = idx + 1
    return expanded


# extractKeywords (inference.js:1019).
_STOP_WORDS = ["近", "靠近", "想找", "尋找", "住在", "一間", "想要", "預算", "大約",
               "希望", "位於", "位在", "位處", "在", "含", "有", "附", "座落於", "座落"]
_LOC_SUFFIXES = ["路", "街", "大道", "區"]
_SPLIT_RE = re.compile(r"\s+|[,，、。]")


def extract_keywords(text: str) -> list[str]:
    """extractKeywords (inference.js:1019)."""
    expanded_text = expand_query_intent(text)
    out: list[str] = []
    for k in _SPLIT_RE.split(expanded_text):
        # .filter(k => k.length > 1 && !k.match(/^\d+$/))
        if len(k) <= 1 or re.fullmatch(r"\d+", k):
            continue
        clean = k
        for sw in _STOP_WORDS:
            if clean.startswith(sw):
                clean = clean[len(sw):]
        for suffix in _LOC_SUFFIXES:
            if clean.endswith(suffix) and len(clean) > len(suffix):
                for p in ["位", "於", "在", "處"]:
                    if clean.startswith(p):
                        clean = clean[len(p):]
        if len(clean) > 1:
            out.append(clean)
    return out


def parse_constraints_from_text(text: str) -> dict:
    """parseConstraintsFromText (inference.js:204)."""
    budget = None
    limit = None
    min_budget = None
    max_budget = None
    gender_unrestricted = False
    has_gender_mention = False
    has_budget_mention = False
    has_room_type_mention = False
    wants_utility_billing = False
    max_electricity_price = None
    require_balcony = False
    require_window = False
    require_parking = False
    require_waste = False
    require_subsidy = False
    is_social_housing = False
    exclude_rooftop = False
    exclude_wooden = False
    exclude_haunted = False

    if any(s in text for s in ("不限女", "不限性別", "男生", "男士")):
        gender_unrestricted = True
        has_gender_mention = True
    elif ("限女" in text) or ("限男" in text):
        has_gender_mention = True

    negative_words = r"(謝絕|不要|拒絕|禁|❌|不接受|不想|討厭|避免|不要有|不要找)"
    if re.search(negative_words + r"[^。！？\n]*(頂加|加蓋|頂樓)", text):
        exclude_rooftop = True
    if re.search(negative_words + r"[^。！？\n]*木板", text):
        exclude_wooden = True
    if re.search(negative_words + r"[^。！？\n]*凶宅", text):
        exclude_haunted = True

    if re.search(r"(要有|必須|希望|想找)[^。！？\n]*陽台", text):
        require_balcony = True
    elif "陽台" in text:
        require_balcony = True

    if re.search(r"(要有|必須|希望|想找)[^。！？\n]*窗", text):
        require_window = True
    elif "窗" in text:
        require_window = True

    if ("車位" in text) or ("停車" in text):
        require_parking = True
    if ("子母車" in text) or ("垃圾" in text):
        require_waste = True

    if any(s in text for s in ("補助", "補貼", "報稅", "入籍")):
        require_subsidy = True
    if ("社宅" in text) or ("社會住宅" in text):
        is_social_housing = True

    if "以上" in text:
        limit = "above"
    elif any(s in text for s in ("以下", "以內", "內")):
        limit = "below"

    if any(s in text for s in ("台水", "台電", "獨立電錶", "獨立電表")):
        wants_utility_billing = True
    elec_match = re.search(r"度\s*(\d+(?:\.\d+)?)\s*[元塊]", text)
    if elec_match:
        max_electricity_price = float(elec_match.group(1))

    cn = {"一": "1", "二": "2", "兩": "2", "三": "3", "四": "4", "五": "5",
          "六": "6", "七": "7", "八": "8", "九": "9", "十": "10", "半": "30"}
    rt = text
    for c, d in cn.items():
        rt = rt.replace(c, d)

    max_walk_mins = None
    walk_match = re.search(r"(?:走路|步行)[^\d]*(\d+)[^\d]*(?:分鐘|分)", rt)
    if walk_match:
        max_walk_mins = int(walk_match.group(1))

    max_scooter_mins = None
    scooter_match = re.search(r"(?:機車|騎車)[^\d]*(\d+)[^\d]*(?:分鐘|分)", rt)
    if scooter_match:
        max_scooter_mins = int(scooter_match.group(1))

    # Range budget (萬/千 normalization). Mirrors JS replace callbacks.
    def _wan_repl(m: re.Match) -> str:
        val = float(m.group(1)) * 10000
        if m.group(2):
            val += int(m.group(2)) * 1000
        return str(int(val) if val == int(val) else val)

    rt_range = re.sub(r"(\d+(?:\.\d+)?)萬(\d*)", _wan_repl, rt)
    rt_range = re.sub(r"(\d+)千", lambda m: str(int(m.group(1)) * 1000), rt_range)

    range_match = re.search(r"(\d{3,})\s*[-~～至到]\s*(\d{3,})", rt_range)
    if range_match:
        min_budget = int(range_match.group(1))
        max_budget = int(range_match.group(2))
        has_budget_mention = True

    if not has_budget_mention:
        if "萬" in rt:
            m = re.search(r"(\d+(?:\.\d+)?)萬(\d*)", rt)
            if m:
                budget = float(m.group(1)) * 10000 + (int(m.group(2)) * 1000 if m.group(2) else 0)
                has_budget_mention = True
        if not budget:
            rt = rt.replace("千", "000").replace("k", "000").replace("K", "000")
            m2 = re.search(r"(\d{4,})", rt)
            if m2:
                budget = int(m2.group(1))
                has_budget_mention = True
        if not budget:
            m3 = re.search(r"(\d+)", rt)
            if m3:
                val = int(m3.group(1))
                if val >= 1000:
                    budget = val
                    has_budget_mention = True
                elif val < 100:
                    has_budget_cue = re.search(r"預算|月租|租金|房租|元|塊|[kK]|千|萬|以下|以內|以上", text)
                    if has_budget_cue:
                        budget = val * 1000
                        has_budget_mention = True

    wants_room_type = None
    if "套房" in text:
        has_room_type_mention = True
        wants_room_type = "套房"
    elif "雅房" in text:
        has_room_type_mention = True
        wants_room_type = "雅房"
    elif "工作室" in text:
        has_room_type_mention = True
        wants_room_type = "工作室"

    pet_mention = any(s in text for s in ("養貓", "養狗", "寵物", "毛小孩", "毛孩"))
    pet_negated = bool(re.search(
        r"(不要|不想|不可|沒有|別|禁|討厭|避免|謝絕|拒絕|怕)[^。！？\n]{0,4}(養貓|養狗|養寵|寵物|毛小孩|毛孩|貓|狗)",
        text))
    exclude_pet = pet_mention and pet_negated
    wants_pet = pet_mention and not pet_negated

    return {
        "budget": budget, "minBudget": min_budget, "maxBudget": max_budget, "limit": limit,
        "genderUnrestricted": gender_unrestricted, "hasGenderMention": has_gender_mention,
        "hasBudgetMention": has_budget_mention, "hasRoomTypeMention": has_room_type_mention,
        "wantsRoomType": wants_room_type, "wantsUtilityBilling": wants_utility_billing,
        "maxElectricityPrice": max_electricity_price, "requireBalcony": require_balcony,
        "requireWindow": require_window, "requireParking": require_parking,
        "requireWaste": require_waste, "requireSubsidy": require_subsidy,
        "isSocialHousing": is_social_housing, "excludePet": exclude_pet,
        "excludeRooftop": exclude_rooftop, "excludeWooden": exclude_wooden,
        "excludeHaunted": exclude_haunted, "maxWalkMins": max_walk_mins,
        "maxScooterMins": max_scooter_mins, "wantsPet": wants_pet,
        "requireElevator": any(s in text for s in (
            "電梯", "升降梯", "不爬樓", "不用爬", "不想爬", "不要爬", "腿不好", "膝蓋不好")),
        "requireCooking": any(s in text for s in (
            "開伙", "開火", "自炊", "煮飯", "炒菜", "在家煮", "自己煮")),
        "requireWaterDispenser": "飲水機" in text,
        "requirePrivateWasher": ("獨洗" in text) or ("個人洗衣機" in text),
        "requireGuard": any(s in text for s in ("代收", "包裹", "管理員", "警衛")),
        "originalText": text,
    }


def build_prop_text(prop: dict) -> str:
    """buildPropText (inference.js:770)."""
    parts = [prop.get("text") or ""]
    for f in ("furniture", "features", "building_type", "room_type"):
        if prop.get(f):
            parts.append(str(prop[f]).replace("/", " "))
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


def prop_has_feature(prop_text: str, feature: str) -> bool:
    """propHasFeature (inference.js:796) — direct or synonym hit."""
    if feature in prop_text:
        return True
    syns = PROP_SYNONYMS.get(feature)
    if syns:
        return any(s in prop_text for s in syns)
    return False


def filter_hard_exclusions(properties: list[dict], c: dict) -> list[dict]:
    """filterHardExclusions (inference.js:618)."""
    candidates = []
    for prop in properties:
        text = prop.get("text") or ""
        if c["excludeRooftop"] and (prop.get("is_rooftop") or "頂加" in text):
            continue
        if c["excludeWooden"] and prop.get("is_wooden_partition"):
            continue
        if c["requireSubsidy"] and any(s in text for s in ("不可補助", "不可報稅", "不可入籍")):
            continue
        if c["isSocialHousing"] and ("社會住宅" not in text) and ("社宅" not in text):
            continue

        if c["wantsPet"] and any(s in text for s in ("禁養", "不可養", "不可寵", "謝絕寵物")):
            continue
        if c["excludePet"]:
            pet_text = build_prop_text(prop)
            if ("可養" in pet_text) or ("寵物友善" in pet_text):
                continue
        if c["requireElevator"] and any(s in text for s in ("無電梯", "沒有電梯", "沒電梯")):
            continue
        if c["requireCooking"] and any(s in text for s in ("禁開伙", "不可開伙", "不可開火", "禁炊")):
            continue

        # Commute time filtering. JS: parseFloat(prop.distance) then NaN check.
        try:
            dist = float(prop.get("distance"))
        except (TypeError, ValueError):
            dist = float("nan")
        if not math.isnan(dist) and dist > 0:
            if c["maxWalkMins"] is not None:
                walk_mins = round(dist / 0.075)
                if walk_mins > c["maxWalkMins"] + 3:
                    continue
            if c["maxScooterMins"] is not None:
                scooter_mins = max(1, round(dist / 0.417))
                if scooter_mins > c["maxScooterMins"] + 2:
                    continue

        if c["maxElectricityPrice"]:
            billing = prop.get("electricity_billing") or ""
            m = re.search(r"(\d+(?:\.\d+)?)", billing)
            if m and float(m.group(1)) > c["maxElectricityPrice"]:
                continue

        if c["wantsUtilityBilling"]:
            billing = prop.get("electricity_billing") or ""
            if "度" in billing:
                m = re.search(r"(\d+(?:\.\d+)?)", billing)
                if m and float(m.group(1)) >= 5:
                    continue

        if c["hasGenderMention"] and c["genderUnrestricted"]:
            furniture = prop.get("furniture") or ""
            is_female_only = ("限女" in text) or ("限女" in furniture)
            if is_female_only:
                continue
        if c["hasBudgetMention"]:
            rent = prop.get("rent")
            if c["maxBudget"] is not None and rent is not None and rent > c["maxBudget"]:
                continue
            if c["budget"] is not None and rent is not None:
                effective_limit = c["limit"] or "below"
                if effective_limit == "below" and rent > c["budget"]:
                    continue
                if effective_limit == "above" and rent < c["budget"]:
                    continue
        candidates.append(prop)
    return candidates


def calculate_rule_based_score(candidates: list[dict], query_keywords: list[str],
                               text: str, c: dict, k: int = K_PROD) -> list[dict]:
    """calculateRuleBasedScore (inference.js:1041).

    K is parameterized (JS hard-codes .slice(0,15) at inference.js:1220). The
    sort key mirrors the JS exactly: (kScore + rms*20) descending.
    """
    has_location_mention = any(
        kw.endswith("路") or kw.endswith("街") or kw.endswith("大道")
        or ("區" in kw) or ("正門" in kw) or ("側門" in kw) or ("男宿" in kw)
        for kw in query_keywords
    )

    ignore_list = ["房", "推薦", "附近", "一下", "預算", "大概", "想要", "需求", "尋找"]
    semantic_map = {
        "垃圾": ["子母車", "代收垃圾", "垃圾處理", "垃圾子車"],
        "電費": ["台電", "獨立電錶", "台水台電"],
        "陽台": ["陽台", "露台"],
        "電梯": ["電梯", "華廈", "大樓"],
        "車位": ["停車", "車位", "車庫"],
    }

    pre_scored = []
    for prop in candidates:
        k_score = 0.0
        match_count = 0.0
        total_requirements = 0
        p_text = build_prop_text(prop).lower()

        for kw in query_keywords:
            if len(kw) < 2 or kw in ignore_list:
                continue
            total_requirements += 1
            is_match = prop_has_feature(p_text, kw)

            if ("樓梯" in kw) or ("電梯" in kw):
                elevator_kws = ["電梯", "華廈", "大樓"]
                is_match = (bool_field_state(prop, "has_elevator") == "yes"
                            or any(alt in p_text for alt in elevator_kws))
            elif ("垃圾" in kw) or ("追車" in kw):
                waste_kws = ["子母車", "代收垃圾", "垃圾處理", "垃圾子車"]
                is_match = (bool_field_state(prop, "has_waste_disposal") == "yes"
                            or any(alt in p_text for alt in waste_kws))
            elif "陽台" in kw:
                is_match = (bool_field_state(prop, "has_balcony") == "yes"
                            or "陽台" in p_text)
            elif "窗" in kw:
                is_match = (bool_field_state(prop, "has_window") == "yes"
                            or "窗" in p_text)
            elif ("車位" in kw) or ("停車" in kw):
                is_match = (bool_field_state(prop, "has_parking") == "yes"
                            or "車位" in p_text or "停車" in p_text)
            elif ("電" in kw) or ("錢" in kw) or ("省" in kw):
                if ("電費" in kw) or ("台電" in kw) or ("省" in kw):
                    power_kws = ["台電", "獨立電錶", "台水台電"]
                    eb = prop.get("electricity_billing")
                    notes = prop.get("notes") or []
                    is_match = (
                        (eb is not None and "台電" in eb)
                        or any("台電" in n for n in notes)
                        or any(alt in p_text for alt in power_kws)
                    )

            if not is_match:
                for group, alternates in semantic_map.items():
                    if (group in kw) or (kw in group):
                        if any(alt in p_text for alt in alternates):
                            is_match = True
                            break

            if is_match:
                match_count += 1
                k_score += 15

        is_commute_explicit = any(s in text for s in ("近", "走", "分鐘", "公里"))
        walk_mins = prop.get("walk_mins") or 0
        distance = prop.get("distance") or 0
        has_commute_signal = (walk_mins > 0) or (distance > 0)
        if c["maxWalkMins"] is not None and is_commute_explicit and has_commute_signal:
            total_requirements += 1
            prop_walk = prop.get("walk_mins") or math.ceil(distance / 0.08)
            if prop_walk <= c["maxWalkMins"]:
                match_count += 1
                k_score += 20

        if c["maxScooterMins"] is not None and is_commute_explicit and has_commute_signal:
            total_requirements += 1
            prop_scooter = prop.get("scooter_mins") or max(1, math.ceil(distance / 0.5))
            if prop_scooter <= c["maxScooterMins"]:
                match_count += 1
                k_score += 15

        if has_location_mention:
            total_requirements += 1
            loc_match = False
            p_raw = prop.get("text") or ""
            for kw in query_keywords:
                if kw in p_raw:
                    if kw.endswith("路") or kw.endswith("街") or kw.endswith("大道"):
                        k_score += 15
                        loc_match = True
                    if "區" in kw:
                        k_score += 5
                        loc_match = True
                    if ("正門" in kw) or ("側門" in kw):
                        k_score += 10
                        loc_match = True
            if loc_match:
                match_count += 1

        if c["hasRoomTypeMention"]:
            total_requirements += 1
            rt_match = False
            p_raw = prop.get("text") or ""
            for rt in ("套房", "雅房", "工作室"):
                if (rt in text) and (rt in p_raw):
                    rt_match = True
            if rt_match:
                match_count += 1
                k_score += 10

        if c["hasBudgetMention"]:
            total_requirements += 2
            rent = prop.get("rent") or 0
            if c["minBudget"] is not None and c["maxBudget"] is not None:
                if c["minBudget"] <= rent <= c["maxBudget"]:
                    match_count += 2
                    k_score += 10
                elif rent < c["minBudget"]:
                    match_count += 1.5
                    k_score += 5
                else:
                    diff = rent - c["maxBudget"]
                    if diff <= 1000:
                        match_count += 0.5
                        k_score += 1
            elif c["budget"] is not None:
                diff = rent - c["budget"]
                if abs(diff) <= 500:
                    match_count += 2
                    k_score += 10
                elif rent < c["budget"]:
                    match_count += 1.5
                    k_score += 3
                elif diff <= 1500:
                    match_count += 0.5
                    k_score += 1

        if c["wantsUtilityBilling"]:
            total_requirements += 1
            eb = prop.get("electricity_billing")
            utility_match = bool(eb) and (
                "台電" in eb or "台水" in eb or eb == "含電費" or eb == "獨立電錶")
            if utility_match:
                match_count += 1
                k_score += 10

        rms = (match_count / total_requirements) if total_requirements > 0 else 1.0
        pre_scored.append({"prop": prop, "kScore": k_score, "rms": rms})

    # Stable sort by (kScore + rms*20) descending — mirrors inference.js:1219.
    pre_scored.sort(key=lambda x: x["kScore"] + x["rms"] * 20, reverse=True)
    return pre_scored[:k]


# =====================================================================
# Metrics (importable — reused by T7 vector A/B)
# =====================================================================

def dcg(rels: list[float]) -> float:
    """DCG with log2(i+2) discount — mirrors eval_ce_query_expansion.py:94."""
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels: list[float], k: int = NDCG_K) -> float:
    """NDCG@k — mirrors eval_ce_query_expansion.py:98."""
    ideal = sorted(ranked_rels, reverse=True)
    idcg = dcg(ideal[:k])
    if idcg == 0:
        return 0.0
    return dcg(ranked_rels[:k]) / idcg


def recall_at_k(ranked_idxs: list[int], relevant_idxs: set[int], k: int) -> float:
    """Recall@k = (# relevant in top-k) / (total relevant)."""
    if not relevant_idxs:
        return 0.0
    top = set(ranked_idxs[:k])
    return len(top & relevant_idxs) / len(relevant_idxs)


# =====================================================================
# Loaders + fuzzy join (importable)
# =====================================================================

def load_properties(path: Path = PROPERTIES) -> list[dict]:
    """Load property_data.json. Mirrors initData filter (inference.js:104):
    drop crawler shells with no address / rent<=0."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [p for p in raw if p and p.get("address") and (p.get("rent") or 0) > 0]


def load_labels(path: Path = TRAIN) -> list[dict]:
    """Load recommendation_train.json graded-relevance pairs."""
    return json.loads(path.read_text(encoding="utf-8"))


def _tokens(s: str) -> frozenset[str]:
    return frozenset(s.split())


def build_fuzzy_join(properties: list[dict], blobs: list[str],
                     min_jaccard: float = JOIN_MIN_JACCARD,
                     min_overlap_frac: float = JOIN_MIN_OVERLAP_FRAC,
                     cache_path: Path | None = None) -> dict[str, int]:
    """Fuzzy-join each distinct train.property blob to a property_data idx by max
    token-set overlap. Accept above threshold (Jaccard>=min_jaccard OR overlap
    fraction of shorter set >=min_overlap_frac). Returns {blob: idx}.

    Cached to disk (cache_path) keyed by inputs; pure stdlib JSON.
    """
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (cached.get("n_props") == len(properties)
                and cached.get("n_blobs") == len(blobs)):
            return cached["map"]

    prop_tokens = [(p["idx"], _tokens(p.get("text") or "")) for p in properties]
    join: dict[str, int] = {}
    for blob in blobs:
        bt = _tokens(blob)
        if not bt:
            continue
        best_idx = None
        best_score = 0.0
        for idx, pt in prop_tokens:
            if not pt:
                continue
            inter = len(bt & pt)
            if inter == 0:
                continue
            union = len(bt | pt)
            jacc = inter / union
            overlap_frac = inter / min(len(bt), len(pt))
            # Rank by jaccard; accept by either threshold.
            if jacc > best_score and (jacc >= min_jaccard or overlap_frac >= min_overlap_frac):
                best_score = jacc
                best_idx = idx
        if best_idx is not None:
            join[blob] = best_idx

    if cache_path:
        cache_path.write_text(json.dumps(
            {"n_props": len(properties), "n_blobs": len(blobs), "map": join},
            ensure_ascii=False), encoding="utf-8")
    return join


def build_ground_truth(labels: list[dict], blob_to_idx: dict[str, int]
                       ) -> dict[str, dict[int, float]]:
    """Build per-query graded ground truth keyed by joined property_data idx.

    relevance: 3/2/1 positive, 0/-1 non-relevant. Mirrors eval_ce_query_expansion
    convention: rel = max(0, relevance). When multiple blobs map to the same idx
    for a query, keep the max relevance.
    """
    gt: dict[str, dict[int, float]] = {}
    for s in labels:
        blob = s["property"]
        idx = blob_to_idx.get(blob)
        if idx is None:
            continue
        rel = max(0, s.get("relevance", s.get("label", 0)))
        q = s["query"]
        bucket = gt.setdefault(q, {})
        bucket[idx] = max(bucket.get(idx, 0.0), float(rel))
    return gt


# =====================================================================
# Recall pipeline driver
# =====================================================================

def rule_based_recall(properties: list[dict], query: str, k: int) -> list[dict]:
    """Full ported recall: parse -> hard filter -> score -> top-K.

    Mirrors recommend()'s recall stage (inference.js:1251 onward, recall portion
    only). Excludes NER augmentation (browser-only ONNX worker) — see caveats.
    """
    constraints = parse_constraints_from_text(query)
    candidates = filter_hard_exclusions(properties, constraints)
    keywords = extract_keywords(query)
    return calculate_rule_based_score(candidates, keywords, query, constraints, k=k)


def measure_asset_sizes() -> tuple[list[tuple[str, float]], float]:
    """Return [(name, MB), ...] and total MB for the recall-relevant assets."""
    sizes = []
    total = 0.0
    for p in RECALL_ASSETS:
        mb = p.stat().st_size / (1024 * 1024) if p.exists() else 0.0
        sizes.append((p.name, mb))
        total += mb
    return sizes, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="cap number of eval queries for speed (0=all)")
    args = ap.parse_args()

    properties = load_properties()
    labels = load_labels()

    distinct_blobs = sorted({s["property"] for s in labels})
    cache_path = ROOT / "tests" / ".rule_based_join_cache.json"
    blob_to_idx = build_fuzzy_join(properties, distinct_blobs, cache_path=cache_path)
    join_rate = len(blob_to_idx) / len(distinct_blobs) if distinct_blobs else 0.0

    gt = build_ground_truth(labels, blob_to_idx)

    # Eligible eval queries: >=2 candidates AND >=1 positive (relevance>=1) after join.
    evalable = {q: rels for q, rels in gt.items()
                if len(rels) >= 2 and any(r >= 1 for r in rels.values())}
    queries = list(evalable.keys())
    if args.sample:
        queries = queries[:args.sample]

    idx_to_prop = {p["idx"]: p for p in properties}

    recall15_list, recall30_list, ndcg5_list = [], [], []
    for qi, q in enumerate(queries):
        relevant_idxs = {idx for idx, r in evalable[q].items() if r >= 1}

        # Recall stage at K=30 (superset of K=15 ranking) — run once at max K.
        ranked = rule_based_recall(properties, q, k=K_VECTOR)
        ranked_idxs = [item["prop"]["idx"] for item in ranked]

        recall15_list.append(recall_at_k(ranked_idxs, relevant_idxs, K_PROD))
        recall30_list.append(recall_at_k(ranked_idxs, relevant_idxs, K_VECTOR))

        # NDCG@5 over the recall ranking using graded relevance (0 if not labeled
        # for this query — unlabeled-in-topK contributes 0, standard for sparse GT).
        ranked_rels = [evalable[q].get(idx, 0.0) for idx in ranked_idxs]
        ndcg5_list.append(ndcg_at_k(ranked_rels, NDCG_K))

        if (qi + 1) % 500 == 0:
            print(f"  ...{qi+1}/{len(queries)} queries scored")

    n = len(queries)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0

    asset_sizes, total_mb = measure_asset_sizes()

    print("\n" + "=" * 64)
    print(" T0 BASELINE — rule-based recall (ported from inference.js)")
    print("=" * 64)
    print(f" Properties (after shell filter) : {len(properties)}")
    print(f" Distinct train.property blobs   : {len(distinct_blobs)}")
    print(f" Fuzzy-join match-rate           : {len(blob_to_idx)}/{len(distinct_blobs)}"
          f"  ({join_rate*100:.1f}%)")
    print(f" Eval queries (>=2 cand, >=1 pos): {n}")
    print("-" * 64)
    print(f" Recall@{K_PROD:<2} (production .slice)    : {mean(recall15_list):.4f}")
    print(f" Recall@{K_VECTOR:<2} (planned vector K)    : {mean(recall30_list):.4f}")
    print(f" NDCG@{NDCG_K} (recall-stage ranking)   : {mean(ndcg5_list):.4f}")
    print("-" * 64)
    print(" Frontend first-load weight (recall-relevant assets):")
    for name, mb in asset_sizes:
        print(f"   {name:<32} {mb:7.2f} MB")
    print(f"   {'TOTAL':<32} {total_mb:7.2f} MB")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
