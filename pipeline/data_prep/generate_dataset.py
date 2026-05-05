"""
generate_dataset.py
Synthesizes training, validation, and test datasets from rental CSV data.
Generates simulated natural language queries and constructs positive/negative 
query-property pairs for Sentence-Pair classification model training.
"""
import csv
import json
import random
import re
from typing import Dict, List, Any

random.seed(42)

# ============================================================
# 1. Configuration & Templates
# ============================================================
class Templates:
    ROOM = {
        "套房": ["套房", "找套房", "想租套房", "要一間套房", "單人套房"],
        "雅房": ["雅房", "找雅房", "想租雅房", "要一間雅房"],
        "住宅": ["住宅", "整層住宅", "整層", "家庭式", "整層出租"],
        "套/雅房": ["套房", "雅房", "套房或雅房"],
    }
    BUILDING = {
        "透天厝": ["透天", "透天厝"],
        "公寓": ["公寓", "公寓大樓"],
        "大樓": ["大樓", "電梯大樓", "有電梯的"],
        "華廈": ["華廈", "華廈大樓"],
    }
    REGION = {
        "南區": ["南區", "台中南區", "南區附近"],
        "大里區": ["大里", "大里區", "台中大里"],
        "東區": ["東區", "台中東區"],
    }
    FURNITURE = {
        "冷氣機": ["有冷氣", "要冷氣", "含冷氣", "怕熱"],
        "冷氣": ["有冷氣", "要冷氣", "怕熱"],
        "洗衣機": ["有洗衣機", "獨洗", "獨立洗衣"],
        "冰箱": ["有冰箱", "需要冰東西"],
        "電視": ["有電視"],
        "有線電視": ["有第四台", "有電視"],
        "書桌椅": ["有書桌", "附書桌"],
        "書桌": ["有書桌", "附書桌"],
        "床": ["有床", "附床"],
        "衣櫃": ["有衣櫃", "衣服很多"],
        "熱水器": ["有熱水器"],
        "（電）熱水器": ["有熱水器"],
        "電梯": ["有電梯", "不想爬樓梯", "不想走樓梯", "搬東西方便", "懶得爬樓梯"],
        "陽台": ["有陽台", "要陽台", "想曬衣服", "衣服容易乾", "通風好", "要有對外窗"],
        "曬衣場": ["獨曬", "有曬衣", "可曬衣"],
        "機車停車位": ["有車位", "機車位", "有停車位", "好停車"],
        "飲水機": ["有飲水機", "不用買水"],
        "寬頻網路": ["有網路", "上網方便"],
    }
    INCLUDED = {
        "水費": ["含水費", "包水費", "水費包含"],
        "電費": ["含電費", "包電費", "台水台電", "照台水台電收費", "不要電費太貴", "一度5塊太貴了", "電費依台水台電"],
        "網路費": ["含網路", "包網路"],
        "管理費": ["含管理費"],
        "瓦斯": ["含瓦斯"],
    }
    PREFIXES = ["想找", "想租", "找", "要", "求", "想要", "幫我找", "需要", "我要找", "有沒有", "請問有", "想住", "要租", "有推薦"]
    CONNECTORS = [" ", "的", " 要", " 有", " 而且"]

# ============================================================
# 2. Data Loading & Normalization
# ============================================================
import os

def load_properties(csv_path: str = None) -> List[Dict[str, Any]]:
    """Reads CSV properties into structured dict format."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "../../data/raw/nchu_rental_info.csv")
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    properties = []
    for row in rows:
        rent_str = row.get("租金", "")
        rent_match = re.search(r"(\d[\d,]*)", rent_str.replace(",", ""))
        rent_num = int(rent_match.group(1)) if rent_match else 0
        dist = float(row.get("距離(km)", "0") or "0")

        furniture = [s.strip() for s in row.get("家具設施", "").split("/") if s.strip()]
        included = [s.strip() for s in row.get("租金包含", "").split("/") if s.strip()]
        security = [s.strip() for s in row.get("安全管理", "").split("/") if s.strip()]
        notes = [s.strip() for s in row.get("備註", "").split("/") if s.strip()]

        addr = row.get("地址", "")
        road = ""
        road_match = re.search(r"([^區市台]*(?:路|街|大道)(?:[一二三四五六七八九十]|[\d])?段?)", addr)
        if road_match:
            road = road_match.group(1).strip()

        region = next((r for r in ["南區", "大里區", "西區", "東區", "北區", "烏日"] if r in addr), "")

        properties.append({
            "address": addr, "region": region, "road": road,
            "room_type": row.get("格局", ""), "building_type": row.get("類型", ""),
            "size": row.get("室內坪數", ""), "rent": rent_num, "rent_str": rent_str,
            "furniture": furniture, "included": included, "security": security,
            "notes": notes, "distance": dist, "url": row.get("網址", ""),
            "img": row.get("圖片網址", ""), "floor": row.get("樓層", ""),
        })

    return properties


def property_to_text(prop: Dict[str, Any]) -> str:
    """Consolidates property keys into a canonical descriptive string."""
    parts = [p for p in (prop["room_type"], prop["building_type"], prop["region"], prop["road"]) if p]
    
    if prop["rent"]: parts.append(f"{prop['rent']}元")
    if prop["distance"]: parts.append(f"距離{prop['distance']}km")

    key_furniture = []
    for f in prop["furniture"]:
        short = f.replace("（電）", "").replace("機車停車位", "機車位").replace("書桌椅", "書桌")
        if short not in key_furniture:
            key_furniture.append(short)
        if len(key_furniture) >= 5: break
            
    if key_furniture: parts.append(" ".join(key_furniture))
    if prop["included"]: parts.append("含" + "".join(prop["included"][:3]))

    parts.extend([note for note in prop["notes"] if "寵物" in note or "限" in note])
    return " ".join(parts)


# ============================================================
# 3. Query Generation Utilities
# ============================================================
class QueryGenerator:
    """Handles logic for synthesizing natural language queries."""
    
    @staticmethod
    def expr_budget(rent: int) -> List[str]:
        exprs = [f"預算{rent}", f"月租{rent}", f"{rent}元"]
        for above in [500, 1000, 2000]:
            ceiling = rent + above
            exprs.extend([f"預算{ceiling}以下", f"{ceiling}以內", f"月租{ceiling}內"])
            
        if rent >= 1000:
            k = rent // 1000
            exprs.extend([f"{k}K", f"{k}k以下", f"預算{k}千"])
            
        cn_map = {1:"一", 2:"二", 3:"三", 4:"四", 5:"五", 6:"六", 7:"七", 8:"八", 9:"九"}
        if 1000 <= rent < 10000 and (k := rent // 1000) in cn_map:
            exprs.extend([f"{cn_map[k]}千", f"預算{cn_map[k]}千以下", f"{cn_map[k]}千以內"])
            
        if rent >= 10000:
            w, r = rent // 10000, (rent % 10000) // 1000
            if w in cn_map:
                exprs.append(f"{cn_map[w]}萬" if r == 0 else f"{cn_map[w]}萬{cn_map[r]}千")
                
        for below in [500, 1000, 2000]:
            floor = max(rent - below, 1000)
            exprs.extend([f"預算{floor}以上", f"{floor}元以上"])
            
        return exprs

    @staticmethod
    def expr_distance(dist_km: float) -> List[str]:
        exprs = []
        if dist_km <= 1.0: exprs += ["學校附近", "中興大學旁", "近中興", "學校旁邊", "中興大學門口", "正門附近", "走路就到"]
        if dist_km <= 0.5: exprs += ["非常近", "校門口", "走路五分鐘"]
        
        walk, ride = round(dist_km / 0.08), round(dist_km / 0.6)
        if walk <= 15: exprs.extend([f"走路{walk}分鐘", f"步行{walk}分鐘內"])
        if ride <= 10: exprs.extend([f"騎車{ride}分鐘", f"機車{ride}分鐘"])
        return exprs

    @classmethod
    def extract_features(cls, prop: Dict[str, Any]) -> Dict[str, List[str]]:
        features = {
            "budget": cls.expr_budget(prop["rent"]) if prop["rent"] else [],
            "room": Templates.ROOM.get(prop["room_type"], [prop["room_type"]]) if prop["room_type"] else [],
            "building": Templates.BUILDING.get(prop["building_type"], []) if prop["building_type"] else [],
            "region": Templates.REGION.get(prop["region"], []) if prop["region"] else [],
            "road": [prop["road"]] if prop["road"] else [],
            "distance": cls.expr_distance(prop["distance"]) if prop["distance"] else [],
            "furniture": list({t for f in prop["furniture"] for t in Templates.FURNITURE.get(f, [])}),
            "included": list({t for inc in prop["included"] for t in Templates.INCLUDED.get(inc, [])}),
            "special": [],
        }
        for note in prop["notes"]:
            if "可養寵物" in note: features["special"].extend(["可養寵物", "可以養貓", "可以養狗", "養寵物"])
            if "限女" in note: features["special"].extend(["限女生", "女生宿舍"])
            if "限男" in note: features["special"].extend(["限男生", "男生宿舍"])
            
        return {k: v for k, v in features.items() if v}

    @classmethod
    def build_queries(cls, prop: Dict[str, Any], num_queries: int = 40) -> List[str]:
        features = cls.extract_features(prop)
        keys = list(features.keys())
        queries = []

        # Strategy 1: Single Property Features
        for exprs in features.values():
            for e in random.sample(exprs, min(3, len(exprs))):
                queries.append(f"{random.choice(Templates.PREFIXES)}{e}")

        # Strategy 2: Dual Combos
        for _ in range(15):
            if len(keys) >= 2:
                k1, k2 = random.sample(keys, 2)
                e1, e2 = random.choice(features[k1]), random.choice(features[k2])
                conn = random.choice(Templates.CONNECTORS)
                pref = random.choice(Templates.PREFIXES) if random.random() > 0.3 else ""
                queries.append(f"{pref}{e1}{conn}{e2}")

        # Strategy 3: Tri-Combos
        for _ in range(12):
            if len(keys) >= 3:
                parts = [random.choice(features[k]) for k in random.sample(keys, 3)]
                pref = random.choice(Templates.PREFIXES) if random.random() > 0.4 else ""
                queries.append(f"{pref}{' '.join(parts)}")

        # Strategy 4: Raw Description
        for _ in range(8):
            if len(keys) >= 4:
                parts = [random.choice(features[k]) for k in random.sample(keys, random.randint(4, min(6, len(keys))))]
                queries.append(" ".join(parts))

        # Strategy 5: Colloquial Slang & Implicit Intent Mapping
        colloquials = ["想在中興大學附近租房", "學校旁邊有沒有房子", "想找便宜的租屋", "有空房嗎"]
        if prop["rent"] and prop["rent"] <= 5000: colloquials.extend(["便宜的房子", "平價租屋", "想省錢"])
        if prop["distance"] and prop["distance"] <= 1.0: colloquials.extend(["走路就能到學校", "學校附近租房", "不想騎車"])
        
        # Implicit Intents
        full_text = " ".join(prop.get("notes", [])) + " ".join(prop.get("furniture", []))
        if "子母車" in full_text or "垃圾" in full_text: colloquials.extend(["不想追垃圾車", "沒時間等垃圾車"])
        if "瓦斯爐" in full_text or "開火" in full_text: colloquials.extend(["喜歡自己煮", "想省伙食費"])
        if "陽台" in full_text or "獨立洗衣" in full_text or "獨洗" in full_text: colloquials.extend(["衣服不想曬房間", "房間容易潮濕", "需要通風"])
        if "木板" not in full_text and prop.get("building_type") in ["大樓", "華廈", "透天厝"]: colloquials.extend(["怕吵", "淺眠需要安靜", "隔音要好"])
        if "寵物" in full_text or "貓" in full_text: colloquials.extend(["有主子", "帶毛小孩"])
        
        queries.extend(random.sample(colloquials, min(3, len(colloquials))))

        final_queries = list(set([cls.inject_noise(q) for q in queries]))
        random.shuffle(final_queries)
        return final_queries[:num_queries]

    @staticmethod
    def inject_noise(text: str) -> str:
        if random.random() > 0.3:
            return text
        if len(text) > 5 and random.random() > 0.5:
            drop_idx = random.randint(0, len(text) - 1)
            text = text[:drop_idx] + text[drop_idx+1:]
        replacements = {
            "中興大學": ["興大", "中興", "NCHU"],
            "套房": ["小套房", "套"],
            "雅房": ["雅"],
            "可以": ["可", "能"],
            "有沒有": ["有沒", "有", "求"],
            "的": ["", "滴"],
        }
        for k, v in replacements.items():
            if k in text and random.random() > 0.5:
                text = text.replace(k, random.choice(v), 1)
        text = text.replace(" ", "") if random.random() > 0.7 else text
        return text


# ============================================================
# 4. Dataset construction & Compatibility Matching
# ============================================================
def is_compatible(query: str, prop: Dict[str, Any]) -> bool:
    """Verifies property constraints to avoid false negatives in training data."""
    for rt in ["套房", "雅房", "住宅"]:
        if rt in query and prop["room_type"] != rt: return False
    
    if (match := re.search(r"(\d+)(?:元)?(?:以下|以內|內)", query)) and prop["rent"] > int(match.group(1)):
        return False
        
    for reg in ["南區", "大里", "東區", "西區"]:
        if reg in query and reg not in prop.get("address", "") and reg not in prop.get("region", ""):
            return False

    roads = re.findall(r"([^區市台]*(?:路|街|大道)(?:[一二三四五六七八九十]|[\d])?段?)", query)
    for road in roads:
        if road not in prop.get("address", "") and road not in prop.get("road", ""):
            return False
            
    for feat, terms in Templates.FURNITURE.items():
        if any(t in query for t in terms) and not any(feat in f for f in prop["furniture"]):
            return False

    # Hard Exclusion Checks for auto-labeling FB queries
    if re.search(r"(謝絕|不要|拒絕|禁|❌|不接受)[^。！？\n]*(頂加|加蓋|頂樓)", query):
        if "頂加" in prop.get("building_type", "") or "加蓋" in prop.get("floor", "") or "加蓋" in " ".join(prop.get("notes", [])):
            return False
            
    if re.search(r"(謝絕|不要|拒絕|禁|❌|不接受)[^。！？\n]*木板", query):
        if "木板" in " ".join(prop.get("notes", [])):
            return False

    if "台水" in query or "台電" in query:
        full_notes_fees = " ".join(prop.get("other_fees", [])) + " " + " ".join(prop.get("notes", []))
        if re.search(r"(\d+(?:\.\d+)?)[元塊]?(?:/度|度)", full_notes_fees):
            return False

    return True


def compute_relevance_score(query: str, prop: Dict[str, Any]) -> int:
    """Computes a graded relevance score (0-3) based on 4 verifiable dimensions.
    Allows for 'Soft Mismatches' to enable meaningful NDCG evaluation.
    
    Grading Logic:
      - 0: Hard Conflict (Gender mismatch, Room type mismatch, explicit exclusions)
      - 1: Partial Match (Only location or 1 dimension matches)
      - 2: Good Match (Matches most dimensions, minor mismatch like slightly over budget)
      - 3: Perfect Match (All specified constraints satisfied)
    """
    
    # --- Part A: Hard Conflicts (Must be 0) ---
    
    # 1. Gender Restriction
    for note in prop.get("notes", []):
        if "限女" in note and "限男" in query: return 0
        if "限男" in note and "限女" in query: return 0
        if "限男" in note and "限妹子" in query: return 0 # Colloquial

    # 2. Room Type Mismatch (Fundamental)
    room_types = ["套房", "雅房", "住宅"]
    query_room = next((rt for rt in room_types if rt in query), None)
    if query_room and query_room not in prop.get("room_type", ""):
        return 0

    # 3. Explicit Exclusions
    if re.search(r"(謝絕|不要|拒絕|禁|❌|不接受)[^。！？\n]*(頂加|加蓋|頂樓)", query):
        if "頂加" in prop.get("building_type", "") or "加蓋" in prop.get("floor", "") or "加蓋" in " ".join(prop.get("notes", [])):
            return 0
            
    # --- Part B: Dimension Scoring (Soft Matching) ---
    satisfied = 0
    total_specified = 0

    # 1. Budget (Soft: 10% buffer for score 2)
    budget_match = re.search(r"(\d+)(?:元)?(?:以下|以內|內)", query)
    if budget_match:
        total_specified += 1
        ceiling = int(budget_match.group(1))
        if prop["rent"] <= ceiling:
            satisfied += 1
        elif prop["rent"] <= ceiling * 1.15: # Soft match
            satisfied += 0.5 

    # 2. Features / Furniture
    features_needed = [feat for feat, terms in Templates.FURNITURE.items() if any(t in query for t in terms)]
    if features_needed:
        total_specified += 1
        found_count = sum(1 for feat in features_needed if any(feat in f for f in prop.get("furniture", [])))
        satisfied += (found_count / len(features_needed))

    # 3. Location / Region
    region_specified = next((reg for reg in ["南區", "大里", "東區", "西區", "北區", "烏日"] if reg in query), None)
    roads_in_query = re.findall(r"([^區市台]*(?:路|街|大道)(?:[一二三四五六七八九十]|[\d])?段?)", query)
    
    if region_specified or roads_in_query:
        total_specified += 1
        loc_match = False
        if region_specified and (region_specified in prop.get("address", "") or region_specified in prop.get("region", "")):
            loc_match = True
        if roads_in_query and any(road in prop.get("address", "") for road in roads_in_query):
            loc_match = True
        
        if loc_match: satisfied += 1

    # --- Part C: Final Mapping ---
    if total_specified == 0:
        # 無具體指定條件的查詢（如「有沒有平件的房子」）視為「優良」而非「完美」
        # 因為沒有格局/地點/預算等關鍵條件的驗證，不應等同於「所有條件全滿足」
        return 2
        
    score_ratio = satisfied / total_specified
    
    if score_ratio >= 0.85: return 3  # 絕大多數條件滿足 → Perfect
    if score_ratio >= 0.65: return 2  # 多數符合但有明顯偏差 → Good  (0.5 to 0.65 落入 Partial)
    if score_ratio >= 0.15: return 1  # 部分符合：如「兩維度只符合一個」→ Partial
    return 0


def create_dataset_pairs(properties: List[Dict[str, Any]], neg_per_pos: int = 1) -> List[Dict[str, Any]]:
    """Constructs matching and non-matching pairs for sequence classification."""
    samples = []
    prop_texts = [property_to_text(p) for p in properties]

    for idx, prop in enumerate(properties):
        queries = QueryGenerator.build_queries(prop)

        for query in queries:
            relevance = compute_relevance_score(query, prop)
            samples.append({"query": query, "property": prop_texts[idx], "label": 1, "relevance": relevance, "property_idx": idx})
            
            neg_found = 0
            other_indices = [i for i in range(len(properties)) if i != idx]
            
            # Hard Negative Mining
            # 1. Filter all incompatible properties
            incompatible_indices = [i for i in other_indices if not is_compatible(query, properties[i])]
            
            if incompatible_indices:
                # 2. Score by character overlap (Jaccard-like) to find properties that look similar but fail hard constraints
                query_chars = set(query)
                scored = []
                for i in incompatible_indices:
                    overlap = len(query_chars.intersection(set(prop_texts[i])))
                    scored.append((i, overlap))
                
                # 3. Sort descending by overlap, take the top candidates as hard negatives
                scored.sort(key=lambda x: x[1], reverse=True)
                # Add some randomness so we don't always pick the exact same negative
                top_k = min(len(scored), neg_per_pos * 3)
                hard_candidates = [idx for idx, score in scored[:top_k]]
                random.shuffle(hard_candidates)
                
                for neg_idx in hard_candidates[:neg_per_pos]:
                    samples.append({"query": query, "property": prop_texts[neg_idx], "label": 0, "relevance": 0, "property_idx": neg_idx})

    random.shuffle(samples)
    return samples


import os

# ============================================================
# 5. Pipeline Orchestration
# ============================================================
def main():
    print("=" * 60)
    print("Step 1: Loading rental properties from CSV...")
    properties = load_properties()
    print(f"  Loaded {len(properties)} properties")

    print("\nStep 2: Generating query-property pairs...")
    # 增加 num_queries 從 40 到 60，neg_per_pos 從 1 到 2
    all_samples = []
    prop_texts = [property_to_text(p) for p in properties]
    for idx, prop in enumerate(properties):
        queries = QueryGenerator.build_queries(prop, num_queries=60)
        for query in queries:
            relevance = compute_relevance_score(query, prop)
            all_samples.append({"query": query, "property": prop_texts[idx], "label": 1, "relevance": relevance, "property_idx": idx})
            
            # Hard Negative Mining: 取跟詢問「字元重疊度最高」的不相容房源作為困難負樣本
            other_indices = [i for i in range(len(properties)) if i != idx]
            incompatible = [i for i in other_indices if not is_compatible(query, properties[i])]
            
            if incompatible:
                # 依語意相似度排序：找「看起來很像但其實不符合」的房源
                query_chars = set(query)
                incompatible.sort(
                    key=lambda i: len(query_chars & set(prop_texts[i])),
                    reverse=True
                )
                for neg_idx in incompatible[:3]:  # 1:3 比例，提升困難樣本比例
                    all_samples.append({"query": query, "property": prop_texts[neg_idx], "label": 0, "relevance": 0, "property_idx": neg_idx})
    
    # 載入 FB 真實貼文
    fb_queries_path = os.path.join(os.path.dirname(__file__), "../../data/raw/fb_queries.json")
    if os.path.exists(fb_queries_path):
        print("\nStep 2.5: Injecting Facebook crawled queries...")
        with open(fb_queries_path, "r", encoding="utf-8") as f:
            fb_queries = json.load(f)
        
        fb_samples = []
        prop_texts = [property_to_text(p) for p in properties]
        for query in fb_queries:
            pos_found = False
            for idx, prop in enumerate(properties):
                # 如果符合相容性，視為正樣本
                if is_compatible(query, prop):
                    rel = compute_relevance_score(query, prop)
                    fb_samples.append({"query": query, "property": prop_texts[idx], "label": 1, "relevance": rel})
                    pos_found = True
                    break
            
            # 若有正樣本，則隨機挑選一個不相容的當作負樣本
            if pos_found:
                other_indices = list(range(len(properties)))
                random.shuffle(other_indices)
                for neg_idx in other_indices:
                    if not is_compatible(query, properties[neg_idx]):
                        fb_samples.append({"query": query, "property": prop_texts[neg_idx], "label": 0, "relevance": 0})
                        break
                        
        print(f"  Generated {len(fb_samples)} pairs from FB queries.")
        all_samples.extend(fb_samples)
        
    pos = sum(1 for s in all_samples if s["label"] == 1)
    neg = len(all_samples) - pos
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Positive: {pos}, Negative: {neg}, Ratio: 1:{neg/pos:.1f}")

    random.shuffle(all_samples)
    train_bound, dev_bound = int(len(all_samples) * 0.8), int(len(all_samples) * 0.9)
    train_data, dev_data, test_data = all_samples[:train_bound], all_samples[train_bound:dev_bound], all_samples[dev_bound:]

    print(f"\nStep 3: Saving datasets...")
    def clean(samples): return [{"query": s["query"], "property": s["property"], "label": s["label"], "relevance": s.get("relevance", s["label"])} for s in samples]

    for filename, subset in zip(
        ["../../data/processed/recommendation_train.json", "../../data/processed/recommendation_dev.json", "../../data/processed/recommendation_test.json"], 
        [train_data, dev_data, test_data]
    ):
        out_path = os.path.join(os.path.dirname(__file__), filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clean(subset), f, ensure_ascii=False, indent=2)

    prop_texts = [property_to_text(p) for p in properties]
    prop_out = os.path.join(os.path.dirname(__file__), "../../data/processed/property_texts.json")
    with open(prop_out, "w", encoding="utf-8") as f:
        json.dump(prop_texts, f, ensure_ascii=False, indent=2)

    print("\nDataset generation complete!")
    print(f"  Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

if __name__ == "__main__":
    main()
