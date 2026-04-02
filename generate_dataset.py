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
        "冷氣機": ["有冷氣", "要冷氣", "含冷氣"],
        "冷氣": ["有冷氣", "要冷氣"],
        "洗衣機": ["有洗衣機", "獨洗", "獨立洗衣"],
        "冰箱": ["有冰箱"],
        "電視": ["有電視"],
        "有線電視": ["有第四台", "有電視"],
        "書桌椅": ["有書桌", "附書桌"],
        "書桌": ["有書桌", "附書桌"],
        "床": ["有床", "附床"],
        "衣櫃": ["有衣櫃"],
        "熱水器": ["有熱水器"],
        "（電）熱水器": ["有熱水器"],
        "電梯": ["有電梯"],
        "陽台": ["有陽台", "要陽台"],
        "曬衣場": ["獨曬", "有曬衣", "可曬衣"],
        "機車停車位": ["有車位", "機車位", "有停車位"],
        "飲水機": ["有飲水機"],
        "寬頻網路": ["有網路"],
    }
    INCLUDED = {
        "水費": ["含水費", "包水費", "水費包含"],
        "電費": ["含電費", "包電費", "台水台電"],
        "網路費": ["含網路", "包網路"],
        "管理費": ["含管理費"],
        "瓦斯": ["含瓦斯"],
    }
    PREFIXES = ["想找", "想租", "找", "要", "求", "想要", "幫我找", "需要", "我要找", "有沒有", "請問有", "想住", "要租", "有推薦"]
    CONNECTORS = [" ", "的", " 要", " 有", " 而且"]

# ============================================================
# 2. Data Loading & Normalization
# ============================================================
def load_properties(csv_path: str = "nchu_rental_info.csv") -> List[Dict[str, Any]]:
    """Reads CSV properties into structured dict format."""
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

        # Strategy 5: Colloquial Slang
        colloquials = ["想在中興大學附近租房", "學校旁邊有沒有房子", "想找便宜的租屋", "有空房嗎"]
        if prop["rent"] and prop["rent"] <= 5000: colloquials.extend(["便宜的房子", "平價租屋"])
        if prop["distance"] and prop["distance"] <= 1.0: colloquials.extend(["走路就能到學校", "學校附近租房"])
        queries.extend(random.sample(colloquials, min(3, len(colloquials))))

        final_queries = list(set(queries))
        random.shuffle(final_queries)
        return final_queries[:num_queries]


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

    return True


def create_dataset_pairs(properties: List[Dict[str, Any]], neg_per_pos: int = 1) -> List[Dict[str, Any]]:
    """Constructs matching and non-matching pairs for sequence classification."""
    samples = []
    prop_texts = [property_to_text(p) for p in properties]

    for idx, prop in enumerate(properties):
        queries = QueryGenerator.build_queries(prop)

        for query in queries:
            samples.append({"query": query, "property": prop_texts[idx], "label": 1, "property_idx": idx})
            
            neg_found = 0
            other_indices = [i for i in range(len(properties)) if i != idx]
            random.shuffle(other_indices)
            
            for neg_idx in other_indices:
                if not is_compatible(query, properties[neg_idx]):
                    samples.append({"query": query, "property": prop_texts[neg_idx], "label": 0, "property_idx": neg_idx})
                    if (neg_found := neg_found + 1) >= neg_per_pos: break

    random.shuffle(samples)
    return samples


# ============================================================
# 5. Pipeline Orchestration
# ============================================================
def main():
    print("=" * 60)
    print("Step 1: Loading rental properties from CSV...")
    properties = load_properties()
    print(f"  Loaded {len(properties)} properties")

    print("\nStep 2: Generating query-property pairs...")
    all_samples = create_dataset_pairs(properties, neg_per_pos=1)
    
    pos = sum(1 for s in all_samples if s["label"] == 1)
    neg = len(all_samples) - pos
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Positive: {pos}, Negative: {neg}, Ratio: 1:{neg/pos:.1f}")

    random.shuffle(all_samples)
    train_bound, dev_bound = int(len(all_samples) * 0.8), int(len(all_samples) * 0.9)
    train_data, dev_data, test_data = all_samples[:train_bound], all_samples[train_bound:dev_bound], all_samples[dev_bound:]

    print(f"\nStep 3: Saving datasets...")
    def clean(samples): return [{"query": s["query"], "property": s["property"], "label": s["label"]} for s in samples]

    for filename, subset in zip(
        ["recommendation_train.json", "recommendation_dev.json", "recommendation_test.json"], 
        [train_data, dev_data, test_data]
    ):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(clean(subset), f, ensure_ascii=False, indent=2)

    prop_texts = [property_to_text(p) for p in properties]
    with open("property_texts.json", "w", encoding="utf-8") as f:
        json.dump(prop_texts, f, ensure_ascii=False, indent=2)

    print("\nDataset generation complete!")
    print(f"  Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

if __name__ == "__main__":
    main()
