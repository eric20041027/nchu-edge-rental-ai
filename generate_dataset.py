"""
generate_dataset.py — 從 nchu_rental_info.csv 自動生成推薦模型的 train/dev/test 資料集

讀取真實房源 CSV，模擬中興大學學生的自然語言查詢輸入，
產生 query-property 配對資料 (正例 + 負例) 用於 sentence-pair classification 訓練。
"""
import csv
import json
import random
import re
import os

random.seed(42)

# ============================================================
# 1. 讀取 CSV 房源資料
# ============================================================
def load_properties(csv_path="nchu_rental_info.csv"):
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    properties = []
    for row in rows:
        # 解析租金數字
        rent_str = row.get("租金", "")
        rent_match = re.search(r"(\d[\d,]*)", rent_str.replace(",", ""))
        rent_num = int(rent_match.group(1)) if rent_match else 0

        # 解析距離
        dist = float(row.get("距離(km)", "0") or "0")

        # 解析家具清單
        furniture = [s.strip() for s in row.get("家具設施", "").split("/") if s.strip()]
        included = [s.strip() for s in row.get("租金包含", "").split("/") if s.strip()]
        security = [s.strip() for s in row.get("安全管理", "").split("/") if s.strip()]
        notes = [s.strip() for s in row.get("備註", "").split("/") if s.strip()]

        # 解析地址中的路/街/巷
        addr = row.get("地址", "")
        road = ""
        # 匹配 路/街/大道
        road_match = re.search(r"([^區市台]*(?:路|街|大道)(?:[一二三四五六七八九十]|[\d])?段?)", addr)
        if road_match:
            road = road_match.group(1).strip()

        region = ""
        for r in ["南區", "大里區", "西區", "東區", "北區", "烏日"]:
            if r in addr:
                region = r
                break

        properties.append({
            "address": addr,
            "region": region,
            "road": road,
            "room_type": row.get("格局", ""),
            "building_type": row.get("類型", ""),
            "size": row.get("室內坪數", ""),
            "rent": rent_num,
            "rent_str": rent_str,
            "furniture": furniture,
            "included": included,
            "security": security,
            "notes": notes,
            "distance": dist,
            "url": row.get("網址", ""),
            "img": row.get("圖片網址", ""),
            "floor": row.get("樓層", ""),
        })

    return properties


# ============================================================
# 2. 將房屋資訊轉為模型可讀的文本描述 (Canonical Property Text)
# ============================================================
def property_to_text(prop):
    """將一筆房源轉成簡潔的文字描述，作為 sentence-pair 的 property 端"""
    parts = []

    if prop["room_type"]:
        parts.append(prop["room_type"])
    if prop["building_type"]:
        parts.append(prop["building_type"])
    if prop["region"]:
        parts.append(prop["region"])
    if prop["road"]:
        parts.append(prop["road"])
    if prop["rent"]:
        parts.append(f"{prop['rent']}元")
    if prop["distance"]:
        parts.append(f"距離{prop['distance']}km")

    # 取前 5 個主要家具
    key_furniture = []
    for f in prop["furniture"]:
        short = f.replace("（電）", "").replace("機車停車位", "機車位").replace("書桌椅", "書桌")
        if short not in key_furniture:
            key_furniture.append(short)
        if len(key_furniture) >= 5:
            break
    if key_furniture:
        parts.append(" ".join(key_furniture))

    if prop["included"]:
        parts.append("含" + "".join(prop["included"][:3]))

    # 備註中的重要資訊
    for note in prop["notes"]:
        if "寵物" in note or "限" in note:
            parts.append(note)

    return " ".join(parts)


# ============================================================
# 3. 生成模擬中興學生的自然語言查詢
# ============================================================

# 預算表達模板
def budget_expressions(rent):
    """根據實際租金生成各種預算表達方式"""
    exprs = []
    # 精確值
    exprs.append(f"預算{rent}")
    exprs.append(f"月租{rent}")
    exprs.append(f"{rent}元")

    # 以下/以內
    for above in [500, 1000, 2000]:
        ceiling = rent + above
        exprs.append(f"預算{ceiling}以下")
        exprs.append(f"{ceiling}以內")
        exprs.append(f"月租{ceiling}內")

    # K 表達
    if rent >= 1000:
        k_val = rent // 1000
        exprs.append(f"{k_val}K")
        exprs.append(f"{k_val}k以下")
        exprs.append(f"預算{k_val}千")

    # 中文數字
    cn_map = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}
    if 1000 <= rent < 10000:
        k = rent // 1000
        if k in cn_map:
            exprs.append(f"{cn_map[k]}千")
            exprs.append(f"預算{cn_map[k]}千以下")
            exprs.append(f"{cn_map[k]}千以內")
    if rent >= 10000:
        w = rent // 10000
        remainder = (rent % 10000) // 1000
        if w in cn_map:
            if remainder == 0:
                exprs.append(f"{cn_map[w]}萬")
            elif remainder in cn_map:
                exprs.append(f"{cn_map[w]}萬{cn_map[remainder]}千")

    # 以上 (搜尋比當前租金更高的)
    for below in [500, 1000, 2000]:
        floor = max(rent - below, 1000)
        exprs.append(f"預算{floor}以上")
        exprs.append(f"{floor}元以上")

    return exprs


# 房型表達模板
ROOM_TEMPLATES = {
    "套房": ["套房", "找套房", "想租套房", "要一間套房", "單人套房"],
    "雅房": ["雅房", "找雅房", "想租雅房", "要一間雅房"],
    "住宅": ["住宅", "整層住宅", "整層", "家庭式", "整層出租"],
    "套/雅房": ["套房", "雅房", "套房或雅房"],
}

# 建築類型模板
BUILDING_TEMPLATES = {
    "透天厝": ["透天", "透天厝"],
    "公寓": ["公寓", "公寓大樓"],
    "大樓": ["大樓", "電梯大樓", "有電梯的"],
    "華廈": ["華廈", "華廈大樓"],
}

# 地區表達模板
REGION_TEMPLATES = {
    "南區": ["南區", "台中南區", "南區附近"],
    "大里區": ["大里", "大里區", "台中大里"],
    "東區": ["東區", "台中東區"],
}

# 距離表達模板
def distance_expressions(dist_km):
    exprs = []
    if dist_km <= 1.0:
        exprs += ["學校附近", "中興大學旁", "近中興", "學校旁邊", "中興大學門口", "正門附近", "走路就到"]
    if dist_km <= 0.5:
        exprs += ["非常近", "校門口", "走路五分鐘"]

    walk_min = round(dist_km / 0.08)
    ride_min = round(dist_km / 0.6)
    if walk_min <= 15:
        exprs.append(f"走路{walk_min}分鐘")
        exprs.append(f"步行{walk_min}分鐘內")
    if ride_min <= 10:
        exprs.append(f"騎車{ride_min}分鐘")
        exprs.append(f"機車{ride_min}分鐘")

    return exprs


# 家具設施模板
FURNITURE_QUERY_MAP = {
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

# 費用包含模板
INCLUDED_QUERY_MAP = {
    "水費": ["含水費", "包水費", "水費包含"],
    "電費": ["含電費", "包電費", "台水台電"],
    "網路費": ["含網路", "包網路"],
    "管理費": ["含管理費"],
    "瓦斯": ["含瓦斯"],
}

# 通用句型
QUERY_PREFIXES = [
    "想找", "想租", "找", "要", "求", "想要",
    "幫我找", "需要", "我要找", "有沒有",
    "請問有", "想住", "要租", "有推薦",
]

QUERY_CONNECTORS = [" ", "的", " 要", " 有", " 而且"]


def generate_queries_for_property(prop, num_queries=40):
    """根據一筆房源的特徵，生成多種模擬學生查詢"""
    queries = []

    # 蒐集各維度的表達
    budget_exprs = budget_expressions(prop["rent"]) if prop["rent"] else []
    room_exprs = ROOM_TEMPLATES.get(prop["room_type"], [prop["room_type"]]) if prop["room_type"] else []
    building_exprs = BUILDING_TEMPLATES.get(prop["building_type"], []) if prop["building_type"] else []
    region_exprs = REGION_TEMPLATES.get(prop["region"], []) if prop["region"] else []
    distance_exprs = distance_expressions(prop["distance"]) if prop["distance"] else []

    furniture_exprs = []
    for f in prop["furniture"]:
        if f in FURNITURE_QUERY_MAP:
            furniture_exprs.extend(FURNITURE_QUERY_MAP[f])

    included_exprs = []
    for inc in prop["included"]:
        if inc in INCLUDED_QUERY_MAP:
            included_exprs.extend(INCLUDED_QUERY_MAP[inc])

    # 特殊需求
    special_exprs = []
    for note in prop["notes"]:
        if "可養寵物" in note:
            special_exprs += ["可養寵物", "可以養貓", "可以養狗", "養寵物"]
        if "限女" in note:
            special_exprs += ["限女生", "女生宿舍"]
        if "限男" in note:
            special_exprs += ["限男生", "男生宿舍"]

    all_features = {
        "budget": budget_exprs,
        "room": room_exprs,
        "building": building_exprs,
        "region": region_exprs,
        "road": [prop["road"]] if prop["road"] else [],
        "distance": distance_exprs,
        "furniture": list(set(furniture_exprs)),
        "included": list(set(included_exprs)),
        "special": special_exprs,
    }

    # 策略 1: 單特徵查詢 (每種類型各取 3 個)
    for key, exprs in all_features.items():
        for expr in random.sample(exprs, min(3, len(exprs))):
            prefix = random.choice(QUERY_PREFIXES)
            queries.append(f"{prefix}{expr}")

    # 策略 2: 雙特徵組合 (增加到 15 個)
    feature_keys = [k for k, v in all_features.items() if v]
    for _ in range(15):
        if len(feature_keys) < 2:
            break
        k1, k2 = random.sample(feature_keys, 2)
        e1 = random.choice(all_features[k1])
        e2 = random.choice(all_features[k2])
        conn = random.choice(QUERY_CONNECTORS)
        prefix = random.choice(QUERY_PREFIXES) if random.random() > 0.3 else ""
        queries.append(f"{prefix}{e1}{conn}{e2}")

    # 策略 3: 三特徵組合 (增加到 12 個)
    for _ in range(12):
        if len(feature_keys) < 3:
            break
        keys = random.sample(feature_keys, 3)
        parts = [random.choice(all_features[k]) for k in keys]
        prefix = random.choice(QUERY_PREFIXES) if random.random() > 0.4 else ""
        queries.append(f"{prefix}{' '.join(parts)}")

    # 策略 4: 完整描述 (4+ 特徵，增加到 8 個)
    for _ in range(8):
        if len(feature_keys) < 4:
            break
        n = random.randint(4, min(6, len(feature_keys)))
        keys = random.sample(feature_keys, n)
        parts = [random.choice(all_features[k]) for k in keys]
        queries.append(" ".join(parts))

    # 策略 5: 口語化句型
    colloquial = [
        "想在中興大學附近租房",
        "學校旁邊有沒有房子",
        "想找便宜的租屋",
        "有空房嗎",
    ]
    if prop["rent"] and prop["rent"] <= 5000:
        colloquial.append("便宜的房子")
        colloquial.append("平價租屋")
    if prop["distance"] and prop["distance"] <= 1.0:
        colloquial.append("走路就能到學校")
        colloquial.append("學校附近租房")
    queries.extend(random.sample(colloquial, min(3, len(colloquial))))

    # 去重 + 截斷
    queries = list(set(queries))
    random.shuffle(queries)
    return queries[:num_queries]


# ============================================================
# 4. 判斷查詢與房源是否相容 (用於確保負例真的是負例)
# ============================================================
def is_compatible(query, prop):
    """
    檢查房源是否符合查詢的要求。
    用於確保在生成「負例」時，不小心挑到其實符合條件的房源。
    """
    # 1. 房型檢查
    for rt in ["套房", "雅房", "住宅"]:
        if rt in query and prop["room_type"] != rt:
            return False
    
    # 2. 預算檢查 (解析查詢中的數字)
    # 匹配 "6000以下", "6000以內", "6k", "六千"
    budget_match = re.search(r"(\d+)(?:元)?(?:以下|以內|內)", query)
    if budget_match:
        limit = int(budget_match.group(1))
        if prop["rent"] > limit:
            return False
    
    # 3. 地區/路段檢查
    for reg in ["南區", "大里", "東區", "西區"]:
        if reg in query and reg not in prop.get("address", "") and reg not in prop.get("region", ""):
            return False
    
    if prop.get("road") and prop["road"] in query:
        # 如果 query 提到了一條路，但這間房子不在那條路上
        # (這裡簡單處理：如果 query 有路名，且 prop 有路名，要一致)
        # 但有些 query 可能包含多個關鍵字，所以我們只做正向匹配的否定
        pass
    
    # 更精確的路段排除：如果查詢中提到某路，但房源描述中完全沒有該路名
    # 我們需要從 query 中識別出可能是路名的部分。
    # 這裡暫時依賴 Sentence-Pair 模型學習，但為了負例挖掘，我們加一個簡單檢查
    roads_in_query = re.findall(r"([^區市台]*(?:路|街|大道)(?:[一二三四五六七八九十]|[\d])?段?)", query)
    for road in roads_in_query:
        if road not in prop.get("address", "") and road not in prop.get("road", ""):
            return False
            
    # 4. 特定家具 (如果有提到，則房源必須具備)
    for feat, terms in FURNITURE_QUERY_MAP.items():
        # 如果候選語句中出現了該家具的關鍵字
        if any(term in query for term in terms):
            # 房源的家具清單中也必須有該特徵
            # (這裡放寬一點，只要 prop["furniture"] 包含 feat 或其簡寫)
            if not any(feat in f for f in prop["furniture"]):
                return False

    return True


# ============================================================
# 5. 產生正負配對資料
# ============================================================
def generate_dataset(properties, num_neg_per_pos=1):
    """
    為每筆房源生成正例 (matching) 和負例 (non-matching) 配對。
    正例: 查詢確實描述了該房源的特徵
    負例: 必須「不符合」查詢條件的房源才作為負例
    """
    all_samples = []
    property_texts = [property_to_text(p) for p in properties]

    for idx, prop in enumerate(properties):
        prop_text = property_texts[idx]
        queries = generate_queries_for_property(prop)

        for query in queries:
            # 正例
            all_samples.append({
                "query": query,
                "property": prop_text,
                "label": 1,
                "property_idx": idx
            })

            # 負例挖掘
            neg_samples_found = 0
            # 隨機打亂候選房源索引來挑選負例
            other_indices = [i for i in range(len(properties)) if i != idx]
            random.shuffle(other_indices)
            
            for neg_idx in other_indices:
                neg_prop = properties[neg_idx]
                
                # 關鍵邏輯：如果這個房源「也符合」使用者的查詢，就不能把它當作負例標記為 0
                if is_compatible(query, neg_prop):
                    continue
                
                all_samples.append({
                    "query": query,
                    "property": property_texts[neg_idx],
                    "label": 0,
                    "property_idx": neg_idx
                })
                neg_samples_found += 1
                if neg_samples_found >= num_neg_per_pos:
                    break

    random.shuffle(all_samples)
    return all_samples


# ============================================================
# 5. 主程式：分割並儲存 train/dev/test
# ============================================================
def main():
    print("=" * 60)
    print("Step 1: Loading rental properties from CSV...")
    properties = load_properties()
    print(f"  Loaded {len(properties)} properties")

    print("\nStep 2: Generating query-property pairs...")
    all_samples = generate_dataset(properties, num_neg_per_pos=1)
    print(f"  Generated {len(all_samples)} total samples")

    # 統計正負例
    pos = sum(1 for s in all_samples if s["label"] == 1)
    neg = len(all_samples) - pos
    print(f"  Positive: {pos}, Negative: {neg}, Ratio: 1:{neg/pos:.1f}")

    # 分割 train/dev/test (80/10/10)
    random.shuffle(all_samples)
    n = len(all_samples)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)

    train_data = all_samples[:train_end]
    dev_data = all_samples[train_end:dev_end]
    test_data = all_samples[dev_end:]

    print(f"\nStep 3: Saving datasets...")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Dev:   {len(dev_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # 儲存 (不含 property_idx，那只是 debug 用)
    def clean(samples):
        return [{"query": s["query"], "property": s["property"], "label": s["label"]} for s in samples]

    with open("recommendation_train.json", "w", encoding="utf-8") as f:
        json.dump(clean(train_data), f, ensure_ascii=False, indent=2)

    with open("recommendation_dev.json", "w", encoding="utf-8") as f:
        json.dump(clean(dev_data), f, ensure_ascii=False, indent=2)

    with open("recommendation_test.json", "w", encoding="utf-8") as f:
        json.dump(clean(test_data), f, ensure_ascii=False, indent=2)

    # 同時儲存房源描述文本 (供 precompute_embeddings.py 使用)
    property_texts = [property_to_text(p) for p in properties]
    with open("property_texts.json", "w", encoding="utf-8") as f:
        json.dump(property_texts, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"  recommendation_train.json  ({len(train_data)} samples)")
    print(f"  recommendation_dev.json    ({len(dev_data)} samples)")
    print(f"  recommendation_test.json   ({len(test_data)} samples)")
    print(f"  property_texts.json        ({len(property_texts)} property descriptions)")

    # 印出幾個範例
    print("\n--- Sample Pairs ---")
    for s in train_data[:5]:
        tag = "POS" if s["label"] == 1 else "NEG"
        print(f"  [{tag}] Q: {s['query']}")
        print(f"       P: {s['property'][:60]}...")
        print()


if __name__ == "__main__":
    main()
