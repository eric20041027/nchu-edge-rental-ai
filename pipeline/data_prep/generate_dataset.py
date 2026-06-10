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
        "西區": ["西區", "台中西區", "勤美附近", "西區附近"],
        "太平區": ["太平", "太平區", "台中太平"],
        "西屯區": ["西屯", "逢甲附近", "西屯區", "七期附近"],
        "北屯區": ["北屯", "北屯區", "台中北屯"],
        "南屯區": ["南屯", "南屯區", "台中南屯"],
        "北區": ["北區", "一中附近", "北區附近"],
        "中區": ["中區", "台中火車站附近", "台中中區"],
    }
    FURNITURE = {
        "冷氣機": ["有冷氣", "要冷氣", "含冷氣", "怕熱", "吹冷氣", "夏天", "西曬"],
        "冷氣": ["有冷氣", "要冷氣", "怕熱", "吹冷氣", "西曬"],
        "變頻冷氣": ["變頻", "省電冷氣", "省電費", "吹整天也不心疼"],
        "洗衣機": ["有洗衣機", "獨洗", "獨立洗衣", "不想共用洗衣機", "獨洗獨曬", "衣服很多", "愛乾淨"],
        "冰箱": ["有冰箱", "需要冰東西", "存糧", "喝冰水"],
        "電視": ["有電視", "追劇", "打電動"],
        "有線電視": ["有第四台", "有電視", "看新聞"],
        "書桌椅": ["有書桌", "附書桌", "辦公", "工作", "打電腦"],
        "書桌": ["有書桌", "附書桌", "要辦公"],
        "床": ["有床", "附床", "睡覺"],
        "衣櫃": ["有衣櫃", "衣服很多", "收納", "收藏控"],
        "熱水器": ["有熱水器", "洗澡"],
        "（電）熱水器": ["有熱水器"],
        "電梯": ["有電梯", "不想爬樓梯", "不想走樓梯", "搬東西方便", "懶人", "有電梯", "高樓層"],
        "陽台": ["有陽台", "要陽台", "想曬衣服", "衣服容易乾", "通風好", "要有對外窗", "曬衣", "獨洗獨曬", "不悶熱", "採光好"],
        "曬衣場": ["獨曬", "有曬衣", "可曬衣", "曬衣服", "曬衣空間"],
        "機車停車位": ["有車位", "機車位", "有停車位", "好停車", "愛車有位子"],
        "飲水機": ["有飲水機", "不用買水", "不想出門", "居家族"],
        "寬頻網路": ["有網路", "上網方便", "打報告", "打遊戲", "光纖", "晚上要工作", "網速快"],
        "垃圾處理": ["子母車", "垃圾車", "不用追垃圾車", "垃圾子母車", "下班晚", "沒時間倒垃圾"],
        "可開伙": ["開伙", "煮飯", "小廚房", "可以煮", "黑晶爐", "想省伙食費", "大廚", "自炊"],
        "可租補": ["可申請補助", "想報稅", "可以申請補助", "要報稅", "可以補助", "可報稅", "租金補貼", "要租補", "可以租補"],
    }
    INCLUDED = {
        "水費": ["含水費", "包水費", "水費包含"],
        "電費": ["含電費", "包電費", "台水台電", "照台水台電收費", "不要電費太貴", "一度5塊太貴了", "電費依台水台電", "台電帳單"],
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
        notes = [s.strip() for s in row.get("特色", "").split("/") if s.strip()]
        
        # Check for inclusions
        included = []
        if "含水" in (row.get("水費", "") + row.get("特色", "")): included.append("水費")
        if "含電" in (row.get("電費", "") + row.get("特色", "")): included.append("電費")
        if "含網" in (row.get("家具設施", "") + row.get("特色", "")): included.append("網路費")

        addr = row.get("地址", "")
        road = ""
        road_match = re.search(r"([^區市台]*(?:路|街|大道)(?:[一二三四五六七八九十]|[\d])?段?)", addr)
        if road_match:
            road = road_match.group(1).strip()

        region = next((r for r in ["南區", "大里區", "西區", "東區", "北區", "烏日"] if r in addr), "")

        # Room type from '類型'
        type_str = row.get("類型", "")
        room_type = "套房" if "套房" in type_str else ("雅房" if "雅房" in type_str else ("住宅" if "住宅" in type_str else ""))
        building_type = type_str.replace("套房", "").replace("雅房", "").replace("住宅", "").strip()

        walk = int(row.get("walk_mins", "0") or "0")
        scooter = int(row.get("scooter_mins", "0") or "0")

        properties.append({
            "address": addr, "region": region, "road": road,
            "room_type": room_type, "building_type": building_type,
            "size": row.get("室內坪數", ""), "rent": rent_num, "rent_str": rent_str,
            "furniture": furniture, "included": included, "security": [],
            "notes": notes, "distance": dist, "url": row.get("網址", ""),
            "img": row.get("圖片網址", ""), "floor": row.get("樓層", ""),
            "walk_mins": walk, "scooter_mins": scooter,
        })

    return properties


def property_to_text(prop: Dict[str, Any]) -> str:
    """Consolidates property keys into a canonical descriptive string."""
    parts = [p for p in (prop["room_type"], prop["building_type"], prop["region"], prop["road"]) if p]

    if prop["rent"]: parts.append(f"{prop['rent']}元")
    if prop["distance"]: parts.append(f"距離{prop['distance']}km")

    walk = prop.get("walk_mins", 0) or 0
    scooter = prop.get("scooter_mins", 0) or 0
    if walk > 0:
        parts.append(f"步行{walk}分鐘")
    if scooter > 0:
        parts.append(f"騎車{scooter}分鐘")

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
    def expr_distance(dist_km: float, walk_mins: int = 0, scooter_mins: int = 0) -> List[str]:
        exprs = []
        if dist_km <= 1.0: exprs += ["學校附近", "中興大學旁", "近中興", "學校旁邊", "中興大學門口", "正門附近", "走路就到"]
        if dist_km <= 0.5: exprs += ["非常近", "校門口", "走路五分鐘"]

        # Use actual commute times from CSV when available; fall back to estimated
        walk = walk_mins if walk_mins > 0 else round(dist_km / 0.08)
        ride = scooter_mins if scooter_mins > 0 else round(dist_km / 0.6)
        if walk <= 20: exprs.extend([f"走路{walk}分鐘", f"步行{walk}分鐘內", f"走路約{walk}分"])
        if ride <= 15: exprs.extend([f"騎車{ride}分鐘", f"機車{ride}分鐘", f"騎車約{ride}分"])
        return exprs

    @classmethod
    def extract_features(cls, prop: Dict[str, Any]) -> Dict[str, List[str]]:
        features = {
            "budget": cls.expr_budget(prop["rent"]) if prop["rent"] else [],
            "room": Templates.ROOM.get(prop["room_type"], [prop["room_type"]]) if prop["room_type"] else [],
            "building": Templates.BUILDING.get(prop["building_type"], []) if prop["building_type"] else [],
            "region": Templates.REGION.get(prop["region"], []) if prop["region"] else [],
            "road": [prop["road"]] if prop["road"] else [],
            "distance": cls.expr_distance(
                prop["distance"],
                prop.get("walk_mins", 0) or 0,
                prop.get("scooter_mins", 0) or 0,
            ) if prop["distance"] else [],
            "furniture": list({t for f in prop["furniture"] for t in Templates.FURNITURE.get(f, [])}),
            "included": list({t for inc in prop["included"] for t in Templates.INCLUDED.get(inc, [])}),
            "special": [],
        }
        for note in prop["notes"]:
            if "可養寵物" in note or "可寵" in note: features["special"].extend(["可養寵物", "可以養貓", "可以養狗", "養寵物"])
            if "限女" in note: features["special"].extend(["限女生", "女生宿舍"])
            if "限男" in note: features["special"].extend(["限男生", "男生宿舍"])
            if "租補" in note or "補助" in note:
                features["furniture"].extend(Templates.FURNITURE["可租補"])
            
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

        # Strategy 5: Colloquial Slang & Deep Lifestyle Inference
        colloquials = ["想在中興大學附近租房", "學校旁邊有沒有房子", "想找便宜的租屋", "有空房嗎"]

        # Financial / Value Inference
        if prop["rent"] and prop["rent"] <= 5000:
            colloquials.extend(["便宜的房子", "平價租屋", "想省錢", "預算有限", "追求CP值"])

        # Proximity Inference
        walk_mins = prop.get("walk_mins", 0) or 0
        proximity_thresh = (walk_mins <= 15 and walk_mins > 0) or (prop["distance"] and prop["distance"] <= 0.8)
        if proximity_thresh:
            walk_str = f"走路{walk_mins}分鐘" if walk_mins > 0 else "走路就到"
            colloquials.extend([
                "走路就能到學校", "學校附近租房", "不想騎車", "正門口附近", "睡晚一點",
                f"{walk_str}能到學校 超方便",
            ])

        # Lifestyle Features Inference
        full_text = " ".join(prop.get("notes", [])) + " ".join(prop.get("furniture", []))

        if any(k in full_text for k in ["飲水機", "子母車", "垃圾", "電梯"]):
            colloquials.extend(["不想出門買水", "懶人首選", "不想追垃圾車", "沒時間倒垃圾", "生活機能要好"])

        if any(k in full_text for k in ["瓦斯爐", "開火", "廚房"]):
            colloquials.extend(["喜歡自己煮", "想省伙食費", "不想天天外食", "居家自炊", "可以煮泡麵"])

        if any(k in full_text for k in ["陽台", "對外窗", "採光", "隔音"]):
            colloquials.extend(["怕悶熱", "房間要通風", "需要對外窗", "淺眠怕吵", "想要採光好", "衣服要乾"])

        if any(k in full_text for k in ["寵物", "貓", "狗"]):
            colloquials.extend(["有主子", "帶毛小孩", "寵物友善", "不離不棄"])

        if any(k in full_text for k in ["全新", "獨洗", "禁菸"]):
            colloquials.extend(["愛乾淨", "稍微潔癖", "不想跟人共用", "怕菸味"])

        queries.extend(random.sample(colloquials, min(3, len(colloquials))))

        # Strategy 6: Situational / Persona-based queries
        # These simulate real-world user phrasing with identity context + constraints
        situational = []

        # Student personas
        if prop["distance"] and prop["distance"] <= 2.0:
            situational += [
                f"大一新生找宿舍 預算{prop['rent']+500}以下",
                f"交換學生一年 想租學校附近",
                f"研究所剛入學 不太熟台中 想住近一點",
            ]
        # Budget-tight personas
        if prop["rent"] and prop["rent"] <= 6000:
            situational += [
                f"剛畢業薪水不高 想找{prop['rent']}左右的",
                f"打工族 月租盡量壓在{prop['rent']+300}以下",
                f"存錢中 希望房租不超過{prop['rent']+200}",
            ]
        # Comfort-seeking personas
        if prop.get("building_type") in ("大樓", "華廈"):
            situational += [
                "不想爬樓梯 一定要有電梯",
                "腿不好 想住有電梯的大樓",
                "搬家方便 大樓優先",
            ]
        # Pet-owner personas
        if any(k in full_text for k in ["可養", "寵物"]):
            situational += [
                "有一隻貓 請問可以養嗎",
                "養了一隻小狗 找可以帶寵物的房",
                "毛孩子跟我一起住 求寵物友善房東",
            ]
        # Work-from-home personas
        if any(k in full_text for k in ["網路", "寬頻"]):
            situational += [
                "在家工作 網路速度很重要",
                "接案族 需要穩定網路和書桌",
                "遠距上班 白天在家 希望安靜且有網路",
            ]
        # Safety-conscious personas
        if any(k in full_text for k in ["保全", "門禁", "監視器"]):
            situational += [
                "女生單獨住 安全最重要",
                "一個人住怕不安全 有門禁比較好",
                "希望有門禁或保全 住得安心",
            ]
        # Subsidy personas
        if any(k in full_text for k in ["租補", "補助"]):
            situational += [
                "想申請租金補貼 房東可以配合嗎",
                "第一次租屋想辦理租屋補助",
                "希望可以報稅 有沒有可租補的",
            ]

        if situational:
            queries.extend(random.sample(situational, min(4, len(situational))))

        # Strategy 7: Negative requirement queries
        # Real users often state what they DON'T want
        negatives_pool = []
        if prop["rent"] and prop["rent"] >= 8000:
            negatives_pool.append("不要太貴 預算有限")
        if prop.get("distance") and prop["distance"] <= 1.5:
            negatives_pool.append("不想騎很遠 最好走路就到")
        negatives_pool += [
            "不要頂加",
            "不要暗房 要有對外窗",
            "不要太吵 附近不要夜市",
        ]
        if negatives_pool:
            chosen_neg = random.choice(negatives_pool)
            # Combine with a positive feature to make it realistic
            if features.get("budget"):
                queries.append(f"{random.choice(features['budget'])} {chosen_neg}")
            else:
                queries.append(chosen_neg)

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


class FeatureEngine:
    """Advanced semantic feature extraction for rental properties."""
    
    @staticmethod
    def extract_all(prop: Dict[str, Any]) -> Dict[str, Any]:
        features = {}
        
        # 1. Billing Type (台電 vs 獨立電表)
        elec = str(prop.get("電費", "")) + " " + str(prop.get("另計費用", ""))
        if "台電" in elec:
            features["billing_type"] = "taipower"
        elif any(kw in elec for kw in ["一度5", "一度6", "一度4"]):
            features["billing_type"] = "fixed"
        else:
            features["billing_type"] = "standard"

        # 2. Service Level (Garbage & Parcels)
        notes = " ".join(prop.get("notes", []) + [prop.get("特色", "")])
        has_garbage = any(kw in notes for kw in ["子母車", "垃圾代收", "垃圾處理", "不用等垃圾車"])
        has_parcel = any(kw in notes for kw in ["管理員", "代收包裹", "包裹代收", "管理室"])
        
        if has_garbage and has_parcel:
            features["service_level"] = "five_star"
        elif has_garbage or has_parcel:
            features["service_level"] = "basic"
        else:
            features["service_level"] = "none"

        # 3. Safety Level
        has_safety = any(kw in notes for kw in ["監視器", "門禁", "保全", "住警器", "磁扣", "感應"])
        features["safety_level"] = "high" if has_safety else "standard"

        # 4. CP Value (Price per Ping)
        try:
            rent_val = str(prop.get("租金", "5000")).replace("元", "").replace(",", "")
            area_val = str(prop.get("室內坪數", "5")).replace("坪", "")
            rent = float(re.search(r"\d+", rent_val).group())
            area = float(re.search(r"\d+", area_val).group())
            
            if area > 0:
                ppp = rent / area
                features["ppp"] = round(ppp, 1)
                # Average in Taichung South Dist is ~800-1200 per ping
                features["cp_tag"] = "high_cp" if ppp < 850 else "standard"
            else:
                features["cp_tag"] = "unknown"
        except:
            features["cp_tag"] = "unknown"

        # 5. Geo Tier
        try:
            dist = float(prop.get("距離(km)", 2.0) or 2.0)
        except:
            dist = 2.0
            
        if dist < 0.5:
            features["geo_tier"] = "core"
        elif dist < 1.5:
            features["geo_tier"] = "active"
        else:
            features["geo_tier"] = "quiet"

        # 6. Aesthetics / Condition
        if any(kw in notes for kw in ["全新", "首租", "第一手"]):
            features["condition"] = "new"
        elif any(kw in notes for kw in ["翻新", "裝潢", "設計師"]):
            features["condition"] = "renovated"
        else:
            features["condition"] = "standard"

        return features


# ============================================================
# 4. Dataset construction & Compatibility Matching
# ============================================================
def is_compatible(query: str, prop: Dict[str, Any]) -> bool:
    """Verifies property constraints to avoid false negatives in training data."""
    for rt in ["套房", "雅房", "住宅"]:
        if rt in query and prop["room_type"] != rt: return False
    
    if (match := re.search(r"(\d+)(?:元)?(?:以下|以內|內)", query)) and prop["rent"] > int(match.group(1)):
        return False
        
    for reg in ["南區", "大里", "東區", "西區", "太平", "西屯", "北屯", "南屯", "北區", "中區"]:
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
    """Computes a graded relevance score (0-3) based on verifiable dimensions.

    Grading Logic:
      - 0: Hard Conflict (gender/room-type mismatch, explicit exclusions, hard budget miss)
      - 1: Partial Match  (≥15% dimensions satisfied)
      - 2: Good Match     (≥65% dimensions satisfied)
      - 3: Perfect Match  (≥85% dimensions satisfied)

    Fixes vs. previous version:
      - Budget direction: "以上" treated as floor (user can pay ≥ X), "以下/以內" as ceiling
      - is_strict only triggers on truly strong keywords; removed "找/想要/需要" which
        appeared in virtually every query, making the flag meaningless
      - satisfied is clamped to ≥ 0 before final ratio (no negative scores)
      - Geo-tier bonus capped so satisfied never exceeds total_specified
    """

    # --- Part A: Hard Conflicts (always return 0) ---

    # 1. Gender restriction
    for note in prop.get("notes", []):
        if "限女" in note and "限男" in query: return 0
        if "限男" in note and "限女" in query: return 0
        if "限男" in note and "限妹子" in query: return 0

    # 2. Room type mismatch
    room_types = ["套房", "雅房", "住宅"]
    query_room = next((rt for rt in room_types if rt in query), None)
    if query_room and query_room not in prop.get("room_type", ""):
        return 0

    # 3. Explicit feature exclusions
    if re.search(r"(謝絕|不要|拒絕|禁|❌|不接受|非)[^。！？\n]*(頂加|加蓋|頂樓|無窗|暗房|漏水|壁癌)", query):
        notes_all = " ".join(prop.get("notes", [])) + prop.get("building_type", "") + prop.get("floor", "")
        if re.search(r"(頂加|加蓋|頂樓|無窗|暗房|漏水|壁癌)", notes_all):
            return 0

    # --- Part B: Dimension Scoring ---
    satisfied = 0.0
    total_specified = 0

    # is_strict: only truly emphatic phrasing — NOT generic "找/想要" which appear everywhere
    is_strict = any(kw in query for kw in [
        "一定要", "絕對", "必須", "絕對不要", "謝絕", "禁止", "急尋", "求租", "【", "#", "＃"
    ])

    # --- 必要條件硬性衝突（優先於所有其他評分，直接返回 0）---
    # 當查詢含明確「必要條件」關鍵字，且 property 明確違反時，不給任何分數
    _MANDATORY_CHECKS = [
        # (query 觸發詞列表, property 違反判斷函式)
        (
            ["一定要電梯", "必須有電梯", "不爬樓梯", "不想爬樓梯", "不要爬樓梯", "要有電梯", "腿不好", "膝蓋不好"],
            lambda p: "電梯" not in " ".join(p.get("furniture", [])) and p.get("building_type", "") not in ("大樓", "華廈"),
        ),
        (
            ["獨洗", "獨立洗衣", "一定要獨洗", "不共用洗衣機"],
            lambda p: not any(kw in " ".join(p.get("furniture", [])) for kw in ["洗衣機", "獨洗", "個人洗衣"]),
        ),
        (
            ["一定要開伙", "必須能煮飯", "要有廚房", "自炊族", "想在家煮飯"],
            lambda p: not any(kw in " ".join(p.get("notes", [])) + " ".join(p.get("furniture", [])) for kw in ["開伙", "瓦斯", "廚房", "自炊"]),
        ),
        (
            ["禁菸", "不接受抽菸", "怕菸味"],
            lambda p: any("可菸" in n or "允許抽菸" in n for n in p.get("notes", [])),
        ),
        (
            ["走路就到", "不騎車", "步行可達", "走路五分鐘以內"],
            lambda p: (p.get("walk_mins", 99) or 99) > 20,
        ),
    ]
    # 注意：只在查詢包含強制性語氣時才觸發（避免過度懲罰普通提及）
    if is_strict or any(kw in query for kw in ["一定要", "必須", "謝絕", "禁"]):
        for trigger_kws, violation_fn in _MANDATORY_CHECKS:
            if any(kw in query for kw in trigger_kws) and violation_fn(prop):
                return 0

    adv = FeatureEngine.extract_all(prop)

    # 1. Budget — direction-aware
    #    "以上"       → floor: user willing to pay ≥ X  (rent must be ≥ X)
    #    "以下/以內/內" → ceiling: user budget cap ≤ X   (rent must be ≤ X)
    #    bare number   → treat as ceiling (most natural renter phrasing)
    budget_match = re.search(r"(\d{4,5})\s*元?\s*(以上|以下|以內|內|左右|上下)?", query)
    if budget_match:
        total_specified += 1
        budget_val = int(budget_match.group(1))
        direction = budget_match.group(2) or ""
        rent = prop["rent"]

        if direction == "以上":
            # user wants a property priced at ≥ budget_val
            if rent >= budget_val:
                satisfied += 1
            elif rent >= budget_val * 0.9:   # within 10% below floor → soft pass
                satisfied += 0.5
            else:
                return 0   # too cheap — doesn't meet stated minimum
        else:
            # ceiling mode (以下 / 以內 / bare number)
            if rent <= budget_val:
                satisfied += 1
            elif rent <= budget_val * 1.1:   # within 10% over budget → soft penalty
                if is_strict: return 0
                satisfied += 0.3
            else:
                return 0   # clearly over budget

    # 2. Features / Furniture
    features_needed = [feat for feat, terms in Templates.FURNITURE.items()
                       if any(t in query for t in terms)]
    if features_needed:
        total_specified += 1
        has_furniture = prop.get("furniture", [])
        found_count = sum(1 for feat in features_needed
                          if any(feat in f for f in has_furniture))
        if is_strict and found_count < len(features_needed):
            return 0
        satisfied += found_count / len(features_needed)

    # 2.5 Lifestyle intents
    lifestyle_intents = {
        "懶人":   ["電梯", "子母車", "飲水機"],
        "自炊":   ["開火", "瓦斯爐", "廚房"],
        "潔癖":   ["全新", "禁菸", "獨洗"],
        "外送":   ["飲水機", "管理員", "包裹"],
        "怕悶熱": ["陽台", "對外窗", "採光"],
    }
    for intent, reqs in lifestyle_intents.items():
        if intent in query:
            total_specified += 1
            prop_text = " ".join(prop.get("furniture", [])) + " ".join(prop.get("notes", []))
            match_count = sum(1 for r in reqs if r in prop_text)
            if is_strict and match_count == 0: return 0
            satisfied += match_count / len(reqs)

    # 3. Location / Region
    region_specified = next(
        (reg for reg in ["南區", "大里", "東區", "西區", "北區", "烏日",
                         "太平", "西屯", "北屯", "南屯", "中區"] if reg in query),
        None,
    )
    roads_in_query = re.findall(
        r"([^區市台]*(?:路|街|大道)(?:[一二三四五六七八九十]|[\d])?段?)", query
    )
    if region_specified or roads_in_query:
        total_specified += 1
        loc_match = False
        if region_specified and (
            region_specified in prop.get("address", "") or
            region_specified in prop.get("region", "")
        ):
            loc_match = True
        if roads_in_query and any(road in prop.get("address", "") for road in roads_in_query):
            loc_match = True
        if loc_match:
            # geo_tier bonus: 0.0 (no extra) to 0.15 (core district), capped at 1.0
            bonus = 0.15 if adv["geo_tier"] == "core" else 0.0
            satisfied += min(1.0, 1.0 + bonus)
        # loc mismatch: no penalty (just 0 added), already hurts score_ratio

    # 4. Pet constraint
    if any(k in query for k in ["寵物", "貓", "狗", "毛孩", "主子"]):
        total_specified += 1
        note_str = " ".join(prop.get("notes", []))
        if "可寵" in note_str or "可養寵" in note_str:
            satisfied += 1
        elif "禁寵" in note_str or "不寵" in note_str:
            return 0
        else:
            # Not mentioned: penalise in strict mode, partial credit otherwise
            if is_strict: return 0
            satisfied += 0.2

    # 5. Trash / service level
    if any(k in query for k in ["垃圾", "子母車", "下班晚", "追垃圾車", "管理員"]):
        total_specified += 1
        if adv["service_level"] == "five_star":
            satisfied += 1
        elif adv["service_level"] == "basic":
            satisfied += 0.7
        else:
            if is_strict: return 0
            satisfied += 0.1

    # 6. Billing / electricity
    if any(kw in query for kw in ["省錢", "台電", "台水", "帳單", "自繳"]):
        total_specified += 1
        note_str = " ".join(prop.get("notes", []))
        if adv["billing_type"] == "taipower" or any(
            k in note_str for k in ["台電", "台水", "帳單", "自繳"]
        ):
            satisfied += 1
        else:
            if is_strict: return 0
            # Non-taipower is a soft miss, not a negative
            satisfied += 0.0

    # 7. Cooking
    if any(kw in query for kw in ["煮飯", "開火", "自炊", "下廚", "瓦斯", "廚房"]):
        total_specified += 1
        all_notes = " ".join(prop.get("notes", [])) + prop.get("building_type", "")
        if any(k in all_notes for k in ["開火", "瓦斯", "廚房", "自炊", "下廚"]):
            satisfied += 1
        else:
            if is_strict: return 0
            satisfied += 0.0   # hard miss without penalising to negative

    # 8. Safety
    if any(kw in query for kw in ["安全", "女性", "監視器"]):
        total_specified += 1
        if adv["safety_level"] == "high": satisfied += 1

    # 9. Aesthetics / condition
    if any(kw in query for kw in ["漂亮", "質感", "新", "裝潢", "設計"]):
        total_specified += 1
        if adv["condition"] == "new": satisfied += 1
        elif adv["condition"] == "renovated": satisfied += 0.8

    # --- Part C: Final mapping ---
    if total_specified == 0:
        return 2   # no verifiable constraints → assume decent but not perfect

    # Clamp to [0, total_specified] to prevent negative ratios
    satisfied = max(0.0, min(satisfied, float(total_specified)))
    score_ratio = satisfied / total_specified

    if score_ratio >= 0.85: return 3
    if score_ratio >= 0.65: return 2
    if score_ratio >= 0.15: return 1
    return 0


def create_dataset_pairs(properties: List[Dict[str, Any]], neg_per_pos: int = 2) -> List[Dict[str, Any]]:
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

    # ============================================================
    # Object-level Split: 先切割物件，再生成樣本
    # 確保同一租屋物件的查詢對不會同時出現在訓練集與測試集中
    # 違反此原則將導致 Data Leakage，NDCG 分數嚴重虛高
    # ============================================================
    prop_indices = list(range(len(properties)))
    random.shuffle(prop_indices)
    n = len(prop_indices)
    train_prop_idx = set(prop_indices[:int(n * 0.8)])
    dev_prop_idx   = set(prop_indices[int(n * 0.8):int(n * 0.9)])
    test_prop_idx  = set(prop_indices[int(n * 0.9):])

    prop_texts = [property_to_text(p) for p in properties]

    print(f"\nStep 2: Generating query-property pairs (Object-level Split)...")
    print(f"  Property split: {len(train_prop_idx)} train / {len(dev_prop_idx)} dev / {len(test_prop_idx)} test")

    # ============================================================
    # 語意衝突負樣本偵測表
    # 偵測查詢中的「條件聲明」，並挑選「表面優質但恰好違反此條件」的房源
    # 這強迫模型學習「禁/不/謝絕」等否定語意的語義權重
    # ============================================================
    CONFLICT_DIMENSIONS = [
        # --- 原有四個維度 ---
        {
            "query_keywords": ["可養寵", "可寵", "可以養貓", "可以養狗", "帶貓", "帶狗", "有寵物", "2隻貓", "3隻貓", "毛孩子", "主子"],
            "prop_conflict":  lambda p: any("禁寵" in n or "禁止寵物" in n or "不可養寵" in n for n in p.get("notes", [])),
            "conflict_type": "pet",
        },
        {
            "query_keywords": ["謝絕頂加", "不要頂加", "禁頂加", "非頂加", "不接受頂加", "不找頂加"],
            "prop_conflict":  lambda p: "頂加" in p.get("building_type", "") or "加蓋" in p.get("floor", ""),
            "conflict_type": "rooftop",
        },
        {
            "query_keywords": ["限女", "女生專用", "女學生", "僅限女", "女性優先"],
            "prop_conflict":  lambda p: any("限男" in n for n in p.get("notes", [])),
            "conflict_type": "gender_female",
        },
        {
            "query_keywords": ["限男", "男生", "僅限男", "男性優先"],
            "prop_conflict":  lambda p: any("限女" in n for n in p.get("notes", [])),
            "conflict_type": "gender_male",
        },
        # --- 新增：台電計費衝突（查詢要求台電帳單，房源為自訂電費）---
        {
            "query_keywords": ["台電", "台水台電", "自繳電費", "照台電收", "台電帳單", "電費依台電"],
            "prop_conflict":  lambda p: any(
                re.search(r"一度[456789]", n) or "電費另計" in n or "代收電費" in n
                for n in p.get("notes", [])
            ),
            "conflict_type": "electricity_billing",
        },
        # --- 新增：禁菸衝突（查詢禁菸，房源允許抽菸）---
        {
            "query_keywords": ["禁菸", "不抽菸", "不接受抽菸", "怕菸味", "無菸"],
            "prop_conflict":  lambda p: any("可菸" in n or "允許抽菸" in n for n in p.get("notes", [])),
            "conflict_type": "smoking",
        },
        # --- 新增：電梯必要衝突（查詢明確要電梯，房源無電梯）---
        {
            "query_keywords": ["一定要電梯", "必須有電梯", "不爬樓梯", "不想爬樓梯", "不要爬樓梯", "要有電梯", "腿不好", "膝蓋不好"],
            "prop_conflict":  lambda p: "電梯" not in " ".join(p.get("furniture", [])) and p.get("building_type", "") not in ("大樓", "華廈"),
            "conflict_type": "elevator_required",
        },
        # --- 新增：開伙必要衝突（查詢明確要能煮飯，房源禁止開伙）---
        {
            "query_keywords": ["一定要開伙", "必須能煮飯", "要有廚房", "喜歡自己煮", "自炊族", "想在家煮飯"],
            "prop_conflict":  lambda p: not any(
                kw in " ".join(p.get("notes", [])) + " ".join(p.get("furniture", []))
                for kw in ["開伙", "瓦斯", "廚房", "自炊"]
            ),
            "conflict_type": "cooking_required",
        },
        # --- 新增：距離衝突（查詢要求走路可達，房源距離太遠）---
        {
            "query_keywords": ["走路就到", "不騎車", "走路上學", "步行可達", "不想騎車", "走路五分鐘"],
            "prop_conflict":  lambda p: (p.get("walk_mins", 99) or 99) > 20,
            "conflict_type": "distance_walking",
        },
        # --- 新增：獨立洗衣衝突（查詢明確要獨洗，房源無獨洗）---
        {
            "query_keywords": ["獨洗", "獨立洗衣", "不共用洗衣", "不想共用洗衣機", "一定要獨洗"],
            "prop_conflict":  lambda p: not any(
                kw in " ".join(p.get("furniture", []))
                for kw in ["洗衣機", "獨洗", "個人洗衣"]
            ),
            "conflict_type": "laundry_private",
        },
    ]
    # Precompute property character sets for faster Jaccard sorting
    prop_char_sets = [set(t) for t in prop_texts]

    def generate_samples_for_split(prop_idx_set):
        """為指定的物件子集生成樣本，負樣本只從同子集內的不相容物件中挑選。"""
        samples = []
        # Convert set to list for stable sampling
        pool = list(prop_idx_set)
        
        for idx in prop_idx_set:
            prop = properties[idx]
            queries = QueryGenerator.build_queries(prop, num_queries=12)
            for query in queries:
                relevance = compute_relevance_score(query, prop)
                samples.append({"query": query, "property": prop_texts[idx], "label": 1, "relevance": relevance, "property_idx": idx})

                other_in_split = [i for i in pool if i != idx]
                query_chars = set(query)

                # --- 策略一：Jaccard 字面困難負樣本 ---
                # 優化：先過濾不相容，再排序
                incompatible = [i for i in other_in_split if not is_compatible(query, properties[i])]
                if incompatible:
                    # 使用預先計算好的 char sets 加速
                    incompatible.sort(key=lambda i: len(query_chars & prop_char_sets[i]), reverse=True)
                    for neg_idx in incompatible[:2]:
                        samples.append({"query": query, "property": prop_texts[neg_idx], "label": 0, "relevance": 0, "property_idx": neg_idx})
                
                # --- 策略一點五：隨機簡單負樣本 ---
                if incompatible:
                    # 精簡：隨機簡單負樣本對模型幫助不大，只留 1 個即可
                    random_negs = random.sample(incompatible, min(1, len(incompatible)))
                    for neg_idx in random_negs:
                        samples.append({"query": query, "property": prop_texts[neg_idx], "label": 0, "relevance": -1, "property_idx": neg_idx})

                # --- 策略二：語意誤導型負樣本（條件衝突，非字面不符）---
                for dim in CONFLICT_DIMENSIONS:
                    if any(kw in query for kw in dim["query_keywords"]):
                        semantic_neg_cands = [
                            i for i in other_in_split
                            if dim["prop_conflict"](properties[i])
                        ]
                        if semantic_neg_cands:
                            # 優先挑租金較低（看起來更吸引人）的語意陷阱
                            semantic_neg_cands.sort(key=lambda i: properties[i].get("rent", 9999))
                            ctype = dim.get("conflict_type", "semantic")
                            for chosen in semantic_neg_cands[:2]:
                                samples.append({"query": query, "property": prop_texts[chosen], "label": 0, "relevance": 0, "property_idx": chosen, "conflict_type": ctype})
                        break  # 每個查詢只加一類語意衝突負樣本，但該類生成2個

                # --- 策略三：必要條件預算上限負樣本 ---
                # 當查詢明確設定預算上限，強制配對租金超出上限的房源
                # 這類樣本表面上相似（同區域/設施），但模型必須學會拒絕超預算房源
                budget_ceil_match = re.search(r"(\d{4,5})\s*元?\s*(以下|以內|內)", query)
                if budget_ceil_match:
                    budget_ceil = int(budget_ceil_match.group(1))
                    # 找出租金明顯超出上限（>10%）的房源，優先選租金與上限接近的（最難的）
                    over_budget = [
                        i for i in other_in_split
                        if properties[i].get("rent", 0) > budget_ceil * 1.1
                    ]
                    if over_budget:
                        # 按租金升序排列：選最接近預算上限的（最難的負樣本）
                        over_budget.sort(key=lambda i: properties[i].get("rent", 9999))
                        for chosen in over_budget[:2]:
                            samples.append({"query": query, "property": prop_texts[chosen], "label": 0, "relevance": 0, "property_idx": chosen, "conflict_type": "budget_ceiling"})

        random.shuffle(samples)
        return samples


    train_data = generate_samples_for_split(train_prop_idx)
    dev_data   = generate_samples_for_split(dev_prop_idx)
    test_data  = generate_samples_for_split(test_prop_idx)

    external_query_files = [
        "../../data/raw/fb_queries.json",
        "../../data/raw/llm_queries.json",
        "../../data/raw/mined_hard_negatives.json",
        "../../data/raw/silver_labeled_queries.json",
        "../../data/raw/budget_hard_traps.json",
        "../../data/raw/hard_traps.json"
    ]
    
    total_external_samples = 0
    for filename in external_query_files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            print(f"\nStep 2.5: Injecting external queries from {os.path.basename(filename)}...")
            with open(filepath, "r", encoding="utf-8") as f:
                external_queries = json.load(f)

            external_samples = []
            for item in external_queries:
                # [New Logic] Handle pre-paired LLM samples or Mined Hard Negatives
                if isinstance(item, dict) and "query" in item:
                    # Support both "property" (text) and "property_text" fields
                    p_text = item.get("property") or item.get("property_text")
                    if p_text:
                        record = {
                            "query": item["query"],
                            "property": p_text,
                            "label": item.get("label", 0),
                            "relevance": item.get("relevance", 0),
                            "is_hard": item.get("is_hard", True),
                        }
                        # Propagate conflict_type so trainer applies 2x loss weight
                        if "conflict_type" in item:
                            record["conflict_type"] = item["conflict_type"]
                        elif item.get("category"):
                            _CAT_TO_CTYPE = {
                                "自炊衝突": "cooking_required",
                                "女生安全衝突": "gender_female",
                                "衛浴獨立衝突": "laundry_private",
                                "寵物衝突": "pet",
                                "停車衝突": "distance_walking",
                                "費用衝突": "electricity_billing",
                                "設備衝突": "elevator_required",
                            }
                            ctype = _CAT_TO_CTYPE.get(item["category"])
                            if ctype:
                                record["conflict_type"] = ctype
                        external_samples.append(record)
                # [Legacy Logic] Handle list of query strings
                elif isinstance(item, str):
                    query = item
                    pos_found = False
                    for idx in train_prop_idx:
                        if is_compatible(query, properties[idx]):
                            rel = compute_relevance_score(query, properties[idx])
                            external_samples.append({"query": query, "property": prop_texts[idx], "label": 1, "relevance": rel, "is_hard": False})
                            pos_found = True
                            break
                    if pos_found:
                        cands = list(train_prop_idx)
                        random.shuffle(cands)
                        for neg_idx in cands:
                            if not is_compatible(query, properties[neg_idx]):
                                external_samples.append({"query": query, "property": prop_texts[neg_idx], "label": 0, "relevance": 0, "is_hard": True})
                                break

            print(f"  Generated {len(external_samples)} pairs from {os.path.basename(filename)}.")
            train_data.extend(external_samples)
            total_external_samples += len(external_samples)
            
    if total_external_samples > 0:
        random.shuffle(train_data)

    pos = sum(1 for s in train_data if s["label"] == 1)
    neg = len(train_data) - pos
    print(f"\n  Train: {len(train_data)} samples (Pos: {pos}, Neg: {neg}, Ratio 1:{neg/pos:.1f})")
    print(f"  Dev:   {len(dev_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    print(f"\nStep 3: Saving datasets...")
    def clean(samples):
        out = []
        for s in samples:
            record = {
                "query": s["query"],
                "property": s["property"],
                "label": s["label"],
                "relevance": s.get("relevance", s["label"]),
                "is_hard": s.get("is_hard", False),
            }
            if "conflict_type" in s:
                record["conflict_type"] = s["conflict_type"]
            out.append(record)
        return out

    for filename, subset in zip(
        ["../../data/processed/recommendation_train.json", "../../data/processed/recommendation_dev.json", "../../data/processed/recommendation_test.json"],
        [train_data, dev_data, test_data]
    ):
        out_path = os.path.join(os.path.dirname(__file__), filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clean(subset), f, ensure_ascii=False, indent=2)

    prop_out = os.path.join(os.path.dirname(__file__), "../../data/processed/property_texts.json")
    with open(prop_out, "w", encoding="utf-8") as f:
        json.dump(prop_texts, f, ensure_ascii=False, indent=2)

    print("\nDataset generation complete!")
    print(f"  Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")

if __name__ == "__main__":
    main()

