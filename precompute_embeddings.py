"""
precompute_embeddings.py
Precomputes and normalizes property metadata from CSV for frontend inference.
Extracts core features (e.g., regions, roads) into descriptive strings for Sentence-Pair classification.
"""
import json
import csv
import re
from typing import Dict, List, Any

def process_property_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Parses and normalizes a single CSV row into a structured dictionary."""
    rent_str = row.get("租金", "")
    rent_match = re.search(r"(\d[\d,]*)", rent_str.replace(",", ""))
    rent_num = int(rent_match.group(1)) if rent_match else 0

    dist = float(row.get("距離(km)", "0") or "0")
    furniture = [s.strip() for s in row.get("家具設施", "").split("/") if s.strip()]
    included = [s.strip() for s in row.get("租金包含", "").split("/") if s.strip()]
    notes = [s.strip() for s in row.get("備註", "").split("/") if s.strip()]

    addr = row.get("地址", "")
    region = next((r for r in ["南區", "大里區", "西區", "東區", "北區", "烏日"] if r in addr), "")

    room_type = row.get("格局", "")
    building_type = row.get("類型", "")
    
    road = ""
    road_match = re.search(r"([^區市台]*(?:路|街|大道)(?:[一二三四五六七八九十]|[\d])?段?)", addr)
    if road_match:
        road = re.sub(r"\d+$", "", road_match.group(1).strip())

    parts = [p for p in (room_type, building_type, region, road) if p]
    if rent_num: parts.append(f"{rent_num}元")
    if dist: parts.append(f"距離{dist}km")

    key_furniture = []
    for f in furniture:
        short = f.replace("（電）", "").replace("機車停車位", "機車位").replace("書桌椅", "書桌")
        if short not in key_furniture:
            key_furniture.append(short)
        if len(key_furniture) >= 5: break
            
    if key_furniture: parts.append(" ".join(key_furniture))
    if included: parts.append("含" + "".join(included[:3]))
    parts.extend([note for note in notes if "寵物" in note or "限" in note])

    return {
        "text": " ".join(parts),
        "url": row.get("網址", ""),
        "address": addr,
        "room_type": room_type,
        "building_type": building_type,
        "rent": rent_num,
        "rent_str": rent_str,
        "size": row.get("室內坪數", ""),
        "floor": row.get("樓層", ""),
        "furniture_str": row.get("家具設施", ""),
        "distance": dist,
        "img": row.get("圖片網址", ""),
        "notes": notes,
        "contact": row.get("聯絡人", ""),
        "phone": row.get("電話", ""),
    }

def load_properties(csv_path: str = "nchu_rental_info.csv") -> List[Dict[str, Any]]:
    """Loads CSV data and yields normalized property dictionaries."""
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [process_property_row(row) for row in rows]

def main() -> None:
    print("=" * 60)
    print("Precomputing property data for frontend...")

    properties = load_properties()
    print(f"  Loaded {len(properties)} properties")

    output = [
        {
            "idx": i,
            "text": prop["text"],
            "url": prop["url"],
            "address": prop["address"],
            "room_type": prop["room_type"],
            "building_type": prop["building_type"],
            "rent": prop["rent"],
            "rent_str": prop["rent_str"],
            "size": prop["size"],
            "floor": prop["floor"],
            "furniture": prop["furniture_str"],
            "distance": prop["distance"],
            "img": prop["img"],
            "contact": prop["contact"],
            "phone": prop["phone"],
        }
        for i, prop in enumerate(properties)
    ]

    with open("property_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved property_data.json ({len(output)} properties)")
    print("\n--- Sample Descriptions ---")
    for p in output[:3]:
        print(f"  [{p['idx']}] {p['text'][:80]}...")

if __name__ == "__main__":
    main()
