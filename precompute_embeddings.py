"""
precompute_embeddings.py — 預先計算所有房源的模型 logits 分數

針對每筆 CSV 房源，使用訓練好的 ONNX 模型計算推薦分數的基準特徵。
前端只需將使用者查詢送進 ONNX model 配對每間房屋即可。

由於 sentence-pair classification 模型需要 query+property 一起輸入，
這裡預先將房源描述文本存好，前端在推論時逐一配對計算。
"""
import json
import csv
import re

def load_properties(csv_path="nchu_rental_info.csv"):
    """載入 CSV 並產生每筆房源的描述文本"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    properties = []
    for row in rows:
        # 解析租金
        rent_str = row.get("租金", "")
        rent_match = re.search(r"(\d[\d,]*)", rent_str.replace(",", ""))
        rent_num = int(rent_match.group(1)) if rent_match else 0

        dist = float(row.get("距離(km)", "0") or "0")
        furniture = [s.strip() for s in row.get("家具設施", "").split("/") if s.strip()]
        included = [s.strip() for s in row.get("租金包含", "").split("/") if s.strip()]
        notes = [s.strip() for s in row.get("備註", "").split("/") if s.strip()]

        addr = row.get("地址", "")
        region = ""
        for r in ["南區", "大里區", "西區", "東區", "北區", "烏日"]:
            if r in addr:
                region = r
                break

        # 組合描述文本 (與 generate_dataset.py 的 property_to_text 一致)
        parts = []
        room_type = row.get("格局", "")
        building_type = row.get("類型", "")

        if room_type: parts.append(room_type)
        if building_type: parts.append(building_type)
        if region: parts.append(region)
        if rent_num: parts.append(f"{rent_num}元")
        if dist: parts.append(f"距離{dist}km")

        key_furniture = []
        for f in furniture:
            short = f.replace("（電）", "").replace("機車停車位", "機車位").replace("書桌椅", "書桌")
            if short not in key_furniture:
                key_furniture.append(short)
            if len(key_furniture) >= 5:
                break
        if key_furniture:
            parts.append(" ".join(key_furniture))

        if included:
            parts.append("含" + "".join(included[:3]))

        for note in notes:
            if "寵物" in note or "限" in note:
                parts.append(note)

        prop_text = " ".join(parts)

        properties.append({
            "text": prop_text,
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
        })

    return properties


def main():
    print("=" * 60)
    print("Precomputing property data for frontend...")

    properties = load_properties()
    print(f"  Loaded {len(properties)} properties")

    # 儲存前端需要的房源資料 (包含描述文本)
    output = []
    for i, prop in enumerate(properties):
        output.append({
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
        })

    with open("property_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved property_data.json ({len(output)} properties)")
    print("\n--- Sample Descriptions ---")
    for p in output[:3]:
        print(f"  [{p['idx']}] {p['text'][:80]}...")

    print("\n" + "=" * 60)
    print("Done! Frontend will use:")
    print("  - ONNX model for sentence-pair classification")
    print("  - property_data.json for property descriptions")
    print("  - Each query will be paired with all properties at inference time")


if __name__ == "__main__":
    main()
