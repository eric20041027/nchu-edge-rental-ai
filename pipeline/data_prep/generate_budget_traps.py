import json
import os

def generate_budget_traps():
    traps = []
    # Pattern: Low Budget Query vs High Rent Property (with matching amenities)
    budgets = [4000, 5000, 6000, 7000]
    amenities = ["獨立洗衣機", "獨洗", "陽台", "電梯", "台電計費"]
    
    for b in budgets:
        for a in amenities:
            # Case 1: Perfect amenities but rent is way over
            traps.append({
                "query": f"想找{b}以內，要有{a}的套房",
                "property": f"台中市南區 質感裝潢套房 附{a} 租金{b + 5000}元/月 近興大",
                "relevance": 0,
                "label": 0,
                "reason": "budget_conflict"
            })
            # Case 2: Perfect budget but missing the critical amenity
            traps.append({
                "query": f"預算{b}，一定要有{a}",
                "property": f"超划算套房 租金只要{b-500}元 採光好 地點優 (無{a}，需使用公共設施)",
                "relevance": 0,
                "label": 0,
                "reason": "amenity_conflict"
            })

    output_path = "data/raw/budget_hard_traps.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(traps, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(traps)} budget and amenity hard traps.")

if __name__ == "__main__":
    generate_budget_traps()
