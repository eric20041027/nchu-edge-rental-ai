"""stage4 第三輪 Task1 — 降採樣模板化重複正樣本,騰訓練訊號給設施隱喻標靶。

查證(本機 recommendation_train.json):
  正樣本 7022 筆只有 381 種設施骨架;單一骨架「冰箱 冷氣 床 桌子 椅子 可養其他寵物」
  佔 31%,前 2 種佔 43%。模型反覆看幾乎一樣的正樣本 → 向量擠向模板中心(塌縮),
  新標靶(台電/報稅)少量訊號被淹沒。

策略(溫和、可逆):正樣本按「設施骨架」分組,每組上限 CAP(預設 120),超過確定性
抽樣保留;負樣本、unique 正樣本、低頻骨架全不動。降的是雜訊不是訊號。

⚠ 實證警語(2026-06-28,stage4 第三輪複核):此降採樣【誤砍 ab_eval semantic
GT 正樣本】—— 129 個 semantic GT 房源中 65 個(50%)的正樣本被砍(模板骨架正好是
這些 query 的答案房源)→ 重訓後 ab_eval semantic R@30 0.46→0.25 崩盤。
**降採樣是重訓退步的元兇,非補 text。** 若再用,須先保護 ab_eval GT 房源的正樣本
(或乾脆不降採樣,只靠補 text + 設施 pair)。本輪 round3b notebook 的 B 路徑改不降採樣。

骨架 = property text 去掉地名/路名/價格/房型後的設施詞序列(同骨架=設施組成幾乎相同)。

用法:
    python pipeline/data_prep/balance_train_data.py            # CAP=120
    python pipeline/data_prep/balance_train_data.py --cap 80   # 調上限
產物:data/processed/recommendation_train_balanced.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/processed/recommendation_train.json"
OUT = ROOT / "data/processed/recommendation_train_balanced.json"
DEFAULT_CAP = 120
SEED = 42  # 確定性抽樣,可重現(沿用 trainer _seed_everything 慣例)


def skeleton(text: str) -> str:
    """property text → 設施骨架(去地名/路名/價格/房型,留設施詞)。"""
    t = re.sub(r"\d+元?", "", text)
    t = re.sub(r"(套房|雅房|工作室|公寓|電梯大樓|大樓|透天厝|透天|住宅|獨立|分租|整層)", "", t)
    # 去含 區/路/街/段/里 的地名詞
    t = " ".join(w for w in t.split() if not re.search(r"區|路|街|段|里", w))
    return t.strip()


def balance(rows: list[dict], cap: int) -> tuple[list[dict], dict]:
    """每設施骨架的正樣本上限 cap;負樣本與其他全保留。回 (balanced_rows, report)。"""
    import random
    rng = random.Random(SEED)

    positives_by_skel: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("label") == 1:
            positives_by_skel[skeleton(r["property"])].append(i)

    drop: set[int] = set()
    capped_groups = 0
    for skel, idxs in positives_by_skel.items():
        if len(idxs) > cap:
            capped_groups += 1
            keep = set(rng.sample(idxs, cap))
            drop.update(i for i in idxs if i not in keep)

    balanced = [r for i, r in enumerate(rows) if i not in drop]
    pos_before = sum(1 for r in rows if r.get("label") == 1)
    pos_after = sum(1 for r in balanced if r.get("label") == 1)
    report = {
        "cap": cap, "total_before": len(rows), "total_after": len(balanced),
        "positives_before": pos_before, "positives_after": pos_after,
        "positives_dropped": len(drop), "skeletons": len(positives_by_skel),
        "groups_capped": capped_groups,
    }
    return balanced, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    args = ap.parse_args()

    rows = json.loads(SRC.read_text(encoding="utf-8"))
    balanced, rep = balance(rows, args.cap)
    OUT.write_text(json.dumps(balanced, ensure_ascii=False), encoding="utf-8")

    print(f"降採樣模板化正樣本 (CAP={rep['cap']})")
    print("-" * 56)
    print(f"  總筆數    : {rep['total_before']} → {rep['total_after']}")
    print(f"  正樣本    : {rep['positives_before']} → {rep['positives_after']}"
          f"  (砍 {rep['positives_dropped']})")
    print(f"  設施骨架  : {rep['skeletons']} 種,其中 {rep['groups_capped']} 種超過上限被降")
    print(f"  負樣本/unique 正樣本: 不動")
    print(f"→ {OUT.name}")


if __name__ == "__main__":
    main()
