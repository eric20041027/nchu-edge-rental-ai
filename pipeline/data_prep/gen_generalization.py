"""階段④ 泛化訓練 query 生成器(特徵驅動,可重跑)。

問題:bi-encoder「假泛化」來自訓練資料 73.5% 重複模板 + 102 條同義詞表。
解法:每個結構化特徵維度(走路近/便宜/電梯/透天…)的 ground-truth 由
property_data 欄位**客觀算出**(非憑感覺標 idx),再配多套口語/隱喻/跨域類比
表達模板,程式笛卡兒組合出多樣 query。GT 可驗、生成可重跑、零標點。

設計:
  FEATURES — 每維度 = {gt: 資料 predicate, templates: 多 gen_type 表達}。
  生成器對每維度套所有模板 → 正樣本;對「硬衝突」維度配 is_hard 負樣本。
  holdout 用獨立 HOLDOUT_TEMPLATES(風格刻意不同),生成時就隔離,絕不進訓練。

產物(對齊 tests/eval_generalization.py --check 的 schema):
  data/processed/generalization_queries.json  — 訓練(query/property/label/is_hard)
  tests/fixtures/generalization_eval.json      — 評估(query/relevant_idxs + meta caveat)
  tests/fixtures/generalization_holdout.json   — holdout(隔離)

用法:
    python pipeline/data_prep/gen_generalization.py            # 生成(預設規模)
    python pipeline/data_prep/gen_generalization.py --n 1000   # 目標訓練 query 數
    python tests/eval_generalization.py --check                # 生成後驗
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
PD = json.loads((ROOT / "frontend/assets/property_data.json").read_text(encoding="utf-8"))
BY_IDX = {p["idx"]: p for p in PD if "idx" in p}


# 同源鐵則:property 文字必須與上線 embedding 的 field 對齊。
# 查證(frontend/assets/property_embeddings.json: field='text')→ 上線 embedding 用 `text`。
# 若這裡用 ce_text(資訊較多),bi-encoder 會學 query↔ce_text 映射,但線上比對的是
# text 向量 → 分布偏移破同源。故 property 文字用 `text`,對齊上線 embedding。
def prop_text(idx: int) -> str:
    p = BY_IDX[idx]
    return p.get("text") or ""


# ── GT predicates:全部從 property_data 欄位客觀算 ─────────────────────────
CHEAP_THR = 7000  # rent p25(便宜桶)

def _walk(thr: int) -> list[int]:
    return [p["idx"] for p in PD if isinstance(p.get("walk_mins"), (int, float)) and p["walk_mins"] <= thr]

def _scooter(thr: int) -> list[int]:
    return [p["idx"] for p in PD if isinstance(p.get("scooter_mins"), (int, float)) and p["scooter_mins"] <= thr]

def _flag(field: str) -> list[int]:
    return [p["idx"] for p in PD if p.get(field) is True]

def _cheap() -> list[int]:
    return [p["idx"] for p in PD if 0 < p.get("rent", 0) <= CHEAP_THR]

def _btype(kw: str) -> list[int]:
    return [p["idx"] for p in PD if kw in str(p.get("building_type", ""))]

def _zhengceng() -> list[int]:
    return [p["idx"] for p in PD if "整層" in str(p.get("building_type", "")) or p.get("room_type") == "住宅"]

def _geo(tier: str) -> list[int]:
    return [p["idx"] for p in PD if p.get("geo_tier") == tier]


# ── 特徵維度:gt(客觀 GT)+ templates(多 gen_type 口語表達,零標點)──────────
# templates 每條 = (query 文字, gen_type)。表達刻意涵蓋口語/隱喻/跨域/生活推理/negation。
FEATURES: dict[str, dict] = {
    "walk_near": {
        "gt": lambda: _walk(10),
        "templates": [
            ("走路就到學校 懶得騎車", "口語"),
            ("睡到最後一刻再衝去上課的那種近", "隱喻"),
            ("出門幾步路就是校門口", "口語"),
            ("根本住在學校隔壁的感覺", "隱喻"),
            ("不用通勤直接走過去就好", "生活推理"),
            ("離興大近到可以中午回來睡午覺", "生活推理"),
            ("校門口走路五分鐘以內", "口語"),
            ("步行可達不必牽車", "口語"),
        ],
    },
    "scooter_near": {
        "gt": lambda: _scooter(7),
        "templates": [
            ("騎車就到不必太近", "口語"),
            ("機車代步幾分鐘的距離沒問題", "口語"),
            ("遠一點沒差有車就好", "生活推理"),
            ("騎個車十分鐘內到校就行", "口語"),
            ("通勤靠機車不挑特別近", "negation"),
            ("有摩托車所以距離我不太計較", "生活推理"),
        ],
    },
    "cheap": {
        "gt": _cheap,
        "templates": [
            ("最便宜能住人就好 學生沒錢", "口語"),
            ("預算很緊只求有地方睡", "生活推理"),
            ("窮學生找最省的", "口語"),
            ("錢包很扁挑最便宜的", "隱喻"),
            ("一個月七千以內的平價房", "口語"),
            ("能省則省租金越低越好", "口語"),
            ("不挑環境便宜第一", "negation"),
        ],
    },
    "elevator": {
        "gt": lambda: _flag("has_elevator"),
        "templates": [
            ("有電梯不用扛行李爬樓", "生活推理"),
            ("懶得爬樓梯要電梯大樓", "口語"),
            ("搬東西方便的有電梯", "生活推理"),
            ("受不了走樓梯一定要電梯", "口語"),
            ("買菜回來不想爬樓", "生活推理"),
        ],
    },
    "balcony": {
        "gt": lambda: _flag("has_balcony"),
        "templates": [
            ("要有陽台曬衣服", "生活推理"),
            ("想種點花草要陽台", "生活推理"),
            ("衣服想曬太陽不要烘的", "生活推理"),
            ("有個小陽台透透氣", "口語"),
        ],
    },
    "parking": {
        "gt": lambda: _flag("has_parking"),
        "templates": [
            ("要有車位停機車", "口語"),
            ("有地方停車不怕被拖吊", "生活推理"),
            ("附停車位的房", "口語"),
            ("車子有地方放的", "口語"),
        ],
    },
    "window": {
        "gt": lambda: _flag("has_window"),
        "templates": [
            ("採光好有對外窗", "口語"),
            ("不要暗房一定要窗戶", "negation"),
            ("有窗戶通風採光的", "口語"),
            ("白天不開燈也亮的房間", "生活推理"),
        ],
    },
    "subsidy": {
        "gt": lambda: _flag("has_subsidy"),
        "templates": [
            ("可以申請租屋補助的", "口語"),
            ("能報租金補貼省一點", "生活推理"),
            ("房東願意配合補助申請", "口語"),
        ],
    },
    "toutian": {
        "gt": lambda: _btype("透天"),
        "templates": [
            ("透天裡的套房安靜唸書", "生活推理"),
            ("獨棟透天的房間", "口語"),
            ("不要大樓喜歡透天厝", "negation"),
            ("透天的住起來像家", "隱喻"),
        ],
    },
    "zhengceng": {
        "gt": _zhengceng,
        "templates": [
            ("一家人要住的大整層", "生活推理"),
            ("整層住家空間夠大", "口語"),
            ("全家搬來要整層的", "生活推理"),
            ("能住一整層不是只有一間", "口語"),
        ],
    },
    "quiet": {
        "gt": lambda: _geo("quiet"),
        "templates": [
            ("安靜環境好好唸書", "生活推理"),
            ("不要吵的清靜地段", "negation"),
            ("讀書需要安靜的住處", "生活推理"),
            ("遠離喧鬧的安靜角落", "隱喻"),
        ],
    },
}

# ── 硬負樣本配對:對某維度 query 配「表面相似但硬衝突」的房源 idx ────────────
# 走路近 query 配「走路最遠」房源、便宜 query 配「最貴」房源 → is_hard 負樣本。
# 衝突 idx 由資料反向算(取該維度 predicate 的補集中最極端者),非憑感覺。
def _hard_conflict(feat: str) -> int | None:
    if feat == "walk_near":  # 配走路最遠
        return max(PD, key=lambda p: p.get("walk_mins", -1))["idx"]
    if feat == "cheap":      # 配最貴
        return max(PD, key=lambda p: p.get("rent", -1))["idx"]
    if feat == "elevator":   # 配無電梯
        no = [p["idx"] for p in PD if p.get("has_elevator") is False]
        return no[0] if no else None
    return None


# ── 評估集 / holdout:風格刻意與訓練不同。GT 同樣資料算。零標點。 ─────────────
# (query, feat_key) — feat_key 指向 FEATURES 的 gt 算 relevant_idxs。
EVAL_TEMPLATES = [
    ("校門口走路就到的套房", "walk_near"),
    ("走路五分鐘以內到學校", "_walk5"),
    ("騎車就到的房不必太近", "scooter_near"),
    ("便宜大碗能住就好", "cheap"),
    ("一個月七千有找的平價房", "cheap"),
    ("懶人要電梯不爬樓", "elevator"),
    ("曬衣服要陽台的", "balcony"),
    ("停機車要有車位", "parking"),
    ("採光好白天不開燈", "window"),
    ("可以報租屋補助的", "subsidy"),
    ("喜歡透天不愛大樓", "toutian"),
    ("全家要住整層的", "zhengceng"),
    ("安靜好唸書的地段", "quiet"),
]

# holdout:更口語/隱喻但真人會打,生成時隔離,絕不進訓練(不用台語/網路梗)。
HOLDOUT_TEMPLATES = [
    ("早上爬不起來越靠近學校越好", "walk_near"),
    ("錢不多找最便宜能住的", "cheap"),
    ("下班只想放鬆不想再爬樓梯", "elevator"),
    ("想曬棉被要有外面空間", "balcony"),
    ("白天房間不要黑漆漆的", "window"),
    ("讀書的人受不了吵", "quiet"),
    ("住起來有家的感覺的透天", "toutian"),
]


def _resolve_eval(key: str) -> list[int]:
    if key == "_walk5":
        return _walk(5)
    return FEATURES[key]["gt"]()


def build_train(target_n: int) -> list[dict]:
    """每條表達 query 綁該維度 GT 的多個房源 → 正樣本 pair(bi-encoder 學 query↔房源群)。

    放量靠「query × GT 房源」而非複製模板:獨特 query 數固定(表達多樣性),
    但每條綁 per_q 間 GT 房源,自然展開到 target_n。每 pair 的 GT 都資料算。
    per_q 依 target_n / 總表達數動態定,平均分攤到各維度。
    """
    rows: list[dict] = []
    hold_q = {q for q, _ in HOLDOUT_TEMPLATES}
    # 先收集所有(feat, query, gt_list),過 holdout 隔離 + 去重。
    units: list[tuple[str, str, list[int]]] = []
    seen_q: set[str] = set()
    for feat, spec in FEATURES.items():
        gt = spec["gt"]()
        if not gt:
            continue
        for q, _gtype in spec["templates"]:
            if q in seen_q or q in hold_q:  # 去重 + holdout 隔離鐵則
                continue
            seen_q.add(q)
            units.append((feat, q, gt))

    # per_q:每條 query 綁幾間 GT 房源(扣掉硬負樣本配額後均攤)。
    per_q = max(1, target_n // max(1, len(units)))
    for feat, q, gt in units:
        for idx in gt[:per_q]:  # 取 GT 前 per_q 間(穩定、可重現)
            rows.append({"query": q, "property": prop_text(idx), "label": 1,
                         "relevance": 2, "is_hard": False, "src_idx": idx, "feat": feat})

    # 硬負樣本:每維度第一條 query 配硬衝突房源(資料反向算)。
    for feat, spec in FEATURES.items():
        conflict = _hard_conflict(feat)
        if conflict is None or not spec["templates"]:
            continue
        q0 = spec["templates"][0][0]
        if q0 not in hold_q:
            rows.append({"query": q0, "property": prop_text(conflict), "label": 0,
                         "relevance": 0, "is_hard": True, "src_idx": conflict, "feat": feat})
    return rows[:target_n] if len(rows) > target_n else rows


CAVEAT = ("評估 query 與訓練 query 皆 Claude 生成,非完美 holdout;數字僅作相對 Δ 與趨勢"
          "判讀,不宣稱絕對泛化。ground-truth 為特徵欄位客觀算 + 用戶抽查。")


def main() -> None:
    ap = argparse.ArgumentParser(description="階段④ 泛化 query 生成器")
    ap.add_argument("--n", type=int, default=1000, help="目標訓練 query 數(上限)")
    args = ap.parse_args()

    train = build_train(args.n)
    out = ROOT / "data/processed/generalization_queries.json"
    out.write_text(json.dumps(train, ensure_ascii=False, indent=1), encoding="utf-8")
    pos = sum(1 for r in train if r["label"] == 1)
    hard = sum(1 for r in train if r["is_hard"])
    print(f"[gen] 訓練 {len(train)} 筆(正 {pos} / 硬負 {hard})→ {out.name}")

    eval_obj = {
        "meta": {"created": "2026-06-23", "source": "特徵驅動生成器(階段④)",
                 "caveat_same_origin": CAVEAT, "selection": "GT 由 property_data 欄位客觀算"},
        "queries": [{"query": q, "bucket": "semantic", "n_relevant": len(_resolve_eval(k)),
                     "relevant_idxs": _resolve_eval(k), "note": k} for q, k in EVAL_TEMPLATES],
    }
    ev = ROOT / "tests/fixtures/generalization_eval.json"
    ev.write_text(json.dumps(eval_obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gen] 評估 {len(EVAL_TEMPLATES)} 筆 → {ev.name}")

    hold_obj = {
        "meta": {"created": "2026-06-23", "isolation": "生成時隔離 絕不進訓練 風格刻意更口語",
                 "caveat_same_origin": CAVEAT},
        "queries": [{"query": q, "n_relevant": len(FEATURES[k]["gt"]()),
                     "relevant_idxs": FEATURES[k]["gt"](), "note": k} for q, k in HOLDOUT_TEMPLATES],
    }
    ho = ROOT / "tests/fixtures/generalization_holdout.json"
    ho.write_text(json.dumps(hold_obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gen] holdout {len(HOLDOUT_TEMPLATES)} 筆 → {ho.name}")


if __name__ == "__main__":
    main()
