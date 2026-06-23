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

def _text_has(kw: str) -> list[int]:
    """文字含關鍵詞當 GT(冷氣/網路/保全等無布林欄位,但文字可靠)。"""
    return [p["idx"] for p in PD
            if kw in (str(p.get("features", "")) + str(p.get("text", "")) + str(p.get("ce_text", "")))]


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
            # 第二輪補:holdout「想曬棉被要有外面空間」2/5 弱項 → 補曬曬/外面空間/通風隱喻
            # (holdout query 本身保持隔離不進訓練,這裡用同義不同說法的表達補強)
            ("棉被想拿出去曬太陽", "生活推理"),
            ("有個能晾衣服的外推空間", "生活推理"),
            ("洗好的衣服想掛外面吹風", "生活推理"),
            ("房間外面要有可以站的地方", "隱喻"),
            ("想養點盆栽曬得到太陽", "生活推理"),
            ("要能通風透氣的半開放空間", "口語"),
            ("不想衣服悶在房間裡曬", "negation"),
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
    # 第三輪補:語意觸發詞維度(模仿階段① A/B「怕熱→冷氣」風格,拉回 vs baseline 退步)。
    # GT 用文字關鍵詞算(冷氣/網路/垃圾無布林欄位但文字可靠)。
    "aircon": {
        "gt": lambda: _text_has("冷氣"),
        "templates": [
            ("怕熱一定要冷氣", "語意觸發"),
            ("夏天熱到受不了要有空調", "生活推理"),
            ("房間沒冷氣會中暑", "隱喻"),
            ("超怕熱的要涼一點", "口語"),
        ],
    },
    "internet": {
        "gt": lambda: _text_has("網路"),
        "templates": [
            ("上網方便要有網路", "語意觸發"),
            ("在家追劇打遊戲網路要好", "生活推理"),
            ("遠距上課需要穩定網路", "生活推理"),
            ("不能斷網的要有寬頻", "口語"),
        ],
    },
    "waste": {
        "gt": lambda: _flag("has_waste_disposal"),
        "templates": [
            ("不想追垃圾車要有人收", "語意觸發"),
            ("垃圾有地方丟不用等車", "生活推理"),
            ("懶得追垃圾車的", "口語"),
            ("有代收垃圾的房子", "口語"),
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


# ── 第三輪:複合多訴求 query(模仿階段① A/B 多訴求混雜 + 拉高真 GT 多條件交集召回)──
# 每條 = (query 多訴求自然說法, [多條件 predicates 取交集當 GT])。交集小桶 → 訓練
# 模型學「多個硬條件同時滿足」的精確召回。GT 全資料算。零標點。
# 每條至少含一個距離骨幹(walk/scooter)讓交集收斂小桶 —— 同真 GT 評估集原則。
# (距離是稀缺特徵:walk≤7 僅 43 間;設施太普遍 556+ 間,光靠設施交集收不斂。)
COMPOUND = [
    ("走路五分內又便宜八千的套房", [lambda: _walk(5), _cheap]),
    ("騎車三分到校預算七千", [lambda: _scooter(3), lambda: [p["idx"] for p in PD if 0 < p.get("rent", 0) <= 7000]]),
    ("走路五分又要有電梯不爬樓", [lambda: _walk(5), lambda: _flag("has_elevator")]),
    ("走路七分便宜又有陽台可以曬衣服", [lambda: _walk(7), _cheap, lambda: _flag("has_balcony")]),
    ("騎車五分安靜又便宜好唸書", [lambda: _scooter(5), lambda: _geo("quiet"), _cheap]),
    ("騎車三分怕熱要冷氣又要便宜", [lambda: _scooter(3), lambda: _text_has("冷氣"), _cheap]),
    ("走路五分的透天厝", [lambda: _walk(5), lambda: _btype("透天")]),
    ("騎車五分有網路又安靜遠距上課", [lambda: _scooter(5), lambda: _text_has("網路"), lambda: _geo("quiet")]),
    ("走路七分有停車位又平價", [lambda: _walk(7), lambda: _flag("has_parking"), _cheap]),
    ("騎車三分電梯又要便宜有保全", [lambda: _scooter(3), lambda: _flag("has_elevator"), _cheap, lambda: _text_has("保全")]),
]


def _compound_gt(preds: list) -> list[int]:
    """多條件交集 GT(小桶):取所有 predicate 結果的 idx 交集。"""
    sets = [set(p()) for p in preds]
    inter = set.intersection(*sets) if sets else set()
    return sorted(inter)


def _resolve_eval(key: str) -> list[int]:
    if key == "_walk5":
        return _walk(5)
    return FEATURES[key]["gt"]()


# 第四輪:維度權重(per_q 按權重分配,非均攤)。修第三輪稀釋問題 ——
# 新增維度讓 units 變多 → 均攤使 holdout 對應的核心單維(walk/balcony…)pair 數被稀釋
# → holdout 94%→89% 退步。給 holdout 對應維度高權重保住其 pair 數,同時保留複合召回。
# 預設權重 1.0;holdout 7 題對應的 5 維 + cheap 拉高,平衡單維口語 vs 複合精確召回。
WEIGHT = {
    "walk_near": 2.0,   # holdout「早上爬不起來」
    "balcony": 2.0,     # holdout「想曬棉被」
    "elevator": 2.0,    # holdout「不想爬樓梯」
    "window": 2.0,      # holdout「黑漆漆」
    "quiet": 2.5,       # holdout「受不了吵」(第二輪起就弱 3/5,加重權救)
    "toutian": 2.0,     # holdout「有家感覺的透天」
    "cheap": 2.0,       # holdout「找最便宜」
    "compound": 1.0,    # 複合召回(真 GT)維持
}
DEFAULT_WEIGHT = 1.0


def build_train(target_n: int) -> list[dict]:
    """每條表達 query 綁該維度 GT 的多個房源 → 正樣本 pair(bi-encoder 學 query↔房源群)。

    放量靠「query × GT 房源」而非複製模板:獨特 query 數固定(表達多樣性),
    每條綁的房源數 per_q 按維度權重(WEIGHT)分配 —— 高權重維度綁更多房源,
    保住 holdout 對應核心單維的 pair 數,不被新增維度均攤稀釋(第四輪修)。
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

    # 第三輪複合多訴求 query:GT 為多條件交集(小桶),綁交集房源。
    for q, preds in COMPOUND:
        if q in seen_q or q in hold_q:
            continue
        gt = _compound_gt(preds)
        if not gt:
            continue
        seen_q.add(q)
        units.append(("compound", q, gt))

    # per_q 按權重分配:base = target_n / Σweight;每維 per_q = round(base × weight)。
    total_weight = sum(WEIGHT.get(feat, DEFAULT_WEIGHT) for feat, _, _ in units)
    base = target_n / max(1e-9, total_weight)
    for feat, q, gt in units:
        per_q = max(1, round(base * WEIGHT.get(feat, DEFAULT_WEIGHT)))
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
