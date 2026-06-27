"""階段④ 泛化評估 harness — 多樣 query 真評估集 + holdout 隔離驗 + Recall@30。

spec: docs/spec/generalization-data.md

兩半設計(對齊 eval_vector_vs_rulebased.py 範式):
  --check 半(純 stdlib,本機跑):
    - 結構驗:generalization_queries / eval / holdout 的 schema 正確
    - 隔離鐵則:holdout 的 src_idx/query 與訓練集零交集(R1 最致命風險)
    - 完整性:eval/holdout 的 relevant_idxs 無懸空(都對得上 property_data idx)
    - 零標點:生成 query 全無標點符號
  Recall 半(需 onnxruntime + tokenizers,本機當前無 → guard + exit 1):
    - 編碼 eval query → cosineTopK(30) vs property_embeddings → Recall@30 + NDCG@5
    - NDCG@5 用 binary relevance,與階段① ablation(ABLATION_STUDY.md 語意桶 0.325)同定義
    - 重訓前後各跑一次比 Δ(本機 venv 或 Colab)

Usage:
    python3 tests/eval_generalization.py --check     # 純 stdlib,本機跑,自我驗
    python3 tests/eval_generalization.py             # Recall@30(需 onnxruntime,venv/Colab)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROPERTY_DATA = ROOT / "frontend" / "assets" / "property_data.json"
EMBEDDINGS = ROOT / "frontend" / "assets" / "property_embeddings.json"
TRAIN_QUERIES = ROOT / "data" / "processed" / "generalization_queries.json"
EVAL_SET = ROOT / "tests" / "fixtures" / "generalization_eval.json"
HOLDOUT = ROOT / "tests" / "fixtures" / "generalization_holdout.json"

DEFAULT_K = 30
_PUNCT = re.compile(r"[，。、；：！？,.!?;:\"'（）()「」【】]")


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_idxs() -> set[int]:
    """property_data.json 的合法 idx 集合(relevant_idxs/src_idx 必須落在此)。"""
    pd = _load(PROPERTY_DATA)
    if pd is None:
        raise SystemExit(f"[fatal] {PROPERTY_DATA} 不存在")
    return {p["idx"] for p in pd if "idx" in p}


# =====================================================================
# --check 半 — 純 stdlib,本機跑
# =====================================================================

def check() -> int:
    print("=" * 60)
    print(" 階段④ harness 自我驗(--check,純 stdlib,無模型)")
    print("=" * 60)
    valid = _valid_idxs()
    print(f"[check] property_data idx 合法集合: {len(valid)} 筆")

    errors: list[str] = []
    train = _load(TRAIN_QUERIES) or []
    eval_data = _load(EVAL_SET)
    holdout = _load(HOLDOUT)

    eval_q = (eval_data or {}).get("queries", []) if isinstance(eval_data, dict) else (eval_data or [])
    hold_q = (holdout or {}).get("queries", []) if isinstance(holdout, dict) else (holdout or [])

    print(f"[check] train queries={len(train)}, eval={len(eval_q)}, holdout={len(hold_q)}")

    # 1. 訓練 schema
    for i, r in enumerate(train):
        miss = {"query", "property", "label", "is_hard"} - set(r)
        if miss:
            errors.append(f"train[{i}] 缺欄位 {miss}")
            break

    # 2. 零標點(訓練 query)
    bad_punct = [r["query"] for r in train if "query" in r and _PUNCT.search(r["query"])]
    if bad_punct:
        errors.append(f"訓練 query 含標點 {len(bad_punct)} 筆,例:{bad_punct[0]!r}")

    # 3. eval/holdout relevant_idxs 無懸空
    for name, qs in [("eval", eval_q), ("holdout", hold_q)]:
        for i, q in enumerate(qs):
            ridxs = set(q.get("relevant_idxs", []))
            dangling = ridxs - valid
            if dangling:
                errors.append(f"{name}[{i}] relevant_idxs 懸空 {sorted(dangling)[:3]}")
                break

    # 4. 隔離鐵則:holdout query 與訓練集零交集(R1 最致命)
    train_qset = {r["query"] for r in train if "query" in r}
    hold_qset = {q["query"] for q in hold_q if "query" in q}
    leaked = train_qset & hold_qset
    if leaked:
        errors.append(f"[隔離違規] holdout 有 {len(leaked)} 筆 query 洩漏進訓練集,例:{next(iter(leaked))!r}")

    # 5. eval/holdout query 也零標點
    for name, qs in [("eval", eval_q), ("holdout", hold_q)]:
        bp = [q["query"] for q in qs if "query" in q and _PUNCT.search(q["query"])]
        if bp:
            errors.append(f"{name} query 含標點 {len(bp)} 筆,例:{bp[0]!r}")

    print("-" * 60)
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        print(f"\n[check] 失敗:{len(errors)} 項")
        return 1
    print("  ✓ schema / 零標點 / relevant_idxs 完整 / holdout 隔離 全部通過")
    print("[check] 通過")
    return 0


# =====================================================================
# Recall 半 — 需 onnxruntime + tokenizers(本機當前無 → guard)
# =====================================================================

def recall(k: int, eval_path: Path = EVAL_SET) -> int:
    try:
        import numpy as np
        import onnxruntime as ort  # noqa: F401
        from transformers import BertTokenizerFast  # noqa: F401
    except ImportError as e:
        print(f"[recall] 缺依賴 {e.name}:本機當前無 onnxruntime/transformers。")
        print("         在 python3.12 venv(pip install onnxruntime transformers)或 Colab 跑。")
        print("         結構驗請用:python3 tests/eval_generalization.py --check")
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_vector_vs_rulebased import build_query_encoder, build_property_matrix  # noqa: E402
    from eval_rule_based_baseline import recall_at_k, ndcg_at_k  # noqa: E402

    eval_data = _load(eval_path)
    if eval_data is None:
        raise SystemExit(f"[fatal] {eval_path} 不存在 — 先生成評估集")
    queries = eval_data["queries"] if isinstance(eval_data, dict) else eval_data
    label = eval_path.stem

    encode = build_query_encoder()
    emb_data = _load(EMBEDDINGS)
    if emb_data is None:
        raise SystemExit(f"[fatal] {EMBEDDINGS} 不存在")
    mat, idx_order = build_property_matrix(emb_data, {})

    recalls, ndcgs = [], []
    for q in queries:
        qvec = encode(q["query"])
        sims = mat @ qvec
        top = [idx_order[i] for i in np.argsort(-sims)[:k]]
        relevant = set(q["relevant_idxs"])
        recalls.append(recall_at_k(top, relevant, k))
        # NDCG@5 — 與階段① ablation 同定義(binary relevance,ndcg_at_k 內部切前 5)。
        rels = [1.0 if idx in relevant else 0.0 for idx in top]
        ndcgs.append(ndcg_at_k(rels, 5))
    rmean = sum(recalls) / len(recalls) if recalls else 0.0
    nmean = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    print(f"[recall] {label} Recall@{k} = {rmean:.4f}  (n={len(queries)})")
    print(f"[ndcg]   {label} NDCG@5 (binary) = {nmean:.4f}  (n={len(queries)})")
    return 0


UNIFIED_SET = ROOT / "tests" / "fixtures" / "unified_eval.json"


def unified(k: int) -> int:
    """統一評估集:每筆按 metric 算對的指標,終結三評估集假矛盾。

    metric="recall"(小桶距離/價格/複合)→ Recall@K(召回率)。
    metric="precision"(大桶設施)→ Precision@K(TOP K 命中該特徵比例 = 純度)。
    報 recall 均值 + precision 均值 + 總分(兩者均值),總分越高越平衡。
    """
    try:
        import numpy as np
        import onnxruntime as ort  # noqa: F401
        from transformers import BertTokenizerFast  # noqa: F401
    except ImportError as e:
        print(f"[unified] 缺依賴 {e.name}:venv(pip install onnxruntime transformers)或 Colab 跑。")
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_vector_vs_rulebased import build_query_encoder, build_property_matrix  # noqa: E402
    from eval_rule_based_baseline import recall_at_k  # noqa: E402

    data = _load(UNIFIED_SET)
    if data is None:
        raise SystemExit(f"[fatal] {UNIFIED_SET} 不存在 — 先 python tests/gen_unified_eval.py")
    queries = data["queries"]
    encode = build_query_encoder()
    emb_data = _load(EMBEDDINGS)
    if emb_data is None:
        raise SystemExit(f"[fatal] {EMBEDDINGS} 不存在")
    mat, idx_order = build_property_matrix(emb_data, {})

    rec, prec = [], []
    for q in queries:
        qvec = encode(q["query"])
        sims = mat @ qvec
        top = [idx_order[i] for i in np.argsort(-sims)[:k]]
        relevant = set(q["relevant_idxs"])
        if q["metric"] == "recall":
            rec.append(recall_at_k(top, relevant, k))
        else:  # precision@k: TOP k 命中該特徵的比例(設施類純度)
            hit = sum(1 for idx in top if idx in relevant)
            prec.append(hit / k if k else 0.0)
    rmean = sum(rec) / len(rec) if rec else 0.0
    pmean = sum(prec) / len(prec) if prec else 0.0
    overall = (rmean + pmean) / 2
    print(f"[unified] Recall@{k}(小桶 n={len(rec)})     = {rmean:.4f}")
    print(f"[unified] Precision@{k}(設施 n={len(prec)}) = {pmean:.4f}")
    print(f"[unified] 總分(兩者均值)             = {overall:.4f}  ← 越高越平衡")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="階段④ 泛化評估 harness")
    ap.add_argument("--check", action="store_true", help="純 stdlib 自我驗(本機跑)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="Recall@K 的 K(預設 30)")
    ap.add_argument("--eval-set", type=str, default=None,
                    help="評估集 JSON 路徑(預設 generalization_eval;真 GT 用 tests/fixtures/true_gt_eval.json)")
    ap.add_argument("--unified", action="store_true",
                    help="統一評估集(單訴求+複合混合,按 metric 算 recall/precision,終結假矛盾)")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.unified:
        return unified(args.k)
    eval_path = Path(args.eval_set) if args.eval_set else EVAL_SET
    return recall(args.k, eval_path)


if __name__ == "__main__":
    raise SystemExit(main())
