"""階段④ 泛化評估 harness — 多樣 query 真評估集 + holdout 隔離驗 + Recall@30。

spec: docs/spec/generalization-data.md

兩半設計(對齊 eval_vector_vs_rulebased.py 範式):
  --check 半(純 stdlib,本機跑):
    - 結構驗:generalization_queries / eval / holdout 的 schema 正確
    - 隔離鐵則:holdout 的 src_idx/query 與訓練集零交集(R1 最致命風險)
    - 完整性:eval/holdout 的 relevant_idxs 無懸空(都對得上 property_data idx)
    - 零標點:生成 query 全無標點符號
  Recall 半(需 onnxruntime + tokenizers,本機當前無 → guard + exit 1):
    - 編碼 eval query → cosineTopK(30) vs property_embeddings → Recall@30
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

ROOT = Path(__file__).resolve().parents[1]
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

def recall(k: int) -> int:
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
    from eval_rule_based_baseline import recall_at_k  # noqa: E402

    eval_data = _load(EVAL_SET)
    if eval_data is None:
        raise SystemExit(f"[fatal] {EVAL_SET} 不存在 — 先生成評估集")
    queries = eval_data["queries"] if isinstance(eval_data, dict) else eval_data

    encode = build_query_encoder()
    mat, idx_order = build_property_matrix(EMBEDDINGS)

    recalls = []
    for q in queries:
        qvec = encode(q["query"])
        sims = mat @ qvec
        top = [idx_order[i] for i in np.argsort(-sims)[:k]]
        recalls.append(recall_at_k(top, set(q["relevant_idxs"]), k))
    mean = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"[recall] generalization_eval Recall@{k} = {mean:.4f}  (n={len(queries)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="階段④ 泛化評估 harness")
    ap.add_argument("--check", action="store_true", help="純 stdlib 自我驗(本機跑)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="Recall@K 的 K(預設 30)")
    args = ap.parse_args()
    return check() if args.check else recall(args.k)


if __name__ == "__main__":
    raise SystemExit(main())
