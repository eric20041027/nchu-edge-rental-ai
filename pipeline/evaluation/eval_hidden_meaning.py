"""跑現役 bi-encoder 對隱藏含義評估集,按桶算 Precision@30 / Recall@30。

隱藏含義 query(口語、不含特徵詞)→ bi-encoder 純向量召回 TOP30 →
對照客觀 GT,看 TOP30 有多少符合「背後條件」。

指標按桶(評估集 meta 已標):
  - precision 桶(GT>30):Precision@30 = TOP30 命中數 / 30
  - recall 桶(GT≤30):Recall@30 = TOP30 命中數 / GT 總數

純向量召回(不套 hard-exclusion),測的是 bi-encoder 本身的隱藏含義理解力。
複用 eval_vector_vs_rulebased.py 的 encode/matrix(1:1 鏡像前端),不重造。

用法:python pipeline/evaluation/eval_hidden_meaning.py
本機需 onnxruntime + transformers(無 torch 亦可)。
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/hidden_meaning_eval.json"
K = 30


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "evvr", ROOT / "pipeline/evaluation/eval_vector_vs_rulebased.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ev = _load_harness()
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    queries = data["queries"]
    props = json.loads((ROOT / "frontend/assets/property_data.json").read_text(encoding="utf-8"))
    idx_to_prop = {i: p for i, p in enumerate(props)}
    emb = ev.load_property_embeddings()
    encode = ev.build_query_encoder()
    prop_mat, emb_idxs = ev.build_property_matrix(emb, idx_to_prop)

    import numpy as np

    n_props = len(props)
    print(f"隱藏含義評估 — 現役 bi-encoder,純向量 TOP{K}")
    print("基準 = 隨機撈 30 個的期望命中率(GT率);倍數 = 模型/基準 → 是否真懂隱藏含義")
    print("-" * 86)
    print(f"{'metric':<10}{'query':<18}{'meaning':<18}{'score':>7}{'base':>7}{'ratio':>7}  判定")
    print("-" * 86)

    p_scores, r_scores = [], []
    understood = 0
    for q in queries:
        relevant = set(q["relevant_idxs"])
        if not relevant:
            continue
        qv = encode(q["query"])
        scores = prop_mat @ qv
        order = np.argsort(-scores)
        top = [emb_idxs[int(r)] for r in order if emb_idxs[int(r)] in idx_to_prop][:K]
        hits = sum(1 for idx in top if idx in relevant)
        base = len(relevant) / n_props  # 隨機撈 30 的期望命中率
        if q["metric"] == "precision":
            val = hits / K
            p_scores.append(val)
        else:
            val = hits / len(relevant)
            r_scores.append(val)
        ratio = val / base if base > 0 else 0.0
        verdict = "懂" if ratio >= 1.5 else ("略懂" if ratio >= 1.1 else "沒懂")
        if ratio >= 1.5:
            understood += 1
        print(f"{q['metric']:<10}{q['query'][:16]:<18}{q['hidden_meaning'][:16]:<18}"
              f"{val:>7.3f}{base:>7.3f}{ratio:>6.1f}x  {verdict}")

    n_eval = len(p_scores) + len(r_scores)
    print("-" * 86)
    if p_scores:
        print(f"  Precision@{K} 平均 ({len(p_scores)} 大桶): {sum(p_scores)/len(p_scores):.3f}")
    if r_scores:
        print(f"  Recall@{K}    平均 ({len(r_scores)} 小桶): {sum(r_scores)/len(r_scores):.3f}")
    print(f"  顯著理解(≥1.5x 隨機)的隱藏含義: {understood}/{n_eval}")


if __name__ == "__main__":
    main()
