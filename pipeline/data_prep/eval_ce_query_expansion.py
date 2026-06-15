"""A/B the CE query-layer expansion: raw query vs semanticExpandQuery output.

The extension map (data/semantic_rules.json → semanticExpandQuery in
inference-worker.js) rewrites the query BEFORE it is fed to the cross-encoder
at inference time. It does NOT participate in CE training — the CE never saw
expanded queries during training, so feeding "原始query + 擴展詞" is a
train/inference distribution shift, analogous to the prop.text enrichment OOD
risk already shown NO-GO at the property layer (eval_ce_text_enrichment.py).

This harness quantifies whether query expansion helps or hurts CE *ranking*
(NDCG@5), using the SAME production CE ONNX + tokenizer the frontend uses,
on the held-out test split (object-level isolated, no leakage).

For each query with >=2 candidates and graded relevance, we score every
candidate property OLD (raw query) vs NEW (expanded query), rank, and compute
NDCG@5. We report overall and, crucially, the subset where expansion actually
fired (queries the map left unchanged are identical OLD/NEW and only dilute).

Requires: onnxruntime, transformers, numpy (Python 3.12; onnxruntime has no
3.14 wheels yet). Mirrors eval_ce_text_enrichment.py's deps.

Usage:
    python pipeline/data_prep/eval_ce_query_expansion.py
    python pipeline/data_prep/eval_ce_query_expansion.py --sample 150

Result (2026-06-15, full test split): query expansion is NET-NEUTRAL for CE
ranking (NDCG@5 0.9302 → 0.9302, Δ≈0) but HIGH-VARIANCE on individual queries
(of 16 rank-changing queries: 7 big wins / 9 big losses, ±0.37–0.50). Big wins
are colloquial/abbreviated queries the CE can't parse raw (夏天→冷氣,
獨洗獨曬→洗衣機 陽台); losses are OOD noise (愛乾淨→獨洗 禁菸). Confirms the
train/inference distribution shift is real but not a NO-GO — keep as-is, but do
NOT assume expansion helps CE. See docs/ce_text_layer_decision.md (property-side
NO-GO) for the symmetric property-layer finding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "frontend" / "models" / "custom_onnx_model_dir" / "my_custom_model_quant.onnx"
TOK_DIR = ROOT / "frontend" / "models" / "custom_onnx_model_dir"
TEST = ROOT / "data" / "processed" / "recommendation_test.json"
RULES = ROOT / "data" / "semantic_rules.json"
MAX_LENGTH = 64  # matches inference-worker.js MAX_LENGTH
K = 5


def load_expansion_map() -> dict[str, str]:
    """Canonical extension map — same source sync_semantic_rules.py feeds the JS."""
    rules = json.loads(RULES.read_text(encoding="utf-8"))["rules"]
    return {k: " ".join(v) for k, v in rules.items()}


def expand_query(query: str, emap: dict[str, str]) -> str:
    """Mirror semanticExpandQuery (inference-worker.js) WITH the negation guard
    we just added (so the A/B reflects current production behavior)."""
    expanded = query
    NEGATORS = "不沒無非免勿"
    for key, expansion in emap.items():
        frm = 0
        while (idx := query.find(key, frm)) != -1:
            negated = idx > 0 and query[idx - 1] in NEGATORS
            if not negated:
                expanded += " " + expansion
                break
            frm = idx + 1
    return expanded


def make_scorer(tok, sess):
    """Match probability for the 'match' class — mirrors worker scorePair softmax."""
    def score(query: str, text: str) -> float:
        enc = tok(query, text, max_length=MAX_LENGTH, padding="max_length",
                  truncation=True, return_tensors="np")
        logits = sess.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc.get("token_type_ids",
                                      np.zeros_like(enc["input_ids"])).astype(np.int64),
        })[0][0]
        # 2-logit classifier → softmax, take match (index 1)
        m = float(np.max(logits))
        e0, e1 = np.exp(logits[0] - m), np.exp(logits[1] - m)
        return float(e1 / (e0 + e1))
    return score


def dcg(rels: list[float]) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels: list[float], k: int = K) -> float:
    ideal = sorted(ranked_rels, reverse=True)
    idcg = dcg(ideal[:k])
    if idcg == 0:
        return 0.0
    return dcg(ranked_rels[:k]) / idcg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="cap number of eval queries for speed (0=all)")
    args = ap.parse_args()

    data = json.loads(TEST.read_text(encoding="utf-8"))
    emap = load_expansion_map()

    # group by query; relevance -1 (random easy neg) → 0
    groups: dict[str, list[tuple[str, float]]] = {}
    for s in data:
        rel = s.get("relevance", s["label"])
        rel = max(0, rel)
        groups.setdefault(s["query"], []).append((s["property"], float(rel)))

    # keep queries with >=2 candidates AND at least one positive (else NDCG trivial)
    evalable = {q: cands for q, cands in groups.items()
                if len(cands) >= 2 and any(r > 0 for _, r in cands)}
    queries = list(evalable.keys())
    if args.sample:
        queries = queries[:args.sample]

    tok = BertTokenizerFast.from_pretrained(str(TOK_DIR))
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    score = make_scorer(tok, sess)

    old_ndcgs, new_ndcgs = [], []
    fired_old, fired_new = [], []   # subset where expansion changed the query
    n_fired = 0

    for qi, q in enumerate(queries):
        cands = evalable[q]
        eq = expand_query(q, emap)
        fired = eq != q
        if fired:
            n_fired += 1

        old_scored = sorted(((score(q, t), r) for t, r in cands),
                            key=lambda x: x[0], reverse=True)
        new_scored = sorted(((score(eq, t), r) for t, r in cands),
                            key=lambda x: x[0], reverse=True)
        n_old = ndcg_at_k([r for _, r in old_scored])
        n_new = ndcg_at_k([r for _, r in new_scored])
        old_ndcgs.append(n_old)
        new_ndcgs.append(n_new)
        if fired:
            fired_old.append(n_old)
            fired_new.append(n_new)

        if (qi + 1) % 50 == 0:
            print(f"  ...{qi+1}/{len(queries)} queries scored")

    def report(label, olds, news):
        if not olds:
            print(f"  {label}: (no queries)")
            return
        mo, mn = np.mean(olds), np.mean(news)
        wins = sum(1 for a, b in zip(olds, news) if b > a + 1e-9)
        loss = sum(1 for a, b in zip(olds, news) if b < a - 1e-9)
        same = len(olds) - wins - loss
        print(f"  {label} (n={len(olds)}):")
        print(f"      OLD raw-query  NDCG@{K} = {mo:.4f}")
        print(f"      NEW expanded   NDCG@{K} = {mn:.4f}")
        print(f"      Δ = {mn-mo:+.4f}   |  win {wins} / tie {same} / lose {loss}")

    print(f"\n===== CE query-expansion A/B  (CE={MODEL.name}, test split) =====")
    print(f"eval queries: {len(queries)}  |  expansion fired on: {n_fired}\n")
    print("ALL eval queries (fired + unchanged):")
    report("ALL", old_ndcgs, new_ndcgs)
    print("\nFIRED-ONLY subset (queries the map actually expanded — the real signal):")
    report("FIRED", fired_old, fired_new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
