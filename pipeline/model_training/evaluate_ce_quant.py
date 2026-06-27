"""
evaluate_ce_quant.py
Compare Dynamic INT8 vs Static INT8 Cross-Encoder on NDCG@5.
Uses recommendation_test.json (1064 queries, relevance 0-3).
"""
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
# onnxruntime / transformers imported lazily in evaluate()/main() so the module
# (and match_prob) can be imported with only numpy — the CI test job has no torch
# stack. See tests/test_evaluate_ce_quant.py.

BASE_DIR  = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "../../frontend/models/custom_onnx_model_dir"
DATA_PATH = BASE_DIR / "../../data/processed/recommendation_test.json"
MAX_LENGTH = 128


def match_prob(logits):
    """MATCH-class softmax prob from [NOT_MATCH, MATCH] logits.

    Mirrors frontend inference-worker.js: exp1 / (exp0 + exp1). Ranking by
    this (not by the raw NOT_MATCH logit) is the whole point of the fix.
    """
    logits = np.asarray(logits, dtype=np.float64)
    ex = np.exp(logits - np.max(logits))
    return float(ex[1] / ex.sum())


def ndcg_at_k(relevances, k=5):
    """Graded NDCG@k with exponential gain: (2^rel - 1) / log2(i+2)"""
    def dcg(rels):
        return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 1.0  # no relevant docs → perfect by convention
    return dcg(relevances) / idcg


def evaluate(model_path, tokenizer, queries_data, label):
    import onnxruntime as ort

    path = Path(model_path)
    if not path.exists():
        print(f"\n  {label}: NOT FOUND — {path}")
        return None

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    size_mb = path.stat().st_size / 1024 / 1024

    ndcg_scores = []
    t0 = time.time()

    for query, pairs in queries_data.items():
        # skip queries with no positive relevance
        if all(p["relevance"] <= 0 for p in pairs):
            continue

        scores = []
        for p in pairs:
            enc = tokenizer(
                query, p["property"],
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            logits = sess.run(None, {
                "input_ids":      enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "token_type_ids": enc.get(
                    "token_type_ids",
                    np.zeros_like(enc["input_ids"])
                ).astype(np.int64),
            })[0][0]  # (2,) = [NOT_MATCH, MATCH]
            # Bug fix: was float(logits[0]) = NOT_MATCH, which ranked in reverse.
            scores.append(match_prob(logits))

        # rank by model score, get relevance order
        ranked_idx = np.argsort(scores)[::-1]
        ranked_rels = [max(pairs[i]["relevance"], 0) for i in ranked_idx]
        ndcg_scores.append(ndcg_at_k(ranked_rels, k=5))

    elapsed = time.time() - t0
    mean_ndcg = np.mean(ndcg_scores)
    std_ndcg  = np.std(ndcg_scores) / np.sqrt(len(ndcg_scores))  # SE

    print(f"\n{'='*52}")
    print(f"  {label}  ({size_mb:.1f} MB)")
    print(f"  NDCG@5    : {mean_ndcg:.4f} ± {std_ndcg:.4f}")
    print(f"  Queries   : {len(ndcg_scores)}")
    print(f"  Time      : {elapsed:.1f}s")
    return mean_ndcg, std_ndcg


def main():
    from transformers import BertTokenizerFast

    raw = json.load(open(DATA_PATH))
    queries_data = defaultdict(list)
    for item in raw:
        if item["relevance"] >= 0:  # skip -1 (ambiguous)
            queries_data[item["query"]].append(item)

    tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
    print(f"Test set: {len(queries_data)} queries\n")

    r_dyn    = evaluate(MODEL_DIR / "my_custom_model_quant.onnx",
                        tokenizer, queries_data, "Dynamic INT8 (current)")
    r_static = evaluate(MODEL_DIR / "my_custom_model_static_int8.onnx",
                        tokenizer, queries_data, "Static INT8 (candidate)")

    if r_dyn and r_static:
        delta = r_static[0] - r_dyn[0]
        verdict = "✅ improvement" if delta >= 0 else f"⚠️  regression {delta:+.4f}"
        print(f"\n{'='*52}")
        print(f"  NDCG@5 delta : {delta:+.4f}  ({verdict})")
        dyn_mb    = (MODEL_DIR / "my_custom_model_quant.onnx").stat().st_size / 1024 / 1024
        static_mb = (MODEL_DIR / "my_custom_model_static_int8.onnx").stat().st_size / 1024 / 1024
        print(f"  Size delta   : {static_mb - dyn_mb:+.1f} MB  ({dyn_mb:.1f} → {static_mb:.1f})")


if __name__ == "__main__":
    main()
