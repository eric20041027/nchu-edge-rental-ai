"""T7 A/B harness — VECTOR recall vs RULE-BASED recall (go/no-go gate, CP5).

On the SAME T1 eval query set (tests/fixtures/ab_eval_queries.json, 278 queries =
78 semantic + 200 keyword), compute Recall@15, Recall@30 and NDCG@5 for BOTH
recall methods, per bucket and overall, and print a verdict against spec
Success #1 / #2.

This is the go/no-go gate for the whole vector-retrieval effort
(docs/spec/vector-retrieval-plan.md CP5):
  - PASS  -> proceed to "remove rule-based recall path" wind-down.
  - FAIL  -> keep the toggle, record the conclusion, go back and tune T2/T4.

=== Reuse (T0) ===
We import tests/eval_rule_based_baseline.py and REUSE its exact, already-verified
definitions so both columns use identical conventions:
  - load_properties           (same shell filter as inference.js:104)
  - recall_at_k, ndcg_at_k    (same metric defs, mirror eval_ce_query_expansion)
  - rule_based_recall         (parse -> filterHardExclusions -> score -> top-K)
  - parse_constraints_from_text, filter_hard_exclusions  (the hard-filter port)
Metrics are NOT reimplemented here.

=== Ground truth ===
The T1 fixture already stores `relevant_idxs` per query (property_data idx, from
T0 build_ground_truth, relevance>=1) so NO re-join is needed. The fixture stores
BINARY relevance only (relevant_idxs), not graded. Therefore:
  - Recall@K is the HEADLINE metric (this is a RECALL comparison) and uses the
    binary relevant set directly.
  - NDCG@5 is computed with BINARY relevance (rel=1 for idxs in relevant_idxs,
    else 0). It is clearly labelled "NDCG@5 (binary)" so it is not confused with
    the T0 graded NDCG@5=0.2469. Both methods use the identical binary convention,
    so the A/B delta is still apples-to-apples; only the absolute value differs
    from the graded T0 number.

=== Vector recall path — mirrors T5 production (frontend/js) EXACTLY ===
T5 recall (inference.js:1409-1428) = encodeQuery -> cosineTopK(all) -> keep only
hits whose prop is in the filterHardExclusions "allowed" set, in cosine order,
take top-30. We replicate that offline 1:1:
  1. ONNX query encoder via onnxruntime InferenceSession(CPUExecutionProvider)
     — same pattern as export_bi_encoder.py:289.
  2. Tokenize with the rbt6 tokenizer (transformers BertTokenizerFast from
     frontend/models/bi_encoder_dir): single sentence, max_length 64, truncation,
     padding='max_length'. Build input_ids + attention_mask int64. NO
     token_type_ids — mirrors bi-encoder-worker.js:79-97 and the exported graph.
  3. session.run(["embedding"], feed)[0] -> (768,), already mean-pooled + L2-norm.
  4. property_embeddings.json -> (count, dim) float array from the flat `vecs`
     (float16-rounded, stored as JSON numbers). Property vectors are L2-norm.
  5. cosineTopK: dot(q, each prop) — both unit-norm so dot == cosine — rank desc.
  6. Hard-exclusion intersection (the T5 fairness step): parse_constraints_from_text
     + filter_hard_exclusions give the allowed property set; take cosine hits over
     ALL props but keep only allowed ones, top-K. Same K as rule-based (15 & 30).

  NOTE on the embeddings idx caveat (T5 finding): embeddings.idxs reference the
  ORIGINAL 704 records, but load_properties() drops crawler shells. We build an
  idx->prop map from the loaded properties and skip any embedding idx not present
  (exactly the "idxToProp + reference identity" approach T5 used to avoid the
  position-misalignment bug). The allowed set is by idx, matching how the T1
  fixture keys relevant_idxs.

=== Dependency reality ===
This box has NO onnxruntime / transformers / numpy. The harness:
  - Guards those imports; if missing it tells you what to install and exits 1
    (mirrors the T3/T4 guards) — UNLESS you pass --check.
  - --check runs the half that needs NO model: loads the T1 fixture, parses
    property_embeddings.json (count*dim integrity), imports T0, and actually
    COMPUTES + PRINTS the RULE-BASED column + per-bucket counts. This proves half
    the harness works here and produces the real rule-based baseline numbers.

Usage:
    # Runs here (pure stdlib) — rule-based column + embeddings integrity:
    python3 tests/eval_vector_vs_rulebased.py --check

    # Full A/B (needs onnxruntime + transformers, e.g. on Colab):
    python3 tests/eval_vector_vs_rulebased.py
    python3 tests/eval_vector_vs_rulebased.py --k 15 30 --sample 50 --bucket semantic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import T0 — reuse its loaders, metrics, and rule-based recall (NO reimplement).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_rule_based_baseline import (  # noqa: E402
    filter_hard_exclusions,
    load_properties,
    ndcg_at_k,
    parse_constraints_from_text,
    recall_at_k,
    rule_based_recall,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "ab_eval_queries.json"
EMBEDDINGS = ROOT / "frontend" / "assets" / "property_embeddings.json"
BI_ENCODER_DIR = ROOT / "frontend" / "models" / "bi_encoder_dir"
ONNX_PATH = BI_ENCODER_DIR / "bi_encoder_quant.onnx"

# K values — same as T0 / rule-based comparator. 15 = production .slice(0,15),
# 30 = planned vector-recall K (spec Resolved #3).
DEFAULT_KS = [15, 30]
NDCG_K = 5
MAX_LEN = 64  # mirrors bi-encoder-worker.js MAX_LEN and the exported graph.


# =====================================================================
# Fixture + embeddings loaders (the --check half — pure stdlib)
# =====================================================================

def load_eval_queries(path: Path = FIXTURE) -> tuple[list[dict], dict]:
    """Load the T1 fixture. Returns (queries, meta).

    Each query: {query, bucket, n_relevant, relevant_idxs, semantic_trigger}.
    relevant_idxs are property_data idx (binary relevance, T0 build_ground_truth).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"], data.get("meta", {})


def load_property_embeddings(path: Path = EMBEDDINGS) -> dict:
    """Load + validate property_embeddings.json (pure stdlib, no numpy).

    Returns {dim, count, idxs, vecs} where `vecs` is the flat float16-rounded
    list (count*dim). --check validates count*dim integrity only; the vector path
    reshapes `vecs` into a (count, dim) numpy matrix in build_property_matrix.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    dim = data["dim"]
    count = data["count"]
    idxs = data["idxs"]
    vecs = data["vecs"]
    if len(idxs) != count:
        raise ValueError(f"idxs len {len(idxs)} != count {count}")
    if len(vecs) != count * dim:
        raise ValueError(f"vecs len {len(vecs)} != count*dim {count*dim}")
    return {"dim": dim, "count": count, "idxs": idxs, "vecs": vecs}


# =====================================================================
# Recall drivers
# =====================================================================

def rule_based_ranked_idxs(properties: list[dict], query: str, k: int) -> list[int]:
    """Rule-based top-k property_data idxs (reuses T0 rule_based_recall)."""
    ranked = rule_based_recall(properties, query, k=k)
    return [item["prop"]["idx"] for item in ranked]


def _l2_normalize(vec: "list[float] | object") -> object:
    """L2-normalize a numpy 1-D vector (query emb is already unit-norm from the
    graph; this is a defensive no-op-ish renorm so cosine == dot holds exactly)."""
    import numpy as np

    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def build_query_encoder():
    """Build the ONNX query encoder. Returns encode(text) -> np.float32 (dim,).

    Mirrors bi-encoder-worker.js encode() + export_bi_encoder.py ort session:
      - BertTokenizerFast from frontend/models/bi_encoder_dir
      - single sentence, max_length 64, truncation, padding='max_length'
      - feed input_ids + attention_mask int64 (NO token_type_ids)
      - session.run(["embedding"], feed)[0][0]  -> (dim,), unit-norm
    """
    import numpy as np
    import onnxruntime as ort
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(str(BI_ENCODER_DIR))
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}

    def encode(text: str):
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            return_tensors="np",
        )
        feed = {}
        for name in ("input_ids", "attention_mask"):
            if name in input_names:
                feed[name] = enc[name].astype(np.int64)
        emb = session.run(["embedding"], feed)[0].astype(np.float32)[0]
        return _l2_normalize(emb)

    return encode


def build_property_matrix(emb: dict, idx_to_prop: dict):
    """Reconstruct the (count, dim) property matrix (numpy) from flat float16
    `vecs`, and the parallel idxs list. Property vectors are already L2-norm."""
    import numpy as np

    dim, count, idxs, vecs = emb["dim"], emb["count"], emb["idxs"], emb["vecs"]
    mat = np.asarray(vecs, dtype=np.float32).reshape(count, dim)
    return mat, idxs


def vector_ranked_idxs(query: str, encode, prop_mat, emb_idxs: list[int],
                       properties: list[dict], idx_to_prop: dict, k: int) -> list[int]:
    """Vector top-k property_data idxs — mirrors T5 inference.js:1409-1428 exactly.

    1) encode query -> unit-norm vec.
    2) cosineTopK over ALL embedding rows (dot == cosine, both unit-norm).
    3) keep hits whose idx is in the filterHardExclusions allowed set, in cosine
       order, take top-k.
    """
    import numpy as np

    q = encode(query)
    scores = prop_mat @ q  # (count,) cosine similarities
    order = np.argsort(-scores)  # descending

    constraints = parse_constraints_from_text(query)
    allowed = {p["idx"] for p in filter_hard_exclusions(properties, constraints)}

    out: list[int] = []
    for row in order:
        idx = emb_idxs[int(row)]
        if idx not in idx_to_prop:  # embedding for a dropped crawler shell
            continue
        if idx not in allowed:
            continue
        out.append(idx)
        if len(out) >= k:
            break
    return out


# =====================================================================
# Metric aggregation
# =====================================================================

def _binary_rels(ranked_idxs: list[int], relevant: set[int]) -> list[float]:
    """Graded-relevance vector for NDCG (binary: 1 if relevant else 0)."""
    return [1.0 if idx in relevant else 0.0 for idx in ranked_idxs]


def score_method(queries: list[dict], rank_fn, ks: list[int]) -> dict:
    """Run a ranking fn over all queries; return per-bucket + overall means.

    rank_fn(query_text, max_k) -> ranked property_data idxs (len up to max_k).
    """
    max_k = max(ks)
    buckets: dict[str, dict[str, list[float]]] = {}

    def _slot(bucket: str) -> dict[str, list[float]]:
        return buckets.setdefault(
            bucket, {f"recall@{k}": [] for k in ks} | {f"ndcg@{NDCG_K}": []}
        )

    for q in queries:
        relevant = set(q["relevant_idxs"])
        if not relevant:
            continue
        ranked = rank_fn(q["query"], max_k)
        for scope in (q["bucket"], "all"):
            slot = _slot(scope)
            for k in ks:
                slot[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
            rels = _binary_rels(ranked, relevant)
            slot[f"ndcg@{NDCG_K}"].append(ndcg_at_k(rels, NDCG_K))

    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return {
        b: {m: mean(vals) for m, vals in metrics.items()} | {"n": len(next(iter(metrics.values())))}
        for b, metrics in buckets.items()
    }


# =====================================================================
# Output
# =====================================================================

def _bucket_counts(queries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"all": 0}
    for q in queries:
        counts[q["bucket"]] = counts.get(q["bucket"], 0) + 1
        counts["all"] += 1
    return counts


def print_rule_only(rb: dict, ks: list[int], counts: dict[str, int]) -> None:
    """--check output: rule-based column only (vector side runtime-pending)."""
    print("\n" + "=" * 68)
    print(" T7 --check : RULE-BASED column (vector side runtime-pending)")
    print("=" * 68)
    for bucket in ("semantic", "keyword", "all"):
        if bucket not in rb:
            continue
        n = rb[bucket].get("n", counts.get(bucket, 0))
        print(f"\n  {bucket} ({n})")
        for k in ks:
            print(f"    Recall@{k:<3}      {rb[bucket][f'recall@{k}']:.4f}")
        print(f"    NDCG@{NDCG_K} (bin)   {rb[bucket][f'ndcg@{NDCG_K}']:.4f}")
    print("\n" + "-" * 68)
    print(" Rule-based numbers are REAL (no model needed). Vector column pending")
    print(" onnxruntime + transformers — run without --check on Colab.")
    print("=" * 68)


def print_ab_table(rb: dict, vec: dict, ks: list[int]) -> bool:
    """Full A/B table + GO/NO-GO verdict. Returns True if PASS."""
    print("\n" + "=" * 72)
    print(" T7 A/B — VECTOR recall vs RULE-BASED recall (CP5 go/no-go)")
    print("=" * 72)
    header = f"  {'':<16}{'RULE-BASED':>12}{'VECTOR':>12}{'Δ':>12}"
    for bucket in ("semantic", "keyword", "all"):
        if bucket not in rb or bucket not in vec:
            continue
        n = rb[bucket].get("n", 0)
        print(f"\n  {bucket} ({n})")
        print(header)
        for k in ks:
            m = f"recall@{k}"
            r, v = rb[bucket][m], vec[bucket][m]
            print(f"  Recall@{k:<9}{r:>12.4f}{v:>12.4f}{v - r:>+12.4f}")
        m = f"ndcg@{NDCG_K}"
        r, v = rb[bucket][m], vec[bucket][m]
        print(f"  NDCG@{NDCG_K}(bin){'':<3}{r:>12.4f}{v:>12.4f}{v - r:>+12.4f}")

    # Verdict — spec Success #1: vector Recall@K >= rule-based (overall),
    # and notably higher on the semantic bucket.
    print("\n" + "-" * 72)
    sem_ok = all(vec["semantic"][f"recall@{k}"] >= rb["semantic"][f"recall@{k}"]
                 for k in ks) if "semantic" in vec else False
    all_ok = all(vec["all"][f"recall@{k}"] >= rb["all"][f"recall@{k}"]
                 for k in ks) if "all" in vec else False
    sem_higher = (vec["semantic"].get("recall@30", 0) >
                  rb["semantic"].get("recall@30", 0)) if "semantic" in vec else False

    print(" Success #1 — vector Recall@K >= rule-based (same query set / K / harness):")
    print(f"   overall (all K):           {'PASS' if all_ok else 'FAIL'}")
    print(f"   semantic >= rule-based:    {'PASS' if sem_ok else 'FAIL'}")
    print(f"   semantic NOTABLY higher@30:{'PASS' if sem_higher else 'FAIL'}")
    verdict = all_ok and sem_ok and sem_higher
    print("\n" + "=" * 72)
    print(f" GO/NO-GO: {'GO  — vector recall wins, proceed to wind-down.' if verdict else 'NO-GO — keep toggle, record conclusion, tune T2/T4.'}")
    print("=" * 72)
    return verdict


# =====================================================================
# Main
# =====================================================================

def run_check(queries: list[dict], properties: list[dict], emb: dict,
              ks: list[int]) -> int:
    """--check path — pure stdlib. Validates everything not needing the model and
    computes+prints the real RULE-BASED column."""
    counts = _bucket_counts(queries)
    print("=" * 68)
    print(" T7 --check : harness self-test (no ONNX / no model)")
    print("=" * 68)
    print(f" T1 fixture loaded              : {len(queries)} queries")
    print(f"   buckets                      : "
          + ", ".join(f"{b}={counts[b]}" for b in sorted(counts) if b != "all"))
    print(f" property_embeddings.json       : count={emb['count']} dim={emb['dim']}"
          f"  (vecs {len(emb['vecs'])} == count*dim {emb['count']*emb['dim']} OK)")
    print(f" properties (after shell filter): {len(properties)}")
    print(f" T0 import (loaders/metrics/recall): OK")

    def rb_rank(query: str, max_k: int) -> list[int]:
        return rule_based_ranked_idxs(properties, query, max_k)

    rb = score_method(queries, rb_rank, ks)
    print_rule_only(rb, ks, counts)
    return 0


def run_full(queries: list[dict], properties: list[dict], emb: dict,
             ks: list[int]) -> int:
    """Full A/B — needs onnxruntime + transformers + numpy."""
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        from transformers import BertTokenizerFast  # noqa: F401
    except ImportError as e:
        print(f"[T7] Missing runtime dependency: {e.name}", file=sys.stderr)
        print("[T7] The VECTOR side needs: pip install onnxruntime transformers numpy",
              file=sys.stderr)
        print("[T7] (This dev box has none — run on Colab, OR use --check here to",
              file=sys.stderr)
        print("      compute+print the rule-based column with pure stdlib.)",
              file=sys.stderr)
        return 1

    if not ONNX_PATH.exists():
        print(f"[T7] ONNX encoder not found: {ONNX_PATH}", file=sys.stderr)
        return 1

    idx_to_prop = {p["idx"]: p for p in properties}
    encode = build_query_encoder()
    prop_mat, emb_idxs = build_property_matrix(emb, idx_to_prop)

    def rb_rank(query: str, max_k: int) -> list[int]:
        return rule_based_ranked_idxs(properties, query, max_k)

    def vec_rank(query: str, max_k: int) -> list[int]:
        return vector_ranked_idxs(query, encode, prop_mat, emb_idxs,
                                  properties, idx_to_prop, max_k)

    print(f"[T7] Scoring {len(queries)} queries (rule-based + vector)...")
    rb = score_method(queries, rb_rank, ks)
    vec = score_method(queries, vec_rank, ks)
    passed = print_ab_table(rb, vec, ks)
    return 0 if passed else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="T7 A/B: vector vs rule-based recall")
    ap.add_argument("--k", type=int, nargs="+", default=DEFAULT_KS,
                    help="K values to compare (default: 15 30)")
    ap.add_argument("--sample", type=int, default=0,
                    help="cap number of eval queries for speed (0=all)")
    ap.add_argument("--bucket", choices=["semantic", "keyword", "all"], default="all",
                    help="restrict eval to one bucket (default: all)")
    ap.add_argument("--check", action="store_true",
                    help="self-test + compute the rule-based column only "
                         "(no ONNX / no model needed; runs anywhere)")
    args = ap.parse_args()

    ks = sorted(set(args.k))
    queries, _meta = load_eval_queries()
    if args.bucket != "all":
        queries = [q for q in queries if q["bucket"] == args.bucket]
    if args.sample:
        queries = queries[:args.sample]

    properties = load_properties()
    emb = load_property_embeddings()

    if args.check:
        return run_check(queries, properties, emb, ks)
    return run_full(queries, properties, emb, ks)


if __name__ == "__main__":
    raise SystemExit(main())
