"""Build the curated A/B eval query set (Task T1) for vector-recall vs rule-based.

=== WHY this set exists (the recall blind-spot fact) ===
The production recall stage (frontend/js/inference.js: recommend -> extractKeywords
-> calculateRuleBasedScore, ~lines 1301-1307) scores candidates using the RAW
query keywords with NO semantic expansion. The semantic-expansion map
(semanticExpandQuery, data/semantic_rules.json `rules` — e.g. 怕熱→冷氣,
不想共用洗衣機→獨洗, 遠距工作→網路) is applied ONLY at the CE rerank layer
(inference-worker.js:306), NOT at recall.

NOTE: the OFFLINE T0 port (eval_rule_based_baseline.py) DOES apply the GENERATED
INTENT_MAP via expand_query_intent() inside extract_keywords — that is the
faithful port of `expandQueryIntent`, a *separate* generated rule set baked into
inference.js. The data/semantic_rules.json `rules` here are the CE-layer
expansion map; many of its colloquial keys (and especially their inflected /
real-world phrasings) are NOT covered by INTENT_MAP, so the literal feature word
never enters the recall keyword bag. Those are the genuine RECALL BLIND SPOTS:
the user expresses a need colloquially, the matching property feature word is
absent from the query, and rule-based recall cannot bridge the gap. Vector recall
can. These are the highest-value queries for proving vector recall helps.

=== CRITICAL nuance: the offline T0 port ALSO expands colloquial triggers ===
The T0 harness's extract_keywords applies expand_query_intent (INTENT_MAP, the
GENERATED rule set baked into inference.js). It turns out EVERY key of
data/semantic_rules.json `rules` is also present in INTENT_MAP. So in the OFFLINE
T7 A/B (which runs the T0 rule-based port as the baseline), a query that merely
*contains* a colloquial trigger is NOT automatically a blind spot — the baseline
already injects the feature word (怕熱 -> 冷氣 etc.) into its keyword bag.

Therefore trigger-membership ALONE is necessary but not sufficient. We add an
EMPIRICAL gate: a query is a genuine blind spot only if the full T0 rule-based
recall (INTENT_MAP expansion included, the actual A/B baseline) FAILS to retrieve
at least one of its relevant listings within K=30. Those are exactly the queries
where vector recall has room to win — provably, against the real baseline.

=== Bucket criterion ===
- "semantic": (1) query contains a non-negated colloquial-intent trigger (a key
  of data/semantic_rules.json `rules`) whose literal feature keyword is ABSENT
  from the query text, AND (2) the T0 rule-based recall MISSES >=1 relevant
  listing at K=30 (the empirical blind-spot gate above). Negated triggers
  (不/沒/無/... immediately before the phrase) are skipped — a negation flips the
  intent and would pollute the bucket.
- "keyword": straightforward keyword-style query (literal feature / location /
  budget terms), NO qualifying colloquial trigger. The control bucket.

A query is only included if it has >=1 ground-truth relevant listing AFTER the
T0 fuzzy-join (relevance>=1 joined to a property_data idx), so the T7 A/B harness
can compute Recall@K / NDCG@5 on it without re-joining.

Reuses the T0 harness (eval_rule_based_baseline.py) verbatim for the join +
ground-truth machinery: load_properties, load_labels, build_fuzzy_join,
build_ground_truth. The join/relevance convention is therefore IDENTICAL to T0.

Usage:
    python3 tests/build_ab_eval_queries.py            # build + write + verify
    python3 tests/build_ab_eval_queries.py --max-keyword 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import eval_rule_based_baseline as t0

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_RULES = ROOT / "data" / "semantic_rules.json"
OUT = ROOT / "tests" / "fixtures" / "ab_eval_queries.json"
JOIN_CACHE = ROOT / "tests" / ".rule_based_join_cache.json"

NEGATORS = t0.NEGATORS  # 不沒無非免勿

# Targets (honest caps; report real achievable counts).
MAX_KEYWORD_DEFAULT = 200  # cap the control bucket so totals stay ~150-300


def load_semantic_rules(path: Path = SEMANTIC_RULES) -> dict[str, list[str]]:
    """Load the CE-layer colloquial->feature expansion map (`rules`)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["rules"]


def find_semantic_trigger(query: str, rules: dict[str, list[str]]
                          ) -> tuple[str, list[str]] | None:
    """Return (trigger_phrase, expansions) if `query` is a genuine blind spot.

    A genuine blind spot requires, for some rule key `trigger`:
      1. `trigger` occurs in `query` (non-negated — no NEGATOR char immediately
         before the occurrence; idx==0 is never negated, mirroring the JS guard).
      2. NONE of `trigger`'s expanded feature words appears literally in `query`
         (if the user already typed the feature word, recall is not blind).

    When multiple triggers qualify, the LONGEST trigger wins (most specific /
    least ambiguous colloquial phrase).
    """
    candidates: list[tuple[str, list[str]]] = []
    for trigger, expansions in rules.items():
        if not _is_non_negated_occurrence(query, trigger):
            continue
        # Blind spot only if the literal feature word is absent from the query.
        if any(exp in query for exp in expansions):
            continue
        candidates.append((trigger, expansions))
    if not candidates:
        return None
    candidates.sort(key=lambda t: len(t[0]), reverse=True)
    return candidates[0]


def _is_non_negated_occurrence(query: str, trigger: str) -> bool:
    """True if `trigger` occurs in `query` at least once NOT preceded by a
    negator char (mirrors expand_query_intent's negation guard)."""
    frm = 0
    while (idx := query.find(trigger, frm)) != -1:
        negated = idx > 0 and query[idx - 1] in NEGATORS
        if not negated:
            return True
        frm = idx + 1
    return False


def build(max_keyword: int) -> dict:
    properties = t0.load_properties()
    labels = t0.load_labels()
    rules = load_semantic_rules()

    distinct_blobs = sorted({s["property"] for s in labels})
    blob_to_idx = t0.build_fuzzy_join(properties, distinct_blobs, cache_path=JOIN_CACHE)
    join_rate = len(blob_to_idx) / len(distinct_blobs) if distinct_blobs else 0.0

    gt = t0.build_ground_truth(labels, blob_to_idx)

    # Usable queries: >=1 positive (relevance>=1) joined to an existing idx.
    usable: dict[str, list[int]] = {}
    for q, rels in gt.items():
        pos = sorted(idx for idx, r in rels.items() if r >= 1)
        if pos:
            usable[q] = pos

    semantic_q: list[dict] = []
    keyword_q: list[dict] = []
    n_trigger = 0  # queries with a qualifying trigger (pre empirical gate)
    for q in sorted(usable):
        relevant = usable[q]
        hit = find_semantic_trigger(q, rules)
        if hit is not None:
            n_trigger += 1
            trigger, _ = hit
            # Empirical blind-spot gate: does the ACTUAL A/B baseline (T0
            # rule-based recall, INTENT_MAP expansion included) miss a relevant
            # listing at K=30? If it already retrieves all of them, vector recall
            # cannot win here, so it is NOT a useful semantic A/B query.
            ranked = t0.rule_based_recall(properties, q, k=t0.K_VECTOR)
            got = {it["prop"]["idx"] for it in ranked}
            if not set(relevant).issubset(got):
                semantic_q.append({
                    "query": q, "bucket": "semantic",
                    "n_relevant": len(relevant), "relevant_idxs": relevant,
                    "semantic_trigger": trigger,
                })
                continue
            # Trigger present but baseline already covers it -> drop (neither a
            # blind spot nor a clean keyword control).
            continue
        keyword_q.append({
            "query": q, "bucket": "keyword",
            "n_relevant": len(relevant), "relevant_idxs": relevant,
            "semantic_trigger": None,
        })

    # Keep ALL semantic (blind spots are scarce + the point of the set). Cap the
    # keyword control bucket so the total stays in the ~150-300 band. Sort the
    # keyword bucket by n_relevant desc so the control keeps the richest GT.
    keyword_q.sort(key=lambda x: x["n_relevant"], reverse=True)
    keyword_kept = keyword_q[:max_keyword]

    queries = semantic_q + keyword_kept

    return {
        "meta": {
            "created": "2026-06-22",
            "source": "recommendation_train.json",
            "join": "T0 fuzzy-join (token-set)",
            "join_match_rate": round(join_rate, 4),
            # relevant_idxs 是對「重跑當下的 property_data.json」fuzzy-join 的 idx。
            # 房源擴量後須重跑本 builder,relevant_idxs 才會對到新 idx(否則指錯房源)。
            "joined_property_count": len(properties),
            "caveat_new_listings": (
                "GT 來源是 recommendation_train.json 的 query-property pair(舊房源)。"
                "擴量新增的房源幾乎不在 train pair 內,故鮮少成為 relevant_idxs。"
                "擴量後 recall 絕對值下降主要是『候選池變大、relevant 標的固定』的數學必然,"
                "非品質退步。要公平評估新房源需人工/LLM 補標(stage4 第二輪)。"
            ),
            "notes": (
                "Semantic bucket = PROVEN recall blind spots: query carries a "
                "colloquial trigger (data/semantic_rules.json `rules` key, "
                "feature word absent from query) AND the T0 rule-based recall "
                "(INTENT_MAP expansion included -- the real A/B baseline) misses "
                ">=1 relevant listing at K=30, so vector recall has room to win. "
                "Empirical gate matters because every `rules` key is also in the "
                "baseline's INTENT_MAP, so trigger-membership alone is not enough. "
                "Keyword bucket = control (literal feature/location/budget, no "
                "qualifying trigger). relevant_idxs are property_data idx from T0 "
                "build_ground_truth (relevance>=1); no re-join needed in T7. "
                f"trigger-bearing queries={n_trigger}, of which "
                f"semantic(blind)={len(semantic_q)}; keyword={len(keyword_kept)} "
                f"(capped at {max_keyword} from {len(keyword_q)} available)."
            ),
        },
        "queries": queries,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-keyword", type=int, default=MAX_KEYWORD_DEFAULT,
                    help="cap on the keyword control bucket size")
    args = ap.parse_args()

    out = build(args.max_keyword)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    queries = out["queries"]
    sem = [q for q in queries if q["bucket"] == "semantic"]
    kw = [q for q in queries if q["bucket"] == "keyword"]
    print("=" * 64)
    print(" T1 A/B EVAL QUERY SET")
    print("=" * 64)
    print(f" Output            : {OUT.relative_to(ROOT)}")
    print(f" join_match_rate   : {out['meta']['join_match_rate']*100:.1f}%")
    print(f" semantic queries  : {len(sem)}")
    print(f" keyword queries   : {len(kw)}")
    print(f" TOTAL             : {len(queries)}")
    print("-" * 64)

    # Verification: sample semantic queries with trigger + the MISSED relevant
    # prop text (confirms the rule-based baseline really fails to retrieve it).
    properties = t0.load_properties()
    idx_to_text = {p["idx"]: (p.get("text") or "") for p in properties}
    print(" Sample SEMANTIC queries (PROVEN blind spots vs rule-based@30):")
    for q in sem[:12]:
        ranked = t0.rule_based_recall(properties, q["query"], k=t0.K_VECTOR)
        got = {it["prop"]["idx"] for it in ranked}
        missed = [i for i in q["relevant_idxs"] if i not in got]
        rec = len(set(q["relevant_idxs"]) & got) / len(q["relevant_idxs"])
        show = (missed or q["relevant_idxs"])[0]
        ptext = idx_to_text.get(show, "")[:64]
        print(f"  [{q['semantic_trigger']}] {q['query']}")
        print(f"      rule-based Recall@30={rec:.2f}; missed relevant idx "
              f"{show}: {ptext}")
    print("-" * 64)
    print(" Sample KEYWORD control queries:")
    for q in kw[:5]:
        print(f"  ({q['n_relevant']} rel) {q['query']}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
