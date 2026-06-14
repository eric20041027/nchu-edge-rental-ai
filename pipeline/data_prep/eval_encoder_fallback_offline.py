"""Offline ceiling test for the bi-encoder intent-layer fallback.

Goal (P0 go/no-go, see open_todos / semantic_expansion_overhaul):
  Before spending 205MB of transformers.js on the frontend, prove OFFLINE that
  the encoder fallback actually earns its weight. Concretely:

    1. Replicate the frontend literal layer (expandQueryIntent in inference.js)
       in Python to find the LITERAL-MISS subset: colloquial queries the 132
       hand-written rules do NOT catch (these are the ones fallback exists for).
    2. Encode the literal-miss queries with text2vec (SAME model / mean-pool /
       L2-norm as build_intent_prototypes.py), cosine vs the 132 prototypes,
       apply thr=0.55 / top-3 + the negation guard.
    3. Judge each hit with a CATEGORY heuristic: every eval query carries a
       topic category (寵物衝突 / 設備衝突 / 噪音衝突 / ...). We map each of the
       132 rules to one of those topics. A fallback hit is:
         - CORRECT   : top-1 routed rule's topic == query's category topic
         - MISROUTE  : top-1 routed rule's topic conflicts with query's topic
         - MISS      : nothing >= thr (fallback stayed silent)

Decision rule (from the task brief): if literal-miss queries are caught
correctly at a high enough rate (e.g. >50%), it is worth wiring up
transformers.js; if not, prefer distillation or dropping the fallback.

The category heuristic is an APPROXIMATION (the rule->topic map is hand-made),
so the headline number is a ceiling estimate, not gospel. Queries with no
topical category (semantic_positive / hard_negative) are reported separately
and NEVER counted in the headline correct-rate.

Encoder spec MUST match build_intent_prototypes.py exactly, else cosine is
meaningless:
  model=shibing624/text2vec-base-chinese, mean-pool over mask, L2 norm,
  max_length=64.

Usage:
    python pipeline/data_prep/eval_encoder_fallback_offline.py            # full run (needs torch)
    python pipeline/data_prep/eval_encoder_fallback_offline.py --limit 200
    python pipeline/data_prep/eval_encoder_fallback_offline.py --dump-misroutes 40
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "semantic_rules.json"
CORPORA = [ROOT / "data" / "raw" / "llm_queries.json",
           ROOT / "data" / "raw" / "hard_traps.json"]

MODEL = "shibing624/text2vec-base-chinese"
MAX_LENGTH = 64
THR = 0.55
TOP_K = 3
NEGATORS = "不沒無非免勿"

# --- Category -> topic. The corpora label each query with the topic it WANTS
#     (the "衝突" categories mean: query intends this topic, property violates it). ---
CATEGORY_TOPIC = {
    "寵物衝突": "pet",
    "設備衝突": "appliance",
    "費用衝突": "budget",
    "地點衝突": "location",
    "噪音衝突": "noise",
    "自炊衝突": "cooking",
    "女生安全衝突": "safety",
    "衛浴獨立衝突": "bath",
    "停車衝突": "parking",
}
# Untyped categories: no topic to verify against. Reported, never in headline.
UNTYPED = {"semantic_positive", "hard_negative", "?"}

# --- Rule (intent key) -> topic, keyed off the rule's expansion tokens. This is
#     the hand-made approximation the judge rests on. Built from semantic_rules:
#     a rule belongs to topic T if its expansion contains any of T's anchor tokens.
TOPIC_ANCHORS = {
    "pet":       ["可寵", "養寵", "可養狗", "可養貓", "寵物"],
    "cooking":   ["可伙", "廚房", "瓦斯爐", "電磁爐", "流理台", "開火", "自炊", "排油煙機", "抽油煙機", "天然瓦斯"],
    "appliance": ["冰箱", "洗衣機", "全家電", "全配", "全家具", "書桌", "床架", "床墊", "熱水器", "冷氣", "變頻"],
    "budget":    ["低租金", "經濟實惠", "實惠", "便宜", "低價", "補助", "租補", "台電", "台水"],
    "location":  ["走路10分", "騎車10分", "興大路", "機車停車位", "車位", "停車場", "停車"],
    "noise":     ["隔音", "氣密窗", "靜巷", "禁菸"],
    "safety":    ["管理員", "門禁", "監視器", "女性友善", "刷卡"],
    "bath":      ["獨衛", "獨立衛浴", "套房", "浴缸"],
    "parking":   ["車位", "停車場", "機車停車位", "停車"],
}
# location/parking overlap on parking tokens; resolve parking first when both match.
TOPIC_PRIORITY = ["pet", "cooking", "safety", "bath", "noise", "parking",
                  "location", "appliance", "budget"]


def load_rules() -> dict[str, list[str]]:
    return json.loads(CANON.read_text(encoding="utf-8"))["rules"]


def rule_topics(rules: dict[str, list[str]]) -> dict[str, str | None]:
    """Map each rule key to a single topic via its expansion tokens."""
    out: dict[str, str | None] = {}
    for key, expansion in rules.items():
        joined = " ".join(expansion)
        topic = None
        for t in TOPIC_PRIORITY:
            if any(tok in joined for tok in TOPIC_ANCHORS[t]):
                topic = t
                break
        out[key] = topic
    return out


def literal_expand(query: str, rules: dict[str, list[str]]) -> bool:
    """Return True iff the literal layer (expandQueryIntent) would catch query.

    Mirrors inference.js: substring match of each intent key with a negation
    guard (skip a hit whose preceding char is a negator; sentence-start idx==0
    is never negated). A query is a literal HIT if any rule fires un-negated.
    """
    for intent in rules:
        from_ = 0
        while (idx := query.find(intent, from_)) != -1:
            negated = idx > 0 and query[idx - 1] in NEGATORS
            if not negated:
                return True
            from_ = idx + 1
    return False


def load_corpus() -> list[dict]:
    rows = []
    for path in CORPORA:
        if not path.exists():
            print(f"[warn] corpus missing: {path}", file=sys.stderr)
            continue
        for r in json.loads(path.read_text(encoding="utf-8")):
            q = r.get("query")
            if q:
                rows.append({"query": q, "category": r.get("category", "?"),
                             "src": path.name})
    # de-dupe by query, keep first (categories are consistent within a corpus)
    seen, uniq = set(), []
    for r in rows:
        if r["query"] not in seen:
            seen.add(r["query"])
            uniq.append(r)
    return uniq


def build_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()

    def mean_pool(last_hidden, mask):
        m = mask.unsqueeze(-1).float()
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def encode(texts: list[str]):
        import numpy as np
        vecs = []
        B = 64
        with torch.no_grad():
            for i in range(0, len(texts), B):
                batch = texts[i:i + B]
                enc = tok(batch, padding=True, truncation=True,
                          max_length=MAX_LENGTH, return_tensors="pt")
                out = model(**enc).last_hidden_state
                v = mean_pool(out, enc["attention_mask"])
                v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                vecs.append(v.cpu().numpy())
        return np.concatenate(vecs, 0)

    return encode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="cap literal-miss queries encoded (debug/speed)")
    ap.add_argument("--dump-misroutes", type=int, default=0,
                    help="print N example misroutes for inspection")
    ap.add_argument("--dump-correct", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep thr x top_k over typed literal-miss and exit")
    args = ap.parse_args()

    rules = load_rules()
    r_topic = rule_topics(rules)
    keys = list(rules)
    print(f"[setup] rules={len(rules)}  "
          f"rules-with-topic={sum(t is not None for t in r_topic.values())}")

    corpus = load_corpus()
    print(f"[setup] corpus unique queries: {len(corpus)}")

    # 1) literal coverage split
    miss = [r for r in corpus if not literal_expand(r["query"], rules)]
    hit = len(corpus) - len(miss)
    print(f"[literal] caught={hit} ({hit/len(corpus):.1%})  "
          f"miss={len(miss)} ({len(miss)/len(corpus):.1%})")

    if args.limit:
        miss = miss[:args.limit]
        print(f"[literal] (limited to {len(miss)} miss queries)")

    # 2) encode literal-miss + cosine vs prototypes
    import numpy as np
    encode = build_encoder()
    proto = encode(keys)                       # (132, dim)
    qv = encode([r["query"] for r in miss])    # (N, dim)
    sims = qv @ proto.T                         # cosine (both L2-normed)

    if args.sweep:
        typed = [(i, r) for i, r in enumerate(miss)
                 if r["category"] not in UNTYPED and CATEGORY_TOPIC.get(r["category"])]
        print(f"\ntyped literal-miss judged: {len(typed)}")
        print(f"{'thr':>5} {'top_k':>5} {'correct':>8} {'misroute':>9} {'miss':>6} {'prec':>6}")
        for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
            for top_k in (1, 3):
                sw = Counter()
                for i, r in typed:
                    topic = CATEGORY_TOPIC[r["category"]]
                    routed = []
                    for j in np.argsort(-sims[i]):
                        if sims[i][j] < thr:
                            break
                        intent = keys[j]
                        kidx = r["query"].find(intent)
                        if not (kidx > 0 and r["query"][kidx - 1] in NEGATORS):
                            routed.append(intent)
                        if len(routed) >= top_k:
                            break
                    if not routed:
                        sw["miss"] += 1
                    elif r_topic.get(routed[0]) == topic:
                        sw["correct"] += 1
                    else:
                        sw["misroute"] += 1
                n = sum(sw.values())
                fired = sw["correct"] + sw["misroute"]
                prec = sw["correct"] / fired if fired else 0
                print(f"{thr:>5.2f} {top_k:>5} {sw['correct']/n:>7.1%} "
                      f"{sw['misroute']/n:>8.1%} {sw['miss']/n:>5.1%} {prec:>5.1%}")
        return 0

    # 3) judge per query
    cnt = Counter()                 # outcome among TYPED literal-miss queries
    untyped_fired = 0
    untyped_total = 0
    per_topic = defaultdict(Counter)
    misroute_ex, correct_ex = [], []

    for i, r in enumerate(miss):
        cat = r["category"]
        row = sims[i]
        order = np.argsort(-row)
        # thr + top_k + negation guard (mirror encoderFallbackExpand)
        routed = []
        for j in order:
            if row[j] < THR:
                break
            intent = keys[j]
            kidx = r["query"].find(intent)
            negated = kidx > 0 and r["query"][kidx - 1] in NEGATORS
            if not negated:
                routed.append((intent, float(row[j])))
            if len(routed) >= TOP_K:
                break

        if cat in UNTYPED:
            untyped_total += 1
            if routed:
                untyped_fired += 1
            continue

        topic = CATEGORY_TOPIC.get(cat)
        if topic is None:
            continue
        if not routed:
            cnt["miss"] += 1
            per_topic[topic]["miss"] += 1
            continue
        top_intent, top_score = routed[0]
        top_topic = r_topic.get(top_intent)
        if top_topic == topic:
            cnt["correct"] += 1
            per_topic[topic]["correct"] += 1
            if len(correct_ex) < args.dump_correct:
                correct_ex.append((r["query"], top_intent, top_score))
        else:
            cnt["misroute"] += 1
            per_topic[topic]["misroute"] += 1
            if len(misroute_ex) < args.dump_misroutes:
                misroute_ex.append((r["query"], cat, top_intent, top_topic, top_score))

    typed_total = sum(cnt.values())
    print("\n===== HEADLINE (typed literal-miss queries) =====")
    print(f"typed literal-miss queries judged: {typed_total}")
    if typed_total:
        c, m, ms = cnt["correct"], cnt["miss"], cnt["misroute"]
        print(f"  CORRECT  (routed to right topic): {c:5d}  ({c/typed_total:.1%})")
        print(f"  MISROUTE (routed to wrong topic): {ms:5d}  ({ms/typed_total:.1%})")
        print(f"  MISS     (fallback stayed silent):{m:5d}  ({m/typed_total:.1%})")
        fired = c + ms
        if fired:
            print(f"  precision-of-fired (correct / fired): {c/fired:.1%}")
    print(f"\nuntyped queries (semantic_positive/hard_negative): "
          f"{untyped_fired}/{untyped_total} fired "
          f"({untyped_fired/untyped_total:.1%})" if untyped_total else "")

    print("\n----- per-topic (typed) -----")
    for t in TOPIC_PRIORITY:
        pc = per_topic.get(t)
        if not pc:
            continue
        tot = sum(pc.values())
        print(f"  {t:9s} n={tot:4d}  correct={pc['correct']/tot:5.1%}  "
              f"misroute={pc['misroute']/tot:5.1%}  miss={pc['miss']/tot:5.1%}")

    if misroute_ex:
        print("\n----- example MISROUTES -----")
        for q, cat, intent, ttopic, s in misroute_ex:
            print(f"  [{cat}] {q!r}\n      -> {intent!r} (topic={ttopic}, cos={s:.3f})")
    if correct_ex:
        print("\n----- example CORRECT -----")
        for q, intent, s in correct_ex:
            print(f"  {q!r} -> {intent!r} (cos={s:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
