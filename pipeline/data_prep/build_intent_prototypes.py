"""Build intent prototype vectors for the bi-encoder fallback layer.

Single source: data/semantic_rules.json (the same 132 rules used by
sync_semantic_rules.py). For each rule KEY (the colloquial intent phrase,
e.g. "想養狗"), encode it with the bi-encoder and store the L2-normalized
sentence vector. At query time the frontend encodes a single user query with
the SAME encoder (same tokenizer + mean-pool + L2 norm) and takes cosine
against these prototypes; a hit above THR routes to that rule's expansion.

This is the offline half of the alternative architecture from
semantic_expansion_overhaul: prototypes are precomputed once here, so the
frontend only ever runs ONE encode (the query) instead of 1 + 132.

Encoder spec (MUST stay in lockstep with the frontend encoder, else cosine is
meaningless):
  - model:      shibing624/text2vec-base-chinese  (true bi-encoder / dual tower)
  - pooling:    mean-pool over attention_mask
  - postproc:   L2 normalize
  - tokenizer:  padding + truncation, max_length=64

Deliberately standalone: does NOT import or depend on the un-committed local
verify_intent_encoder.py / distill_intent_encoder.py from a previous session,
and does NOT reuse the stale 136-rule intent_prototypes.json (this repo is now
at 132 rules after the geo-term removal).

Usage:
    # Verify rule source is in sync WITHOUT loading the model (CI / regression):
    python pipeline/data_prep/build_intent_prototypes.py --check

    # Build prototypes (requires torch + transformers; downloads model once):
    python pipeline/data_prep/build_intent_prototypes.py

Output: data/intent_prototypes.json
    {
      "model": "shibing624/text2vec-base-chinese",
      "dim": 768,
      "thr": 0.55,          # cosine threshold for a fallback hit
      "top_k": 3,           # max prototypes to route to per query
      "rules": { "<intent>": "<space-joined expansion>", ... },
      "prototypes": { "<intent>": [<dim floats>], ... }
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "semantic_rules.json"
OUT = ROOT / "data" / "intent_prototypes.json"

MODEL = "shibing624/text2vec-base-chinese"
DIM = 768
MAX_LENGTH = 64
# Fallback tuning (validated in semantic_expansion_overhaul: thr=0.55 recovers
# 83% of literal-miss colloquial queries with text2vec's 0.294 anisotropy).
THR = 0.55
TOP_K = 3


def load_rules() -> dict[str, str]:
    """Return {intent_phrase: space-joined expansion} from the canonical file.

    semantic_rules.json stores rules as {key: [token, ...]}; we join tokens
    with spaces so the prototype payload mirrors expandQueryIntent's intentMap
    (which appends the space-joined string).
    """
    data = json.loads(CANON.read_text(encoding="utf-8"))
    rules = data["rules"]
    return {k: " ".join(v) for k, v in rules.items()}


def check(rules: dict[str, str]) -> int:
    """Verify the canonical source is well-formed and, if prototypes already
    exist, that they are in sync with the current rule set. No model load."""
    if not rules:
        print("[check] FAIL: no rules in semantic_rules.json", file=sys.stderr)
        return 1
    print(f"[check] semantic_rules.json: {len(rules)} rules")

    if not OUT.exists():
        print(f"[check] {OUT.name} not built yet (skeleton stage) — OK")
        print("[check] run without --check to build (needs torch + transformers)")
        return 0

    proto = json.loads(OUT.read_text(encoding="utf-8"))
    p_rules = proto.get("rules", {})
    p_vecs = proto.get("prototypes", {})

    errs = []
    if proto.get("model") != MODEL:
        errs.append(f"model mismatch: {proto.get('model')} != {MODEL}")
    if set(p_rules) != set(rules):
        missing = set(rules) - set(p_rules)
        extra = set(p_rules) - set(rules)
        errs.append(f"rule keys out of sync (missing {len(missing)}, extra {len(extra)})")
    for k in rules:
        v = p_vecs.get(k)
        if not isinstance(v, list) or len(v) != proto.get("dim", DIM):
            errs.append(f"prototype for {k!r} missing or wrong dim")
            break

    if errs:
        for e in errs:
            print(f"[check] FAIL: {e}", file=sys.stderr)
        print("[check] rebuild prototypes: python pipeline/data_prep/build_intent_prototypes.py",
              file=sys.stderr)
        return 1
    print(f"[check] {OUT.name}: {len(p_vecs)} prototypes in sync with rules — OK")
    return 0


def build(rules: dict[str, str]) -> int:
    """Encode each intent phrase and write prototypes. Imports heavy deps lazily
    so --check stays dependency-free."""
    try:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        print(f"[build] missing dependency: {e}\n"
              f"        pip install torch transformers", file=sys.stderr)
        return 1

    def mean_pool(last_hidden, mask):
        m = mask.unsqueeze(-1).float()
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def l2norm(v):
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()

    keys = list(rules)
    with torch.no_grad():
        enc = tok(keys, padding=True, truncation=True,
                  max_length=MAX_LENGTH, return_tensors="pt")
        out = model(**enc).last_hidden_state
        vecs = l2norm(mean_pool(out, enc["attention_mask"]))

    vecs = vecs.cpu().numpy()
    dim = int(vecs.shape[1])
    if dim != DIM:
        print(f"[build] WARN: encoder dim {dim} != expected {DIM}", file=sys.stderr)

    prototypes = {k: [round(float(x), 5) for x in vecs[i]] for i, k in enumerate(keys)}
    payload = {
        "model": MODEL,
        "dim": dim,
        "thr": THR,
        "top_k": TOP_K,
        "rules": rules,
        "prototypes": prototypes,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"[build] {OUT.name}: {len(prototypes)} prototypes × {dim}d "
          f"({size_kb:.0f} KB)")
    print("[build] frontend must encode queries with the SAME model/pooling/"
          "L2-norm for cosine to be valid")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify rule source / existing prototypes are in sync; no model load")
    args = ap.parse_args()

    if not CANON.exists():
        print(f"[error] canonical rules not found: {CANON}", file=sys.stderr)
        return 1

    rules = load_rules()
    return check(rules) if args.check else build(rules)


if __name__ == "__main__":
    raise SystemExit(main())
