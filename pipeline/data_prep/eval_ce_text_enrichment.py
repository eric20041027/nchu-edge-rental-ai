"""A/B the CE text-layer enrichment: raw prop.text vs deduped buildCEText.

The cross-encoder (65% of the final score) was fed prop.text — a short
structured token string. nchu's prop.text lacks the structural-field-derived
feature words dd carries, so nchu was under-ranked at the text layer
(data_source_misalignment residual). We now feed buildCEText (buildPropText,
deduped). This harness quantifies the per-source CE-score shift to confirm
nchu rises WITHOUT dd regressing.

Runs the SAME production CE ONNX (my_custom_model_quant.onnx) + tokenizer the
frontend uses. For each feature query, scores every property OLD (text) vs NEW
(buildCEText), then reports mean score by source.

Usage:
    python pipeline/data_prep/eval_ce_text_enrichment.py
    python pipeline/data_prep/eval_ce_text_enrichment.py --sample 120   # subsample props for speed
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
PROPS = ROOT / "frontend" / "assets" / "property_data.json"
MAX_LENGTH = 128

# mirror inference.js BOOL_FIELD_FEATURES + PROP_SYNONYMS-relevant assembly.
BOOL_FIELD_FEATURES = {
    "has_elevator": "電梯", "has_window": "對外窗", "has_balcony": "陽台",
    "has_parking": "車位 停車場", "has_waste_disposal": "垃圾處理",
    "is_rooftop": "頂樓", "water_dispenser": "飲水機", "private_washer": "獨洗",
    "has_subsidy": "補助", "is_taipower": "台電",
}

# feature queries where nchu was historically under-ranked at the text layer.
QUERIES = [
    "我要有電梯的套房", "要有陽台", "想找有冰箱的房間", "需要保全管理員比較安全",
    "有對外窗採光好的", "要有飲水機", "可以停機車的", "獨立電錶台電計費",
]


def build_prop_text(p: dict) -> str:
    parts = [p.get("text") or ""]
    for f in ("furniture", "features", "building_type", "room_type"):
        v = p.get(f)
        if v:
            parts.append(str(v).replace("/", " "))
    for f in ("notes", "other_fees"):
        v = p.get(f)
        if isinstance(v, list):
            parts.append(" ".join(v))
    for bk, wd in BOOL_FIELD_FEATURES.items():
        if p.get(bk) is True:
            parts.append(wd)
    eb = p.get("electricity_billing")
    if eb and eb != "不明":
        parts.append(str(eb))
    return " ".join(parts)


def build_ce_text(p: dict) -> str:
    seen, out = set(), []
    for tok in build_prop_text(p).split():
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return " ".join(out)


def src(p: dict) -> str:
    return "nchu" if "nchu" in (p.get("url") or "") else "dd"


def make_scorer(tok, sess):
    def score(query: str, text: str) -> float:
        enc = tok(query, text, max_length=MAX_LENGTH, padding="max_length",
                  truncation=True, return_tensors="np")
        logits = sess.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc.get("token_type_ids",
                                      np.zeros_like(enc["input_ids"])).astype(np.int64),
        })[0][0]
        return float(logits) if logits.ndim == 0 else float(logits[0])
    return score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="subsample N props per source for speed (0=all)")
    args = ap.parse_args()

    props = json.loads(PROPS.read_text(encoding="utf-8"))
    dd = [p for p in props if src(p) == "dd"]
    nchu = [p for p in props if src(p) == "nchu"]
    if args.sample:
        dd, nchu = dd[:args.sample], nchu[:args.sample]
    print(f"props: dd={len(dd)} nchu={len(nchu)} | CE={MODEL.name}")

    tok = BertTokenizerFast.from_pretrained(str(TOK_DIR))
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    score = make_scorer(tok, sess)

    # aggregate mean CE score per source, OLD vs NEW, across all queries
    agg = {"dd": {"old": [], "new": []}, "nchu": {"old": [], "new": []}}
    print(f"\n{'query':<22} {'src':>4} {'old':>7} {'new':>7} {'Δ':>7}")
    for q in QUERIES:
        for label, group in (("dd", dd), ("nchu", nchu)):
            olds = [score(q, p.get("text") or "") for p in group]
            news = [score(q, build_ce_text(p)) for p in group]
            agg[label]["old"] += olds
            agg[label]["new"] += news
            mo, mn = np.mean(olds), np.mean(news)
            print(f"{q:<22} {label:>4} {mo:>7.3f} {mn:>7.3f} {mn-mo:>+7.3f}")

    print("\n===== OVERALL mean CE score (sigmoid-ish logit) =====")
    for label in ("dd", "nchu"):
        mo, mn = np.mean(agg[label]["old"]), np.mean(agg[label]["new"])
        print(f"  {label:>4}: old={mo:.3f}  new={mn:.3f}  Δ={mn-mo:+.3f}")
    gap_old = np.mean(agg["dd"]["old"]) - np.mean(agg["nchu"]["old"])
    gap_new = np.mean(agg["dd"]["new"]) - np.mean(agg["nchu"]["new"])
    print(f"\n  dd–nchu gap: old={gap_old:+.3f}  new={gap_new:+.3f}  "
          f"(narrowed by {gap_old-gap_new:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
