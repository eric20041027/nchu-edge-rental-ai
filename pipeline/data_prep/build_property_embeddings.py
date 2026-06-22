"""T4 — Precompute property embeddings with the trained bi-encoder -> frontend JSON.

The offline half of vector retrieval: encode every property in
``frontend/assets/property_data.json`` with the SAME trained bi-encoder the query
side uses (T3), and emit a static Float32/Float16 vector file the frontend loads
once. 同源 requirement: same model, same mask-aware mean-pool, same L2-normalize
as ``BiEncoder`` (train_bi_encoder.py) and as the exported query ONNX (T3). Because
both sides are unit-norm, cosine == dot product on-device.

This is a thin, focused script (not a class on EmbeddingPrecomputer): the existing
``EmbeddingPrecomputer`` defaults to MiniLM and writes a pickle for backend vector
search. T4 must instead use the trained rbt6 bi-encoder and emit FRONTEND STATIC
JSON, so reusing its loop machinery buys nothing — we mirror the proven
``build_intent_prototypes.py`` packaging precedent instead (dim + flat vectors +
metadata) so the frontend loader pattern is reusable.

=== Text field choice: `text` (NOT `ce_text`) — empirically grounded ===
T2 trained on ``recommendation_train.json``'s ``property`` field. Measured against
the 704 snapshot:
  * property_data ``text``    : 33/704 EXACT matches in train, avg len ~50 chars.
  * property_data ``ce_text`` : 0/704 exact matches, avg len ~134 chars (adds
    distance/walk/scooter minutes + an extended feature list).
The train ``property`` text (avg ~44 chars) is distributionally the ``text`` field,
not ``ce_text``. Embedding ``ce_text`` would push the document side out of the
distribution the encoder learned (同源 by MODEL is necessary but the INPUT TEXT
must also match what the encoder saw), hurting cosine recall. So we embed ``text``.
Override with --field ce_text if a future retrain uses the long form.

=== Output schema: frontend/assets/property_embeddings.json ===
    {
      "model": "rbt6_bi_encoder",   # provenance tag (matches T3 query encoder)
      "dim": 768,                    # embedding dimension (rbt6 hidden_size)
      "count": 704,                  # number of property vectors
      "dtype": "float16",            # "float16" (default) or "float32"
      "field": "text",               # which property field was embedded
      "idxs": [0, 1, 2, ...],        # property_data idx for each row, IN ORDER
      "vecs": [<count*dim numbers>]  # flat, row-major; reshape to (count, dim)
    }
Frontend reconstruction (mirrors initEncoderFallback's Float32Array.from):
    const flat = Float32Array.from(raw.vecs);   // float16 list -> JS numbers -> f32
    const dim  = raw.dim;
    // property i vector = flat.subarray(i*dim, (i+1)*dim)  (already L2-normalized)
All vectors are L2-normalized, so cosine(query, prop) == dot(query, prop).

=== File size / scaling (spec Success #4: ≤~5 MB increase at ~1萬筆) ===
JSON of rounded floats, not raw binary. float16 rounds vectors to 4 decimals
(~half the digits of float32's ~7), roughly halving the JSON payload with cosine
drift far below retrieval granularity (top-30 recall unaffected). Measured JSON
sizes (dim 768): 704 props -> float16 ≈ ~4.5 MB / float32 ≈ ~6.2 MB; ~1万筆 ->
float16 ≈ ~64 MB / float32 ≈ ~88 MB. So at 704, float16 fits the spec Success #4
≤5 MB budget while float32 (~6.2 MB) just overshoots it -> float16 is the default.
NOTE: at ~1万筆 even float16 JSON-of-floats blows the budget — scaling to 1万 will
need a binary .bin (base64 or a typed-array fetch) instead of a JSON float list.
704 is the current target; float16 keeps us in budget and is the tested path.

Usage (run on Colab/GPU AFTER T2 training):
    python -m pipeline.data_prep.build_property_embeddings
    python -m pipeline.data_prep.build_property_embeddings --dtype float32
    python -m pipeline.data_prep.build_property_embeddings --field ce_text
    python -m pipeline.data_prep.build_property_embeddings --check   # no model load

Requires trained weights at ``config.bi_encoder_saved_dir`` (or --saved-dir).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
PROPERTY_DATA = ROOT / "frontend" / "assets" / "property_data.json"
OUT = ROOT / "frontend" / "assets" / "property_embeddings.json"

MODEL_TAG = "rbt6_bi_encoder"
MAX_LENGTH = 64
DEFAULT_FIELD = "text"
DEFAULT_DTYPE = "float16"
# Rounding precision per dtype: float16 ~= 4 decimals, float32 ~= 7 decimals.
_ROUND = {"float16": 4, "float32": 7}


def load_properties(path: Path, field: str) -> tuple[List[int], List[str]]:
    """Return (idxs, texts) for every property, using the chosen text field.

    Falls back to ``text`` per-record if the chosen field is empty for that record
    (matches the spec's "ce_text if present else text" intent while defaulting to
    the distribution the encoder was trained on).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    idxs: List[int] = []
    texts: List[str] = []
    for rec in data:
        value = rec.get(field) or rec.get("text") or ""
        idxs.append(int(rec["idx"]))
        texts.append(str(value))
    return idxs, texts


def check(field: str) -> int:
    """Validate property_data + any existing output, WITHOUT loading the model."""
    if not PROPERTY_DATA.exists():
        print(f"[check] FAIL: {PROPERTY_DATA} not found", file=sys.stderr)
        return 1
    idxs, texts = load_properties(PROPERTY_DATA, field)
    empties = sum(1 for t in texts if not t.strip())
    print(f"[check] property_data.json: {len(idxs)} records, field={field!r}, "
          f"{empties} empty (fell back to text)")

    if not OUT.exists():
        print(f"[check] {OUT.name} not built yet — OK")
        print("[check] run without --check to build (needs torch + transformers + weights)")
        return 0

    payload = json.loads(OUT.read_text(encoding="utf-8"))
    dim = payload.get("dim")
    count = payload.get("count")
    vecs = payload.get("vecs", [])
    errs = []
    if count != len(payload.get("idxs", [])):
        errs.append(f"count {count} != len(idxs) {len(payload.get('idxs', []))}")
    if not isinstance(dim, int) or dim <= 0:
        errs.append(f"bad dim {dim}")
    elif len(vecs) != count * dim:
        errs.append(f"len(vecs) {len(vecs)} != count*dim {count * dim}")
    if errs:
        for e in errs:
            print(f"[check] FAIL: {e}", file=sys.stderr)
        return 1
    print(f"[check] {OUT.name}: {count} vecs x {dim}d in sync — OK")
    return 0


def build(field: str, dtype: str, saved_dir: Path, batch_size: int) -> int:
    """Encode all properties and write the frontend JSON. Heavy deps lazy-loaded."""
    if dtype not in _ROUND:
        print(f"[build] FAIL: --dtype must be one of {list(_ROUND)}", file=sys.stderr)
        return 1

    # Weights guard (helpful message, same intent as T3's guard).
    cfg_json = saved_dir / "config.json"
    has_weights = saved_dir.exists() and any(
        (saved_dir / n).exists() for n in ("pytorch_model.bin", "model.safetensors")
    )
    if not (cfg_json.exists() and has_weights):
        print(
            f"[build] FAIL: no trained bi-encoder weights at {saved_dir}\n"
            "        T4 requires the T2-trained weights (save_pretrained output).\n"
            "        Train on Colab/GPU first, or pass --saved-dir / set "
            "BI_ENCODER_SAVED_DIR.\n"
            "        Expected: config.json + (pytorch_model.bin | model.safetensors).",
            file=sys.stderr,
        )
        return 1

    try:
        import numpy as np
        import torch
        from transformers import AutoModel, BertTokenizerFast
    except ImportError as e:
        print(f"[build] missing dependency: {e}\n        pip install torch transformers",
              file=sys.stderr)
        return 1

    from pipeline.model_training.train_bi_encoder import BiEncoder

    idxs, texts = load_properties(PROPERTY_DATA, field)
    print(f"[build] encoding {len(texts)} properties (field={field!r}, dtype={dtype})")

    tokenizer = BertTokenizerFast.from_pretrained(str(saved_dir))
    encoder = AutoModel.from_pretrained(str(saved_dir))
    model = BiEncoder(encoder).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_vecs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            emb = model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            )  # (b, H) already mean-pooled + L2-normalized
            all_vecs.append(emb.cpu().numpy().astype(np.float32))

    vecs = np.concatenate(all_vecs, axis=0)  # (count, dim), unit-norm
    dim = int(vecs.shape[1])

    # float16 path: round-trip through np.float16 then back so the stored JSON
    # floats reflect the true on-device float16 precision (frontend re-normalizes
    # implicitly via cosine; drift is well below retrieval granularity).
    if dtype == "float16":
        vecs = vecs.astype(np.float16).astype(np.float32)
    decimals = _ROUND[dtype]

    flat = [round(float(x), decimals) for x in vecs.reshape(-1)]
    payload = {
        "model": MODEL_TAG,
        "dim": dim,
        "count": int(vecs.shape[0]),
        "dtype": dtype,
        "field": field,
        "idxs": idxs,
        "vecs": flat,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"[build] {OUT.name}: {vecs.shape[0]} vecs x {dim}d ({size_mb:.2f} MB, {dtype})")

    _validate_output(payload)
    return 0


def _validate_output(payload: dict) -> None:
    """CP3-style sanity: vectors are unit-norm after reconstruction."""
    import numpy as np

    dim = payload["dim"]
    count = payload["count"]
    arr = np.asarray(payload["vecs"], dtype=np.float32).reshape(count, dim)
    norms = np.linalg.norm(arr, axis=1)
    max_dev = float(np.abs(norms - 1.0).max())
    # float16 rounding loosens unit-norm slightly; 2e-2 is comfortably tight.
    assert max_dev < 2e-2, f"reconstructed vectors not unit-norm (max dev {max_dev})"
    print(f"[build] unit-norm check passed (max |‖v‖-1| = {max_dev:.2e})")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute property embeddings -> frontend static JSON (T4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--field", default=DEFAULT_FIELD, choices=["text", "ce_text"],
                        help="property field to embed (default: text — matches T2 train distribution)")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["float16", "float32"],
                        help="stored vector precision (default: float16, halves file size)")
    parser.add_argument("--saved-dir", default=None,
                        help="trained bi-encoder dir (default: config.bi_encoder_saved_dir)")
    parser.add_argument("--batch-size", type=int, default=64, help="encode batch size")
    parser.add_argument("--check", action="store_true",
                        help="validate property_data + existing output; no model load")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not PROPERTY_DATA.exists():
        print(f"[error] property data not found: {PROPERTY_DATA}", file=sys.stderr)
        return 1
    if args.check:
        return check(args.field)

    from pipeline.model_training.config import ModelTrainingConfig

    config = ModelTrainingConfig()
    saved_dir = Path(args.saved_dir) if args.saved_dir else config.bi_encoder_saved_dir
    return build(args.field, args.dtype, saved_dir, args.batch_size)


if __name__ == "__main__":
    raise SystemExit(main())
