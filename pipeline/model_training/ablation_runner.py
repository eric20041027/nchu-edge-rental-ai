"""
ablation_runner.py — Main entry point for the full ablation study.

Usage:
    python -m pipeline.model_training.ablation_runner

Directory layout:
    <project_root>/ablation_results/           RESULTS_ROOT
    <project_root>/saved_models/ablation/      MODELS_ROOT

Run order:
    1. Print banner with all 10 runs + configs
    2. Run all 10 sequentially (skip if metrics.json exists — resumable)
    3. Aggregate and print summary table → ablation_results/summary.json
    4. Group D: evaluate REF and C2_no_FGM checkpoints on noisy_test.json
    5. Regenerate summary including D results
"""
import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import torch

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

RESULTS_ROOT = os.path.join(_PROJECT_ROOT, "ablation_results")
MODELS_ROOT  = os.path.join(_PROJECT_ROOT, "saved_models", "ablation")

from .ablation_config import AblationConfig, ALL_ABLATION_RUNS
from .ablation_train  import run_ablation, compute_flat_ndcg5, STUDENT_CHECKPOINT, MAX_LENGTH

from transformers import BertTokenizerFast, AutoModelForSequenceClassification


# ── Banner ────────────────────────────────────────────────────────────────────

def _print_banner():
    print("\n" + "=" * 72)
    print("  ABLATION STUDY — Chinese Rental Recommendation (rbt6->rbt3 KD)")
    print(f"  Results root : {RESULTS_ROOT}")
    print(f"  Models root  : {MODELS_ROOT}")
    print(f"  Total runs   : {len(ALL_ABLATION_RUNS)} training runs + Group D evaluation")
    print("=" * 72)
    print(f"  {'Run ID':<25} {'Group':<6} {'Description'}")
    print("  " + "-" * 68)
    for cfg in ALL_ABLATION_RUNS:
        ref_tag = " [REF]" if cfg.is_reference else ""
        print(f"  {cfg.run_id:<25} {cfg.group:<6} {cfg.description}{ref_tag}")
    print("=" * 72 + "\n")


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary_table(all_metrics: List[dict], reference_ndcg: float):
    print("\n" + "=" * 72)
    print("  ABLATION STUDY RESULTS")
    print("=" * 72)
    header = f"  {'Run ID':<22} {'Group':<6} {'F1':>8} {'NDCG@5':>9} {'ΔNDCG':>8}"
    print(header)
    print("  " + "-" * 68)
    for m in all_metrics:
        delta = m["ndcg_at_5"] - reference_ndcg
        delta_str = f"{delta:+.4f}" if not m.get("is_reference") else "  —"
        print(f"  {m['run_id']:<22} {m.get('group','?'):<6} "
              f"{m['f1']*100:>7.2f}% {m['ndcg_at_5']:>9.4f} {delta_str:>8}")
    print("=" * 72 + "\n")


# ── Save summary JSON ─────────────────────────────────────────────────────────

def _save_summary(all_metrics: List[dict], reference_ndcg: float):
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    summary = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "reference_ndcg":  reference_ndcg,
        "runs": [
            {
                "run_id":      m["run_id"],
                "group":       m.get("group", "?"),
                "description": m.get("description", ""),
                "ndcg_at_5":   m["ndcg_at_5"],
                "f1":          m["f1"],
                "precision":   m.get("precision", 0.0),
                "recall":      m.get("recall", 0.0),
                "accuracy":    m.get("accuracy", 0.0),
                "delta_ndcg":  round(m["ndcg_at_5"] - reference_ndcg, 6),
            }
            for m in all_metrics
        ],
    }
    summary_path = os.path.join(RESULTS_ROOT, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Summary saved: {summary_path}")
    return summary_path


# ── Load saved metrics (with group + description enrichment) ──────────────────

def _load_metrics(cfg: AblationConfig) -> Optional[dict]:
    p = os.path.join(RESULTS_ROOT, cfg.run_id, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["group"]       = cfg.group
    m["description"] = cfg.description
    m["is_reference"] = cfg.is_reference
    return m


# ── Group D: noisy evaluation ─────────────────────────────────────────────────

def _run_group_d(device: str):
    """Evaluate REF and C2_no_FGM checkpoints on noisy_test.json."""
    noisy_path = os.path.join(_PROJECT_ROOT, "data", "processed", "noisy_test.json")
    if not os.path.exists(noisy_path):
        print(f"\n[Group D] noisy_test.json not found at: {noisy_path}")
        print("  Generate it first: python -m pipeline.data_prep.noise_generator")
        print("  Skipping Group D evaluation.\n")
        return []

    with open(noisy_path, "r", encoding="utf-8") as f:
        noisy_data = json.load(f)
    print(f"\n[Group D] noisy_test.json loaded: {len(noisy_data)} samples")

    tokenizer = BertTokenizerFast.from_pretrained(STUDENT_CHECKPOINT)

    d_results = []
    for run_id, d_run_id in [("REF_v29", "D_noisy_with_FGM"), ("C2_no_FGM", "D_noisy_no_FGM")]:
        out_dir  = os.path.join(RESULTS_ROOT, d_run_id)
        d_metrics_path = os.path.join(out_dir, "metrics.json")

        if os.path.exists(d_metrics_path):
            print(f"  [skip] {d_run_id}: metrics.json found.")
            with open(d_metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            m["group"]       = "D"
            m["description"] = f"Noisy test on {run_id} checkpoint"
            d_results.append(m)
            continue

        model_dir = os.path.join(MODELS_ROOT, run_id)
        if not os.path.isdir(model_dir):
            print(f"  [warn] Model checkpoint not found: {model_dir} — skipping {d_run_id}")
            continue

        print(f"  Evaluating {d_run_id} (checkpoint: {run_id})...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=2, ignore_mismatched_sizes=True,
        )
        model.to(device)
        model.eval()

        ndcg5 = compute_flat_ndcg5(model, tokenizer, noisy_data, device)
        print(f"    NDCG@5 on noisy test: {ndcg5:.4f}")

        os.makedirs(out_dir, exist_ok=True)
        d_m = {
            "run_id":      d_run_id,
            "ndcg_at_5":   round(ndcg5, 6),
            "f1":          0.0,
            "precision":   0.0,
            "recall":      0.0,
            "accuracy":    0.0,
            "source_run":  run_id,
            "test_set":    "noisy_test.json",
            "group":       "D",
            "description": f"Noisy test on {run_id} checkpoint",
        }
        with open(d_metrics_path, "w", encoding="utf-8") as f:
            json.dump(d_m, f, ensure_ascii=False, indent=2)
        d_results.append(d_m)

    return d_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(MODELS_ROOT,  exist_ok=True)

    _print_banner()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] No CUDA GPU detected. Training will be very slow or fail.")

    # ── Step 1-2: Run all training ablations ──────────────────────────────────
    all_metrics: List[dict] = []
    reference_ndcg = 0.0

    for cfg in ALL_ABLATION_RUNS:
        try:
            m = run_ablation(cfg, RESULTS_ROOT, MODELS_ROOT)
            m["group"]       = cfg.group
            m["description"] = cfg.description
            m["is_reference"] = cfg.is_reference
            all_metrics.append(m)
            if cfg.is_reference:
                reference_ndcg = m["ndcg_at_5"]
        except Exception as exc:
            print(f"\n[ERROR] Run {cfg.run_id} failed: {exc}")
            # Attempt to load partial metrics if they exist
            existing = _load_metrics(cfg)
            if existing:
                all_metrics.append(existing)
                if cfg.is_reference:
                    reference_ndcg = existing["ndcg_at_5"]
            else:
                all_metrics.append({
                    "run_id":      cfg.run_id,
                    "group":       cfg.group,
                    "description": cfg.description,
                    "is_reference": cfg.is_reference,
                    "ndcg_at_5":   0.0,
                    "f1":          0.0,
                    "precision":   0.0,
                    "recall":      0.0,
                    "accuracy":    0.0,
                    "error":       str(exc),
                })

    # ── Step 3: Summary after training runs ───────────────────────────────────
    _print_summary_table(all_metrics, reference_ndcg)
    _save_summary(all_metrics, reference_ndcg)

    # ── Step 4: Group D noisy evaluation ─────────────────────────────────────
    d_results = _run_group_d(device)

    if d_results:
        all_metrics_with_d = all_metrics + d_results
        _print_summary_table(all_metrics_with_d, reference_ndcg)
        _save_summary(all_metrics_with_d, reference_ndcg)
        print("\n[Done] All ablation runs complete (including Group D).")
    else:
        print("\n[Done] Training ablation runs complete. Group D skipped or no results.")


if __name__ == "__main__":
    main()
