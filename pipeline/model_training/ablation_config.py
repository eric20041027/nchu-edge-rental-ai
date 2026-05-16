"""
ablation_config.py — Configuration dataclass and run list for the ablation study.

Defines AblationConfig and ALL_ABLATION_RUNS (1 reference + 9 ablations across
groups A/B/C/D).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AblationConfig:
    """Configuration for a single ablation run."""

    run_id: str               # e.g. "A0_CE_only"
    group: str                # "REF", "A", "B", "C", "D"
    description: str          # human-readable label

    # ── Loss components ───────────────────────────────────────────────────────
    enable_ranknet: bool = True
    enable_listnet: bool = True

    # ── Regularization ────────────────────────────────────────────────────────
    enable_fgm: bool = True
    rdrop_alpha: float = 0.05

    # ── Knowledge Distillation ────────────────────────────────────────────────
    distill_alpha_max: float = 0.38
    distill_alpha_min: float = 0.12

    # ── Data augmentation ─────────────────────────────────────────────────────
    enable_noise_augment: bool = False   # mix noisy query copies into training set

    # ── Meta ──────────────────────────────────────────────────────────────────
    is_reference: bool = False   # True for the v2.9 baseline run

    def to_dict(self) -> dict:
        """Serialize all fields to a plain dict (JSON-serializable)."""
        return {
            "run_id":                self.run_id,
            "group":                 self.group,
            "description":           self.description,
            "enable_ranknet":        self.enable_ranknet,
            "enable_listnet":        self.enable_listnet,
            "enable_fgm":            self.enable_fgm,
            "rdrop_alpha":           self.rdrop_alpha,
            "distill_alpha_max":     self.distill_alpha_max,
            "distill_alpha_min":     self.distill_alpha_min,
            "enable_noise_augment":  self.enable_noise_augment,
            "is_reference":          self.is_reference,
        }


# ── All ablation runs ─────────────────────────────────────────────────────────
# Total: 1 reference + 9 ablations = 10 training runs.
# Group D does NOT require separate training — it evaluates REF and C2 checkpoints
# on a noisy test set and is handled by ablation_runner.py.

ALL_ABLATION_RUNS = [
    # ── Reference (= v2.9; reused as anchor for groups A/B/C) ────────────────
    AblationConfig(
        "REF_v29", "REF",
        "v2.9 Reference (CE+RankNet+ListNet+KD+RDrop+FGM)",
        is_reference=True,
    ),

    # ── Group A: Loss function ablation ──────────────────────────────────────
    AblationConfig(
        "A0_CE_only", "A",
        "CE only (Baseline)",
        enable_ranknet=False,
        enable_listnet=False,
    ),
    AblationConfig(
        "A1_CE_RankNet", "A",
        "CE + RankNet",
        enable_ranknet=True,
        enable_listnet=False,
    ),
    AblationConfig(
        "A2_CE_ListNet", "A",
        "CE + ListNet",
        enable_ranknet=False,
        enable_listnet=True,
    ),

    # ── Group B: KD alpha schedule ───────────────────────────────────────────
    AblationConfig(
        "B1_alpha_012", "B",
        "Fixed alpha=0.12",
        distill_alpha_max=0.12,
        distill_alpha_min=0.12,
    ),
    AblationConfig(
        "B2_alpha_025", "B",
        "Fixed alpha=0.25",
        distill_alpha_max=0.25,
        distill_alpha_min=0.25,
    ),
    AblationConfig(
        "B3_alpha_038", "B",
        "Fixed alpha=0.38",
        distill_alpha_max=0.38,
        distill_alpha_min=0.38,
    ),

    # ── Group C: Regularization ablation ─────────────────────────────────────
    AblationConfig(
        "C2_no_FGM", "C",
        "No FGM",
        enable_fgm=False,
    ),
    AblationConfig(
        "C3_no_RDrop", "C",
        "No R-Drop",
        rdrop_alpha=0.0,
    ),
    AblationConfig(
        "C4_no_FGM_no_RDrop", "C",
        "No FGM, No R-Drop",
        enable_fgm=False,
        rdrop_alpha=0.0,
    ),

    # ── V3.0: Best combination from ablation + noise augmentation ────────────
    AblationConfig(
        "V30_optimized", "V30",
        "Best combo: no RDrop, alpha=0.12, +25% noisy augment",
        enable_ranknet=True,
        enable_listnet=True,
        enable_fgm=True,
        rdrop_alpha=0.0,
        distill_alpha_max=0.12,
        distill_alpha_min=0.12,
        enable_noise_augment=True,
    ),

    # ── Group D: No separate training ─────────────────────────────────────────
    # D evaluations use REF_v29, C2_no_FGM, and V30_optimized checkpoints on noisy_test.json.
    # Handled in ablation_runner.py after all training runs complete.
]
