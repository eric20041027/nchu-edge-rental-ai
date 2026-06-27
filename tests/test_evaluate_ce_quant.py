"""match_prob direction check — the bug this guards is reading the wrong
logit index, which silently ranks every property in reverse."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")

# Load the script by path: importing the package pulls torch (via __init__),
# which the torch-free CI test job doesn't have. match_prob only needs numpy.
_spec = importlib.util.spec_from_file_location(
    "evaluate_ce_quant",
    Path(__file__).parent.parent / "pipeline/model_training/evaluate_ce_quant.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
match_prob = _mod.match_prob


def test_match_logit_scores_higher_than_not_match():
    # logits are [NOT_MATCH, MATCH]; a high MATCH logit must score high.
    assert match_prob([0.0, 5.0]) > match_prob([5.0, 0.0])


def test_returns_probability_in_unit_range():
    assert match_prob([0.0, 0.0]) == pytest.approx(0.5)
    assert 0.0 <= match_prob([1.0, 2.0]) <= 1.0
    assert match_prob([-100.0, 100.0]) == pytest.approx(1.0)  # no overflow
