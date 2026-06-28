"""隱藏含義評估集的確定性 self-check（不需模型）：
驗 GT 計算正確、分桶切換正確、只收有區辨力條件。"""
import json
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/hidden_meaning_eval.json"


def _gen_module():
    spec = importlib.util.spec_from_file_location(
        "gen_hm", ROOT / "pipeline/evaluation/gen_hidden_meaning_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bucket_switch_precision_vs_recall():
    """GT>30 → precision(防虛降);GT≤30 → recall。"""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    K = data["meta"]["k"]
    for q in data["queries"]:
        if q["n_relevant"] > K:
            assert q["metric"] == "precision", f"{q['query']} GT={q['n_relevant']}>{K} 應 precision"
        else:
            assert q["metric"] == "recall", f"{q['query']} GT={q['n_relevant']}≤{K} 應 recall"


def test_gt_is_discriminative():
    """每個 query 的 GT 必須有區辨力：非空、非全體(否則指標失效)。"""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    n_props = data["meta"]["property_count"]
    for q in data["queries"]:
        assert 0 < q["n_relevant"] < n_props, \
            f"{q['query']} GT={q['n_relevant']} 無區辨力(空或全體)"


def test_gt_recomputable_from_predicates():
    """GT 必須能從客觀 predicate 重算出來（非合成標註）。抽一個 query 驗。"""
    gen = _gen_module()
    # 飲水機 query 的 GT = water_dispenser is True 的房源
    expected = {i for i, p in enumerate(gen.PD) if p.get("water_dispenser") is True}
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    water_q = next(q for q in data["queries"] if "飲水機" in q["hidden_meaning"]
                   and len(q["relevant_idxs"]) == len(expected))
    assert set(water_q["relevant_idxs"]) == expected
