"""降採樣不變量 self-check(不需模型):
每設施骨架正樣本 ≤ CAP、負樣本不動、無骨架被清空、確定性可重現。"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "bal", ROOT / "pipeline/data_prep/balance_train_data.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sample_rows():
    # 一個會觸發 cap 的小資料:骨架 A 有 5 個正樣本、骨架 B 有 1 個、加負樣本
    rows = []
    for i in range(5):
        rows.append({"query": f"q{i}", "property": f"套房 西區 {i}路 9000元 冰箱 冷氣 床",
                     "label": 1, "is_hard": False})
    rows.append({"query": "qb", "property": "套房 東區 別路 8000元 飲水機 陽台",
                 "label": 1, "is_hard": False})
    rows.append({"query": "qn", "property": "套房 南區 某路 7000元 冰箱", "label": 0,
                 "is_hard": False})
    return rows


def test_cap_enforced_per_skeleton():
    m = _mod()
    rows = _sample_rows()
    balanced, rep = m.balance(rows, cap=2)
    from collections import Counter
    skels = Counter(m.skeleton(r["property"]) for r in balanced if r["label"] == 1)
    assert all(c <= 2 for c in skels.values()), f"有骨架超過 cap: {skels}"


def test_negatives_and_low_freq_untouched():
    m = _mod()
    rows = _sample_rows()
    balanced, rep = m.balance(rows, cap=2)
    # 負樣本全保留
    neg_before = sum(1 for r in rows if r["label"] == 0)
    neg_after = sum(1 for r in balanced if r["label"] == 0)
    assert neg_after == neg_before
    # 低頻骨架(飲水機 那筆,只 1 個)保留
    assert any("飲水機" in r["property"] for r in balanced)


def test_deterministic():
    m = _mod()
    rows = _sample_rows()
    a, _ = m.balance(rows, cap=2)
    b, _ = m.balance(rows, cap=2)
    assert [r["property"] for r in a] == [r["property"] for r in b]


def test_no_skeleton_emptied():
    """每個原本有正樣本的骨架,降採樣後仍至少保留 1 個。"""
    m = _mod()
    rows = _sample_rows()
    balanced, _ = m.balance(rows, cap=2)
    from collections import Counter
    before = {m.skeleton(r["property"]) for r in rows if r["label"] == 1}
    after = {m.skeleton(r["property"]) for r in balanced if r["label"] == 1}
    assert before == after, f"有骨架被清空: {before - after}"
