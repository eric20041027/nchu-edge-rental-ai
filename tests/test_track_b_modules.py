"""Smoke tests for Track B modules: B2 LifestyleMapper, B3 HardConstraintFilter, B4 OSRMClient."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# ─── B2: LifestyleMapper ────────────────────────────────────────────────────
from pipeline.data_prep.lifestyle_mapper import LifestyleMapper

class TestLifestyleMapper:
    def setup_method(self):
        self.mapper = LifestyleMapper()

    def test_expand_single_cluster(self):
        result = self.mapper.expand_query("我有毛孩想養貓")
        assert "可寵" in result or "寵物" in result

    def test_expand_no_match(self):
        q = "我想租套房"
        assert self.mapper.expand_query(q) == q

    def test_infer_features_garbage(self):
        feats = self.mapper.infer_features("不想追垃圾車")
        assert "子母車" in feats

    def test_infer_features_cooking(self):
        feats = self.mapper.infer_features("自己煮省錢")
        assert "廚房" in feats

    def test_matched_clusters(self):
        clusters = self.mapper.matched_clusters("怕熱又怕吵")
        assert "怕熱" in clusters
        assert "怕吵" in clusters

    def test_multiple_clusters(self):
        result = self.mapper.expand_query("怕熱又不想追垃圾車")
        assert "冷氣" in result
        assert "子母車" in result


# ─── B3: HardConstraintFilter ───────────────────────────────────────────────
from pipeline.constraints.hard_constraints import HardConstraintFilter, ParsedQuery

SAMPLE_PROPS = [
    {"rent": "6000", "electricity_billing": "台水台電", "is_rooftop": "False",
     "is_wooden_partition": "False", "text": "套房 南區", "furniture": "", "notes": [], "distance": "1.5"},
    {"rent": "9000", "electricity_billing": "5元/度", "is_rooftop": "False",
     "is_wooden_partition": "False", "text": "雅房 北區", "furniture": "", "notes": [], "distance": "2.0"},
    {"rent": "5500", "electricity_billing": "台電", "is_rooftop": "True",
     "is_wooden_partition": "False", "text": "頂加 西區", "furniture": "", "notes": [], "distance": "0.8"},
    {"rent": "7000", "electricity_billing": "含電費", "is_rooftop": "False",
     "is_wooden_partition": "False", "text": "套房 南區 禁養寵物", "furniture": "", "notes": [], "distance": "1.2"},
]

class TestHardConstraintFilter:
    def setup_method(self):
        self.f = HardConstraintFilter()

    def test_budget_strict_filters_over_budget(self):
        q = ParsedQuery.from_text("預算7000以內")
        result = self.f.filter(SAMPLE_PROPS, q)
        rents = [int(p["rent"]) for p in result]
        assert all(r <= 7000 for r in rents), rents

    def test_pet_filter(self):
        q = ParsedQuery.from_text("我要養貓")
        result = self.f.filter(SAMPLE_PROPS, q)
        assert not any("禁養" in p["text"] for p in result)

    def test_rooftop_exclusion(self):
        q = ParsedQuery.from_text("不要頂加")
        result = self.f.filter(SAMPLE_PROPS, q)
        assert not any(p["is_rooftop"] == "True" for p in result)

    def test_no_constraint_passes_all(self):
        q = ParsedQuery.from_text("租套房")
        result = self.f.filter(SAMPLE_PROPS, q)
        assert len(result) == len(SAMPLE_PROPS)

    def test_taipower_filter(self):
        q = ParsedQuery.from_text("要台電計費")
        result = self.f.filter(SAMPLE_PROPS, q)
        # Property with 5元/度 should be excluded
        assert not any("5元" in p["electricity_billing"] for p in result)


# ─── B4: OSRMClient (unit — no network) ─────────────────────────────────────
from pipeline.osrm_client import OSRMClient

class TestOSRMClientUnit:
    def test_cache_returns_same_object(self, monkeypatch):
        client = OSRMClient()
        cached = {"dist": 1.5, "walk_mins": 20, "scooter_mins": 5}
        client._cache["台中市南區某路1號"] = cached
        result = client.get_commute("台中市南區某路1號")
        assert result is cached

    def test_compute_weight_neutral_fallback(self, monkeypatch):
        client = OSRMClient()
        monkeypatch.setattr(client, "get_commute", lambda addr: None)
        assert client.compute_weight("anywhere") == 0.5

    def test_compute_weight_near(self, monkeypatch):
        client = OSRMClient()
        monkeypatch.setattr(client, "get_commute", lambda addr: {"dist": 1.0, "walk_mins": 13, "scooter_mins": 4})
        weight = client.compute_weight("near")
        assert 0.9 <= weight <= 1.0

    def test_compute_weight_far(self, monkeypatch):
        client = OSRMClient()
        monkeypatch.setattr(client, "get_commute", lambda addr: {"dist": 20.0, "walk_mins": 260, "scooter_mins": 50})
        weight = client.compute_weight("far")
        assert weight == 0.0


# ─── B1: NER imports ────────────────────────────────────────────────────────
class TestNERImports:
    def test_config_importable(self):
        # pipeline.ner_model 頂層 import torch(ner_trainer)→ 無 torch 時 skip。
        pytest.importorskip("torch")
        from pipeline.ner_model import NERConfig
        cfg = NERConfig()
        assert cfg.num_labels == 7
        assert cfg.target_f1 == 0.958

    def test_predictor_importable(self):
        pytest.importorskip("torch")
        from pipeline.ner_model import NERPredictor
        p = NERPredictor()
        assert not p._loaded

    def test_label_maps(self):
        # 即使只 import 子模組,pipeline.ner_model.__init__ 仍會 import torch → skip。
        pytest.importorskip("torch")
        from pipeline.ner_model.config import LABEL2ID, ID2LABEL
        assert "B-LOC" in LABEL2ID
        assert "B-BGT" in LABEL2ID
        assert "B-FEAT" in LABEL2ID
        assert ID2LABEL[LABEL2ID["B-LOC"]] == "B-LOC"
