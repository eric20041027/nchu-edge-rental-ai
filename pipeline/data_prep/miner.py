"""Active learning: Mine hard negatives from previous model version."""
import json
import re
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseProcessor
from .config import DataPrepConfig
from .models import HardNegativeExample, QueryPropertyPair


class HardNegativeMiner(BaseProcessor):
    """Identifies hard negative examples using an existing trained model.

    Hard negatives are query-property pairs where:
    - The model assigns high confidence (likely relevant)
    - BUT: Hard constraints conflict (gender, budget, amenities, etc.)

    This creates challenging training examples that prevent the model from
    relying on spurious correlations.
    """

    def __init__(self, config: DataPrepConfig):
        super().__init__(config)
        self.score_threshold = 0.45
        self.batch_size = 128
        self.model_path = None
        self.tokenizer = None
        self.session = None

    def run(
        self,
        properties: Optional[List[Dict[str, Any]]] = None,
        queries: Optional[List[str]] = None,
        model_path: Optional[str] = None,
    ) -> List[HardNegativeExample]:
        """Mine hard negatives from query-property space.

        Args:
            properties: List of property dicts (or loads from merged CSV)
            queries: List of queries (or loads from LLM augmented samples)
            model_path: Path to ONNX model (optional for scoring)

        Returns:
            List of HardNegativeExample objects
        """
        # Load data
        if properties is None:
            self.log_step("Loading properties from merged CSV")
            import pandas as pd
            df = pd.read_csv(self.config.merged_csv)
            properties = self._parse_properties_from_df(df)
            self.log_result("Properties loaded", len(properties))

        if queries is None:
            self.log_step("Loading queries from checkpoint")
            queries = self._load_queries_from_checkpoint()
            self.log_result("Queries loaded", len(queries))

        # Initialize model if provided
        if model_path:
            self.log_step("Loading scoring model")
            self._load_model(model_path)

        # Mine hard negatives
        self.log_step("Mining hard negatives from query-property space")
        mined = self._mine_hard_negatives(properties, queries)
        self.log_result("Hard negatives mined", len(mined))

        # Save
        output_path = self.config.checkpoint_dir / "hard_negatives.json"
        self._save_hard_negatives(mined, output_path)

        return mined

    def _parse_properties_from_df(self, df) -> List[Dict[str, Any]]:
        """Parse DataFrame into property dicts."""
        properties = []
        for _, row in df.iterrows():
            prop = {
                "id": row.get("網址", ""),
                "text": f"{row.get('地址', '')} {row.get('類型', '')} {row.get('租金', '')} {row.get('家具設施', '')}",
                "address": row.get("地址", ""),
                "rent": self._parse_rent(row.get("租金", "")),
                "furniture": self._parse_list(row.get("家具設施", "")),
                "notes": self._parse_list(row.get("備註", "")),
            }
            properties.append(prop)
        return properties

    def _load_queries_from_checkpoint(self) -> List[str]:
        """Load queries from augmentation checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / "augmented_samples.json"
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return []

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return list(set([s.get("query", "") for s in data]))
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return []

    def _load_model(self, model_path: str) -> None:
        """Load ONNX model for scoring."""
        try:
            import onnxruntime as ort
            from transformers import BertTokenizerFast

            self.tokenizer = BertTokenizerFast.from_pretrained(
                self.config.project_root / "frontend/models/custom_onnx_model_dir"
            )
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self.log_step("Model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load model: {e}")

    def _mine_hard_negatives(
        self,
        properties: List[Dict[str, Any]],
        queries: List[str],
    ) -> List[HardNegativeExample]:
        """Mine hard negatives from query-property combinations."""
        mined = []

        # Sample queries for efficiency
        sampled_queries = random.sample(queries, min(15, len(queries)))
        self.log_result("Sampled queries", len(sampled_queries))

        for query_idx, query in enumerate(sampled_queries):
            # Sample properties for this query
            sampled_props = random.sample(properties, min(300, len(properties)))

            for batch_idx in range(0, len(sampled_props), self.batch_size):
                batch_props = sampled_props[batch_idx : batch_idx + self.batch_size]

                # Score batch
                scores = self._score_batch(query, batch_props)

                # Check for hard conflicts
                for prop, score in zip(batch_props, scores):
                    if score > self.score_threshold and self._is_hard_conflict(
                        query, prop
                    ):
                        example = HardNegativeExample(
                            query=query,
                            property_id=prop.get("id", ""),
                            model_score=float(score),
                            reason=self._extract_conflict_reason(query, prop),
                        )
                        mined.append(example)

            self.log_result(f"Query {query_idx + 1}/{len(sampled_queries)}", len(mined))

        return mined

    def _score_batch(self, query: str, properties: List[Dict[str, Any]]) -> np.ndarray:
        """Score a query against multiple properties."""
        if not self.session or not self.tokenizer:
            # If no model loaded, use simple heuristic scoring
            return np.array([self._heuristic_score(query, p) for p in properties])

        try:
            texts = [p.get("text", "") for p in properties]
            inputs = self.tokenizer(
                [query] * len(texts),
                texts,
                return_tensors="np",
                max_length=64,
                padding="max_length",
                truncation=True,
            )

            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
                "token_type_ids": inputs["token_type_ids"].astype(np.int64),
            }

            logits = self.session.run(None, ort_inputs)[0]
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return probs[:, 1]  # Probability of relevance

        except Exception as e:
            self.logger.warning(f"Model scoring failed: {e}. Using heuristic.")
            return np.array([self._heuristic_score(query, p) for p in properties])

    @staticmethod
    def _heuristic_score(query: str, prop: Dict[str, Any]) -> float:
        """Simple heuristic score if model unavailable."""
        text = prop.get("text", "").lower()
        q = query.lower()

        score = 0.0
        # Match keywords
        keywords = q.split()
        for kw in keywords:
            if len(kw) > 2 and kw in text:
                score += 0.1

        return min(score, 0.95)

    @staticmethod
    def _is_hard_conflict(query: str, prop: Dict[str, Any]) -> bool:
        """Check for hard conflicts between query and property."""
        text = (prop.get("text", "") + " " + " ".join(prop.get("notes", []))).lower()
        q = query.lower()

        # 1. Gender conflict
        if "限女" in text and ("限男" in q or "男生" in q):
            return True
        if "限男" in text and ("限女" in q or "女生" in q):
            return True

        # 2. Budget conflict
        budget_match = re.search(r"(\d+)(?:元|k)?(?:以下|以內|內)", q)
        if budget_match:
            val = int(budget_match.group(1))
            if "k" in budget_match.group(0).lower() and val < 100:
                val *= 1000
            rent = prop.get("rent", 0)
            if rent > val * 1.2:
                return True

        # 3. Pet conflict
        if (
            any(kw in q for kw in ["可養寵", "可寵", "養貓", "養狗"])
            and any(kw in text for kw in ["禁寵", "不可養寵"])
        ):
            return True

        # 4. Rooftop conflict
        if (
            any(kw in q for kw in ["不找頂加", "不要頂加", "非頂加"])
            and any(kw in text for kw in ["頂加", "頂樓加蓋"])
        ):
            return True

        # 5. Smoking conflict
        if ("禁菸" in q or "不抽菸" in q) and "可菸" in text:
            return True

        # 6. Must-have amenities
        must_haves = {
            "電梯": ["電梯"],
            "陽台": ["陽台", "露台"],
            "車位": ["車位", "停車"],
            "獨洗": ["獨洗", "個人洗衣機", "自用洗衣機"],
            "開伙": ["開伙", "廚房", "瓦斯爐"],
            "管理員": ["管理員", "子母車", "收包裹"],
        }

        for key, synonyms in must_haves.items():
            if key in q and not any(syn in text for syn in synonyms):
                return True

        return False

    @staticmethod
    def _extract_conflict_reason(query: str, prop: Dict[str, Any]) -> str:
        """Extract reason for hard conflict."""
        text = prop.get("text", "").lower()
        q = query.lower()

        if "限女" in text and ("限男" in q or "男生" in q):
            return "限女與男性衝突"
        if "限男" in text and ("限女" in q or "女生" in q):
            return "限男與女性衝突"

        budget_match = re.search(r"(\d+)", q)
        if budget_match and prop.get("rent", 0) > int(budget_match.group(1)) * 1.2:
            return "超過預算"

        if any(kw in q for kw in ["可養寵", "養貓", "養狗"]):
            return "禁寵衝突"

        return "硬性約束衝突"

    @staticmethod
    def _parse_rent(rent_str: str) -> float:
        """Parse rent value."""
        try:
            match = re.search(r"\d+", str(rent_str).replace(",", ""))
            return float(match.group()) if match else 0.0
        except (ValueError, AttributeError):
            return 0.0

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse slash-separated list, handling NaN and non-string values."""
        if text is None or (isinstance(text, float) and text != text):  # NaN check
            return []
        if not isinstance(text, str):
            return []
        if not text or not text.strip():
            return []
        return [s.strip() for s in text.split("/") if s.strip()]

    @staticmethod
    def _save_hard_negatives(
        examples: List[HardNegativeExample], path
    ) -> None:
        """Save hard negatives to JSON."""
        data = [
            {
                "query": e.query,
                "property_id": e.property_id,
                "model_score": e.model_score,
                "reason": e.reason,
            }
            for e in examples
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
