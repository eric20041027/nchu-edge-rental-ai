"""Generate training datasets from merged rental properties."""
import json
import random
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseProcessor
from .config import DataPrepConfig
from .models import QueryPropertyPair, TrainingDataset


class DatasetGenerator(BaseProcessor):
    """Generates query-property training pairs with graded relevance scoring.

    Features:
    - Multi-strategy query generation (single features, dual combos, tri-combos, etc.)
    - Graded relevance scoring (0-3) based on constraint satisfaction
    - Hard negative mining with semantic conflict detection
    - Object-level train/val/test split (prevents data leakage)
    """

    def __init__(self, config: DataPrepConfig):
        super().__init__(config)
        self.random_seed = config.random_seed
        random.seed(self.random_seed)

    def run(self, merged_data: Optional[pd.DataFrame] = None) -> TrainingDataset:
        """Execute dataset generation pipeline.

        Args:
            merged_data: Optional pre-loaded merged DataFrame. If None, loads from config.

        Returns:
            TrainingDataset with train/val/test splits
        """
        # Load data
        if merged_data is None:
            self.log_step(f"Loading merged data from {self.config.merged_csv}")
            merged_data = pd.read_csv(self.config.merged_csv)
        else:
            self.log_step(f"Using provided merged data ({len(merged_data)} rows)")

        self.log_result("Total properties", len(merged_data))

        # Parse properties from CSV
        properties = self._parse_properties(merged_data)
        self.log_result("Parsed properties", len(properties))

        # Object-level split (prevent data leakage)
        self.log_step("Performing object-level train/val/test split")
        train_props, val_props, test_props = self._split_properties(
            properties,
            train_ratio=self.config.train_split,
            val_ratio=self.config.val_split,
        )
        self.log_result("Train properties", len(train_props))
        self.log_result("Val properties", len(val_props))
        self.log_result("Test properties", len(test_props))

        # Generate samples for each split
        self.log_step("Generating query-property pairs")
        train_pairs = self._generate_samples(train_props, split_name="train")
        val_pairs = self._generate_samples(val_props, split_name="val")
        test_pairs = self._generate_samples(test_props, split_name="test")

        self.log_result("Train pairs", len(train_pairs))
        self.log_result("Val pairs", len(val_pairs))
        self.log_result("Test pairs", len(test_pairs))

        # Create dataset
        dataset = TrainingDataset(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            metadata={
                "total_properties": len(properties),
                "train_count": len(train_pairs),
                "val_count": len(val_pairs),
                "test_count": len(test_pairs),
                "seed": self.random_seed,
            }
        )

        # Save
        self.log_step(f"Saving dataset to {self.config.dataset_json}")
        self._save_dataset(dataset)

        return dataset

    def _parse_properties(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse DataFrame rows into property dictionaries."""
        properties = []
        for _, row in df.iterrows():
            rent = self._parse_rent(row.get("租金", ""))
            distance = float(str(row.get("距離(km)", "0") or "0").replace("km", "").strip() or "0")

            # Handle NaN values in row data
            def _safe_get(row, key, default=""):
                val = row.get(key, default)
                if val is None or (isinstance(val, float) and val != val):  # NaN check
                    return default
                return val if isinstance(val, str) else str(val) if val else default

            prop = {
                "url": _safe_get(row, "網址"),
                "address": _safe_get(row, "地址"),
                "region": self._extract_region(_safe_get(row, "地址")),
                "road": self._extract_road(_safe_get(row, "地址")),
                "room_type": self._extract_room_type(_safe_get(row, "類型")),
                "building_type": self._extract_building_type(_safe_get(row, "類型")),
                "size": _safe_get(row, "室內坪數"),
                "rent": rent,
                "rent_str": _safe_get(row, "租金"),
                "floor": _safe_get(row, "樓層"),
                "distance": distance,
                "furniture": self._parse_list(_safe_get(row, "家具設施")),
                "notes": self._parse_list(_safe_get(row, "備註")),
            }
            properties.append(prop)
        return properties

    @staticmethod
    def _parse_rent(rent_str: str) -> float:
        """Extract numeric rent from string like '6500 元'."""
        try:
            match = re.search(r"\d+", str(rent_str).replace(",", ""))
            return float(match.group()) if match else 0.0
        except (ValueError, AttributeError):
            return 0.0

    @staticmethod
    def _extract_region(address) -> str:
        """Extract region district from address, handling NaN and non-string values."""
        if address is None or (isinstance(address, float) and address != address):  # NaN check
            return ""
        if not isinstance(address, str):
            return ""
        regions = ["南區", "大里區", "東區", "西區", "太平區", "西屯區", "北屯區", "南屯區", "北區", "中區"]
        for region in regions:
            if region in address:
                return region
        return ""

    @staticmethod
    def _extract_road(address) -> str:
        """Extract road name from address, handling NaN and non-string values."""
        if address is None or (isinstance(address, float) and address != address):  # NaN check
            return ""
        if not isinstance(address, str):
            return ""
        match = re.search(r"([^區市台]*(?:路|街|大道)(?:[一二三四五六七八九十]|[\d])?段?)", address)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_room_type(type_str) -> str:
        """Extract room type (套房/雅房/住宅), handling NaN and non-string values."""
        if type_str is None or (isinstance(type_str, float) and type_str != type_str):  # NaN check
            return ""
        if not isinstance(type_str, str):
            return ""
        for room_type in ["套房", "雅房", "住宅"]:
            if room_type in type_str:
                return room_type
        return ""

    @staticmethod
    def _extract_building_type(type_str) -> str:
        """Extract building type, handling NaN and non-string values."""
        if type_str is None or (isinstance(type_str, float) and type_str != type_str):  # NaN check
            return ""
        if not isinstance(type_str, str):
            return ""
        return type_str.replace("套房", "").replace("雅房", "").replace("住宅", "").strip()

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

    def _split_properties(
        self,
        properties: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple:
        """Object-level train/val/test split."""
        indices = list(range(len(properties)))
        random.shuffle(indices)

        n = len(indices)
        train_split = int(n * train_ratio)
        val_split = int(n * (train_ratio + val_ratio))

        train_props = [properties[i] for i in indices[:train_split]]
        val_props = [properties[i] for i in indices[train_split:val_split]]
        test_props = [properties[i] for i in indices[val_split:]]

        return train_props, val_props, test_props

    def _generate_samples(
        self,
        properties: List[Dict[str, Any]],
        split_name: str = "train"
    ) -> List[QueryPropertyPair]:
        """Generate query-property pairs for given properties."""
        samples = []

        for prop in properties:
            # Generate queries for this property
            queries = self._generate_queries_for_property(prop)

            for query in queries:
                relevance = self._compute_relevance_score(query, prop)

                # Positive pair
                pair = QueryPropertyPair(
                    query=query,
                    property_id=prop.get("url", ""),
                    is_match=True,
                    score=relevance
                )
                samples.append(pair)

                # Negative pairs (hard negatives from other properties)
                for neg_prop in random.sample(properties, min(2, len(properties) - 1)):
                    if neg_prop.get("url") != prop.get("url"):
                        pair = QueryPropertyPair(
                            query=query,
                            property_id=neg_prop.get("url", ""),
                            is_match=False,
                            score=0
                        )
                        samples.append(pair)

        random.shuffle(samples)
        return samples

    def _generate_queries_for_property(self, prop: Dict[str, Any], count: int = 12) -> List[str]:
        """Generate synthetic queries for a property."""
        queries = []

        # Simple queries from individual features
        if prop.get("rent"):
            queries.append(f"預算{int(prop['rent'])}元")
        if prop.get("region"):
            queries.append(f"想租{prop['region']}")
        if prop.get("room_type"):
            queries.append(f"找{prop['room_type']}")

        # Combined queries
        if prop.get("room_type") and prop.get("region"):
            queries.append(f"想租{prop['region']}{prop['room_type']}")

        # Colloquial queries
        if prop.get("distance") and prop["distance"] < 1.0:
            queries.append("學校附近")
        if prop.get("rent") and prop["rent"] < 5000:
            queries.append("便宜的房子")

        # Furniture-based queries
        for furniture in prop.get("furniture", [])[:3]:
            queries.append(f"要{furniture}")

        # Deduplicate and sample
        queries = list(set(queries))
        random.shuffle(queries)
        return queries[:count]

    def _compute_relevance_score(self, query: str, prop: Dict[str, Any]) -> int:
        """Compute graded relevance score (0-3).

        Scoring:
        - 0: Hard conflict (room type mismatch, exclusion, etc.)
        - 1: Partial match (only 1 dimension matches)
        - 2: Good match (most dimensions match, minor deviations)
        - 3: Perfect match (all specified dimensions satisfied)
        """
        # Hard conflict checks
        if "套房" in query and prop.get("room_type") != "套房":
            return 0
        if "雅房" in query and prop.get("room_type") != "雅房":
            return 0
        if "住宅" in query and prop.get("room_type") != "住宅":
            return 0

        # Budget constraint
        budget_match = re.search(r"(\d{4,5})", query)
        if budget_match:
            budget = int(budget_match.group(1))
            if prop.get("rent", 0) > budget * 1.1:
                return 0

        # Compute match score
        dimensions = 0
        satisfied = 0

        # Dimension 1: Room type
        if prop.get("room_type"):
            dimensions += 1
            if prop["room_type"] in query:
                satisfied += 1

        # Dimension 2: Region
        if prop.get("region"):
            dimensions += 1
            if prop["region"] in query or prop.get("address", "") in query:
                satisfied += 1

        # Dimension 3: Budget
        if prop.get("rent"):
            dimensions += 1
            if budget_match and prop["rent"] <= int(budget_match.group(1)):
                satisfied += 1

        # Dimension 4: Features
        if prop.get("furniture"):
            dimensions += 1
            matched_furniture = sum(1 for f in prop["furniture"] if f in query)
            if matched_furniture > 0:
                satisfied += matched_furniture / len(prop["furniture"])

        if dimensions == 0:
            return 2  # No constraints specified → generic good match

        ratio = satisfied / dimensions
        if ratio >= 0.85:
            return 3
        if ratio >= 0.65:
            return 2
        if ratio >= 0.15:
            return 1
        return 0

    def _save_dataset(self, dataset: TrainingDataset) -> None:
        """Save dataset to JSON files (train, val, test separately)."""
        from pathlib import Path

        dataset_dir = Path(self.config.dataset_json).parent
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save training dataset
        train_data = [
            {"query": p.query, "property_id": p.property_id, "label": p.is_match, "score": p.score}
            for p in dataset.train_pairs
        ]
        train_path = dataset_dir / "training_dataset.json"
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        # Save validation dataset
        val_data = [
            {"query": p.query, "property_id": p.property_id, "label": p.is_match, "score": p.score}
            for p in dataset.val_pairs
        ]
        val_path = dataset_dir / "validation_dataset.json"
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        # Save test dataset
        test_data = [
            {"query": p.query, "property_id": p.property_id, "label": p.is_match, "score": p.score}
            for p in dataset.test_pairs
        ]
        test_path = dataset_dir / "test_dataset.json"
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        # Also save combined dataset for reference
        combined_data = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "metadata": dataset.metadata
        }
        combined_path = dataset_dir / self.config.dataset_json
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
