"""Silver labeling: Generate query-label pairs using LLM."""
import json
import re
from typing import Any, Dict, List, Optional

from .base import BaseProcessor
from .config import DataPrepConfig
from .models import QueryPropertyPair


class SilverLabeler(BaseProcessor):
    """Generates silver-labeled query-property pairs using LLM.

    Silver labels are automatically generated (not manually curated) but
    provide signal for model training. Each property gets diverse queries
    at multiple relevance levels (0-3), capturing:
    - Budget-focused queries
    - Facility-focused queries
    - Lifestyle-focused queries
    - Negative exclusion-focused queries
    """

    def __init__(self, config: DataPrepConfig):
        super().__init__(config)
        self.api_key = config.llm_api_key
        self.queries_per_property = 6
        self.client = self._init_client()

    def _init_client(self):
        """Initialize Gemini client."""
        if not self.api_key:
            self.logger.warning("ANTHROPIC_API_KEY not set. Silver labeling disabled.")
            return None

        try:
            from google import genai
            return genai.Client(api_key=self.api_key)
        except ImportError:
            self.logger.warning("google-genai not installed. Install with: pip install google-genai")
            return None

    def run(self, properties: Optional[List[Dict[str, Any]]] = None) -> List[QueryPropertyPair]:
        """Generate silver labels for properties.

        Args:
            properties: List of property dicts (or loads from merged CSV)

        Returns:
            List of QueryPropertyPair objects with silver labels
        """
        if not self.client:
            self.logger.error("LLM client not available. Skipping silver labeling.")
            return []

        # Load properties if needed
        if properties is None:
            self.log_step("Loading properties from merged CSV")
            import pandas as pd
            df = pd.read_csv(self.config.merged_csv)
            properties = self._parse_properties_from_df(df)
            self.log_result("Properties loaded", len(properties))

        # Load existing samples
        output_path = self.config.checkpoint_dir / "silver_labeled.json"
        existing_pairs = self._load_existing_samples(output_path)
        self.log_result("Existing silver labels", len(existing_pairs))

        # Generate new samples
        self.log_step(f"Generating silver labels ({self.queries_per_property} per property)")
        all_pairs = list(existing_pairs)

        for idx, prop in enumerate(properties):
            if (idx + 1) % 10 == 0:
                self.log_result(f"Progress: {idx + 1}/{len(properties)}", len(all_pairs))

            queries_and_labels = self._generate_for_property(prop)
            for query, relevance in queries_and_labels:
                pair = QueryPropertyPair(
                    query=query,
                    property_id=prop.get("id", ""),
                    is_match=relevance >= 2,
                    score=relevance
                )
                all_pairs.append(pair)

        # Save
        self.log_step(f"Saving {len(all_pairs)} silver labels to {output_path}")
        self._save_samples(all_pairs, output_path)

        return all_pairs

    def _generate_for_property(self, prop: Dict[str, Any]) -> List[tuple]:
        """Generate queries and labels for a single property.

        Args:
            prop: Property dictionary

        Returns:
            List of (query, relevance) tuples
        """
        prop_text = self._format_property_description(prop)
        prompt = self._build_labeling_prompt(prop_text, self.queries_per_property)

        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model="models/gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )

            results = json.loads(response.text)
            output = []

            for item in results:
                query = item.get("query", "").strip()
                relevance = int(item.get("relevance", 1))

                # Validate
                if query and 0 <= relevance <= 3:
                    output.append((query, relevance))

            return output

        except Exception as e:
            self.logger.debug(f"Label generation failed: {e}")
            return []

    @staticmethod
    def _format_property_description(prop: Dict[str, Any]) -> str:
        """Format property for LLM prompt."""
        text = prop.get("text", "")
        rent = prop.get("rent", "未知")
        notes = " ".join(prop.get("notes", []))

        return f"房源資訊: {text}\n租金: {rent}元\n備註: {notes}"

    @staticmethod
    def _build_labeling_prompt(prop_text: str, num_queries: int) -> str:
        """Build LLM prompt for silver labeling."""
        return f"""
你是一位專業的租屋大數據標記專家。請針對以下房源，生成 {num_queries} 組「極具挑戰性」且「口語化」的用戶搜尋查詢，並給出精確的相關性評分 (0-3)。

{prop_text}

評分標準:
3 (Perfect): 查詢精確描述了該房源的所有關鍵亮點（地點、預算、特定稀有設施如獨洗、台電等）。
2 (Good): 查詢與房源大致契合，但房源在某些次要維度（如樓層或稍微偏離的核心區域）僅是「還不錯」。
1 (Partial): 查詢與房源只有一點點交集（例如預算對但地點不對，或設施完全不符但價格極低）。
0 (None): 查詢的需求與房源完全衝突（例如限女 vs 男生，或禁寵 vs 要養貓）。

請確保生成多樣化的查詢：包含「預算導向」、「設施導向」、「生活風格導向」以及「帶有負面排除條件的導向」。

輸出為 JSON 陣列，每個元素包含 'query' 和 'relevance' 欄位。
只輸出 JSON，不要任何說明文字。
"""

    def _parse_properties_from_df(self, df) -> List[Dict[str, Any]]:
        """Parse DataFrame into property dicts."""
        properties = []
        for _, row in df.iterrows():
            prop = {
                "id": row.get("網址", ""),
                "text": f"{row.get('地址', '')} {row.get('類型', '')} {row.get('租金', '')}",
                "rent": self._parse_rent(row.get("租金", "")),
                "notes": self._parse_list(row.get("備註", "")),
            }
            properties.append(prop)
        return properties

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
        """Parse slash-separated list."""
        if not text:
            return []
        return [s.strip() for s in text.split("/") if s.strip()]

    def _load_existing_samples(self, path) -> List[QueryPropertyPair]:
        """Load previously generated samples."""
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return [
                        QueryPropertyPair(
                            query=s.get("query", ""),
                            property_id=s.get("property_id", ""),
                            is_match=s.get("is_match", False),
                            score=s.get("score"),
                        )
                        for s in data
                    ]
        except Exception as e:
            self.logger.warning(f"Failed to load existing samples: {e}")
        return []

    @staticmethod
    def _save_samples(pairs: List[QueryPropertyPair], path) -> None:
        """Save samples to JSON."""
        data = [
            {
                "query": p.query,
                "property_id": p.property_id,
                "is_match": p.is_match,
                "score": p.score or 0,
            }
            for p in pairs
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
