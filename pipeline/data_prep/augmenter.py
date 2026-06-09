"""LLM-powered semantic augmentation for training data."""
import json
import re
import time
from typing import Any, Dict, List, Optional

from .base import BaseProcessor
from .config import DataPrepConfig
from .models import QueryPropertyPair


class SemanticAugmenter(BaseProcessor):
    """Generates synthetic training samples using LLM semantic augmentation.

    Two modes:
    1. Negative: Hard negatives with conflicting constraints (looks good but fails)
    2. Positive: Colloquial semantic mappings (user intent → property features)
    """

    def __init__(self, config: DataPrepConfig):
        super().__init__(config)
        self.api_key = config.llm_api_key
        self.batch_size = config.llm_batch_size
        self.client = self._init_client()

    def _init_client(self):
        """Initialize LLM client (Gemini by default)."""
        if not self.api_key:
            self.logger.warning("ANTHROPIC_API_KEY not set. LLM augmentation disabled.")
            return None

        try:
            from google import genai
            return genai.Client(api_key=self.api_key)
        except ImportError:
            self.logger.warning(
                "google-genai not installed. Install with: pip install google-genai"
            )
            return None

    def run(self, target_count: int = 1000) -> List[QueryPropertyPair]:
        """Generate augmented training samples.

        Args:
            target_count: Total samples to generate (default 1000)

        Returns:
            List of QueryPropertyPair objects
        """
        if not self.client:
            self.logger.error("LLM client not available. Skipping augmentation.")
            return []

        self.log_step(f"Generating {target_count} augmented samples")

        # Load existing samples if any
        output_path = self.config.checkpoint_dir / "augmented_samples.json"
        existing_samples = self._load_existing_samples(output_path)
        self.log_result("Existing samples", len(existing_samples))

        all_pairs = []
        for sample in existing_samples:
            all_pairs.append(self._sample_to_pair(sample))

        # Generate until target
        while len(all_pairs) < target_count:
            remaining = target_count - len(all_pairs)
            # Alternate between positive and negative samples
            mode = "positive" if len(all_pairs) % 100 < 50 else "negative"

            self.log_step(
                f"Generating batch: {len(all_pairs)}/{target_count} (mode={mode})"
            )
            batch = self._generate_batch(mode, min(self.batch_size, remaining))

            if batch:
                for sample in batch:
                    all_pairs.append(self._sample_to_pair(sample))
                self.log_result("Batch success, total", len(all_pairs))

                # Save progress
                self._save_samples(all_pairs, output_path)
            else:
                self.logger.warning("Batch failed, retrying in 5 seconds...")
                time.sleep(5)

            time.sleep(2)  # Rate limiting

        self.log_result("Augmentation complete", len(all_pairs))
        return all_pairs[:target_count]

    def _generate_batch(self, mode: str, count: int) -> List[Dict[str, Any]]:
        """Generate a batch of samples using LLM.

        Args:
            mode: "negative" or "positive"
            count: Number of samples to generate

        Returns:
            List of sample dicts from LLM response
        """
        prompt = self._build_prompt(mode, count)

        try:
            response = self.client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
            )
            text = response.text.strip()

            # Extract JSON from response
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)

            # Fallback: try parsing with markdown cleanup
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())

        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            return []

    @staticmethod
    def _build_prompt(mode: str, count: int) -> str:
        """Build LLM prompt for sample generation."""
        if mode == "negative":
            return f"""
你是一個房地產數據專家。請幫我生成 {count} 組「租屋推薦系統」的困難負樣本 (Hard Negatives)。
這些樣本的特徵是：房源描述看起來與查詢 (Query) 非常匹配，但在關鍵約束上存在衝突。

請針對以下類別均勻生成：
1. 寵物衝突：Query 要求「可狗」，Property 描述「僅限貓」或「不可養寵物」。
2. 費用衝突：Query 要求「台水台電」，Property 描述「獨立電表一度5元」。
3. 地點衝突：Query 要求「近興大正門」，Property 描述「近興大南門/後門」。
4. 設備衝突：Query 要求「獨立洗脫烘」，Property 描述「公共洗衣房/投幣式洗衣機」。
5. 垃圾處理衝突：Query 要求「不追垃圾車/要有子母車」，Property 描述「需自行等候垃圾車」。

輸出格式必須是 JSON 陣列，每個元素包含：
{{
    "query": "使用者的口語化查詢",
    "property": "房源描述（要包含誘人條件但關鍵處衝突）",
    "relevance": 0,
    "label": 0,
    "category": "hard_negative"
}}
只輸出 JSON，不要任何說明文字。
"""
        else:  # positive
            return f"""
你是一個房地產數據專家。請幫我生成 {count} 組「租屋推薦系統」的正向語義映射樣本 (Positive Semantic Mapping)。
目標是讓模型學會理解口語化隱喻背後的真實特徵需求。

請針對以下「口語隱喻 → 核心特徵」均勻涵蓋所有類別進行生成：

【A. 煮飯 / 自炊 / 廚房需求】（本類別至少佔 40%，請大量生成）
- 「想在家煮飯」「想自己煮飯」「在家開伙」「喜歡下廚」「自炊」
  → 房源具備「可伙」「廚房」「流理台」「瓦斯爐」「電磁爐」「抽油煙機」「開火」
- 「省餐費」「不想外食」「不吃外食」「省伙食費」
  → 房源具備「可伙」「廚房」「流理台」（有廚房可自炊省錢）
- 「天然瓦斯」「有瓦斯」「要有廚房」「要能煮飯」「煮飯」「開火」
  → 房源具備「天然瓦斯」「瓦斯爐」「可伙」「廚房」
- 「可以煮東西」「喜歡自己煮」「自己煮」
  → 房源具備「廚房」「流理台」「電磁爐」「可伙」

【B. 便利 / 懶人設施】
- 「不想追垃圾車」「下班太晚趕不上垃圾車」 → 「子母車」「垃圾代收」
- 「懶得爬樓梯」「膝蓋不好」 → 「電梯」「大樓」
- 「不想去自助洗」「不喜歡共用洗衣機」 → 「獨立洗衣機」「獨立洗脫烘」

【C. 溫度 / 電費】
- 「怕熱」「想省電費」 → 「變頻冷氣」「台電計費」

【D. 寵物友善】
- 「有貓」「想養狗」「有毛孩」 → 「可養貓」「可養狗」「寵物友善」

【E. 安靜 / 讀書】
- 「讀書」「念書」「打報告」 → 「書桌」「安靜」「寬頻」

注意事項：
- query 要非常口語、符合台灣租屋族習慣用語
- property 要精確包含上述對應特徵關鍵字，讓模型看到明確的語意配對
- 同一類別請生成語氣各異的多個表達（避免重複樣板句）

輸出格式必須是 JSON 陣列，每個元素包含：
{{
    "query": "非常口語、台灣慣用語的查詢",
    "property": "精確匹配該需求的房源描述（要強調對應的解決方案特徵）",
    "relevance": 3,
    "label": 1,
    "category": "semantic_positive"
}}
只輸出 JSON，不要任何說明文字。
"""

    def _load_existing_samples(self, path) -> List[Dict[str, Any]]:
        """Load previously generated samples."""
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load existing samples: {e}")
        return []

    @staticmethod
    def _save_samples(pairs: List[QueryPropertyPair], path) -> None:
        """Save samples to disk."""
        data = [
            {
                "query": p.query,
                "property": p.property_id,
                "label": 1 if p.is_match else 0,
                "relevance": p.score or 0,
            }
            for p in pairs
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _sample_to_pair(sample: Dict[str, Any]) -> QueryPropertyPair:
        """Convert LLM sample to QueryPropertyPair."""
        return QueryPropertyPair(
            query=sample.get("query", ""),
            property_id=sample.get("property", ""),
            is_match=sample.get("label", 0) == 1,
            score=sample.get("relevance", 0),
        )
