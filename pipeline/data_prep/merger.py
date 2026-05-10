"""Merge and deduplicate multiple rental data sources."""
import re
import pandas as pd
from typing import Optional

from .base import BaseProcessor
from .config import DataPrepConfig


class DataMerger(BaseProcessor):
    """Merges multiple rental CSV sources with advanced deduplication.

    Deduplication strategy:
    1. Primary: URL uniqueness
    2. Secondary: Normalized address + rent tolerance (5%)
    """

    def __init__(self, config: DataPrepConfig):
        super().__init__(config)
        self.address_tolerance = 0.05  # 5% price tolerance

    def run(self) -> pd.DataFrame:
        """Execute merge and deduplication.

        Returns:
            Merged and deduplicated DataFrame
        """
        self.log_step("Loading datasets")
        df_main = pd.read_csv(self.config.main_csv)
        df_official = pd.read_csv(self.config.official_csv)

        self.log_result("Main dataset rows", len(df_main))
        self.log_result("Official dataset rows", len(df_official))

        # Combine sources
        df_combined = pd.concat([df_main, df_official], ignore_index=True)
        initial_count = len(df_combined)
        self.log_step(f"Combined into {initial_count} total rows")

        # Step 1: URL deduplication
        self.log_step("Deduplicating by URL")
        df_combined = df_combined.drop_duplicates(subset=["網址"], keep="first")
        self.log_result("After URL dedup", len(df_combined))

        # Step 2: Normalized address + rent tolerance deduplication
        self.log_step("Deduplicating by normalized address + rent tolerance")
        df_combined["_norm_address"] = df_combined["地址"].apply(self._normalize_address)
        df_combined["_rent_val"] = df_combined["租金"].apply(self._parse_rent)
        df_combined = df_combined.sort_values(by=["_norm_address", "_rent_val"])

        final_rows = []
        seen_addresses = {}  # address -> list of prices

        for _, row in df_combined.iterrows():
            addr = row["_norm_address"]
            price = row["_rent_val"]

            is_duplicate = False
            if addr in seen_addresses:
                for seen_price in seen_addresses[addr]:
                    if abs(price - seen_price) / (seen_price + 1e-5) < self.address_tolerance:
                        is_duplicate = True
                        break

            if not is_duplicate:
                final_rows.append(row)
                if addr not in seen_addresses:
                    seen_addresses[addr] = []
                seen_addresses[addr].append(price)

        df_final = pd.DataFrame(final_rows)
        removed = initial_count - len(df_final)
        self.log_result("Duplicates removed", removed)

        # Cleanup temporary columns
        df_final = df_final.drop(columns=["_norm_address", "_rent_val"])

        # Ensure all columns from original exist
        for col in df_main.columns:
            if col not in df_final.columns:
                df_final[col] = ""
        df_final = df_final[df_main.columns]

        # Save
        self.log_step(f"Saving merged data to {self.config.merged_csv}")
        df_final.to_csv(self.config.merged_csv, index=False, encoding="utf-8-sig")

        return df_final

    @staticmethod
    def _normalize_address(address: str) -> str:
        """Normalize Chinese numerals and common address variations."""
        if not isinstance(address, str):
            return ""
        mapping = {
            "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
            "六": "6", "七": "7", "八": "8", "九": "9", "○": "0", "零": "0"
        }
        for cn, ar in mapping.items():
            address = address.replace(cn, ar)
        address = re.sub(r"[\s\-,.，。]", "", address)
        return address

    @staticmethod
    def _parse_rent(rent_str: str) -> float:
        """Extract numeric rent value from string."""
        try:
            match = re.search(r"\d+", str(rent_str))
            return float(match.group()) if match else 0.0
        except (ValueError, AttributeError):
            return 0.0
