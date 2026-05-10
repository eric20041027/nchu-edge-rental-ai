"""Update commute data using geocoding and routing APIs."""
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .base import BaseProcessor
from .config import DataPrepConfig


class CommuteDataUpdater(BaseProcessor):
    """Updates real-world commute data for rental properties.

    Uses:
    - ArcGIS Geocoding: Convert address → coordinates
    - OSRM (Open Source Routing Machine): Calculate walking/driving times
    - NCHU Main Gate: Reference point (24.1252, 120.6818)

    Provides fallback heuristic when APIs unavailable.
    """

    # NCHU Main Gate Coordinates
    NCHU_LAT = 24.1252
    NCHU_LON = 120.6818

    def __init__(self, config: DataPrepConfig):
        super().__init__(config)
        self.timeout = 10
        self.rate_limit_sleep = 0.5

    def run(self, properties_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Update commute data for properties.

        Args:
            properties_df: DataFrame with property data (or loads from config)

        Returns:
            Updated DataFrame with commute columns
        """
        # Load data
        if properties_df is None:
            self.log_step(f"Loading properties from {self.config.merged_csv}")
            properties_df = pd.read_csv(self.config.merged_csv)
        else:
            self.log_step(f"Using provided data ({len(properties_df)} rows)")

        # Ensure columns exist
        for col in ["距離(km)", "walk_mins", "scooter_mins"]:
            if col not in properties_df.columns:
                properties_df[col] = ""

        self.log_step("Updating commute data")
        updated_count = 0

        for idx, row in properties_df.iterrows():
            # Check if already has data
            if (
                row.get("walk_mins")
                and row.get("scooter_mins")
                and str(row.get("walk_mins", "")).strip() != "0"
            ):
                continue

            address = row.get("地址", "")
            if not address:
                continue

            if (idx + 1) % 50 == 0:
                self.log_result(f"Progress: {idx + 1}/{len(properties_df)}", updated_count)

            # Get real commute data
            commute_data = self._get_real_commute(address)

            if commute_data:
                properties_df.at[idx, "距離(km)"] = commute_data["dist"]
                properties_df.at[idx, "walk_mins"] = commute_data["walk_mins"]
                properties_df.at[idx, "scooter_mins"] = commute_data["scooter_mins"]
            else:
                # Fallback: estimate from distance
                distance = float(str(row.get("距離(km)", "1.0") or "1.0"))
                properties_df.at[idx, "walk_mins"] = round(distance * 15)
                properties_df.at[idx, "scooter_mins"] = round(distance * 3) + 2

            updated_count += 1
            time.sleep(self.rate_limit_sleep)

            # Periodic save
            if (idx + 1) % 10 == 0:
                self._save_progress(properties_df)

        # Final save
        self.log_step(f"Saving updated data to {self.config.merged_csv}")
        properties_df.to_csv(self.config.merged_csv, index=False, encoding="utf-8-sig")
        self.log_result("Total updated", updated_count)

        return properties_df

    def _get_real_commute(self, address: str) -> Optional[Dict[str, Any]]:
        """Get real-world commute time from address to NCHU.

        Args:
            address: Property address

        Returns:
            Dict with 'dist', 'walk_mins', 'scooter_mins' or None
        """
        # Geocode address
        lat, lon = self._geocode_address(address)
        if not lat:
            return None

        # Get routing data
        try:
            results = {
                "dist": 0,
                "walk_mins": 0,
                "scooter_mins": 0,
            }

            # Walking time
            walk_url = f"http://router.project-osrm.org/route/v1/foot/{lon},{lat};{self.NCHU_LON},{self.NCHU_LAT}?overview=false"
            walk_resp = requests.get(walk_url, timeout=self.timeout).json()
            if walk_resp.get("routes"):
                route = walk_resp["routes"][0]
                results["dist"] = round(route["distance"] / 1000, 2)
                results["walk_mins"] = round(route["duration"] / 60)

            # Driving/Scooter time
            drive_url = f"http://router.project-osrm.org/route/v1/car/{lon},{lat};{self.NCHU_LON},{self.NCHU_LAT}?overview=false"
            drive_resp = requests.get(drive_url, timeout=self.timeout).json()
            if drive_resp.get("routes"):
                results["scooter_mins"] = round(drive_resp["routes"][0]["duration"] / 60) + 2

            # Sanity check
            if results["dist"] > 0:
                min_expected_walk = round(results["dist"] * 12)
                if results["walk_mins"] < min_expected_walk:
                    results["walk_mins"] = min_expected_walk

            return results

        except Exception as e:
            self.logger.debug(f"Routing error: {e}")
            return None

    def _geocode_address(self, address: str) -> tuple:
        """Convert address to coordinates using ArcGIS.

        Args:
            address: Property address

        Returns:
            (latitude, longitude) or (None, None) on failure
        """
        try:
            # Normalize address
            search_addr = (
                address if "台中" in address else f"台中市{address}"
            )
            search_addr = search_addr.split("樓")[0].split("F")[0].split("f")[0]

            # Query ArcGIS
            encoded_addr = urllib.parse.quote(search_addr)
            geo_url = f"https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates?address={encoded_addr}&f=json&maxLocations=1"

            resp = requests.get(geo_url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("candidates"):
                    loc = data["candidates"][0]["location"]
                    return loc["y"], loc["x"]

        except Exception as e:
            self.logger.debug(f"Geocoding error: {e}")

        return None, None

    def _save_progress(self, df: pd.DataFrame) -> None:
        """Save progress checkpoint."""
        try:
            df.to_csv(self.config.merged_csv, index=False, encoding="utf-8-sig")
        except Exception as e:
            self.logger.warning(f"Failed to save progress: {e}")
