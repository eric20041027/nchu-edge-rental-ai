"""OSRMClient — real road-network distance and commute time calculation.

Wraps ArcGIS geocoding + OSRM routing.  Results are cached in-memory to
avoid redundant API calls during batch processing.
"""
from __future__ import annotations
import urllib.parse
import requests

# NCHU main gate
NCHU_LAT = 24.1252
NCHU_LON = 120.6818

# Walking speed: ~12-15 min/km (conservative)
WALK_MIN_PER_KM = 13.0


class OSRMClient:
    """Compute real walking and scooter commute times to NCHU."""

    def __init__(
        self,
        osrm_server: str = "http://router.project-osrm.org",
        origin_lat: float = NCHU_LAT,
        origin_lon: float = NCHU_LON,
        timeout: int = 10,
    ):
        self.osrm = osrm_server.rstrip("/")
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.timeout = timeout
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_commute(self, address: str) -> dict | None:
        """
        Returns {'dist': km, 'walk_mins': int, 'scooter_mins': int}
        or None if geocoding/routing fails.
        """
        if address in self._cache:
            return self._cache[address]

        coords = self._geocode(address)
        if not coords:
            return None

        lat, lon = coords
        result = self._route(lat, lon)
        if result:
            self._cache[address] = result
        return result

    def compute_weight(self, address: str, mode: str = "walk") -> float:
        """
        Normalised 0-1 weight based on commute distance.
        Closer → higher weight.  Beyond 20 km → 0.
        """
        info = self.get_commute(address)
        if not info:
            return 0.5   # neutral fallback
        dist = info.get("dist", 0)
        return max(0.0, 1.0 - dist / 20.0)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _geocode(self, address: str) -> tuple[float, float] | None:
        search = address if "台中" in address else f"台中市{address}"
        search = search.split("樓")[0].split("F")[0].split("f")[0]
        encoded = urllib.parse.quote(search)
        url = (
            "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer"
            f"/findAddressCandidates?address={encoded}&f=json&maxLocations=1"
        )
        try:
            resp = requests.get(url, timeout=self.timeout)
            data = resp.json()
            if data.get("candidates"):
                loc = data["candidates"][0]["location"]
                return loc["y"], loc["x"]
        except Exception:
            pass
        return None

    def _route(self, lat: float, lon: float) -> dict | None:
        results = {"dist": 0.0, "walk_mins": 0, "scooter_mins": 0}
        olon, olat = self.origin_lon, self.origin_lat

        walk_url = f"{self.osrm}/route/v1/foot/{lon},{lat};{olon},{olat}?overview=false"
        drive_url = f"{self.osrm}/route/v1/car/{lon},{lat};{olon},{olat}?overview=false"

        try:
            w = requests.get(walk_url, timeout=self.timeout).json()
            if w.get("routes"):
                route = w["routes"][0]
                dist_km = round(route["distance"] / 1000, 2)
                walk_mins = round(route["duration"] / 60)
                # Sanity check: enforce minimum based on distance
                min_expected = round(dist_km * WALK_MIN_PER_KM)
                results["dist"] = dist_km
                results["walk_mins"] = max(walk_mins, min_expected)

            d = requests.get(drive_url, timeout=self.timeout).json()
            if d.get("routes"):
                scooter_mins = round(d["routes"][0]["duration"] / 60) + 2  # parking buffer
                results["scooter_mins"] = scooter_mins

            return results
        except Exception:
            return None
