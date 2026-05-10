"""Environment-driven configuration for all crawlers."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

def _env_list(key: str, default: str) -> list[int]:
    return [int(v) for v in os.getenv(key, default).split(",") if v.strip()]

def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))

@dataclass
class CrawlerConfig:
    """Central configuration object."""
    output_csv: Path = field(default_factory=lambda: Path(os.getenv("CRAWLER_OUTPUT_CSV", str(_REPO_ROOT / "data/raw/nchu_rental_info.csv"))))
    nchu_output_csv: Path = field(default_factory=lambda: Path(os.getenv("CRAWLER_NCHU_CSV", str(_REPO_ROOT / "data/raw/nchu_official_raw.csv"))))
    fb_output_json: Path = field(default_factory=lambda: Path(os.getenv("CRAWLER_FB_JSON", str(_REPO_ROOT / "data/raw/fb_queries.json"))))
    target_sections: list[int] = field(default_factory=lambda: _env_list("CRAWLER_SECTIONS", "69,82,92,65,66"))
    room_kinds: list[int] = field(default_factory=lambda: _env_list("CRAWLER_ROOM_KINDS", "2,3"))
    max_pages_591: int = field(default_factory=lambda: _env_int("CRAWLER_591_MAX_PAGES", 3))
    max_pages_ddroom: int = field(default_factory=lambda: _env_int("CRAWLER_DDROOM_MAX_PAGES", 100))
    nchu_pages: int = field(default_factory=lambda: _env_int("CRAWLER_NCHU_PAGES", 15))
    headless: bool = field(default_factory=lambda: os.getenv("CRAWLER_HEADLESS", "false").lower() == "true")
