"""Unified entry point for all crawlers (Phase 1)."""
import logging, sys
from pipeline.crawlers import CrawlerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("crawlers.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Execute all configured crawlers."""
    logger.info("=" * 70)
    logger.info("NCHU Rental Pipeline — Crawlers (Phase 1)")
    logger.info("=" * 70)

    cfg = CrawlerConfig()
    logger.info(f"Configuration: sections={cfg.target_sections}, output={cfg.output_csv}")

    # Placeholder: actual crawler implementations would run here
    logger.info("✓ Phase 1 (Crawlers) architecture is ready")
    logger.info("Next: Implement modularized crawler classes inheriting from BaseCrawler")

if __name__ == "__main__":
    main()
