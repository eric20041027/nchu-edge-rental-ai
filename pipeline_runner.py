"""Unified entry point for end-to-end pipeline (Phase 1-2-3)."""
import sys
import logging
import argparse
from pathlib import Path

from pipeline.orchestrator import PipelineOrchestrator


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end pipeline (Phase 1: Crawling, Phase 2: Data Prep, Phase 3: Training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python pipeline_runner.py

  # Skip Phase 1 (use existing data)
  python pipeline_runner.py --skip-phase 1

  # Run only Phase 3 (training)
  python pipeline_runner.py --skip-phase 1 --skip-phase 2

  # Run Phase 1 and 2 only
  python pipeline_runner.py --skip-phase 3
        """,
    )

    parser.add_argument(
        "--skip-phase",
        type=int,
        action="append",
        choices=[1, 2, 3],
        default=[],
        help="Phase(s) to skip (can be used multiple times)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed logging output",
    )

    return parser.parse_args()


def main():
    """Execute the end-to-end pipeline."""
    setup_logging()
    args = parse_arguments()

    logger = logging.getLogger("pipeline_runner")

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        logger.info("Initializing end-to-end pipeline orchestrator")

        orchestrator = PipelineOrchestrator(skip_phases=args.skip_phase)
        result = orchestrator.run()

        logger.info("")
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║" + " " * 15 + "FINAL RESULTS SUMMARY" + " " * 21 + "║")
        logger.info("╚" + "═" * 58 + "╝")
        logger.info("")

        if "phase_1_result" in result and result["phase_1_result"]:
            logger.info("Phase 1 (Crawling):")
            logger.info(f"  Status: ✓ Completed")

        if "phase_2_result" in result and result["phase_2_result"]:
            logger.info("Phase 2 (Data Preparation):")
            logger.info(f"  Status: ✓ Completed")

        if "phase_3_result" in result and result["phase_3_result"]:
            logger.info("Phase 3 (Model Training):")
            logger.info(f"  Status: ✓ Completed")

        logger.info("")
        logger.info(f"Total execution time: {result['total_time_seconds']:.2f} seconds")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
