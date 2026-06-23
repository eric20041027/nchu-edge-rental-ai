"""階段③ 本機段一鍵入口:crawl(可選)→ 富化 → property_data.json → --check 驗。

spec: docs/spec/data-pipeline-oneshot.md

本機段(純 CPU,無 torch)三步,段內零手動:
  1. (可選 --crawl)run_crawlers → 更新 data/raw/*.csv(需網路)
  2. precompute_embeddings.main()      → frontend/assets/property_data.json(基本富化,無 ce_text)
  3. precompute_ce_text.py --write     → 補 ce_text(CE 精排必需;缺它前端會 OOD)
  4. build_property_embeddings --check  → 驗記錄數/欄位(無需 torch)

向量重算(property_embeddings.json,需 torch)是 Colab 段,不在此:
  python -m pipeline.data_prep.build_property_embeddings
"""
import argparse
import subprocess
import sys
from pathlib import Path

_DATA_PREP = Path(__file__).parent / "data_prep"


def _run_step(label: str, argv: list[str]) -> int:
    """跑一個子步驟(subprocess),回傳 exit code;非 0 即印錯。"""
    print(f"\n[build_frontend_data] → {label}")
    rc = subprocess.call(argv)
    if rc != 0:
        print(f"[build_frontend_data] ✗ {label} 失敗(exit {rc})", file=sys.stderr)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="本機段:富化房源 → property_data.json + 驗證")
    ap.add_argument("--crawl", action="store_true",
                    help="先跑 ddroom/nchu 爬蟲更新 CSV(需網路、pydantic;預設用既有 CSV)")
    args = ap.parse_args()

    if args.crawl:
        print("[build_frontend_data] → crawl(ddroom/nchu)")
        from pipeline import run_crawlers          # lazy:只在 --crawl 時才載入重依賴
        from pipeline.crawlers import CrawlerConfig
        run_crawlers(CrawlerConfig())

    # 富化兩步 + 驗證一步(皆純 CPU)
    steps = [
        ("precompute_embeddings → property_data.json",
         [sys.executable, "-m", "pipeline.data_prep.precompute_embeddings"]),
        ("precompute_ce_text --write → 補 ce_text",
         [sys.executable, str(_DATA_PREP / "precompute_ce_text.py"), "--write"]),
        ("build_property_embeddings --check → 驗證",
         [sys.executable, "-m", "pipeline.data_prep.build_property_embeddings", "--check"]),
    ]
    for label, argv in steps:
        rc = _run_step(label, argv)
        if rc != 0:
            return rc

    print("\n[build_frontend_data] ✓ 本機段完成。下一步 Colab 段重算向量:")
    print("    python -m pipeline.data_prep.build_property_embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
