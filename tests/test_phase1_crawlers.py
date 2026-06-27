"""Phase 1 爬蟲模塊測試。

批次2:移除對孤兒抽象 RentalProperty(已刪)的測試;CSV_COLUMNS 改測現役
shared 的 19 欄 schema(與 nchu_rental_info.csv 一致)。
"""
import pytest
from pipeline.crawlers import CrawlerConfig, CSV_COLUMNS


def test_crawler_config_init():
    """測試爬蟲配置初始化"""
    cfg = CrawlerConfig()
    assert cfg.target_sections is not None
    assert cfg.max_pages_591 > 0
    assert cfg.max_pages_ddroom > 0
    assert cfg.headless is not None


def test_csv_columns_completeness():
    """測試現役 19 欄 schema(shared.CSV_COLUMNS,三 crawler 共用)。"""
    expected_cols = [
        "網址", "地址", "類型", "室內坪數", "租金", "押金", "樓層", "電話",
        "家具設施", "另計費用", "水費", "電費", "租屋補助", "特色", "最短租期",
        "圖片網址", "距離(km)", "walk_mins", "scooter_mins",
    ]
    assert len(CSV_COLUMNS) == 19
    assert CSV_COLUMNS == expected_cols


def test_crawler_config_output_paths():
    """測試爬蟲配置輸出路徑"""
    cfg = CrawlerConfig()
    assert cfg.output_csv is not None
    assert cfg.nchu_output_csv is not None


def test_crawler_config_crawler_params():
    """測試爬蟲參數配置"""
    cfg = CrawlerConfig()
    assert hasattr(cfg, 'headless')
    assert isinstance(cfg.headless, bool)
    assert isinstance(cfg.target_sections, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
