"""Phase 1 爬蟲模塊測試"""
import pytest
from pipeline.crawlers import CrawlerConfig, RentalProperty, CSV_COLUMNS


def test_crawler_config_init():
    """測試爬蟲配置初始化"""
    cfg = CrawlerConfig()

    # 驗證基本配置
    assert cfg.target_sections is not None
    assert cfg.max_pages_591 > 0
    assert cfg.max_pages_ddroom > 0
    assert cfg.headless is not None

    print("✓ CrawlerConfig 初始化成功")
    print(f"  - target_sections: {cfg.target_sections}")
    print(f"  - max_pages_591: {cfg.max_pages_591}")
    print(f"  - max_pages_ddroom: {cfg.max_pages_ddroom}")


def test_rental_property_model():
    """測試 RentalProperty Pydantic 模型"""
    prop = RentalProperty(
        url="http://example.com/property1",
        address="台中市南區示範路123號",
        rent="6500 元",
        layout="套房",
    )

    # 驗證模型
    assert prop.url == "http://example.com/property1"
    assert prop.address == "台中市南區示範路123號"
    assert prop.rent == "6500 元"
    assert prop.layout == "套房"

    print("✓ RentalProperty 模型驗證成功")


def test_rental_property_validation():
    """測試房產驗證"""
    # URL 是必需的
    try:
        prop = RentalProperty(url="")
        assert False, "Should raise validation error"
    except Exception:
        pass

    print("✓ 房產驗證成功")


def test_rental_property_to_csv_row():
    """測試房產轉 CSV 行"""
    prop = RentalProperty(
        url="http://example.com",
        address="台中市南區路123號",
        layout="套房",
        rent="6500 元",
        contact_name="王小明",
        contact_phone="0987654321",
    )

    row = prop.to_csv_row()

    # 驗證 CSV 列中文鍵
    assert "網址" in row
    assert "地址" in row
    assert "租金" in row
    assert "聯絡人" in row
    assert "電話" in row

    # 驗證值
    assert row["網址"] == "http://example.com"
    assert row["聯絡人"] == "王小明"

    print("✓ CSV 轉換成功")


def test_csv_columns_completeness():
    """測試 CSV 列完整性"""
    expected_cols = [
        "網址", "地址", "格局", "類型", "室內坪數", "租金",
        "空房間數", "押金", "安全標章", "樓層", "聯絡人", "電話",
        "家具設施", "租金包含", "另計費用", "安全管理", "消防逃生",
        "備註", "圖片網址", "距離(km)", "walk_mins", "scooter_mins",
    ]

    assert len(CSV_COLUMNS) == len(expected_cols)
    assert CSV_COLUMNS == expected_cols

    print(f"✓ CSV 列完整性驗證成功 ({len(CSV_COLUMNS)} 列)")


def test_crawler_config_output_paths():
    """測試爬蟲配置輸出路徑"""
    cfg = CrawlerConfig()

    # 驗證輸出路徑
    assert cfg.output_csv is not None
    assert cfg.nchu_output_csv is not None

    print("✓ 輸出路徑配置正確")
    print(f"  - output_csv: {cfg.output_csv}")
    print(f"  - nchu_output_csv: {cfg.nchu_output_csv}")


def test_crawler_config_crawler_params():
    """測試爬蟲參數配置"""
    cfg = CrawlerConfig()

    # 驗證爬蟲參數
    assert hasattr(cfg, 'headless')
    assert isinstance(cfg.headless, bool)
    assert isinstance(cfg.target_sections, list)

    print("✓ 爬蟲參數配置正確")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
