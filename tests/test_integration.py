"""集成測試 — Phase 1 + Phase 2 協調器"""
import pytest
from pipeline.crawlers import CrawlerConfig
from pipeline.data_prep import DataPrepConfig, DataPipeline


def test_crawler_module_complete():
    """測試爬蟲模塊完整性"""
    from pipeline.crawlers import (
        CrawlerConfig,
        RentalProperty,
        CSV_COLUMNS,
    )
    from pipeline.crawlers.base import BaseCrawler

    # 驗證所有公開類和常數都能導入
    cfg = CrawlerConfig()
    assert cfg is not None
    assert len(CSV_COLUMNS) == 22
    assert BaseCrawler is not None

    print("✓ 爬蟲模塊完整性驗證成功")


def test_data_prep_module_complete():
    """測試數據準備模塊完整性"""
    from pipeline.data_prep import (
        DataPrepConfig,
        DataMerger,
        DatasetGenerator,
        SemanticAugmenter,
        HardNegativeMiner,
        SilverLabeler,
        CommuteDataUpdater,
        BudgetTrapGenerator,
        EmbeddingPrecomputer,
        DataPipeline,
    )

    cfg = DataPrepConfig()
    assert cfg is not None

    # 驗證所有 9 個處理器都能導入
    pipeline = DataPipeline(cfg)
    assert pipeline is not None

    print("✓ 數據準備模塊完整性驗證成功")


def test_data_pipeline_orchestration():
    """測試 DataPipeline 協調器"""
    cfg = DataPrepConfig()
    pipeline = DataPipeline(cfg)

    # 驗證協調器有所有必需的方法
    assert hasattr(pipeline, 'run')
    assert hasattr(pipeline, 'run_step')
    assert hasattr(pipeline, '_run_merge')
    assert hasattr(pipeline, '_run_generate')
    assert hasattr(pipeline, '_run_augment')
    assert hasattr(pipeline, '_run_mine')
    assert hasattr(pipeline, '_run_embed')

    print("✓ DataPipeline 協調器驗證成功")


def test_pydantic_models():
    """測試 Pydantic 數據模型"""
    from pipeline.crawlers import RentalProperty
    from pipeline.data_prep import (
        MergedRental,
        QueryPropertyPair,
        PropertyEmbedding,
        BudgetTrap,
        HardNegativeExample,
    )

    # 測試爬蟲模型
    prop = RentalProperty(url="http://test.com", address="測試地址")
    assert prop.url == "http://test.com"

    # 測試數據準備模型
    merged = MergedRental(url="http://test.com", address="測試地址")
    assert merged.url == "http://test.com"

    pair = QueryPropertyPair(
        query="想租套房",
        property_id="prop123",
        is_match=True,
        score=3
    )
    assert pair.query == "想租套房"

    embedding = PropertyEmbedding(
        property_id="prop123",
        embedding=[0.1, 0.2, 0.3]
    )
    assert len(embedding.embedding) == 3

    print("✓ Pydantic 數據模型驗證成功")


def test_configuration_hierarchy():
    """測試配置層次"""
    from pipeline.crawlers import CrawlerConfig
    from pipeline.data_prep import DataPrepConfig

    crawler_cfg = CrawlerConfig()
    data_cfg = DataPrepConfig()

    # 驗證兩個配置都能初始化
    assert crawler_cfg is not None
    assert data_cfg is not None

    # 驗證配置參數
    assert hasattr(crawler_cfg, 'target_sections')
    assert hasattr(data_cfg, 'train_split')

    print("✓ 配置層次驗證成功")


def test_base_classes():
    """測試抽象基類"""
    from pipeline.crawlers.base import BaseCrawler
    from pipeline.data_prep import BaseProcessor

    # 驗證基類存在
    assert BaseCrawler is not None
    assert BaseProcessor is not None

    # 驗證它們是抽象的
    assert hasattr(BaseCrawler, 'run')
    assert hasattr(BaseProcessor, 'run')

    print("✓ 抽象基類驗證成功")


def test_end_to_end_imports():
    """測試端到端導入流程"""
    # 爬蟲模塊
    from pipeline.crawlers import CrawlerConfig, RentalProperty
    from pipeline.crawlers_runner import main as crawlers_main

    # 數據準備模塊
    from pipeline.data_prep import DataPrepConfig, DataPipeline
    from pipeline.data_prep_runner import main as dataprep_main

    # 驗證所有導入都成功
    assert CrawlerConfig is not None
    assert RentalProperty is not None
    assert DataPrepConfig is not None
    assert DataPipeline is not None

    print("✓ 端到端導入流程驗證成功")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
