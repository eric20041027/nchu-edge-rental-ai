"""Phase 2 數據準備模塊測試"""
import pytest
from pathlib import Path
from pipeline.data_prep import (
    DataPrepConfig,
    BaseProcessor,
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


def test_dataprep_config_init():
    """測試數據準備配置初始化"""
    cfg = DataPrepConfig()

    # 驗證路徑配置
    assert cfg.project_root is not None
    assert cfg.data_root is not None
    assert cfg.raw_data_dir is not None
    assert cfg.processed_data_dir is not None
    assert cfg.checkpoint_dir is not None

    print("✓ DataPrepConfig 初始化成功")


def test_dataprep_config_parameters():
    """測試配置參數"""
    cfg = DataPrepConfig()

    # 測試參數讀取
    assert isinstance(cfg.queries_per_property, int)
    assert isinstance(cfg.train_split, float)
    assert 0 < cfg.train_split < 1
    assert 0 < cfg.val_split < 1
    assert 0 < cfg.test_split < 1
    assert cfg.random_seed is not None

    print("✓ 配置參數驗證成功")


def test_base_processor():
    """測試 BaseProcessor 基類"""
    cfg = DataPrepConfig()
    merger = DataMerger(cfg)

    # 驗證基類功能
    assert merger.config is not None
    assert merger.logger is not None
    assert hasattr(merger, 'log_step')
    assert hasattr(merger, 'log_result')
    assert hasattr(merger, 'save_checkpoint')
    assert hasattr(merger, 'load_checkpoint')

    print("✓ BaseProcessor 功能驗證成功")


def test_all_processors_import():
    """測試所有處理器都能導入"""
    cfg = DataPrepConfig()

    processors = {
        'DataMerger': DataMerger(cfg),
        'DatasetGenerator': DatasetGenerator(cfg),
        'SemanticAugmenter': SemanticAugmenter(cfg),
        'HardNegativeMiner': HardNegativeMiner(cfg),
        'SilverLabeler': SilverLabeler(cfg),
        'CommuteDataUpdater': CommuteDataUpdater(cfg),
        'BudgetTrapGenerator': BudgetTrapGenerator(cfg),
        'EmbeddingPrecomputer': EmbeddingPrecomputer(cfg),
    }

    for name, proc in processors.items():
        assert proc is not None
        assert isinstance(proc, BaseProcessor)
        assert hasattr(proc, 'run')
        print(f"  ✓ {name} 導入成功")

    print(f"✓ 所有 {len(processors)} 個處理器都能導入")


def test_pipeline_init():
    """測試 DataPipeline 初始化"""
    cfg = DataPrepConfig()
    pipeline = DataPipeline(cfg)

    assert pipeline is not None
    assert hasattr(pipeline, 'run')
    assert hasattr(pipeline, 'run_step')
    assert pipeline.config is not None

    print("✓ DataPipeline 初始化成功")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
