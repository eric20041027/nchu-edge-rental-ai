# Phase 2 — Data Prep 模組重構計劃

## 📊 現狀分析

### 現有檔案結構

```
pipeline/data_prep/
├── merge_sources.py           (合併多源數據 CSV)
├── generate_dataset.py        (生成訓練/驗證/測試集)
├── update_commute_data.py     (更新通勤時間數據)
├── precompute_embeddings.py   (預計算房產嵌入向量)
├── augment_with_llm.py        (LLM 語義擴增)
├── mine_hard_negatives.py     (困難負樣本挖掘)
├── silver_labeling.py         (弱標籤標注)
└── generate_budget_traps.py   (預算陷阱生成)
```

### 特點

- **8 個獨立腳本**，各自負責一個環節
- **無共用基類** — 重複的錯誤處理、日誌、配置邏輯
- **硬編碼路徑** — `../../data/raw` 等相對路徑散落各檔
- **無數據驗證** — Pandas DataFrame 傳來傳去，無型別保障
- **無協調器** — `run_pipeline.sh` 中各步驟獨立呼叫，無依賴管理

---

## 🎯 Phase 2 目標

使用與 Phase 1 **相同的模塊化架構模式** 改造 Data Prep：

1. **共用資料模型** — Pydantic 模型定義數據結構
2. **配置管理** — 路徑、超參數從環境變數讀取
3. **基類抽象** — `BaseProcessor` 提供日誌、錯誤處理
4. **模塊內聚** — 各處理步驟為獨立類，實作公開 API
5. **協調器** — 中央 `DataPipeline` 管理執行流程

---

## 🏗️ 提議架構

### 層次結構

```
pipeline/data_prep/
├── __init__.py
├── config.py          ← DataPrepConfig (合併路徑、參數)
├── models.py          ← MergedRental, Dataset, EmbeddingBatch 等
├── base.py            ← BaseProcessor (日誌、重試、驗證)
│
├── merger.py          ← DataMerger 類 (merge_sources.py 改造)
├── generator.py       ← DatasetGenerator 類 (generate_dataset.py)
├── embedder.py        ← EmbeddingPrecomputer 類
├── augmenter.py       ← SemanticAugmenter 類 (LLM)
├── miner.py           ← HardNegativeMiner 類
├── labeler.py         ← SilverLabeler 類
├── commute_updater.py ← CommuteDataUpdater 類
├── budget_generator.py ← BudgetTrapGenerator 類
│
└── pipeline.py        ← DataPipeline (協調所有步驟)
```

### 資料模型示例

```python
# models.py
from pydantic import BaseModel, Field

class MergedRental(BaseModel):
    """Merged and deduplicated rental record."""
    url: str
    address: str
    rent: float
    ...

class TrainingDataset(BaseModel):
    """Training dataset with query-property pairs."""
    train_pairs: list[tuple[str, str, bool]]  # (query, property_id, is_match)
    val_pairs: list[...]
    test_pairs: list[...]

class PropertyEmbedding(BaseModel):
    """Precomputed property embeddings."""
    property_id: str
    metadata_embedding: list[float]
    ...
```

### 基類設計

```python
# base.py
class BaseProcessor(abc.ABC):
    def __init__(self, config: DataPrepConfig, logger: Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger(...)
    
    @abc.abstractmethod
    def run(self) -> Any:
        """Execute this processing step."""
    
    def save_checkpoint(self, data: Any, name: str) -> Path:
        """Save intermediate checkpoint."""
    
    def load_checkpoint(self, name: str) -> Any:
        """Load intermediate checkpoint."""
```

### 協調器設計

```python
# pipeline.py
class DataPipeline:
    def __init__(self, config: DataPrepConfig):
        self.merger = DataMerger(config)
        self.generator = DatasetGenerator(config)
        self.embedder = EmbeddingPrecomputer(config)
        # ... 等等
    
    def run(self) -> None:
        """Execute full data prep pipeline."""
        merged = self.merger.run()
        dataset = self.generator.run(merged)
        embeddings = self.embedder.run(dataset)
        # ... 等等
```

---

## 📈 實施步驟（預計 4-6 小時）

### Step 1: 設計資料模型 (1-2h)
- [ ] 分析各步驟的輸入/輸出格式
- [ ] 定義 `MergedRental`, `TrainingDataset`, `Embedding` 等 Pydantic 模型
- [ ] 確保與現有 CSV 格式相容

### Step 2: 建立基礎層 (0.5h)
- [ ] 實作 `DataPrepConfig` (路徑、參數)
- [ ] 實作 `BaseProcessor` 抽象基類
- [ ] 建立 `__init__.py` 公開 API

### Step 3: 逐個改造現有模組 (2-3h)
- [ ] `merger.py` ← `merge_sources.py`
- [ ] `generator.py` ← `generate_dataset.py`
- [ ] `embedder.py` ← `precompute_embeddings.py`
- [ ] `augmenter.py` ← `augment_with_llm.py`
- [ ] `miner.py` ← `mine_hard_negatives.py`
- [ ] 其他...

### Step 4: 實作協調器 (0.5h)
- [ ] `pipeline.py` 統一執行順序

### Step 5: 測試與集成 (1-2h)
- [ ] 單元測試各 Processor
- [ ] 集成測試完整 Pipeline
- [ ] 驗證輸出與原版一致

---

## 💡 關鍵決策點

### Q1: 檢查點管理
**選項 A** — 記憶體中傳遞
```python
merged_data = merger.run()
dataset = generator.run(merged_data)  # 直接傳遞
```
**選項 B** — 磁碟檢查點
```python
merger.run()  # 儲存到 checkpoint/merger_output.pkl
dataset = generator.run()  # 從磁碟讀取
```

**建議**: 使用 **選項 A** 加可選檢查點存儲（以效率優先，可靠性次之）

### Q2: LLM 調用策略
**現狀**: `augment_with_llm.py` 同步呼叫 Claude API

**改進選項**:
- [ ] 保持同步（簡單）
- [ ] 改為非同步批量呼叫（効率高，需重構 augmenter.py）
- [ ] 支持本地 LLM fallback

**建議**: 保持同步優先，後續可改為非同步

### Q3: 生成策略
`generate_dataset.py` 目前隨機生成查詢，**無去重**

**改進**:
- 添加 `query_dedup=True` 選項
- 支援 seed 控制可重複性

---

## 📝 預期成果

### 完成後

```python
# 簡潔的公開 API
from pipeline.data_prep import DataPrepConfig, DataPipeline

cfg = DataPrepConfig()
pipeline = DataPipeline(cfg)
pipeline.run()

# 或 step-by-step
from pipeline.data_prep import DataMerger, DatasetGenerator

merger = DataMerger(cfg)
merged_data = merger.run()

generator = DatasetGenerator(cfg)
dataset = generator.run(merged_data)
```

### 檔案列表

- ✨ `pipeline/data_prep/__init__.py` — 公開 API
- ✨ `pipeline/data_prep/config.py` — 配置
- ✨ `pipeline/data_prep/models.py` — 資料模型
- ✨ `pipeline/data_prep/base.py` — 基類
- 🔄 `pipeline/data_prep/merger.py` — 改造自 merge_sources.py
- 🔄 `pipeline/data_prep/generator.py` — 改造自 generate_dataset.py
- ... （其他 6 個）
- ✨ `pipeline/data_prep/pipeline.py` — 協調器

---

## ⏱️ 時間線

- **預計**: 下週一開始
- **工作量**: 4-6 小時（分散 2-3 天）
- **交付物**: 
  - [x] 設計文檔 (本文)
  - [ ] 完整模塊化實現
  - [ ] 10+ 單元測試
  - [ ] 集成測試
  - [ ] Phase 2 完成報告

---

## 🚀 後續連鎖反應

Phase 2 完成後，Phase 3 (Model Training) 可採用**完全相同的模式**：

```
pipeline/model_training/
├── config.py
├── models.py
├── base.py
├── trainer.py
├── exporter.py
├── quantizer.py
├── evaluator.py
└── pipeline.py (ModelPipeline)
```

預計同樣 4-6 小時。

---

## 🎯 核心價值

✅ **一致性** — 全專案採用相同的模塊化模式
✅ **可維護性** — 添加新步驟無需改舊代碼
✅ **可擴展性** — 每個步驟可獨立使用或組合
✅ **可測試性** — 各步驟有清晰的輸入/輸出
✅ **生產就緒** — 配置、重試、日誌、驗證齊全

---

**準備開始 Phase 2？**
