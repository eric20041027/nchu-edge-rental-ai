# Development Guide — 開發者指南

## 環境建置

```bash
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
playwright install chromium
```

**必要條件**：CUDA GPU（訓練需要），Python 3.10+，Windows/Linux

---

## 兩階段蒸餾訓練

```bash
set PYTHONUTF8=1

# 第一步：訓練 rbt6 teacher（~2 小時，RTX 3060）
python -m pipeline.model_training.train_teacher

# 第二步：蒸餾至 rbt3 + ONNX 導出 + INT8 量化（~20 分鐘）
python -m pipeline.model_training.train_and_export_onnx
```

輸出：
- `saved_models/rbt6_teacher/` — Teacher checkpoint（永不被 student 覆蓋）
- `saved_models/rbt3_finetuned/` — Student PyTorch checkpoint
- `frontend/models/custom_onnx_model_dir/my_custom_model_quant.onnx` — 部署模型

---

## 消融實驗

```bash
set PYTHONUTF8=1
python -m pipeline.model_training.ablation_runner
# 已完成的 run 自動跳過（skip if metrics.json exists）
# 結果：ablation_results/summary.json
```

詳見 [ABLATION_STUDY.md](ABLATION_STUDY.md)。

---

## 模型評估

```bash
set PYTHONUTF8=1
python -m pipeline.model_training.evaluate_model
# 輸出：NDCG@5、Bootstrap CI、Phase 1 分類指標
```

---

## 資料流水線（完整重建）

```bash
set PYTHONUTF8=1
python pipeline_runner.py
# 執行全部 6 步：爬取 → 清洗 → 標記 → 訓練集生成 → 困難樣本挖掘 → 匯出
```

---

## 本地前端預覽

```bash
cd frontend && python -m http.server 8000
# 開啟 http://localhost:8000
```

---

## 目錄結構

```text
.
├── data/
│   ├── raw/                 # 原始爬取數據
│   └── processed/           # 訓練集 / 驗證集 / 測試集 / 前端房源 JSON
├── docs/                    # 技術文件
│   ├── MODEL_ARCHITECTURE.md
│   ├── TRAINING_STRATEGY.md
│   ├── ABLATION_STUDY.md
│   ├── DATA_PIPELINE.md
│   ├── EDGE_INFERENCE.md
│   └── DEVELOPMENT.md
├── frontend/
│   ├── index.html
│   ├── benchmark.html       # 效能測試工具
│   ├── sw.js                # Service Worker
│   └── js/
│       ├── app.js           # 主應用邏輯
│       ├── inference.js     # Cross-Encoder 推論介面
│       ├── inference-worker.js  # Cross-Encoder Web Worker
│       └── ner-worker.js        # NER Web Worker
├── pipeline/
│   ├── crawlers/            # 多源爬蟲
│   ├── data_prep/           # 6 步資料流水線
│   ├── model_training/
│   │   ├── train_teacher.py          # rbt6 teacher 訓練
│   │   ├── train_and_export_onnx.py  # rbt3 student 蒸餾 + ONNX + INT8
│   │   ├── ablation_runner.py        # 消融實驗主入口
│   │   ├── ablation_config.py        # 消融配置
│   │   ├── ablation_train.py         # 消融訓練模組
│   │   ├── training_utils.py         # 共用工具（FGM、metrics、callbacks）
│   │   ├── evaluate_model.py         # 多指標評估
│   │   └── quantize_model.py         # 獨立量化腳本
│   ├── ner_model/
│   └── constraints/         # 硬約束邏輯
├── ablation_results/        # 消融實驗結果 JSON
├── saved_models/
│   ├── rbt6_teacher/        # Teacher checkpoint（永不被 student 覆蓋）
│   └── rbt3_finetuned/      # Student checkpoint
└── pipeline_runner.py       # 端到端入口點
```

---

## 模型版本演進

| 版本 | 量化大小 | Teacher F1 | Student F1 | NDCG@5 | 關鍵改動 |
|:---|:---|:---|:---|:---|:---|
| rbt6 FT (v2.2) | 57 MB | — | 84.8% | — | — |
| rbt3 KD v1 (v2.3) | 37 MB | 84.8% | 85.1% | 0.818 | KD 首次啟用 |
| rbt3 R-Drop (v2.4) | 37 MB | — | 76.9% | 0.727 | Teacher 路徑 bug |
| rbt3 KD v2 (v2.5) | 36.8 MB | 78.7% | 76.4% | 0.760 | 同 bug |
| **rbt3 KD v3 (v2.9)** | **38.6 MB** | **85.9%** | **85.5%** | **0.833** | Bug 修復 + 全功能 |
| **rbt3 v3.0** | **36.8 MB** | **85.9%** | **85.4%** | **~0.879** | R-Drop 移除（消融）|

v2.4–v2.8 退步根本原因：Teacher 路徑被 student 覆蓋，pre-trained rbt6 random head 作為 teacher → soft label 噪聲。
