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
- `frontend/models/custom_onnx_model_dir/my_custom_model_quant.onnx` — Cross-Encoder 精排部署模型（現為 C 組房源富化模型，38.7 MB / 38,721,068 bytes；舊版曾備份為 `*.PREV-20260616.onnx`，已於 dead-weight 清理（收尾 B，PR #44）移除）
- `frontend/models/bi_encoder_dir/bi_encoder_quant.onnx` — bi-encoder 向量召回部署模型（INT8，57.0 MB / 59,784,101 bytes）

---

## bi-encoder 向量召回（訓練 + 導出 + embedding 預計算）

```bash
set PYTHONUTF8=1

# 第一步：訓練 bi-encoder（shared-weight encoder，InfoNCE/MNRL）
python -m pipeline.model_training.train_bi_encoder

# 第二步：導出 ONNX + Dynamic INT8（mean-pool + L2-norm baked in graph）
python -m pipeline.model_training.export_bi_encoder

# 第三步：離線預計算房源向量 → frontend/assets/property_embeddings.json
python -m pipeline.data_prep.build_property_embeddings
```

輸出：
- `frontend/models/bi_encoder_dir/bi_encoder_quant.onnx` — 召回部署模型（57.0 MB / 59,784,101 bytes）
- `frontend/assets/property_embeddings.json` — 房源 embedding（704×768 float16，L2-norm）

向量召回 vs rule-based 的 A/B harness（go/no-go gate，T7 判定 GO）：

```bash
python tests/eval_vector_vs_rulebased.py
```

詳見 [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md#向量召回-bi-encodervector-recall) 與 [spec/vector-retrieval.md](spec/vector-retrieval.md)。

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
│   ├── assets/
│   │   └── property_embeddings.json  # 房源 embedding（704×768 float16，L2-norm）
│   └── js/
│       ├── app.js           # 主應用邏輯
│       ├── inference.js     # Cross-Encoder 推論介面
│       ├── inference-worker.js  # Cross-Encoder Web Worker
│       ├── bi-encoder-worker.js # bi-encoder 向量召回 Web Worker
│       └── ner-worker.js        # NER Web Worker
├── pipeline/
│   ├── crawlers/            # 多源爬蟲
│   ├── data_prep/           # 6 步資料流水線
│   │   ├── augment_with_expansion_map.py  # C 組房源富化（property_to_text_enriched）
│   │   ├── precompute_ce_text.py          # 把 C 組富化 ce_text 預算進前端 JSON
│   │   └── build_property_embeddings.py   # 離線預計算房源向量 → property_embeddings.json
│   ├── model_training/
│   │   ├── train_teacher.py          # rbt6 teacher 訓練
│   │   ├── train_and_export_onnx.py  # rbt3 student 蒸餾 + ONNX + INT8
│   │   ├── train_bi_encoder.py       # bi-encoder 向量召回訓練（InfoNCE/MNRL）
│   │   ├── export_bi_encoder.py      # bi-encoder ONNX 導出 + INT8
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
├── tests/
│   └── eval_vector_vs_rulebased.py  # 向量召回 vs rule-based A/B harness（go/no-go gate）
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
| **rbt3 C 組富化** | **38.7 MB** | — | **85.4%** | **0.9475** | 房源文字富化（`property_to_text_enriched`，`MAX_LENGTH=128`）；NDCG@5 +0.0125 / F1 +0.021 vs A baseline，數據見 [property_enrichment_value.md](property_enrichment_value.md) |

> C 組富化的 NDCG@5（0.9475）與上方 v3.0（~0.879）數量級不同，因評測 query 集不同，不可直接比較。

v2.4–v2.8 退步根本原因：Teacher 路徑被 student 覆蓋，pre-trained rbt6 random head 作為 teacher → soft label 噪聲。

> 上表為 **Cross-Encoder 精排**模型演進。另有 **bi-encoder 向量召回**模型（基底 `hfl/rbt6`，INT8，57.0 MB / 59,784,101 bytes），與精排為不同階段、評測指標亦不同（Recall@K / NDCG@5）。T7 A/B（判定 GO）：all Recall@30 0.057 → 0.412、semantic Recall@30 0.007 → 0.547。詳見 [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md#向量召回-bi-encodervector-recall)。
