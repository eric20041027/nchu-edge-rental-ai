# 興大 AI 租屋推薦 (NCHU AI Rental Recommendation)

這是一個專為中興大學學生設計的 AI 租屋推薦系統。使用者只需輸入自然語言需求（例如：「預算 6000 以內、近正門、有冷氣」），系統即可透過微調後的 ALBERT 模型進行語意匹配，提供最適合的房源建議。

## 🌟 核心特徵

- **自然語言辨識**: 採用 Sentence-Pair Classification 模式，精準理解使用者需求。
- **邊緣端推論 (Edge AI)**: 使用 ONNX Runtime Web，模型直接在使用者瀏覽器運行，反應迅速且保護隱私。
- **直覺式介面**: 現代化、響應式設計，支援行動裝置。
- **完整的訓練管線**: 包含自動化合成資料集、模型訓練、匯出與量化流程。

## 🛠 核心技術棧

- **Frontend**:
  - 原生 JavaScript (ES6+)
  - [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WASM 加速)
  - CSS3 (現代化佈局與動畫)
- **Machine Learning**:
  - Python 3.10+
  - PyTorch
  - Hugging Face Transformers (ALBERT Tiny)
  - Hugging Face Datasets
- **Deployment**:
  - 支援 Vercel, GitHub Pages 等靜態託管平台。

## 🚀 快速開始

### 1. 執行網頁應用
本專案為靜態網頁，您可以直接開啟 `index.html` 或使用本地伺服器：

```bash
# 使用 Python 啟動伺服器
python3 -m http.server 8000
```
然後訪問 `http://localhost:8000`。

### 2. 開發與訓練環境設定
如果您需要重新訓練模型或產生資料集，請設定 Python 環境：

```bash
# 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install torch transformers datasets numpy onnx onnxruntime
```

## 🏗 專案架構與資料流

### 目錄結構
```text
├── index.html            # 網頁主進入點
├── styles.css             # 介面樣式
├── app.js                 # UI 邏輯與互動
├── inference.js           # ONNX 推論邏輯與模型載入
├── property_data.json      # 房源完整資訊 (供 UI 顯示)
├── custom_onnx_model_dir/ # 已匯出的 ONNX 模型與標記器
│   ├── model.onnx         # 權重檔
│   ├── model.onnx.data    # 外部權重資料 (>100MB 拆分)
│   └── tokenizer.json     # 標記器設定
├── scripts/               # 訓練與工具腳本
│   ├── generate_dataset.py       # 自動生成訓練集 (由 CSV 轉 JSON)
│   ├── train_and_export_onnx.py  # 模型微調與 ONNX 匯出
│   └── quantize_model.py         # (選用) 模型量化壓縮
└── nchu_rental_info.csv   # 原始房源資料庫
```

### 資料流 (Request Lifecycle)
1. **輸入**: 使用者在文字框輸入租屋需求。
2. **預處理**: `inference.js` 調用 `AutoTokenizer` 將文字轉換為 Token。
3. **推論**: 瀏覽器透過 WASM 執行 `model.onnx`，計算 Query 與各房源的匹配分數。
4. **渲染**: `app.js` 根據分數排序，將 Top-K 房源以卡片形式呈現。

## 📈 模型訓練流程

如果您想要更新房源或優化模型：

1. **更新資料**: 修改 `nchu_rental_info.csv`。
2. **生成資料集**:
   ```bash
   python scripts/generate_dataset.py
   ```
   這會模擬學生口語，自動生成正負配對樣本。
3. **訓練並匯出**:
   ```bash
   python scripts/train_and_export_onnx.py
   ```
   此步驟會微調 `albert-chinese-tiny` 並直接匯出為 `model.onnx`。
4. **部署模型**: 將生成的模型檔案移至 `custom_onnx_model_dir/` 即可。

## 📝 備註
- 模型採用 Sentence-Pair 模式，輸入格式為 `[CLS] 查詢 [SEP] 房屋描述 [SEP]`。
- 由於 ONNX 模型權重較大，建議使用支援 LFS 的 Git 託管。
