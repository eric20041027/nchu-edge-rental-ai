# 興大 AI 租屋推薦 (NCHU AI Rental Recommendation)

這是一個專為中興大學學生設計的 AI 租屋推薦系統。使用者只需輸入自然語言需求（例如：「預算 6000 以內、近正門、有冷氣」），系統即可透過微調後的 ALBERT 模型進行語意匹配，提供最適合的房源建議。

## 核心特徵

- **自然語言辨識**: 採用 Sentence-Pair Classification 模式，精準理解使用者需求。
- **邊緣端推論 (Edge AI)**: 使用 ONNX Runtime Web，模型直接在使用者瀏覽器運行不需將資料回傳伺服器，反應迅速且保護隱私。
- **直覺式介面**: 現代化、響應式設計，支援行動裝置。
- **完整的訓練管線**: 包含自動化合成資料集、模型訓練、匯出與量化流程。

## 核心技術棧

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

## 快速開始

### 1. 執行網頁應用
本專案為靜態網頁，您可以直接開啟 `index.html` 或使用本地伺服器：

```bash
# 使用 Python 啟動伺服器
python3 -m http.server 8000
```
然後訪問 `http://localhost:8000`。

### 2. 開發與訓練環境設定
如果需要重新訓練模型或產生資料集，請設定 Python 環境：

```bash
# 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install torch transformers datasets numpy onnx onnxruntime
```

## 專案架構與資料流

本專案將流程分為資料準備、模型訓練、以及前端推論三個核心階段。

### 核心工作流圖 (Workflow Diagram)

```mermaid
graph TD
    A[nchu_rental_info.csv] --> B(generate_dataset.py)
    A --> C(precompute_embeddings.py)
    B --> D[Training Dataset .json]
    D --> E(train_and_export_onnx.py)
    E --> F[model.onnx]
    F --> G(quantize_model.py)
    G --> H[custom_onnx_model_dir/]
    C --> I[property_data.json]
    H --> J[inference.js]
    I --> J
    K[User Input] --> J
    J --> L[Recommended Listings]
```

### 1. 資料準備 (Data Preparation)
*   **原始資料**: 從 `nchu_rental_info.csv` 讀取房源資訊。
*   **描述生成**: `precompute_embeddings.py` 會讀取 CSV 並生成 `property_data.json`，其中包含每筆房源的「標準化描述文本」，這是 AI 進行比對的基準。

### 2. 模型訓練 (Model Training)
*   **合成資料**: `generate_dataset.py` 模擬學生口語（如：「預算 5k」、「套房」）生成正負配對樣本。
*   **微調與匯出**: `train_and_export_onnx.py` 基於 `albert-chinese-tiny` 進行二分類微調，並匯出為 `model.onnx`。可以使用 `quantize_model.py` 進一步壓縮模型體積。

### 3. Web 推論 (Runtime Inference)
當使用者輸入需求時：
1.  **硬性過濾**: `inference.js` 先根據性別限制、預算範圍（若包含「以下」等詞彙）過濾掉絕對不符的房源。
2.  **粗篩 (Stage 1)**: 利用關鍵字重疊度與價格接近程度，從候選房源中挑選出 Top 30。
3.  **精篩 (Stage 2)**: 將查詢與 Top 30 房源描述送入 **ALBERT ONNX 模型** 進行深度語意評分。
4.  **渲染**: `app.js` 根據最終加權分數進行排序並呈現。

## 目錄結構
```text
├── index.html            # 網頁主進入點
├── styles.css             # 介面樣式
├── app.js                 # UI 邏輯與互動
├── inference.js           # ONNX 推論邏輯與兩階段檢索系統
├── property_data.json      # 房源完整資訊與描述文本
├── custom_onnx_model_dir/ # 模型存放區 (過大時需使用 LFS)
│   ├── model.onnx         # ALBERT 權重
│   ├── model.onnx.data    # 外部權重資料
│   └── tokenizer.json     # 標記器設定
├── generate_dataset.py       # 自動生成模擬訓練集
├── train_and_export_onnx.py  # 模型微調與匯出
├── precompute_embeddings.py   # 生成 property_data.json
├── quantize_model.py         # 模型量化壓縮工具
└── nchu_rental_info.csv   # 原始房源資料庫
```

## 模型訓練流程

如果您想要更新房源或優化模型：

1. **更新資料**: 修改 `nchu_rental_info.csv`。
2. **生成資料集**:
   ```bash
   python generate_dataset.py
   ```
   這會模擬學生口語，自動生成正負配對樣本。
3. **訓練並匯出**:
   ```bash
   python train_and_export_onnx.py
   ```
   此步驟會微調 `albert-chinese-tiny` 並直接匯出為 `model.onnx`。
4. **更新推論資料**:
   ```bash
   python precompute_embeddings.py
   ```
   此步驟會根據原始 CSV 與新模型生成 `property_data.json`。
5. **部署模型**: 將生成的模型檔案與 JSON 移至對應目錄即可。

## 備註
- 模型採用 Sentence-Pair 模式，輸入格式為 `[CLS] 查詢 [SEP] 房屋描述 [SEP]`。
- 由於 ONNX 模型權重較大，建議使用支援 LFS 的 Git 託管。
