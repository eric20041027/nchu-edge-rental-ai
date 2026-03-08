# 🏠 興大 AI 租屋推薦系統

一套完全運行在瀏覽器端的 **AI 智慧租屋推薦平台**，使用者只需用自然語言輸入找房需求，系統即可透過自訓練的 ALBERT NER 模型進行語意分析，並以 Content-Based Filtering 演算法即時推薦最合適的中興大學周邊租屋物件。

> **核心亮點**：整套 AI 推論完全跑在 **WebAssembly (WASM)** 上，無需後端伺服器、無需 API Key，開啟網頁即可使用。

## Key Features

- 🧠 **自訓練 NER 模型** — 基於 `clue/albert_chinese_tiny` 微調，使用超過 10,000 筆標註資料訓練，驗證準確率達 **99.99%**
- ⚡ **瀏覽器端即時推論** — 透過 ONNXRuntime-Web (WebAssembly) 直接在前端執行模型推論，零延遲
- 🔍 **智慧語意解析** — 支援中文數字（六千）、英文縮寫（6K）、口語化表達（獨洗獨曬套房）
- 🏘️ **Content-Based Filtering** — 多維度 CBF 推薦演算法，涵蓋預算、房型、距離、設施等 11 種特徵
- 🗺️ **Google Maps 嵌入** — 每張房屋卡片內嵌即時地圖，一眼掌握位置資訊
- 📱 **響應式深色主題** — 現代 Glassmorphism 設計風格，完美適配手機與桌面

---

## Tech Stack

| 層級 | 技術 |
|------|------|
| **ML 模型** | ALBERT (chinese_tiny) + PyTorch + HuggingFace Transformers |
| **模型格式** | ONNX (Open Neural Network Exchange) + External Data Weights |
| **前端推論** | ONNXRuntime-Web (WebAssembly backend) |
| **Tokenizer** | Xenova/Transformers.js (`AutoTokenizer`) |
| **前端框架** | Vanilla HTML5 + CSS3 + JavaScript (ES Modules) |
| **資料解析** | PapaParse (CSV → JSON) |
| **字體** | Google Fonts — Noto Sans TC |
| **圖示** | FontAwesome 6 |
| **版本控制** | Git + GitHub |

---

## Architecture

### 系統架構圖

```
使用者輸入 (自然語言)
        │
        ▼
┌───────────────────────┐
│   Tokenizer (ALBERT)  │  ← Xenova/Transformers.js
│   字元切分 + 編碼      │
└───────────┬───────────┘
            │ input_ids, attention_mask, token_type_ids
            ▼
┌───────────────────────┐
│   ONNX Runtime (WASM) │  ← model.onnx + model.onnx.data
│   Token Classification│
│   3-Class NER 推論     │
└───────────┬───────────┘
            │ logits → argmax → B-Target / I-Target / O
            ▼
┌───────────────────────┐
│   Feature Extractor   │  ← tagFeatures() in inference.js
│   語意特徵結構化提取    │
│   (預算/房型/距離/設施) │
└───────────┬───────────┘
            │ CBF Feature Vector
            ▼
┌───────────────────────┐
│   Scoring Engine      │  ← recommend() in inference.js
│   Content-Based       │
│   Filtering 評分      │
└───────────┬───────────┘
            │ Top-K Results
            ▼
┌───────────────────────┐
│   UI Renderer         │  ← app.js → index.html
│   房屋卡片 + 地圖渲染  │
└───────────────────────┘
```

### 目錄結構

```
Renting_model_ONNX/
├── index.html                  # 主頁面 (租屋推薦介面)
├── styles.css                  # 深色主題樣式表
├── app.js                      # 前端互動邏輯 + 卡片渲染
├── inference.js                # AI 推論引擎 (NER + CBF 評分)
├── nchu_rental_info.csv        # 租屋物件資料庫 (591 爬蟲資料)
│
├── custom_onnx_model_dir/      # 前端載入的 ONNX 模型目錄
│   ├── model.onnx              # ONNX 模型主檔 (計算圖)
│   ├── model.onnx.data         # ONNX 外部權重檔 (~16MB)
│   ├── tokenizer.json          # ALBERT Tokenizer 字典
│   ├── tokenizer_config.json   # Tokenizer 設定檔
│   └── special_tokens_map.json # 特殊 Token 對應表
│
├── train_and_export_onnx.py    # 模型訓練 + ONNX 匯出腳本
├── train.json                  # 訓練資料集 (BIO 標註格式)
├── test.json                   # 驗證資料集 (BIO 標註格式)
│
├── my_custom_model.onnx        # 訓練產出的原始 ONNX 模型
├── my_custom_model.onnx.data   # 訓練產出的原始權重檔
├── my_trained_albert/          # 微調後的 PyTorch 模型備份
│
├── test_custom_model.html      # 模型功能測試頁面
├── custom_model_output/        # 訓練過程中的 Checkpoint
└── onnx_model_dir/             # 舊版公版模型 (已棄用)
```

### NER 標註格式 (BIO Tagging)

模型使用 3 類標籤進行 Token 級別分類：

| 標籤 ID | 標籤名稱 | 說明 |
|---------|----------|------|
| 0 | `O` | Outside — 非特徵字元（動詞、介系詞、語氣詞等） |
| 1 | `B-Target` | Begin — 特徵實體的開頭字 |
| 2 | `I-Target` | Inside — 特徵實體的內部字 |

**範例：**
```
輸入：想 找 預 算 六 千 內 的 套 房
標籤：O  O  B  I  B  I  I  O  B  I
          ↑預算    ↑六千內      ↑套房
```

### CBF 推薦演算法

`recommend()` 函數依據 11 種特徵維度進行加權評分：

| 特徵維度 | 最高分數 | 說明 |
|---------|----------|------|
| 預算匹配 | 20 分 | 差距 ≤500 元滿分，超出則線性遞減 |
| 地區匹配 | 20 分 | 地址含指定區域關鍵字 |
| 房型匹配 | 20 分 | 格局包含指定房型 |
| 建築類型 | 15 分 | 類型包含指定建築 |
| 距離匹配 | 25 分 | 距離符合需求標準 |
| 家具設施 | 每項 5 分 | 逐項比對物件設施清單 |
| 安全設備 | 每項 5 分 | 逐項比對安全管理清單 |

**硬性排除規則：**
- 預算「以上」→ 低於預算的房屋直接排除
- 預算「以下/以內」→ 超出預算的房屋直接排除
- 性別限制衝突 → 直接排除
- 寵物限制衝突 → 直接排除

---

## Prerequisites

- **Python 3.10+**（用於模型訓練與 ONNX 匯出）
- **pip**（Python 套件管理器）
- **現代瀏覽器**（Chrome / Edge / Firefox，支援 WebAssembly）
- **Git**

### Python 套件依賴

```bash
pip install torch transformers datasets numpy
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/eric20041027/Renting-recommendation-ONNX.git
cd Renting-recommendation-ONNX
```

### 2. 啟動本地開發伺服器

```bash
python3 -m http.server 5002
```

開啟瀏覽器前往 [http://localhost:5002](http://localhost:5002)

### 3. 使用方式

在底部輸入框直接打字描述您的租屋需求，例如：

- `預算六千以下的套房`
- `大里區 獨洗獨曬 有冷氣`
- `騎車五分鐘 8K以內 可養貓`
- `中興大學門口 雅房 台水台電`

系統將即時解析您的需求並推薦最合適的房源。

---

## 模型訓練指南

### 1. 準備訓練資料

訓練資料格式為 JSON 陣列，每筆包含 `text`（字元陣列）和 `tags`（BIO 標籤陣列）：

```json
[
  {
    "text": ["預", "算", "六", "千", "內", "的", "套", "房"],
    "tags": ["O", "O", "B-Target", "I-Target", "I-Target", "O", "B-Target", "I-Target"]
  }
]
```

將訓練資料存為 `train.json`，驗證資料存為 `test.json`。

### 2. 執行訓練與匯出

```bash
python3 train_and_export_onnx.py
```

腳本會自動完成以下流程：
1. 載入 `train.json` 和 `test.json`
2. 使用 `clue/albert_chinese_tiny` 作為基底模型進行 Fine-tuning
3. 每個 Epoch 結束後計算驗證集準確率
4. 訓練完成後將模型匯出為 ONNX 格式

### 3. 部署新模型到前端

```bash
cp my_custom_model.onnx custom_onnx_model_dir/model.onnx
cp my_custom_model.onnx.data custom_onnx_model_dir/model.onnx.data
cp my_trained_albert/tokenizer.json custom_onnx_model_dir/
cp my_trained_albert/tokenizer_config.json custom_onnx_model_dir/
cp my_trained_albert/special_tokens_map.json custom_onnx_model_dir/
```

### 4. 訓練參數調整

在 `train_and_export_onnx.py` 中可調整：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `num_train_epochs` | 3 | 訓練輪數 (資料量大時 3 即可) |
| `learning_rate` | 2e-5 | 學習率 |
| `per_device_train_batch_size` | 8 | 每批次訓練樣本數 |
| `max_length` | 16 | 最大 Token 長度 |
| `test_size` | 0.2 | 驗證集比例 (僅在無 test.json 時生效) |

---

## Available Scripts

| 指令 | 說明 |
|------|------|
| `python3 -m http.server 5002` | 啟動本地開發伺服器 |
| `python3 train_and_export_onnx.py` | 訓練模型並匯出 ONNX |

---

## Deployment

本專案為純前端靜態網站，可部署到任何靜態檔案託管服務：

### GitHub Pages

1. 前往 Repository → Settings → Pages
2. Source 選擇 `main` branch
3. 等待部署完成後即可透過 `https://your-username.github.io/repo-name/` 訪問

> ⚠️ **注意**：模型權重檔案約 16MB，首次載入需要數秒鐘下載。

### Netlify / Vercel

直接連接 GitHub Repository 即可自動部署，無需額外設定。

---

## Troubleshooting

### 模型載入失敗：`Module.MountedFiles` 錯誤

**原因**：ONNX 模型的外部權重檔名不匹配。模型內部硬編碼了 `my_custom_model.onnx.data` 作為權重檔名。

**解決方案**：確保 `inference.js` 中的 `externalData` 正確映射：
```javascript
externalData: [{
    path: 'my_custom_model.onnx.data',  // 模型內部期望的檔名
    data: window.location.origin + '/custom_onnx_model_dir/model.onnx.data'  // 實際檔案位置
}]
```

### Tokenizer 載入失敗：`local_files_only=true` 錯誤

**原因**：`env.localModelPath` 未正確指向模型目錄。

**解決方案**：確認 `inference.js` 中的路徑設定：
```javascript
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = window.location.origin + '/';
```

### 推薦結果不精確

1. **檢查 NER 輸出**：開啟瀏覽器 DevTools Console，查看 `ALBERT Segmented Words:` 日誌
2. **增加訓練資料**：針對識別不佳的句型，生成更多相似的訓練資料
3. **調整評分權重**：修改 `inference.js` 中 `recommend()` 函數的分數分配

---

## License

此專案為中興大學課程專題作品。
