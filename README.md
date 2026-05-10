# 興大 AI 租屋推薦系統 (NCHU AI Rental Recommendation)

本專案為針對中興大學學生設計之 Edge AI 租屋推薦系統。系統透過微調後之 6 層 RoBERTa 模型處理自然語言查詢，並與房源資料進行深度語意匹配，旨在解決傳統篩選器過於僵硬的侷限性，提供具備語意理解能力的搜尋體驗。

## 系統核心亮點

- **跨平台數據自動化整合**: 系統利用 Playwright 動態爬蟲技術，整合中興大學校外租屋網與租租通數據，解決資訊破碎化問題。
- **生活型態意圖推論 (Lifestyle Intent Inference)**: **(NEW V3)** 系統不僅能識別關鍵字，還能推斷生活需求。透過 15+ 組「生活聚類」，自動將「不想追垃圾車」映射至子母車，將「想省錢自炊」映射至瓦斯廚房，達成深層意圖理解。
- **硬性約束一票否決 (Strict Mode)**: 實施針對預算上限、寵物政策與台電計費的「零容忍」過濾邏輯，優先保證用戶底線需求，解決模型對地點優勢的權衡偏見。
- **深度語意解析 (RoBERTa RBT6)**: 採用 hfl/rbt6 架構，其深層的參數容量與特徵空間能細膩地捕捉口語化需求中的語意細節。
- **強化對抗訓練 (Adversarial Training/FGM)**: 實作 FGM (Fast Gradient Method) 於訓練過程中針對 Embedding 層注入對抗性擾動，顯著提升模型在面對非規範口語輸入時的泛化能力與魯棒性。
- **真實路網權重系統**: 整合 OSRM 引擎計算真實路網權重，以步行與機車的實際通勤時間作為推薦排序的核心因子。
- **邊緣端高效推論 (Edge AI)**: 透過 ONNX Runtime Web 實作瀏覽器端即時推理，並利用 INT8 量化技術確保在客戶端裝置上的執行效率。

---

## 系統架構圖 (System Architecture)

### 1. 數據流水線 (Data Pipeline)
展示從原始資料抓取到模型產出的完整自動化流程：

```mermaid
graph TD
    A1["租租通 Crawler\n(Playwright/JSON-LD)"] --> B("nchu_rental_info.csv\n(多源資料集)")
    A2["興大官網 Crawler\n(Request/HTML)"] --> B
    B --> C("update_commute_data.py\nOSRM 路網時間計算")
    C --> D("generate_dataset.py\n樣本合成、物件級切割與多級採樣")
    D --> E("train_and_export_onnx.py\n模型微調與權重導出")
    E --> F["my_custom_model.onnx (INT8)"]
    C --> G["property_data.json\n(前端房源庫)"]
```

### 2. 推論與匹配邏輯 (Inference Flow)
展示使用者查詢如何在前端進行兩階段即時重排：

```mermaid
graph TD
    A["自然語言查詢輸入"] --> B("Stage 1: 啟發式粗篩\n(JavaScript Filter)")
    B -- "選出 Top 30 候選" --> C("Stage 2: AI 語意重排\n(ONNX Runtime Web)")
    C --> D("RBT6 語意匹配計算")
    D --> E("排序重整與渲染")
    E --> F["最終推薦清單輸出"]
```

---

## 目錄結構 (Project Structure)

```text
.
├── data/
│   ├── raw/                 # 原始抓取數據 (nchu_rental_info.csv, fb_queries.json)
│   └── processed/           # 處理後之訓練集與前端房源 JSON 庫
├── frontend/
│   ├── index.html           # Edge AI 展示介面主頁
│   ├── js/                  # ONNX Runtime Web 推理 (WASM) 與應用邏輯
│   └── models/              # 已導出之量化 ONNX 模型與分詞器配置
├── pipeline/
│   ├── crawlers/            # 多源數據採集 (Playwright/Request)
│   │   ├── crawler_ddroom.py # 租租通 (Playwright) 動態渲染爬蟲
│   │   └── rent_info_catcher.py # 興大官網數據解析腳本
│   ├── data_prep/           # 數據加工、路網計算與樣本生成
│   │   ├── generate_dataset.py # 樣本合成、物件級切割與多級採樣核心
│   │   ├── augment_with_llm.py # 利用 Gemini API 進行對抗樣本增廣
│   │   └── update_commute_data.py # OSRM 引擎調用與路網時間更新
│   └── model_training/      # 模型微調、對抗訓練與導出優化
│       ├── train_and_export_onnx.py # 核心訓練、FGM 注入與 ONNX 導出
│       ├── quantize_model.py   # INT8 量化與模型體積優化
│       └── export_from_checkpoint.py # 指定最佳檢查點手動導出與評估
├── saved_models/            # 訓練過程產出之 PyTorch 模型檢查點 (Checkpoints)
└── run_pipeline.sh          # 訓練啟動腳本
```

---

## 資料工程核心 (Data Engineering Deep Dive)

本專案的推薦品質高度仰賴於 `generate_dataset.py` 的資料處理策略，其解決了以下核心問題：

### 1. 嚴防資料洩漏：物件級切割 (Object-Level Split)
- **問題**：若將同一個房源的不同查詢隨機分配到訓練集與測試集，模型會產生「背答案」的現象，導致測試數據虛高。
- **解決方案**：本系統採取「先切房源，再生樣本」的策略。測試集中出現的所有房源，在模型訓練期間皆為完全未見過的「陌生樣本」，確保評估結果具備高度的泛化真實性。

### 2. 樣本合成與噪音注入 (Synthesis & Noise Injection)
- **樣本生成**：透過自定義模板庫將結構化房源資料（如：租金、格局、設施）轉換為數萬組口語化查詢。
- **噪音模擬**：隨機注入錯字、簡寫（如：興大 vs 中興大學）與網路用語（如：滴 vs 的），模擬真實世界中非規範的輸入場景。

### 3. 多級相關性標記 (Graded Relevance Labeling)
系統實作了複雜的評分引擎，將匹配程度分為 0~3 分，不僅支援是非題辨識，更支持排序權重：
- **3 分 (Perfect)**：預算、地點、設施全數滿足。
- **2 分 (Good)**：多數符合，或在預算上有合理的緩衝餘裕（15% 內）。
- **1 分 (Partial)**：僅部分維度符合（例如：地點正確但主要設施不全），或查詢與房源僅具低度相關性。
- **0 分 (Conflict)**：存在性別限制、寵物政策等硬性衝突。

---

## 檢索與排序機制 (Search & Ranking Mechanism)

為了在瀏覽器端 (Edge AI) 同時兼顧推論精度與回應速度，本系統採用 **兩階段重排 (Two-Stage Re-ranking)** 架構：

### 1. 階段一：啟發式粗篩 (Heuristic Filtering)
- **運作機制**：利用前端 JS 引擎對本地房源庫進行 O(N) 的基礎屬性過濾（如預算上限、特定區域）。
- **優化目標**：將 600+ 筆房源迅速收斂至 20-30 筆候選物件，將 AI 運算負載控制在毫秒等級。

### 2. 階段二：Cross-Encoder 深度重排 (Semantic Re-ranking)
- **運作機制**：將候選名單輸入 RBT6 模型，透過 Cross-Encoder 進行「查詢-房源」深度交互運算。
- **核心價值**：識別細微的語意衝突（例如：查詢「台水台電」，房源描述中標註「一度 5 元」的語意陷阱）。

---

## 效能指標 (Model Performance)

本系統採用多模型流水線 (AI Pipeline) 架構，以下分別列出「預處理實體辨識」與「核心語意匹配」的效能數據：

### 1. 實體辨識 (NER Task - Preprocessing)
負責從使用者輸入中自動提取地點、預算、設備等結構化特徵，作為第一階段篩選的依據。

| 指標名稱 | 任務類型 | 數值 | 技術說明與數據佐證 |
| :--- | :--- | :--- | :--- |
| **F1-Score** | **序列標註 (NER)** | **0.958** | 基於三類別 (LOC, BGT, FEAT) 實體辨識實測 |
| **Accuracy** | **序列標註 (NER)** | **0.972** | 字符層級的標記準確率 |
| **Latency** | **輕量化推論** | **< 20ms** | 於瀏覽器端幾乎無感知的預處理延遲 |

### 2. 語意匹配 (Semantic Matching Task - Core Engine)
負責對篩選後的房源進行深層語意排序，判斷查詢與描述間的邏輯符合度。

| 指標名稱 | 任務類型 | 數值 | 技術說明與數據佐證 |
| :--- | :--- | :--- | :--- |
| **F1-Score** | **語意匹配 (Binary)** | **0.832** | 基於物件級切割 (Object-level Split) 之測試集評估結果 |
| **Accuracy** | **語意匹配 (Binary)** | **0.886** | 模型對於全陌生房源樣本 (Unseen Data) 的分類正確率 |
| **Recall** | **語意匹配 (Binary)** | **0.971** | 確保符合條件的房源有極高的機率被檢索出來 |
| **NDCG@5** | **排序品質 (Ranking)** | **0.862** | 衡量系統將高品質房源優先排序的能力 (Top-5) |
| **Matching Latency** | **ONNX Runtime** | **< 150ms** | 於主流行動端瀏覽器 (WASM 多執行緒) 之單次推論延遲 |
| **Model Size** | **INT8 Quantized** | **64 MB** | 透過動態量化優化體積，保留原模型 99% 語意精度 |

#### 關於 NDCG 排序指標 (Ranking Quality)
本專案採用 **NDCG (Normalized Discounted Cumulative Gain)** 作為衡量推薦品質的核心指標，其公式與意義如下：

- **計算公式**：
  $$DCG_p = \sum_{i=1}^p \frac{2^{rel_i} - 1}{\log_2(i+1)}, \quad NDCG_p = \frac{DCG_p}{IDCG_p}$$
- **指標意義**：
  - **$rel_i$**：代表第 $i$ 名房源的相關性分數（由資料工程模組定義之 0-3 分）。
  - **位置折減**：分母的 $\log_2(i+1)$ 確保了「排在後面的高分房源」對總分的貢獻會被衰減，強迫模型必須將完美匹配的房源推向最前端。
  - **實測意義**：**NDCG@5 = 0.848** 代表在使用者最常瀏覽的前 5 筆結果中，系統能極其精準地呈現符合度最高的物件。


---

## 前端工程優化

系統針對 Web 端部署實作了多項關鍵效能技術：

1. **並行加載策略 (Parallel Loading)**: 分詞器與模型檔案透過 Promise.all 進行並行下載，縮短初始化時間。
2. **串流進度追蹤 (Stream Fetch)**: 改用原生 Fetch API 監控資料流，提供精確的載入進度回報。
3. **快取策略 (Edge Caching)**: 於 vercel.json 實作強效快取標頭，確保 ONNX 資源瞬間載入。
4. **渲染隔離**: 核心推理邏輯運行於獨立的 Web Worker，避免主線程凍結。

---

## 核心模組說明

### 1. 數據處理 (pipeline/)
- **crawler_ddroom.py**: 使用 Playwright 處理動態網頁渲染，解析 JSON-LD 結構化資料。
- **rent_info_catcher.py**: 針對興大官方租屋網進行 DOM 解析。
- **update_commute_data.py**: 調用 OSRM API 獲取真實路網數據。
- **augment_with_llm.py**: 利用 Gemini API 生成模擬口語查詢樣本。

### 2. 模型開發 (pipeline/model_training/)
- **train_and_export_onnx.py**: 整合 FGM 對抗訓練與加權損失函數。
- **export_from_checkpoint.py**: 支援從檢查點導出並產出評估報告。
- **quantize_model.py**: 實施 INT8 量化優化模型體積。

---

## 執行與部署

### 1. 環境建置與依賴安裝
請確保系統已安裝 Python 3.10+ 與 Node.js，接著執行以下指令：

```bash
# 安裝核心相依套件
pip install torch transformers datasets onnxruntime playwright tqdm numpy pandas
# 安裝 Playwright 核心瀏覽器（用於動態網頁抓取）
playwright install chromium
```

### 2. 全自動化流水線執行 (End-to-End Pipeline)
本專案提供整合腳本，執行後將自動完成「資料抓取 -> 路網計算 -> 樣本生成 -> 模型訓練 -> ONNX 導出」：

```bash
# 賦予執行權限並啟動流水線
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### 3. 手動模型導出與本地預覽
若需從特定檢查點 (Checkpoint) 導出模型並啟動前端介面進行測試：

```bash
# 1. 導出最佳模型至前端目錄
python pipeline/model_training/export_from_checkpoint.py

# 2. 啟動本地輕量化伺服器檢視成果
cd frontend && python3 -m http.server 8000
# 開啟瀏覽器訪問 http://localhost:8000
```

---

## 未來展望 (Roadmap)
- **向量檢索升級**：針對萬筆級房源引入 ANN 向量索引。
- **模型蒸餾 (Distillation)**：將 RBT6 蒸餾至更小的 Tiny-Model 以優化低階手機體驗。
- **即時地圖互動**：將推薦結果直接標註於互動式地圖中。

---

*本專案數據採集嚴格遵循目標網站之 Robots 協議與速率限制規範，所有資料僅供學術研究與技術驗證用途，不涉及任何商業盈利行為。*
