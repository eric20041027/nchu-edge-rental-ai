# 興大 AI 租屋推薦系統 (NCHU AI Rental Recommendation)

本專案為針對中興大學學生設計之 Edge AI 租屋推薦系統。系統透過微調後之 6 層 RoBERTa 模型處理自然語言查詢，並與房源資料進行深度語意匹配，旨在解決傳統篩選器過於僵硬的侷限性。

## 系統核心技術

- **深度語意解析 (RoBERTa RBT6)**: 採用 hfl/rbt6 模型架構，相較於初版 RBT3，具備更深層的特徵提取能力，能精確識別「採光」、「通風」、「安靜」等口語化非結構化需求。
- **全自動數據流水線**: 整合興大校外租屋網與租租通數據，透過地址正規化與租金容差比對演算法，確保房源資料的完整性與去重品質。
- **真實路網權重**: 整合 OSRM (Open Source Routing Machine) 數據，將「真實步行/行車時間」作為推薦排序的核心因子，而非單純的直線距離。
- **邊緣端高效推論**: 透過 ONNX Runtime Web 實作瀏覽器端推理，並透過 INT8 動態量化技術優化模型體積，兼顧隱私與回應速度。
- **行動端專項優化**: 實作 Mobile-First 響應式佈局，針對觸控操作進行優化，提供接近原生 App 的操作體驗。

## 效能指標 (Model Performance)

| 指標 | 任務類型 | 數值 | 狀態 | 說明 |
| :--- | :--- | :--- | :--- | :--- |
| **F1-Score** | **二分類語意匹配** | **0.832** | 優秀 | 基於 RBT6 模型於 Step 4000 達成之最佳表現 |
| **Accuracy** | **二分類語意匹配** | **0.884** | 穩定 | 判斷查詢與房源是否符合的基礎準確度 |
| **Model Architecture** | **Matching Engine** | **RBT6** | 升級 | 從 3 層升級至 6 層，顯著提升複雜語義辨識率 |
| **Inference Latency** | **N/A** | **< 150ms** | 優化 | 基於 WebAssembly 多線程加速技術 |

> [!NOTE]
> 上述 F1-Score 專指 **「判斷使用者查詢與房源描述是否匹配」** 的二分類任務 (Binary Classification Task)。系統中的 NER (實體辨識) 任務屬於預處理階段，由獨立的輕量化模型負責，不計入此表指標中。


## 前端工程優化

系統針對 Web 端部署實作了多項關鍵效能技術：

1. **並行加載策略 (Parallel Loading)**: 分詞器 (Tokenizer) 與模型檔案透過 Promise.all 進行並行下載，將初始化等待時間減少約 40%。
2. **串流進度追蹤 (Stream Fetch)**: 捨棄傳統封裝函數，改用原生 Fetch API 監控資料流，提供精確至 KB 的載入進度回報，提升使用者心理預期。
3. **快取策略 (Edge Caching)**: 於 vercel.json 實作強效快取標頭 (immutable)，確保 ONNX 資源在使用者重複造訪時能瞬間載入。
4. **渲染隔離**: 核心推理邏輯運行於獨立的 Web Worker，避免複雜計算導致 UI 主線程凍結。

## 檢索與排序機制 (Search & Ranking Mechanism)

為了在瀏覽器端 (Edge AI) 同時兼顧推論精度與回應速度，本系統採用 **兩階段重排 (Two-Stage Re-ranking)** 架構：

### 階段一：啟發式粗篩 (Heuristic Filtering)
- **運作機制**：系統首先利用輕量化的 JavaScript 邏輯，對本地快取的 `property_data.json` 進行初步篩選。
- **篩選因子**：包含租金區間、地理區域、基本設備等硬性約束。
- **目標**：將 600+ 筆原始房源過濾至 Top 20-30 筆最具潛力的候選名單，大幅降低後續 AI 運算的負擔。

### 階段二：Cross-Encoder 深度重排 (Semantic Re-ranking)
- **運作機制**：將候選名單與使用者查詢組成「句子對 (Sentence-Pair)」，輸入 **RBT6 語意匹配模型**。
- **優勢**：相較於單純的向量相似度比對 (Cosine Similarity)，Cross-Encoder 能進行深層的語意交互運算，精確識別如「怕吵」、「要採光」等隱性需求與房源描述間的衝突。

## 系統擴展性設計 (Scalability)

針對未來房源數量增長至數千或數萬筆的場景，本系統已預留以下升級路徑：

1. **向量檢索 (Bi-Encoder Integration)**：將第一階段升級為基於向量的近似最近鄰搜尋 (ANN)，利用 Cosine Similarity 進行大規模預選。
2. **混合索引架構**：房源資料將依區域進行分片加載 (Sharding)，僅在使用者感興趣的範圍內下載特徵向量。
3. **模型蒸餾**：透過模型蒸餾技術進一步壓縮 RBT6，以利於在行動端執行更大規模的並行推論。

## 系統架構圖
1. **Data Crawling**: 多源資料抓取與結構化處理。
2. **Commute Analysis**: 路網座標計算與時間標記。
3. **困難負樣本挖掘 (Hard Negative Mining)**：
   - **生成策略**：系統透過自定義演算法與 LLM 輔助，生成「語意陷阱」樣本。這些樣本在字面上與查詢極度相似（High Lexical Overlap），但在核心關鍵約束上完全互斥。
   - **實例定義**：例如查詢要求「台水台電」，系統會刻意配對描述中包含「環境優美、近興大」但註明「電費一度5元」的房源作為負樣本，強迫模型學習區分誘人條件與核心約束間的衝突，而非僅依賴關鍵字出現頻率進行判讀。
4. **Model Tuning**: 基於 RBT6 的遷移學習與 F1 指標監控。
5. **ONNX Export**: 權重轉換與部署包封裝。

### 執行目錄結構
- `frontend/`: 包含所有 Web 端資源（HTML, CSS, JS, Models）。
- `pipeline/`:
  - `crawlers/`: 各平台數據抓取模組。
  - `data_prep/`: 數據清洗、路網計算與樣本生成。
  - `model_training/`: 訓練、評估與導出腳本。
- `saved_models/`: 存儲訓練過程中的檢查點與狀態檔案。

## 執行與部署

### 環境需求
- Python 3.10+
- Node.js 18+ (用於前端開發與部署)
- PyTorch & Transformers (用於模型訓練)

### 本地部署
1. 克隆專案並安裝相依套件。
2. 執行 `pipeline/model_training/train_and_export_onnx.py` 進行訓練。
3. 使用 `pipeline/model_training/export_from_checkpoint.py` 導出至前端目錄。
4. 於 `frontend/` 目錄下啟動本地伺服器即可檢視結果。

## 合規性與聲明
本專案之爬蟲均遵循目標網站之 robots.txt 協定，並實作速率限制機制以降低伺服器負載。所有資料僅用於學術研究與 AI 技術驗證，不涉及任何商業盈利行為。
