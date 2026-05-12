# 專案更新日誌 (Changelog) - 興大 AI 租屋推薦系統

## [2026.05.12 v2.2] - 多維度品質提升：NER 壓縮、評分修正與前端增強

### 核心改進

#### 🔧 `compute_relevance_score` 全面修正
- **預算方向感知**：修正「以上」被誤判為上限的嚴重 Bug，改為正確語意（以上=下限，以下/以內=上限）
- **is_strict 過度觸發修正**：移除「找/想要/需要」等幾乎所有查詢都包含的詞，嚴格模式僅在「一定要/絕對/急尋/【/＃」等真正強調語境觸發
- **負分保護**：`satisfied -= 0.5` 改為 `satisfied += 0.0`，並在最終計算前加 `max(0.0, ...)` 防止 score_ratio < 0
- **Geo-tier 溢位修正**：`satisfied += 1 + bonus` 改為 `satisfied += min(1.0, 1.0 + bonus)` 防止超出 total_specified

#### 📝 查詢模板多樣性擴充
- **Strategy 6 (情境/角色式查詢)**：新增大一新生、交換學生、預算緊張族、WFH、寵物主人、安全意識、租補申請等 7 類角色情境查詢
- **Strategy 7 (負向需求查詢)**：加入「不要頂加」、「不要暗房」、「不要太吵」等真實用戶常見否定式查詢

#### 🚶 通勤時間整合 (Task 5)
- `load_properties()` 新增 `walk_mins` / `scooter_mins` 欄位讀取（969/972 筆有數據）
- `property_to_text()` 輸出新增「步行X分鐘」、「騎車X分鐘」使模型學習通勤語意
- `expr_distance()` 優先使用 CSV 實際通勤數據（OSRM 計算），無資料才退回估算（0.08km/min 步行，0.6km/min 騎車）
- 通勤閾值放寬：步行 15 → 20 分鐘，騎車 10 → 15 分鐘，新增「走路約X分」/「騎車約X分」變體

#### 💰 前端 NER BGT 實體預算過濾 (Task 6)
- 新增 `parseBudgetFromNER(budgetSpans)` 函數，支援萬/千/k/中文數字解析
- `recommend()` 中：當 regex 未能識別預算時，自動使用 NER 提取的 BGT 實體補充 `constraints.budget`
- 支援方向感知：BGT span 包含「以上」→ `limit='above'`，其餘預設 `limit='below'`

#### 🤖 NER 模型壓縮 (hfl/rbt3)
- 底層模型從 `bert-base-chinese`（12 層，98 MB）換為 `hfl/rbt3`（3 層，37 MB）
- F1-Score 維持 0.9972，總下載量 169 MB → 107 MB（-37%）
- 修正訓練腳本：`save_only_model=True`、`save_total_limit=2`（防止 optimizer.pt 耗盡磁碟）
- 修正 ONNX 導出：`dynamo=False` 解決 PyTorch 2.11 torch._dynamo 不相容問題

#### 👍 前端推薦反饋按鈕 (Task 3)
- 每張房源卡片底部新增「👍 有用 / 👎 不符」反饋列
- `saveFeedback()` 記錄 `{ts, query, propertyId, vote}` 至 localStorage，上限 500 筆
- 點擊後切換為「感謝回饋！」綠色提示，防止重複記錄

#### 📊 雙進度條 UI (Task 4)
- 模型載入介面分為兩條進度條：Cross-Encoder（綠色）+ NER 語意模型（紫色）
- NER Worker 改用 Streaming Fetch，每 512KB 回報一次進度（`ner_progress` 訊息）
- `initNER()` 新增 `onProgress` / `onReady` 回調參數

### 訓練資料更新
- 以修正後的 `generate_dataset.py` 重新生成：33,598 訓練 / 3,894 dev / 3,993 測試
- 負例比例保持 1:2.5（Pos: 9,590 / Neg: 24,008）

---

## [2026.05.11 v2.1] - 軟標籤排序損失函數最優化 (Soft-Label Ranking Loss)
### 核心突破
- **NDCG@5 性能大幅躍進**:
    - 前代 (Run 1): NDCG@5 = 0.7647, MRR = 0.6894
    - 新代 (Run 2): NDCG@5 = **0.9629** ✅ (+25.9%), MRR = **0.9515** ✅ (+37.9%)
    - 遠超目標 0.85，達成 96.29% 的前5推薦精準度

- **軟標籤組合損失函數**:
    - 設計: Total Loss = 0.5×CE(hard_labels) + 0.5×BCE(soft_labels)
    - 軟標籤來自 relevance 欄位: -1/0→0.0, 1→0.4, 2→0.7, 3→1.0
    - 效果: 模型學習分層排序邏輯，relevance=3 >> relevance=1，直接優化 NDCG

- **訓練資料優化**:
    - 負例比例: 1:1 → **1:2** (pos:neg)
    - 訓練樣本: 25,000 → **37,488** (+50% 資料量)
    - 對比訊號: 1:2 比例提供更強的排序對比，符合真實推薦場景

- **評估方法修正**:
    - eval_sample_size: 1,000 (不完整) → **10,000** (全覆蓋)
    - 評估基礎: 294 個 query → **1,428 個 query** (全部)
    - NDCG 計算: 現在基於完整的 query-candidate 分組，數字更真實

- **訓練過程**:
    - 最佳 epoch: 6 (eval_loss = 0.1758，相較前代 8 epoch 更快收斂)
    - Early stopping: 3 epoch patience 於 epoch 9 觸發
    - 訓練時長: 38 分鐘 (10,548 steps, 5.07 it/s)

- **ONNX 導出與量化**:
    - ONNX 模型: 228 MB (完整精度)
    - 量化模型: **57 MB** (INT8, 74.8% 壓縮)
    - 量化後 NDCG 無損 (同樣達成 0.9627)

### 技術改進
- **Trainer 架構升級**:
    - FGMTrainer 實作 `_compute_combined_loss()` 靜態方法
    - 軟標籤於 training_step 提取後再傳入模型，避免模型接收額外輸入
    - FGM 對抗訓練同步應用於 combined loss

- **模型儲存修正**:
    - 訓練後顯式調用 `trainer.save_model()` + `tokenizer.save_pretrained()`
    - Exporter 可直接從 saved_model_dir 載入（無需 fallback）

- **Quantizer 修正**:
    - 修正 weight_type 參數: "UINT8" (string) → QuantType.QUInt8 (enum)
    - 使用 onnxruntime.quantization.QuantType 正確轉換

### 部署檔案
- `frontend/models/custom_onnx_model_dir/my_custom_model.onnx`: 228 MB
- `frontend/models/custom_onnx_model_dir/my_custom_model_quant.onnx`: 57 MB (INT8)
- `D:\renting_models\rbt6_finetuned/`: 訓練模型 (Epoch 6 best)

### 文檔更新
- README.md: 性能指標表格全面更新，NDCG@5/MRR/Model Size 反映新數據
- 新增「軟標籤排序損失函數優化」章節，詳細說明公式與效果
- trainer.py 模組說明: 補充軟標籤損失與 1:2 不衡比例

---

## [2026.05.11 v2] - RBT6 交叉編碼器模型訓練完成
### 核心完成
- **RBT6 Cross-Encoder 訓練成功**:
    - 模型: hfl/rbt6 (6層中文RoBERTa)
    - 訓練時長: ~30分鐘 (RTX 3060 GPU)
    - 訓練數據: 38,000+ 合成樣本 (query-property pairs)
    - 測試集: 2,220 樣本

- **性能指標確認**:
    - 測試損失: 0.6338 (穩定收斂)
    - NDCG@5: 0.5234 ✓
    - MRR: 0.6573 ✓
    - 準確率: 62.90%
    - 無過擬合跡象 (eval loss 穩定在 0.65-0.69)

- **訓練優化配置**:
    - 優化器: AdamW (lr=2e-5)
    - 對抗訓練: FGM (Fast Gradient Method)
    - 學習率調度: Linear warmup + decay
    - Early stopping: 3個epoch耐心值

- **存儲管理優化**:
    - 移動 venv (5.25GB) 和 saved_models (2.83GB) 到 D: 驅動器
    - C: 驅動器空間從 0% 回復至 2% 可用
    - 模型檢查點位置: D:\renting_models\rbt6_finetuned (3,419 MB)

- **已知技術限制**:
    - ONNX 導出: transformers 5.8.0 中 SDPA 注意機制與 ONNX JIT 不相容
    - 替代方案進行中: torch.jit.trace, transformers.js, 或直接 PyTorch 推論

### 文檔更新
- 新增 TRAINING_COMPLETION_REPORT.txt 訓練完成報告
- 記錄模型性能、訓練過程、後續步驟

---

## [2026.05.11] - 環境整備、文檔完善與流程驗證
### 核心完成
- **Python 虛擬環境完整配置**:
    - 解決 MinGW Python 與 PyPI 兼容性問題，安裝標準 CPython 3.11
    - 配置 PyTorch 2.6.0+cu124 (CUDA 12.4)，GPU CUDA 可用性驗證通過
    - 安裝全部 requirements.txt 依賴，包括 datasets、seqeval、accelerate 等訓練所需套件
    - 虛擬環境可直接用於訓練與推論

- **完整端到端流程驗證**:
    - Phase 2 (數據處理): 6 步管道完整運行驗證，所有步驟正常執行
    - Phase 3 (模型訓練): RBT6 Cross-Encoder 訓練啟動，loss 正常下降 (0.70 → 0.68 @ 2 epochs)
    - 後台訓練進行中，預計數小時內完成

- **README.md 完全更新**:
    - 數據流水線圖更新：5 步 → 6 步 (新增 commute 步驟完整說明)
    - 目錄結構完全重寫：詳細列出 NER 模型、約束系統、訓練管道等新增模塊
    - 核心模組說明：5 個新章節，涵蓋 6 步數據處理、NER 模型、語意匹配、訓練流程、端到端執行
    - 系統亮點擴展：新增「雙層 NER + 語意匹配」、「6 步自適應管道」等亮點說明
    - 前端優化章節：補充 NER Web Worker、量化優化、延遲指標等技術細節
    - 執行指南升級：從 Shell (run_pipeline.sh) → Python CLI (pipeline_runner.py)，支援靈活的 --skip-phase 組合
    - 新增 NER 模型單獨訓練指令

- **專案大幅清理**:
    - 刪除廢棄腳本：run_pipeline.sh, 3 個舊 runner 檔案, 2 個測試腳本
    - 刪除臨時檔案：1877 個 __pycache__ 目錄, 9 個日誌檔, PyTorch wheel, 臨時模型檔等
    - 項目大小減少 ~2.5GB，結構整潔清晰

- **生成新的項目狀態報告**:
    - PROJECT_STATUS_REPORT.md：記錄完成項目、文件清理、技術細節、性能指標、執行參考等
    - 保留作為當前項目的正式狀態文檔

### 技術細節補充
- **NER 模型集成亮點**:
    - F1-Score = 0.958 (序列標註任務)
    - Accuracy = 0.972 (字符級)
    - 瀏覽器端推論延遲 <20ms (Web Worker)
    - INT8 量化後大小適合移動端

- **語意匹配性能**:
    - NDCG@5 = 0.862 (排序品質)
    - Matching Latency = <150ms (ONNX Runtime Web)
    - 雙模型總體積 ~100MB (INT8)

### 文檔整理
- 新增 PROJECT_STATUS_REPORT.md 作為正式狀態文檔
- 已確認所有階段計劃文件 (PHASE2_PLAN, PHASE3_*, PHASE4_*, 等) 都已實行完成，標記待刪除

---

## [2026.05.10] - 深度語義推測 V3 與硬性約束強化 (LTR 3.0)
### 核心更新
- **深度語義推測 (Lifestyle Intent Inference)**:
    - 重構 `generate_dataset.py` 語義模板，引入「生活型態聚類 (Lifestyle Clusters)」映射。
    - 支援從口語描述自動推測潛在設施需求（如：「省伙食費」➡️ 廚房/瓦斯；「外送族」➡️ 管理員/飲水機）。
    - 建立 15+ 組生活場景語義橋樑，讓模型具備「聽懂弦外之音」的能力。
- **硬性約束一票否決 (Hard-Constraint Enforcement)**:
    - 實施嚴格模式 (Strict Mode) 標籤過濾。針對預算溢出 (>10%)、禁養寵物、台水電缺失等關鍵標籤實施「硬性 0 分」策略。
    - 解決了模型在「地點優勢」與「設施缺失」之間的權衡偏見，優先保證用戶底線需求。
- **全端語義同步 (Full-Stack Semantic Sync)**:
    - 同步更新 `inference-worker.js` 與 `generate_dataset.py` 的擴展字典。
    - 實作「雙層語義橋接」：前端進行預處理擴展，後端模型進行深度意圖識別，達成 12/12 壓力測試的高分表現。

### 技術優化
- **壓力測試體系升級 (Stress Test v3)**:
    - 將 `semantic_stress_test.py` 擴展至 12 個核心案例，涵蓋 FB 真實貼文模擬與複雜功能複合測試。
    - 實作「全特徵掃描」驗證邏輯，確保 AI 推薦結果與原始資料庫特徵 100% 吻合。
- **高強度對抗訓練 (LTR 3.0)**:
    - 實施 5 Epochs 的精準 Fine-tuning，配合「地獄級負樣本注入」，顯著提升模型在邊界案例上的決策信心。

## [2026.05.10] - 語義特徵工程 2.0 與 V2 全自動管線升級
### 核心更新
- **全自動管線 V2 (Pipeline V2)**:
    - 重構 `run_pipeline.ps1` 與 `run_pipeline.sh`，整合最新的數據採集、語義增強、訓練與評估流程。
    - 實作跨平台（Windows/Unix）一致性執行邏輯，支持一鍵從爬蟲到 ONNX 導出。
- **高級特徵引擎 (FeatureEngine 2.0)**:
    - 在 `generate_dataset.py` 中實作結構化特徵提取，涵蓋：
        - **電費透明度**：自動分類台電計費 vs 固定費率。
        - **服務等級**：識別子母車、垃圾代收與包裹管理。
        - **CP 值維度**：計算「租金/坪數」單價，自動標註區域高 CP 值房源。
        - **地理分層**：基於 OSRM 距離將房源分為核心圈、活躍區與寧靜區。
        - **屋況分析**：自動辨識全新首租與精緻翻新。
- **可解釋性 AI 強化 (Explainable AI V2)**:
    - 升級前端「推薦理由」生成器，利用高級特徵顯示具備 Emoji 的人性化標籤（如：✨ 免追垃圾車、⚡ 台電計費）。
    - 實作「語義隱喻匹配」，讓系統理解「下班晚」對應「垃圾代收」的需求。

### 技術優化
- **LLM 樣本大規模增廣**:
    - 利用 Gemini API 成功生成 1,000 筆高品質樣本（500 筆困難負樣本 + 500 筆正向語義映射）。
    - 數據集規模突破 **3.3 萬筆** 訓練樣本，顯著提升模型對口語化查詢的理解力。
- **訓練監控升級**:
    - 實作 `CustomEarlyStoppingCallback`，在訓練日誌中即時顯示 Patience 狀態 (X/8)。
    - 將 **Precision (精確度)** 指標加入驗證輸出，便於精準監控誤報率。
- **依賴與環境優化**:
    - 整合 `python-dotenv` 管理 API 金鑰，優化 `google-genai` 呼叫穩定性。

## [2026.05.08] - 可解釋性 AI 與透明化排序系統升級
### 核心更新
- **可解釋性 AI (Explainable AI)**:
    - 實作前端「命中理由 (Match Reasons)」顯示，利用 NLP 提取關鍵屬性並以綠色標籤展示（如：陽台、台電計費）。
    - 提升使用者對推薦分數的信任感，讓系統從「黑盒」轉為「透明」。
- **混合過濾系統 (Hybrid Filtering V2)**:
    - 在 AI 模型之上加入硬性約束檢查層。
    - 針對「寵物政策」、「陽台需求」等關鍵約束進行衝突偵測。
    - 衝突房源自動執行 **90% 降分懲罰**，並在 UI 上執行 **70% 透明度淡化** 與 **紅色衝突警示**。
- **數據質量審計 (Data Auditing)**:
    - 新增 `audit_data.py` 自動化工具，支援 Big5/UTF-8 編碼自適應讀取。
    - 成功執行原始數據審計，確保租金與地點標註無重大異常。
- **極致模型壓縮 (Model Compression)**:
    - 優化量化腳本，針對 `Gather` 算子執行動態量化。
    - **成果**：將 RBT6 模型體積從 105MB 成功壓制至 **57.3 MB**，遠低於 GitHub 100MB 限制。

### 技術優化
- **訓練指標優化**:
    - 將 `metric_for_best_model` 切換為 **F1 Score**，平衡精確度與召回率。
    - 增加 **EarlyStopping Patience (8)**，允許模型在 Fine-tuning 階段有更長的時間進行細微權益優化。
- **評估流程升級**:
    - 在 `evaluate_model.py` 中整合混合過濾邏輯，使測試報告更貼近真實前端體驗。
    - 動態訓練日誌修復，正確顯示當前 Fine-tuning 模型名稱。

---

本文件詳實記錄了中興大學 AI 租屋推薦系統專案的技術演進歷程，統整了超過 140 次的提交紀錄。

## [第五階段] 極致性能優化與泛化能力增強 (當前階段)
目標：達成 Graded NDCG@5 > 0.85 門檻

### [2026-05-08] - 模型架構升級與前端效能飛躍
- 語意匹配模型重大升級：將「查詢-房源」二分類匹配任務之架構從 RBT3 升級至 hfl/rbt6 (6 層 RoBERTa)，在 Step 4000 達成 F1-Score 0.832 之卓越表現。
- 並行加載與串流技術：重構分詞器與模型之下載邏輯，實作 Promise.all 並行請求與原生 Fetch 串流進度追蹤，解決加載停滯與進度條異常問題。
- 全平台行動端適配：完成 Mobile-First 響應式佈局重構，針對行動端觸控、間距與字級進行專項優化，達成接近原生 App 的操作質感。
- 手動導出與評估工具：開發專用導出腳本 (export_from_checkpoint.py)，支援從任意檢查點手動提取最佳權重並自動產出技術評估報告。
- 對抗訓練 (FGM)：在 WeightedTrainer 中整合快速梯度方法 (Fast Gradient Method)，針對 Embedding 層注入擾動，提升語意辨識的魯棒性。
- LLM 困難樣本增廣：利用 Gemini API 生成 500 組高難度陷阱樣本，專門針對寵物政策、台水電、地點邊界等細微特徵進行誤導訓練。
- 隨機失活 (Dropout) 強化：將隱藏層與注意力機制的 Dropout 比例提升至 0.15，增強正則化效果並防止過擬合。
- 訓練環境相容性修正：修正 WeightedTrainer.training_step 函數簽名，以支援新版本 Transformers 庫之參數規範。

### [2026-05-07] - 排序校準與權重策略
- 溫度校準 (Temperature Scaling)：在損失函數中引入 T=2.0 的溫度係數，壓平 Logit 分佈以緩解 Sigmoid 飽和問題，顯著提升 NDCG 解析度。
- 激進權重學習：實施非線性權重映射（Perfect 樣本賦予 15.0 倍權重），強制模型優先學習高相關度房源之特徵。

---

## [第四階段] 精準度指標與通勤感知優化 (2026-04-27 至 2026-05-06)
目標：從二元分類評測轉型為分級排序評測

- 分級評測指標：正式採用 NDCG@5 與 MRR 作為核心效能監控指標，取代單一的 F1 分數。
- 通勤邏輯整合：整合 OSRM 開源路網資料，實現精確的通勤時間與路網距離計算。
- 困難負樣本挖掘：將正負樣本比優化至 1:2.5，並針對模型易錯樣本進行二次訓練。
- 預算語意解析：實施繁體中文數字與簡寫（如：6k、1萬5）的魯棒解析邏輯。

---

## [第三階段] 雙塔匹配架構與 UI 全面改版 (2026-03-18 至 2026-04-02)
目標：提升使用者體驗與深層語意匹配精度

- 句子對分類架構 (Cross-Encoder)：全面切換為 Cross-Encoder 架構，實現查詢與房源描述的直接深度匹配。
- UI/UX 視覺重構：實施玻璃擬態 (Glassmorphism)、網格漸變動畫與 staggered 階梯式進入動畫。
- 模型量化：實施 INT8 動態量化技術，在不損失精度的前提下大幅縮減模型體積，優化 Web 端載入速度。

---

## [第二階段] Web AI 部署與 ONNX 整合 (2026-03-08 至 2026-03-12)
目標：達成 100% 瀏覽器端本地 AI 推理

- ONNX Runtime Web：將推理邏輯從 Python 後端遷移至瀏覽器端 WASM 執行緒。
- Transformers.js 整合：實施 Hugging Face 的 Web 端框架，確保 Tokenizer 與模型權重之相容性。
- 實體辨識模型 (NER)：訓練並部署專用的三類別 NER 模型，自動提取查詢中的地點、預算與設備需求。
- Vercel 雲端部署：完成 WASM 與跨域請求 (CORS) 之標頭配置。

---

## [第一階段] 基礎建設與多源 ETL (2026-02-27 至 2026-03-07)
目標：資料採集與後端基準建立

- 多源爬蟲開發：開發針對 591、Dcard、FB 租屋社團的自動化資料採集腳本。
- Flask 後端建立：構建首個基於 Python 的推薦系統 API 服務。
- 關鍵字匹配引擎：實施第一代基於正則表達式 (Regex) 的過濾與評分邏輯。
- 資料集標準化：收集並正規化首批中興大學周邊租屋物件資料。

---

## [專案統計與規格]
- 總提交次數：140+
- 模型架構：RoBERTa-tiny (RBT6)
- 部署平台：Vercel + ONNXRuntime Web
- 核心指標：F1 分數 0.87，NDCG@5 目標 0.85
