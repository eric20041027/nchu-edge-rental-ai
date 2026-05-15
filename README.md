# 興大 AI 租屋推薦系統 (NCHU AI Rental Recommendation)

本專案為針對中興大學學生設計之 **Edge AI 租屋推薦系統**。透過微調並蒸餾的中文 RoBERTa 模型（rbt3 INT8，**38.6 MB**）在瀏覽器端進行即時語意匹配，解決傳統篩選器過於僵硬的侷限，提供具備深層語意理解的搜尋體驗。

---

## 系統核心亮點

- **超輕量 Edge AI**：rbt3 Cross-Encoder（38.6 MB INT8），完全在瀏覽器端執行（ONNX Runtime Web + WASM），無需後端伺服器
- **知識蒸餾**：rbt6 teacher → rbt3 student，NDCG@5 = **0.833 ± 0.014**，超越所有歷史版本
- **雙層語意理解**：NER 抽取地點/預算/設施（F1=0.997）→ Cross-Encoder 深度重排
- **硬性約束零容忍**：預算上限、寵物政策、台電計費一票否決，不被語意優勢覆蓋
- **真實路網通勤時間**：OSRM 計算步行/機車實際路網時間作為排序核心因子
- **生活型態推論**：15+ 語意聚類，「不想追垃圾車」→ 子母車設施，「自炊族」→ 瓦斯廚房

---

## 效能指標 (Model Performance)

### 1. NER 實體辨識

| 指標 | 數值 | 說明 |
|:---|:---|:---|
| **F1-Score** | **0.997** | LOC / BGT / FEAT 三類實體聯合 F1 |
| **延遲** | **< 20ms** | 瀏覽器端 Web Worker 推論延遲 |
| **大小** | **37 MB** (INT8) | bert-base-chinese 98 MB → 37 MB（−62%）|

### 2. Cross-Encoder 語意匹配（v2.9，INT8 量化）

#### Phase 1：單對分類正確率

**測試目標**：給定一個（查詢, 房源）配對，模型能否正確判斷「相關 / 不相關」？以 n=5,000 測試樣本評估模型的二元分類能力，並以三個閾值觀察精準-召回取捨。

**閾值的意義**：模型輸出一個 0–1 的相關性機率分數，閾值決定「超過多少才算 Match」：

| 閾值 | 適用場景 | Accuracy | Precision | Recall | F1 |
|:---|:---|:---|:---|:---|:---|
| 0.5 | 初篩（不遺漏好房源）| 87.8% | 71.2% | **98.0%** | 82.5% |
| **0.7** | **排序引擎實際使用** ✅ | **88.0%** | **82.7%** | 75.0% | **78.7%** |
| 0.9 | 高信心過濾（極嚴格）| 85.6% | 94.5% | 54.3% | 68.9% |

- **0.5**：幾乎不遺漏任何好房源（Recall 98%），適合作為「寧可多選也不遺漏」的粗篩
- **0.7**：精準與召回的平衡點，是 Top-30 重排的實際運作閾值
- **0.9**：只輸出極高信心的結果，可用於推播通知等高精準場景

#### Phase 2：Top-30 重排品質

**測試目標**：給定一個查詢，從 30 個候選房源中重排，模型能否把最相關的放在最前面？以 500 個查詢模擬真實推薦場景，評估 Top-5 排名品質。

| 指標 | **v2.9** | v2.3（舊紀錄）| 說明 |
|:---|:---|:---|:---|
| **Graded NDCG@5** | **0.833 ± 0.014** ✅ | 0.818 | 4 級相關性（0-3）指數增益 NDCG，Bootstrap CI（n=1000）|

**NDCG@5 = 0.833** 代表：在 30 個候選房源中，Top-5 的排列順序與理想排序的相似度為 83.3%。分母採指數增益（$2^{rel} - 1$），使 Perfect match（rel=3）的排名效益是 Partial（rel=1）的 7 倍。

$$NDCG_k = \frac{DCG_k}{IDCG_k}, \quad DCG_k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+2)}$$

**候選池標籤分佈（Top-30 pool）**：Perfect(3)=45.4%、Good(2)=20.4%、Partial(1)=13.0%、None(0)=21.2%

### 3. 模型版本演進

| 版本 | 量化大小 | Teacher F1 | Student F1 | NDCG@5 |
|:---|:---|:---|:---|:---|
| rbt6 FT (v2.2) | 57 MB | — | 84.8% | — |
| rbt3 KD v1 (v2.3) | 37 MB | 84.8% | 85.1% | 0.818 |
| rbt3 R-Drop (v2.4) | 37 MB | — | 76.9% | 0.727 |
| rbt3 KD v2 (v2.5) | 36.8 MB | 78.7% | 76.4% | 0.760 |
| **rbt3 KD v3 (v2.9)** | **38.6 MB** | **85.9%** | **85.5%** | **0.833** ✅ |

v2.4–v2.8 退步的根本原因：負樣本採樣 bug（見[知識蒸餾架構](#知識蒸餾架構knowledge-distillation)）。

---

## 系統架構圖

### 1. 數據流水線

```mermaid
graph TD
    A1["租租通 Crawler\n(Playwright/JSON-LD)"] --> B("merge\n多源資料合併")
    A2["興大官網 Crawler\n(Request/HTML)"] --> B
    B --> C("commute\nOSRM 路網時間計算")
    C --> D("generate\n樣本合成 + 多級相關性標記 0-3")
    D --> E("augment\nLLM 語意增強")
    E --> F("mine\n困難樣本挖掘")
    F --> G1("train_teacher.py\nrbt6 teacher 訓練")
    G1 --> G2("train_and_export_onnx.py\nrbt6→rbt3 蒸餾 + ONNX + INT8")
    G2 --> H["my_custom_model_quant.onnx\n(INT8, 38.6 MB)"]
    C --> J["property_data.json\n(前端房源庫)"]
```

### 2. 推論流程

```mermaid
graph TD
    A["自然語言查詢"] --> B("Stage 1: NER 抽取\nLOC / BGT / FEAT")
    B --> C("Stage 2: 粗篩\nJS Filter + 硬約束 + OSRM 距離")
    C -- "Top 30 候選" --> D("Stage 3: AI 語意重排\nONNX Runtime Web")
    D --> E("rbt3 Cross-Encoder")
    E --> F["最終推薦清單"]
```

---

## 知識蒸餾架構（Knowledge Distillation）

### 為什麼使用蒸餾？

直接訓練 rbt3（3 層，38.6 MB）排序上限約 NDCG@5 ≈ 0.72–0.75。由 rbt6（6 層）作 teacher 教導 rbt3，可讓小模型學到超越其容量限制的排序知識。

### 兩階段訓練

```
階段一：train_teacher.py  — rbt6 teacher
  資料  : 33,598 訓練樣本
  損失  : CE(ls=0.05) + RankNet×1.5 + ListNet + R-Drop + FGM
  存檔  : metric_for_best_model = "loss"（排序損失在 F1 峰值後仍持續下降）
  結果  : F1=0.859，Prec=0.768

階段二：train_and_export_onnx.py  — rbt3 student
  資料  : 同一份訓練資料 + 凍結的 rbt6 teacher
  損失  : (1-α)·L_task + α·T²·KL + R-Drop + FGM
  存檔  : metric_for_best_model = "f1"
  結果  : F1=0.855，NDCG@5=0.833
  導出  : FP32 → INT8（38.6 MB）→ 同步至 frontend/
```

### 蒸餾損失

$$\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\text{task}} + \alpha \cdot T^2 \cdot D_{\mathrm{KL}}\!\left(\sigma\!\left(\frac{z_s}{T}\right) \,\middle\|\, \sigma\!\left(\frac{z_t}{T}\right)\right) + \alpha_{\text{rdrop}}\,\mathcal{L}_{\text{R-Drop}}$$

- $z_s$：student logits，$z_t$：teacher logits（凍結，純推論）
- $T=4.0$：蒸餾溫度。原始 logits 差值 ≈ 6.4 → softmax ≈ [0.002, 0.998]（資訊量趨零）；縮放後差值 ≈ 1.6 → softmax ≈ [0.17, 0.83]（類別間序資訊可傳遞）
- $T^2$ 係數：抵消溫度縮放對梯度幅度的影響，確保 KL loss 與 task loss 在相同數量級

### 動態蒸餾權重 α（餘弦退火）

$$\alpha(t) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \frac{1 + \cos\!\left(\dfrac{\pi\,t}{T_{\text{epoch}}}\right)}{2}$$

參數：$\alpha_{\min}=0.12$，$\alpha_{\max}=0.38$，$T_{\text{epoch}}=10$

| 訓練階段 | $\alpha$ | 效果 |
|:---|:---|:---|
| 初期（$t \to 0$）| 0.38 | teacher 主導，防止 student 初期崩塌 |
| 末期（$t \to T$）| 0.12 | task loss 主導，student 收斂至任務最優點 |

---

## 訓練策略

### 損失函數組合

**Teacher（train_teacher.py）**

$$\mathcal{L}_{\text{teacher}} = \mathcal{L}_{\text{CE}} + 1.5\,\mathcal{L}_{\text{RankNet}} + \mathcal{L}_{\text{ListNet}} + \alpha_{\text{rdrop}}\,\mathcal{L}_{\text{R-Drop}}$$

**Student（train_and_export_onnx.py）**

$$\mathcal{L}_{\text{student}} = (1-\alpha)\underbrace{\left(\mathcal{L}_{\text{CE}} + 1.5\,\mathcal{L}_{\text{RankNet}} + \mathcal{L}_{\text{ListNet}}\right)}_{\mathcal{L}_{\text{task}}} + \alpha\,T^2\,D_{\mathrm{KL}} + \alpha_{\text{rdrop}}\,\mathcal{L}_{\text{R-Drop}}$$

$\mathcal{L}_{\text{CE}}$：label smoothing $\varepsilon=0.05$；$\alpha_{\text{rdrop}}=0.05$

### RankNet 排序損失

$$\mathcal{L}_{\text{RankNet}} = \frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}} \log\!\left(1 + e^{-(s_i - s_j)}\right), \quad s_k = \frac{z_k^{(1)}}{T_{\text{task}}}$$

$\mathcal{P} = \{(i,j) \mid r_i > r_j\}$，$T_{\text{task}}=2.0$

**為何需要 $T_{\text{task}}$**：若 $s_i - s_j \approx 6.0$，則 $e^{-6} \approx 0.0025$，梯度趨近於零（梯度消失）；$T_{\text{task}}=2.0$ 將差值縮至 3.0，使 $e^{-3} \approx 0.050$，維持有效梯度。

### ListNet 列表損失

$$\mathcal{L}_{\text{ListNet}} = -\sum_{i} P_i^* \log P_i, \quad P_i = \text{softmax}\!\left(\frac{s}{T_{\text{task}}}\right)_{\!i}, \quad P_i^* = \text{softmax}(r)_i$$

### 關鍵設計一覽

| 技術 | 說明 |
|:---|:---|
| **FGM 對抗訓練** | embedding 注入梯度方向擾動，提升口語輸入泛化性 |
| **R-Drop（$\alpha=0.05$）** | 雙前向強制 Dropout 一致性，減少預測方差 |
| **metric = "loss"（teacher）** | 多任務損失在 F1 收斂後仍持續下降，loss metric 捕捉更好的排序校準點 |
| **隨機負樣本採樣** | `random.sample(neg_all, n)` 自然混合 ~69% rel=0（硬衝突）+ ~31% rel=-1（輕度不符），維持軟邊界學習信號 |
| **物件級切割** | Train/Dev/Test 按房源分割，測試集房源訓練期間完全未見 |

### 負樣本採樣策略（v2.4–v2.8 的 bug 根源）

| 策略 | rel=0（硬衝突）| rel=−1（輕度不符）| Teacher Prec | NDCG@5 |
|:---|:---|:---|:---|:---|
| Stratified hard-first（v2.4~v2.8 bug）| 100% | 0% | ~0.65 | ~0.76 |
| **Random mix（v2.9）** | **~69%** | **~31%** | **0.768** | **0.833** |

**根本原因**：`neg_hard`（rel=0，約 17,611 筆）> `target_neg`（約 9,590 筆），分層採樣 100% 取 rel=0，rel=−1 完全被排除。模型失去軟邊界學習信號，信心校準惡化。

---

## 資料工程核心

### 1. 物件級切割（防資料洩漏）

先按房源切割 Train/Dev/Test，再從每個房源合成查詢。測試集的房源在訓練期間**完全未見**。

### 2. 多級相關性標記（0–3）

每對（查詢, 房源）由 `compute_relevance_score()` 自動計算 0–3 分。

#### Part A：硬性衝突（直接回傳 0）

| 衝突類型 | 判斷邏輯 |
|:---|:---|
| **性別限制** | 限女 ✕ 查詢找男生（反之亦然）|
| **房型不符** | 查詢要套房但物件為雅房（反之亦然）|
| **明確排除** | 查詢含「謝絕/禁/❌」+ 頂加/漏水/壁癌 |

#### Part B：9 個評分維度（各 0–1，加總後計算比例）

| # | 維度 | 評分邏輯 |
|:---|:---|:---|
| 1 | **預算** | 超 10% → 硬衝突；超 1–10% → 軟扣 0.3 |
| 2 | **家具設施** | 符合需求項目數 / 總需求項目數 |
| 2.5 | **生活型態意圖** | 懶人/自炊/潔癖等對應設施組合命中率 |
| 3 | **地點** | 地區或路名命中；核心地段（< 0.5km）額外 +0.15 |
| 4 | **寵物** | 明確可養 +1；明確禁養 → 0；未提及 +0.2 |
| 5 | **垃圾/管理服務** | 子母車+代收包裹 +1；無 +0.1 |
| 6 | **電費計費** | 台電/台水計費 +1；其他 +0 |
| 7 | **開伙** | 有廚房/瓦斯相關設施 +1；無 +0 |
| 8 | **安全設施** | 有保全/門禁/監視器 +1 |
| 9 | **屋況外觀** | 全新首租 +1；翻新裝潢 +0.8；一般 +0 |

> **`is_strict` 模式**：查詢含「一定要/必須/絕對」等語氣時，任一指定維度 miss 直接回傳 0。

#### Part C：最終分數映射

$$R = \frac{\text{已滿足維度數}}{\text{已指定維度數}}$$

| $R$ | 分數 | 名稱 | 代表案例 |
|:---|:---|:---|:---|
| $\geq 0.85$ | **3** | Perfect | 指定南區 6000 套房，命中 5500 南區套房含冷氣洗衣機 |
| $\geq 0.65$ | **2** | Good | 指定 6000，命中 6400 同地區同格局（預算軟超 7%）|
| $\geq 0.15$ | **1** | Partial | 指定有陽台南區，命中南區無陽台（地點對但設施不全）|
| $< 0.15$ | **0** | Conflict | 想養貓，房源標注禁養寵物 |

> 若查詢不含任何可驗證條件（如「幫我找個房子」），預設回傳 **2**。

### 3. 查詢多樣化（7 類策略）

| 類型 | 說明 |
|:---|:---|
| S1–S4 | 單特徵 / 雙組合 / 三組合 / 多約束原始描述 |
| S5 | 生活型態推論（懶人系→電梯、自炊族→瓦斯、寵物主→可養貓…）|
| S6 | 角色情境（大一新生、WFH、安全意識、租補申請…）|
| S7 | 負向需求（不要頂加、不要暗房、不要太吵…）|
| 噪音 | 錯字、簡寫（興大 vs 中興大學）、網路用語（滴 vs 的）|

### 4. 困難樣本挖掘

基於 Jaccard 字符重疊，找出「表面相似卻違反硬約束」的語意陷阱（禁養寵物、性別限制）作為 hard negatives，double weight 強化學習。

---

## 前端工程優化

1. **雙 Web Worker 並行推論**：NER + Cross-Encoder 各有獨立 Worker，主線程零阻塞
2. **Cache API + Service Worker**：`.onnx` cache-first；HTML/JS stale-while-revalidate；版本號 `v20260515` 控制快取失效
3. **串流進度追蹤**：Fetch API 監控資料流，精確顯示兩個模型各自的百分比進度
4. **NER BGT 預算過濾**：解析萬/千/k/中文數字，支援方向感知（「以上」= 下限，「以內」= 上限）
5. **推薦反饋**：每張卡片附 👍/👎，記錄至 localStorage（最多 500 筆）

---

## 目錄結構

```text
.
├── data/
│   ├── raw/                 # 原始爬取數據
│   └── processed/           # 訓練集 / 驗證集 / 測試集 / 前端房源 JSON
├── frontend/
│   ├── index.html
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
│   │   ├── training_utils.py         # 共用工具（FGM、metrics、callbacks）
│   │   ├── evaluate_model.py         # 多指標評估
│   │   └── quantize_model.py         # 獨立量化腳本
│   ├── ner_model/
│   └── constraints/         # 硬約束邏輯
├── saved_models/
│   ├── rbt6_teacher/        # Teacher checkpoint（永不被 student 覆蓋）
│   └── rbt3_finetuned/      # Student checkpoint
└── pipeline_runner.py       # 端到端入口點
```

---

## 執行與部署

### 環境建置

```bash
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
playwright install chromium
```

### 兩階段蒸餾訓練

```bash
set PYTHONUTF8=1
# 第一步：訓練 rbt6 teacher
python -m pipeline.model_training.train_teacher

# 第二步：蒸餾至 rbt3 + ONNX 導出 + INT8 量化
python -m pipeline.model_training.train_and_export_onnx
```

### 模型評估

```bash
set PYTHONUTF8=1
python -m pipeline.model_training.evaluate_model
# 輸出：NDCG@5、Bootstrap CI、Phase 1 分類指標
```

### 本地前端預覽

```bash
cd frontend && python -m http.server 8000
# 開啟 http://localhost:8000
```

---

## 未來展望

- **向量檢索升級**：房源規模擴增至萬筆時引入 ANN 向量索引（FAISS/Annoy）
- **即時地圖互動**：推薦結果直接標註於互動式地圖
- **使用者反饋微調**：利用 localStorage 累積的 👍/👎 反饋進行線上學習

---

*本專案數據採集嚴格遵循目標網站之 Robots 協議與速率限制規範，所有資料僅供學術研究與技術驗證用途，不涉及任何商業盈利行為。*
