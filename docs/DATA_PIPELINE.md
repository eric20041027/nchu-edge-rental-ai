# Data Pipeline — 資料工程核心

## 1. 物件級切割（防資料洩漏）

先按房源切割 Train/Dev/Test，再從每個房源合成查詢。測試集的房源在訓練期間**完全未見**。

## 2. 多級相關性標記（0–3）

每對（查詢, 房源）由 `compute_relevance_score()` 自動計算 0–3 分。

**實際儲存值範圍：−1、0、1、2、3**，其中 −1 為隨機採樣負樣本的 sentinel 值：`is_compatible()=False`（確定有硬衝突），但非 top-Jaccard 挑選，屬於明顯不符的「簡單負樣本」。訓練時與 rel=0 同等視為 `label=0`，僅用於採樣比例監控。

### Part A：硬性衝突（直接回傳 0）

| 衝突類型 | 判斷邏輯 |
|:---|:---|
| **性別限制** | 限女 ✕ 查詢找男生（反之亦然）|
| **房型不符** | 查詢要套房但物件為雅房（反之亦然）|
| **明確排除** | 查詢含「謝絕/禁/❌」+ 頂加/漏水/壁癌 |

### Part B：9 個評分維度（各 0–1，加總後計算比例）

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

### Part C：最終分數映射

$$R = \frac{\text{已滿足維度數}}{\text{已指定維度數}}$$

| R | 分數 | 名稱 | 代表案例 |
|:---|:---|:---|:---|
| >= 0.85 | **3** | Perfect | 指定南區 6000 套房，命中 5500 南區套房含冷氣洗衣機 |
| >= 0.65 | **2** | Good | 指定 6000，命中 6400 同地區同格局（預算軟超 7%）|
| >= 0.15 | **1** | Partial | 指定有陽台南區，命中南區無陽台（地點對但設施不全）|
| < 0.15 | **0** | Conflict | 想養貓，房源標注禁養寵物 |

> 若查詢不含任何可驗證條件（如「幫我找個房子」），預設回傳 **2**。

## 3. 查詢多樣化（7 類策略）

| 類型 | 說明 |
|:---|:---|
| S1–S4 | 單特徵 / 雙組合 / 三組合 / 多約束原始描述 |
| S5 | 生活型態推論（懶人系→電梯、自炊族→瓦斯、寵物主→可養貓…）|
| S6 | 角色情境（大一新生、WFH、安全意識、租補申請…）|
| S7 | 負向需求（不要頂加、不要暗房、不要太吵…）|
| 噪音 | 錯字、簡寫（興大 vs 中興大學）、網路用語（滴 vs 的）|

## 4. 困難樣本挖掘（Hard Negative Mining）

基於 Jaccard 字符重疊，找出「表面相似卻違反硬約束」的語意陷阱（禁養寵物、性別限制）作為 hard negatives，double weight 強化學習。

**實作**：`pipeline/data_prep/hard_negative_miner.py`

- 計算查詢與房源描述的 Jaccard 字符 n-gram 重疊
- 篩選 `jaccard_sim > 0.3` 且 `is_compatible() = False` 的配對
- 這些樣本在訓練時 `sample_weight` 乘以 2.0

## 5. 通勤時間計算（OSRM）

使用 OSRM（Open Source Routing Machine）計算步行和機車至中興大學的實際路網時間，作為房源排序的重要因子。

**流程**：
1. 爬取房源地址
2. 地理編碼（Google Maps Geocoding API）
3. OSRM 查詢步行/騎車路網時間
4. 儲存至 `property_data.json`

**前端應用**：推薦結果顯示「步行 X 分鐘 / 騎車 Y 分鐘」，支援通勤時間過濾。

## 6. 雙來源欄位對齊（租租通 vs 興大）

`property_data.json` 共 704 筆（其中 3 筆為爬蟲空殼：`address` 空 / `rent=0`，前端 `initData` 載入時過濾，有效 **701** 筆），無顯式 `source` 欄，以 `url` 區分：租租通（dd-room）**559 筆**、興大官網（nchu）**145 筆**。兩來源由不同 crawler 抓取不同欄位子集，造成排序系統性偏袒租租通。

### 根因：crawler 解析不完整（非缺資料）

興大 detail 頁有 6 個二級表格，但 `crawler_nchu` 原只解析「家具設施 / 另計費用 / 備註」三個，漏掉現成的「租金包含 / 安全管理 / 消防逃生」。下游 bool 欄全由 `full_text` 子字串衍生，漏抓即全 miss（例 `has_window` 來自「窗」，但「鐵鋁門/窗」就在沒被抓的「安全管理」表裡）。

**修法**：

1. **crawler 補抓 3 漏表** + 新增 `FEATURES_DB` / `derive_nchu_features()`，從各表衍生 canonical 標籤折入特色。實測收斂：興大特色項 avg **1.6→5.32**、`has_window` **0→70%**、`safety_level high` **0→94%**，硬篩 100% 保留。
2. **bool 設施欄三態判定**（`boolFieldState`）：`yes`→命中、可信 `false`→明確無、**崩塌欄 `false`→未知**（回退文字/交 AI）。某來源整欄崩塌（≈0% true）是「來源性偏誤」非真實差異，硬判 false 會誤殺。崩塌欄以 `COLLAPSED_BOOL_FIELDS` 硬編（資料換版需依新統計更新）。
3. **前端同義橋接**（`PROP_SYNONYMS`）：查詢擴展詞對上兩來源不同用詞（可寵→可養貓、廚房→可開伙、禁菸→無菸、台水→`electricity_billing` 結構欄…）。

### 語意擴展層可驗證性審查

105 條口語意圖規則的擴展詞，逐一以**前端真實比對邏輯**（`buildPropText` + bool 欄 + `PROP_SYNONYMS` + `electricity_billing`）對 704 房源比對命中數。**0 命中 = 無資料支撐的模型臆測**，分類處理：救援可橋接者、刪除真死 token 與整條失效 rule（規則數 132→105、unique token 122→75、0-backing 60→0）。

**實作 / 重跑**：`pipeline/data_prep/audit_expansion_tokens.py`

## 7. 房源文字富化（C 組，2026-06-16）

訓練／打分用的房源文字由舊基底（`generate_dataset.py` 的 furniture[:5] + notes 只留含「寵物/限」）
切換為 **`property_to_text_enriched`**：

- **全 notes**（保留完整備註描述）
- **全 furniture**（不再截斷為前 5 項）

讓「採光」「隔音」等須房源描述細節才匹配得到的語意得以保留。富化後文字較長
（約 98 token），訓練 `MAX_LENGTH` 由 64 提高至 128。C 組 A/B 評測見
[ABLATION_STUDY.md](ABLATION_STUDY.md)。

**實作**：

1. `pipeline/data_prep/augment_with_expansion_map.py` — 產生 `property_to_text_enriched`
   富化文字（全 notes + 全 furniture），供訓練與打分使用。
2. `pipeline/data_prep/precompute_ce_text.py` — 將富化文字 **byte-exact** 預算進前端
   `property_data.json` 的 `ce_text` 欄；以 **`address` + `rent` 為鍵**對齊房源（704/704），
   確保前端 Cross-Encoder 推論所用文字與訓練文字完全一致。

---

## 8. 雜訊測試集生成（Group D 評估用）

`pipeline/data_prep/noise_generator.py` 生成 `data/processed/noisy_test.json`：

| 雜訊類型 | 說明 | 範例 |
|:---|:---|:---|
| 縮寫替換 | 中興大學 → 興大，套房 → 套 | `ABBREV_MAP` |
| 錯字注入 | ~10% 字符替換，TYPO_MAP 約束 | 的→滴，找→揾 |
| 口語化 | 前置/後置填充詞，口語替換 | 幫我找…、…這樣的房子 |
| 數字格式 | 5000→五千/5k/五零零零 | `_NAMED_AMOUNTS` |

```bash
python -m pipeline.data_prep.noise_generator
# 輸出：data/processed/noisy_test.json
```
