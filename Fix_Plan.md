# Fix Plan — 專案改進計畫

> 生成日期：2026-05-15  
> 目標：逐一修復全專案審查中發現的 13 項問題，依優先級排序

---

## 優先級分級

| 標記 | 說明 |
|:---|:---|
| 🔴 高 | 確定的 bug，功能已損壞或邏輯錯誤 |
| 🟡 中 | 潛在的邏輯問題，某些條件下才觸發 |
| 🟢 低 | 程式碼品質、維護性問題 |

---

## 修復清單

### Fix-01 🔴 Feedback 事件委派 selector 錯誤（`app.js`）

**問題**  
`document.getElementById('recommendation-list')` 使用 kebab-case，但實際 DOM ID 為 `recommendationList`（camelCase）。導致 listener 永遠綁定在 `null` 上，`addEventListener` 一路到 `if (!list) return` 就退出，所有 👍/👎 按鈕的點擊事件**靜默失效**，feedback 資料從未寫入 localStorage。

**修復位置**：`frontend/js/app.js` L492-493

**做法**  
將 selector 改為與 L17 相同的 `document.getElementById('recommendationList')`，並同步把 `query` 的讀取來源改為實際使用的 `userRequirement` input 元素（原本查的是 `'query-input'` / `'searchInput'`，都不存在）。

---

### Fix-02 🔴 `evaluate_model.py` 重複執行兩次 ranking loop

**問題**  
評估腳本在 L216-248 用相同的 `random.seed(42)` 重新執行一遍完整的 500 queries ONNX 推論，只為了取得 NDCG@1/3/10、Precision@k、Hit@1。這導致：
- 評估總時間**翻倍**
- 兩個 loop 的樣本對齊依賴 `random.seed` 行為，潛在不一致風險
- 中間有一個空的 `for i, query in enumerate(eval_queries): pass` 死碼迴圈（L216-219）

**修復位置**：`pipeline/model_training/evaluate_model.py` L140-248

**做法**  
在第一個 loop 中同步收集 `ranked_rel_all`（per-query graded relevance 向量），完全刪除第二個重複 loop，在第一個 loop 結束後統一計算所有指標。

---

### Fix-03 🔴 兩份相關性計算函數邏輯不一致

**問題**  
`pipeline/data_prep/generate_dataset.py` 是目前實際使用的訓練資料生成器，有完整的 9 個評分維度（預算方向感知、電費計費、生活型態意圖等）。`pipeline/data_prep/generator.py` 的 `_compute_relevance_score` 是大幅簡化版，缺少關鍵維度。若未來遷移到 `generator.py`，訓練資料標注品質會直接退步。

**修復位置**：`pipeline/data_prep/generator.py`

**做法**  
在 `generator.py` 的 `_compute_relevance_score` 加上棄用警告（`DeprecationWarning`），並在函數體內直接 delegate 給 `generate_dataset.py` 中的 `compute_relevance_score`，確保單一真相來源。

---

### Fix-04 🟡 `compute_loss` 中 `inputs.pop` 的副作用

**問題**  
`inputs.pop("sample_weight", None)` 直接修改傳入的 `inputs` dict。`training_step` 對同一份 `inputs` 呼叫兩次 `compute_loss`（正常 forward + FGM adversarial forward），第二次拿到的 `inputs` 已缺少 `sample_weight`，導致兩次 forward 的 loss 計算邏輯不一致（有無 weight penalty）。

**修復位置**：`pipeline/model_training/train_and_export_onnx.py` L218、`pipeline/model_training/train_teacher.py` L212

**做法**  
改用 `inputs.get("sample_weight", None)` 非破壞性讀取，不修改原始 dict；在模型 forward 之前用 dict comprehension 過濾掉不需傳入模型的欄位。

---

### Fix-05 🟡 `electricity_billing` null 防護缺失（`inference.js`）

**問題**  
`inference.js` L465 的 `prop.electricity_billing.match(...)` 在欄位為 `undefined` 或 `null` 時會拋出 `TypeError: Cannot read properties of undefined`。此欄位在部分房源中可能不存在，特定查詢條件下前端直接報 JS 錯誤、推薦結果消失。

**修復位置**：`frontend/js/inference.js` L465、L471

**做法**  
加 null guard：`const billing = prop.electricity_billing || ""; const match = billing.match(...)`

---

### Fix-06 🟡 FGM 無 `try/finally` 保護

**問題**  
`training_step` 中的 `fgm.attack()` 後若 `compute_loss` 或 `accelerator.backward` 拋出例外，`fgm.restore()` 不會被執行，embedding 參數殘留被擾動的值，後續所有訓練步驟都建立在損壞的 embedding 上，且不會有任何錯誤訊息提示。

**修復位置**：`pipeline/model_training/train_and_export_onnx.py` L322-329、`pipeline/model_training/train_teacher.py` L277-284

**做法**  
用 `try/finally` 包住 adversarial backward，確保 `fgm.restore()` 一定執行。

---

### Fix-07 🟡 三個 Intent Map 不同步

**問題**  
前端有兩份各自定義的生活型態映射（`inference-worker.js` 的 `semanticExpandQuery` + `inference.js` 的 `expandQueryIntent`），Python 端還有第三份 `lifestyle_mapper.py`。三者有多項不一致，且 worker 中有**重複 key**（`"潔癖"` 出現兩次），JS 靜默覆蓋後者。  
具體差異：
- `"首選"` vs `"首租"`（語意完全不同）
- `"怕吵"` 展開中「靜巷」vs「寧靜」
- `inference.js` 中的 `expandQueryIntent` 比 worker 少了許多 key（懶人、自炊等）

**修復位置**：`frontend/js/inference-worker.js` L149-182、`frontend/js/inference.js` L533-548

**做法**  
1. 刪除 `inference-worker.js` 的重複 key `"潔癖"`（L172）
2. 將 `"首選"` 修正為 `"首租"`
3. 補齊 `inference.js` 的 `expandQueryIntent` 缺少的 key（`"懶人"`、`"自炊"`、`"外送族"` 等）使兩個前端 map 一致

---

### Fix-08 🟡 RankNet 數值穩定性

**問題**  
`torch.log(1 + torch.exp(-(s_i - s_j)))` 是手動 softplus，當 `s_j - s_i` 非常大時，`torch.exp(...)` 會溢出成 `inf`，導致 loss 為 `nan`，訓練靜默崩潰。雖然 T_task=2.0 已緩解，但仍有理論風險。

**修復位置**：`pipeline/model_training/train_and_export_onnx.py` L278、`pipeline/model_training/train_teacher.py` L253

**做法**  
改用 `F.softplus(-(s_i - s_j)) * mask`，PyTorch 內建數值穩定實作。

---

### Fix-09 🟢 API key warning 訊息誤導（`labeler.py`）

**問題**  
`labeler.py` 的 warning 訊息說 `ANTHROPIC_API_KEY not set`，但實際使用的是 **Google Gemini** API（`google.genai`）。開發者或 CI 日誌讀到這條 warning 會去設定錯誤的 API key。

**修復位置**：`pipeline/data_prep/labeler.py`

**做法**  
將 warning 改為 `"GOOGLE_API_KEY not set. Silver labeling (Gemini) disabled."`

---

### Fix-10 🟢 `evaluate_model.py` 重複 `sys.path.append`

**問題**  
L25 完全重複 L24 的 `sys.path.append`，相同路徑加入兩次，無實際副作用但是明顯多餘。

**修復位置**：`pipeline/model_training/evaluate_model.py` L25

**做法**  
刪除重複行。

---

### Fix-11 🟢 `app.js` 死碼清除

**問題**  
`const isRelevant = true` 後面緊接 `if (!isRelevant && inputText.length > 1)` 永遠不成立的 if 區塊（L194-203），包含完整的 DOM 操作程式碼。這個死碼增加維護者的認知負擔，且看起來像功能尚未完成的殘留。

**修復位置**：`frontend/js/app.js` L194-203

**做法**  
刪除 `isRelevant` 變數宣告及整個 if 區塊。

---

### Fix-12 🟢 ONNX 匯出中多餘的重複載入模型

**問題**  
`export_to_onnx` 函數先執行 `model.config.attn_implementation = "eager"`，緊接著又用 `AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_DIR, attn_implementation="eager")` 重新從磁碟載入模型。原本對記憶體中模型的修改毫無作用，且多一次磁碟 I/O。

**修復位置**：`pipeline/model_training/train_and_export_onnx.py` L579-584

**做法**  
移除 `model.config.attn_implementation = "eager"` 這行（因為下一行的 `from_pretrained` 已透過參數指定 `attn_implementation="eager"`，功效相同）。

---

### Fix-13 🟢 大量重複程式碼提取至共用模組

**問題**  
下列程式碼在 `train_and_export_onnx.py` 和 `train_teacher.py` 中**完全相同**：
- `FGM` class
- `compute_metrics` function
- `CleanLogCallback` class
- `CustomEarlyStoppingCallback` class

兩份 copy 在之後的修改中必須同步，維護成本高。

**修復位置**：新建 `pipeline/model_training/training_utils.py`，並更新兩個訓練腳本的 import

**做法**  
1. 新建 `training_utils.py`，將四個共用元件移入
2. 在兩個訓練腳本頂部改為 `from .training_utils import FGM, compute_metrics, CleanLogCallback, CustomEarlyStoppingCallback`
3. 刪除原腳本中的重複定義

---

## 實施順序

```
Fix-01 → Fix-02 → Fix-03 → Fix-04 → Fix-05
→ Fix-06 → Fix-07 → Fix-08 → Fix-09 → Fix-10
→ Fix-11 → Fix-12 → Fix-13
```

完成後統一 commit：
```
fix: apply 13-item codebase improvement plan
```
