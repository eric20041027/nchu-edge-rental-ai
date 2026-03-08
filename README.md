# 興大 AI 租屋推薦系統 (前端 ML 離線推論版)

本專題致力於打造一個**完全無需後端伺服器**的租屋推薦平台。透過技術整合，我們將原本運行於 Python 後端的大型自然語言處理 (NLP) 模型輕量化，並成功將其完整遷移至使用者的瀏覽器端 (Frontend) 進行離線推論。

## 💡 專題核心特色 (Capstone Highlights)

1. **線下建模與模型輕量化**
   - 原始模型為基於 Transformer 架構的 ALBERT (A Lite BERT)。
   - 透過 ONNX (Open Neural Network Exchange) 格式，將 PyTorch 模型轉換為靜態計算圖，大幅縮減體積至僅 16MB，同時保有原本的特徵萃取準確度。
2. **純前端 Web ML 獨立運算**
   - 移除全部 Python (Flask) 後端依賴。
   - 結合 `Transformers.js` (負責斷詞) 與 `ONNXRuntime-Web` (負責矩陣運算)。
   - 達成使用者端「零延遲、免伺服器運算成本、極高隱私安全」的 Web 3.0 體驗。
3. **Vercel 靜態無伺服器部署**
   - 專案能以 100% 靜態網頁 (HTML/CSS/JS) 的形式直接免費託管於 Vercel 或 GitHub Pages。

---

## 🔬 ONNX 張量 (Tensor) 轉換細節解說

在 `inference.js` 中，我們完美還原了 Python 模型推論的數學運算底層邏輯。要讓前端瀏覽器執行 AI 神經網路，必須經過以下嚴謹的張量 (Tensor) 轉換流程：

### 1. 斷詞與編碼 (Tokenization)
當使用者輸入: `"預算六千以內的套房"`。
首先由 `Transformers.js` 的 `AutoTokenizer` 將自然語言切分成模型能理解的整數陣列 (Token IDs)：
*   **`input_ids`**: `[101, 7521, 5050, 1063, 1283, 809, 1058, 4638, ... 102]` (101代表句首 `[CLS]`，102代表句尾 `[SEP]`)。
*   **`attention_mask`**: 告訴模型哪些字是真正有意義的 (為 1)，哪些是為了補齊長度填充的 Pad (為 0)。
*   **`token_type_ids`**: 區分句子段落 (在這裡都是 0)。

### 2. 建立 ONNX 運算張量 (Tensor Creation)
WebAssembly (WASM) 無法直接讀取 JavaScript 的一般陣列，我們必須將這些陣列轉換為低階的記憶體結構。我們宣告了高精度的 64位元整數矩陣 (`BigInt64Array`)：
```javascript
const input_ids_tensor = new ort.Tensor('int64', BigInt64Array.from(tokens.input_ids), [1, sequence_length]);
```
這個 `[1, sequence_length]` 代表維度 (Dimension)：即「1個 Batch (一次處理一句話)」，長度為 `sequence_length`。

### 3. ONNXRuntime 矩陣推論
我們將組裝好的 Tensors 輸入進掛載好的 ONNX Session。底層引擎會調用瀏覽器的 CPU/GPU 資源進行龐大的線性代數運算 (Linear Algebra Calculation)，最後輸出一組名為 `logits` 的浮點數矩陣 (`Float32Array`)。
*   `logits` 的維度是 `[1, sequence_length, 2]`。代表每個字都有 2 個機率值 (分別代表它是 `B-Tag` 還是 `I-Tag` 實體標籤的可能性)。

### 4. Argmax 解碼與還原 (Decoding)
最後，我們對每一個字計算 `Argmax` (尋找兩者之中機率最高的那個索引值)，決定這個字是標籤 `B (Begin)` 還是 `I (Inside)`。
系統接著將這些抽取出連續的 B/I 標籤字元，還原成如「6000」、「套房」等特徵關鍵字，最後送入 CBF (Content-Based Filtering) 推薦演算法矩陣與 CSV 資料庫比對分數。

## 🚀 開發與本機測試

由於瀏覽器的 CORS (跨源資源共用) 安全限制，無法直接雙擊點開 `index.html` 載入模型。請在本機端使用輕量級伺服器啟動：

```bash
# 在專案目錄下開啟終端機，執行以下指令：
python3 -m http.server 5002

# 然後打開瀏覽器前往：
http://localhost:5002/
```
等待背景完成 16MB 模型快取下載後，即可開始享有純前端的租屋推薦體驗！
