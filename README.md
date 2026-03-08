# 興大 AI 租屋推薦系統 (Frontend-Only ML Architecture)

本專案是一個**完全無伺服器 (Serverless/Frontend-Only)** 的租屋推薦應用程式，專為尋找中興大學附近租屋的大學生所打造。透過最新的 Web Machine Learning 技術，本系統將原本依賴 Python 後端的 Transformer 模型推論，完整移植至前端瀏覽器執行。

使用者能以自然語言輸入租屋需求（如：「預算 6000 以下，近中興正門，獨洗獨曬套房」），由瀏覽器即時下載並執行 ONNX 模型，分析語意特徵並計算分數，找出最佳租屋選項。

## 💡 專題核心特色 (Key Features)

- **100% 離線 AI 推論**：採用 `ONNXRuntime-Web` 直接在瀏覽器解析 16MB 的 ALBERT 模型。
- **免除後端成本與延遲**：無需架設 Python 伺服器，使用者端實現 0 總網路延遲的 CBF (Content-Based Filtering) 推薦。
- **邊緣運算隱私保護**：語意分析在裝置端進行，保障使用者輸入需求隱私。
- **無縫靜態部署**：可秒級部署至 Vercel 或 GitHub Pages，無任何伺服器相依性。
- **現代化響應式 UI**：採用玻璃質感與深色模式的主題風格。

---

## 🛠 Tech Stack

- **核心技術**：HTML5, CSS3, Vanilla JavaScript (ES6+)
- **自然語言處理 (NLP)**：
  - [Transformers.js](https://huggingface.co/docs/transformers.js/index) (負責 Tokenization 斷詞與編碼)
  - [ONNX Runtime Web](https://onnxruntime.ai/docs/api/javascript/api/interfaces/Session.html) (負責 WebAssembly 張量矩陣運算)
  - **模型**：ALBERT (A Lite BERT) 轉換之 `.onnx` 格式
- **資料處理**：[PapaParse](https://www.papaparse.com/) (負責高速解析本機房屋 CSV 檔)
- **部署平台**：[Vercel](https://vercel.com) (Static Web Hosting)

---

## 🚀 Getting Started

本系統為純靜態網頁架構，但由於載入 `.onnx` 模型與 `.csv` 檔案受到瀏覽器 CORS (Cross-Origin Resource Sharing) 安全政策限制，不能直接以 `file://` 協定打開 `index.html`，必須透過本地 HTTP 伺服器執行。

### 1. 取得專案原始碼

```bash
git clone https://github.com/eric20041027/Renting-recommendation-ONNX.git
cd Renting-recommendation-ONNX
```

### 2. 啟動 Local Development Server

您可以使用任何熟悉的終端機 HTTP 伺服器，最簡單的方式是使用 Python 內建的模組 (幾乎所有電腦皆已內建 Python)：

```bash
# 啟動 Python HTTP 伺服器於 Port 5002
python3 -m http.server 5002
```

或者使用 Node.js 的 `http-server`：

```bash
npx http-server -p 5002
```

### 3. 開始測試推論

開啟瀏覽器並前往：
[http://localhost:5002](http://localhost:5002)

網頁一開始會顯示「正在下載 AI 模組與資料...」，此時瀏覽器正在從本機背景快取 16MB 的 `model.onnx` 與房屋資料，載入完畢後即可輸入條件進行秒速推理。

---

## 🏗 Architecture Overview

本專案將複雜的後端 API 與模型推論重構，全數封裝進靜態資料夾中：

### Directory Structure

```text
Renting_model_ONNX/
├── index.html            # 網頁結構與進入點
├── styles.css            # 現代化深色 UI 樣式
├── app.js                # UI 互動邏輯 (接收輸入、處理動畫、卡片渲染)
├── inference.js          # [核心] AI 模型載入、WebAssembly 推理、推薦計分演算法
├── nchu_rental_info.csv  # 靜態房屋資料庫 (爬取整理後之結果)
├── onnx_model_dir/       # 機器學習模型 (ALBERT)
│   ├── model.onnx        # 神經網路靜態計算圖
│   ├── tokenizer.json    # 分詞器字典對照表
│   └── tokenizer_config.json 
├── .gitignore            # Git 忽略設定
└── README.md             # 專案說明文件
```

### Request & Data Flow

在純前端架構中，資料流如下：

1. **Initialization:** 
   載入網頁時，`inference.js` 併發請求下載並實例化 `nchu_rental_info.csv` 與 `model.onnx` Session。
2. **User Input:** 
   使用者輸入需求字串（如：「預算六千套房」）。
3. **Tokenization:** 
   `Transformers.js` 查表將字串轉換為 Token IDs。
4. **ONNX Inference:** 
   WebAssembly 將 Tokens 轉為記憶體連續張量 (Tensor)，送入神經網路得出浮點數機率矩陣。
5. **Feature Extraction:** 
   利用 Argmax 解碼出 `B-Tag` 與 `I-Tag`，拼湊出具體特徵 (如 `預算: 6000`, `房型: 套房`)。
6. **CBF Engine:** 
   將特徵送入基於規則的得分函數 (CBF)，掃描並評分 CSV 內所有房屋。
7. **Rendering:** 
   `app.js` 接收前五高分之房屋物件，動態生成 HTML 卡片渲染於畫面上。

---

## 🔬 Deep Dive: ONNX 張量 (Tensor) 轉換細節

本專題將 Pytorch 後端模型完全解構，其中最大的技術難點在於：如何在 JavaScript (弱型別腳本語言) 中處理神經網路所需的高精度多維矩陣運算？
以下為 `inference.js` 中的實踐步驟：

### 1. 斷詞與編碼 (Tokenization)
首先使用 HuggingFace 的開源 JS 工具載入字典：
```javascript
const tokens = await tokenizer("預算六千套房", { return_tensor: false });
```
此步驟會產出神經網路的四大基礎元件：
*   **`input_ids`**: `[101, 7521, 5050, 1063... 102]` (映射至字典的整數索引，101為句首，102為句尾)。
*   **`attention_mask`**: 代表需要被計算梯度的有效字元 (皆為 1，非 Padding)。
*   **`token_type_ids`**: 區分句型結構 (單一句子皆為 0)。

### 2. 位元陣列轉鑄為 ONNX 張量 (Typed Memory Allocation)
WebAssembly (WASM) 無法直接讀取 JS 的一般陣列，必須手動分配並轉鑄為 C++ 能夠理解的連續記憶體區塊 (`BigInt64Array`)：
```javascript
// 分配 64位元高精度整數記憶體，維度設定為 [Batch Size=1, Sequence Length]
const input_ids_tensor = new ort.Tensor(
    'int64', 
    BigInt64Array.from(tokens.input_ids.map(BigInt)), 
    [1, tokens.input_ids.length]
);
```

### 3. WASM 核心運算 (Session Run)
將組裝好的三個張量封裝入 `feeds`，指令 `onnxruntime-web` 啟動矩陣乘法運算：
```javascript
const results = await session.run(feeds);
const logits = results.logits.data; // <- Float32Array
```
輸出的 `logits` 是一個超扁平的浮點數一維陣列。其被壓平的理論維度為 `[1, sequence_length, 2]`，代表句子裡的每一個字，都會得到 2 個隱藏層機率值。

### 4. 決策邊界與解碼 (Argmax Decoding)
遍歷所有的字元，分別比對它們所屬的兩個機率值。
尋找最大值 (Argmax) 的索引，若第 0 個機率較大則為 `B (Begin)`，若第 1 個機率較大則為 `I (Inside)`。接著將這些 B/I 的字元連續重組，即可還原為過濾條件，並送入後續的推薦記分板。

---

## 🌐 Deployment (產線部署)

本專題已經去除任何 Python / WSGI 依賴，因此部署起來極為輕量且無風險，任何靜態託管平台皆可無縫上線。

### Vercel (推薦)

預設專案已最佳化成符合 Vercel 最原生的形態：
1. 前往 [Vercel](https://vercel.com/)。
2. 點擊 **Add New Project**。
3. 匯入本 GitHub Repository。
4. **Framework Preset** 選擇 `Other`。
5. 不需任何 Build Command 或設定任何 Environmental Variables。
6. 點擊 **Deploy**。
*註：我們已將原先會造成路由衝突的 `vercel.json` 移除，確保 CSS/JS 等靜態資源皆能正常載入。*

### GitHub Pages

因為一切運作皆在前端，只要打開本 repo 的 `Settings -> Pages`，將來源指定至 `main` 分支的 `/(root)` 即可免費託管，讓這份專題報告永久存活。
