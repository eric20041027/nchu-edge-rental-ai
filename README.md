# 興大 AI 租屋推薦系統

本專案是一個無伺服器 (Serverless/Frontend-Only) 的租屋推薦應用程式，專為尋找中興大學附近租屋的大學生所打造。透過 Web Machine Learning 技術，本系統將原本依賴 Python 後端的 Transformer 模型推論，完整移植至前端瀏覽器執行。

使用者能以自然語言輸入租屋需求，由瀏覽器即時下載並執行 ONNX 模型，分析語意特徵並計算分數，找出最佳租屋選項。

## 專題核心特色

- 100% 離線 AI 推論：採用 ONNXRuntime-Web 直接在瀏覽器解析 16MB 的 ALBERT 模型。
- 免除後端成本與延遲：無需架設 Python 伺服器，使用者端實現零延遲的 CBF (Content-Based Filtering) 推薦。
- 網路安全與隱私：語意分析在裝置端進行，保障使用者輸入需求隱私。
- 無縫靜態部署：部署至 Vercel，提供快速且穩定的靜態網頁體驗。

## 技術架構

- 核心技術：HTML5, CSS3, JavaScript (ES6+)
- 自然語言處理 (NLP)：
  - Transformers.js (負責 Tokenization 斷詞與編碼)
  - ONNX Runtime Web (負責 WebAssembly 張量矩陣運算)
  - 模型：ALBERT (A Lite BERT) 轉換之 .onnx 格式
- 資料庫：PapaParse 解析本機房屋 CSV 檔

## 本機開發與執行

因瀏覽器的 CORS 跨網域存取限制，無法直接打開 index.html 載入本地端的 .onnx 模型與 .csv 資料檔，必須透過本地 HTTP 伺服器執行。

1. 取得專案原始碼：
   ```bash
   git clone https://github.com/eric20041027/Renting-recommendation-ONNX.git
   cd Renting-recommendation-ONNX
   ```

2. 啟動本地 HTTP 伺服器：
   ```bash
   python3 -m http.server 5002
   ```

3. 開啟網頁開始測試：
   開啟瀏覽器前往 http://localhost:5002。首次載入會從背景下載 16MB 的模型檔案，完成後即可輸入條件進行本地推理。

## 系統資料流與 ONNX 張量解析

系統採用純前端架構，捨棄 Python 伺服器，資料處理流程如下：

1. 模型初始化：載入網頁時，inference.js 非同步下載 csv 與 model.onnx。
2. 斷詞編碼：使用者輸入字串後，Transformers.js 將字串轉換為 Token IDs。
3. ONNX 推論：WebAssembly 將 Token 轉換為高精度記憶體連續張量 (Tensor)，輸入神經網路得出浮點數機率矩陣。
4. 解碼與特徵擷取：利用 Argmax 解碼出實體標籤，取得具體特徵字串。
5. 推薦演算法：將擷取的特徵送入 CBF 演算法，掃描所有房屋資料並計算推薦分數。
6. 前端渲染：app.js 接收推薦結果並動態生成卡片渲染至畫面。
