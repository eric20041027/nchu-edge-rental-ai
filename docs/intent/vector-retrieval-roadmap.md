# 未來計劃 — 中長期路線(向量檢索 → 反饋微調 → 資料管線一鍵化 → 泛化強化)

> 由 interview-me 產出的確認意圖,2026-06-21。後續 spec / 規劃以此為準。

## 確認的意圖

- **Outcome:** 把線上檢索從「CE 對每筆候選逐一推論」改成「bi-encoder 向量檢索」——
  房源 embedding 預先算好,線上只算一次 query embedding + 暴力 cosine,
  讓房源拓到幾千~一萬筆時瀏覽器端不卡。CE 退居小候選集的精排(re-rank)。
- **User:** 你(以及最終用 demo 的人)—— 房源規模放大後仍能即時拿到推薦。
- **Why now:** 要拓展更多房源;現有 edge 端逐筆 CE 推論會線性爆掉。
  先把可擴展性架構鋪好,反饋微調才有意義。
- **Success:** 房源拓到幾千~一萬筆時,瀏覽器端檢索仍即時(無明顯卡頓),
  且推薦品質不比現在差。
- **Constraint:** 守住 edge-first(瀏覽器 ONNX 跑得動)、
  模型/索引別大到拖垮前端載入。

## 路線順序

1. **向量檢索**(語意召回 + 解推論速度)— ✅ 階段①已落地 main(T0–T7 GO)
2. **反饋微調**(用點擊/回饋線上重排)— ✅ 階段②已落地 main(PR #50)
3. **資料管線一鍵化 + 多平台**(讓房源擴充不再手動)— ✅ 階段③已落地 main(PR #52);後續「先管線後平台」接新 crawler
4. **泛化強化**(LLM 生成多樣 query 破模板 + 真評估集 + 重訓)— 意圖見下方〈階段④確認意圖〉

## 階段②確認意圖

> 由 interview-me 產出的確認意圖,2026-06-23。後續 spec / 規劃以此為準。

- **Outcome:** 在 CE 精排之後加一層**純後處理重排** —— 用 localStorage 累積的
  per-propertyId 👍/👎 對房源加固定調整量(👍 輕推、👎 重罰但不消失),
  讓推薦隨使用者回饋越用越貼個人。**不碰任何模型權重。**
- **User:** 這個瀏覽器的使用者(及 demo 操作者)—— 回饋跨 query、跨 session
  累積,個人化只在本機(per-browser)。
- **Why now:** 階段①向量檢索把可擴展性鋪好,反饋微調才有意義;回饋機制
  (`renting_feedback_log`)已在收資料,現在把它接進排序。
- **Success(行為判準):** 👎 某房源後重跑同/異 query,它明顯下沉;👍 後上升;
  跨 session 仍生效;kill-switch 關掉後行為完全回到階段①。瀏覽器 preview
  親自驗 + 一個確定性 self-check(假回饋 + 假 CE 分數 → 重排順序符合預期)。
- **Constraint:** edge-first、純前端、零後端、零模型重訓;後處理層獨立可關
  (kill-switch,比照 `VECTOR_RECALL_ENABLED`),零侵入 `recommend()`
  既有召回 + CE 邏輯。

### 階段② Out of scope

- 真重訓 bi-encoder / CE
- 👎 泛化到向量鄰居 / 相似房源(只記 propertyId 本人)
- 硬隱藏房源(👎 = 重罰下沉,非消失)
- 離線 Recall@K A/B 數字(回饋即 ground-truth,離線指標循環論證)
- 跨裝置 / 雲端同步回饋

## 階段③確認意圖

> 由 interview-me 產出的確認意圖,2026-06-23。後續 spec / 規劃以此為準。

- **Outcome:** 把現有「爬蟲 → 富化 → 重算向量 → 上線」串成**可重複管線**(段內零手動):
  新房源資料丟進去 → 產出可換進前端的 `property_data.json` + `property_embeddings.json`。
  **先把管線打通,之後再一個個接新租屋平台 crawler 餵進這條管線。**
  - **現實修正(2026-06-23 查證 code 後):** 向量重算 `build_property_embeddings.py` 用
    **PyTorch** 跑 bi-encoder forward,本機 CPU(無 torch)跑不動 → 管線**分兩段**:
    **本機段**(crawl → 富化 → `property_data.json`,純 CPU 可跑可驗)+
    **Colab 段**(向量重算 → `property_embeddings.json`,需 torch)。
    不是「一條命令」而是「兩條命令、各自段內零手動」。重算用已訓練權重,**非重訓**。
- **User:** 維護者 —— 擴充房源從「手動跑一串 script」變成「跑一條命令」;
  最終受益是 demo 使用者(房源更多更有料)。
- **Why now:** 階段①②鋪好可擴展性與品質;roadmap 原始 why 就是「拓展更多房源」,
  但流程手動是量上不去的根因 → 先解流程(痛點 = 量少 + 流程手動,流程是量的瓶頸)。
- **Success(行為判準):** 兩段各自跑通、段內零手動 —— **本機段**一條命令:餵新房源 →
  富化 → 產出 `property_data.json`(本機親驗,含 `build_property_embeddings.py --check`
  驗記錄數/欄位,無需 torch);**Colab 段**一條命令:重算向量 → `property_embeddings.json`。
  **不綁量化拓量門檻**(拓到幾筆是自然結果)。
- **Constraint:** 中興大學附近為主軸(`geo_tier` / `distance` 相對中興算,不動地理語境);
  不跨城市、不引入 `city` 欄位;edge-first 不變;**先管線、後平台**。

### 階段③ Out of scope

- FB 來源(社團自由文字、資料品質爛、難富化 → 砍掉)
- 跨城市 / 跨地理語境(geo_tier/distance 維持相對中興)
- 量化拓量門檻(拓到 N 筆當自然結果,非驗收條件)
- 重訓模型(屬階段④;階段③只串既有 script 不重訓)
- 後端服務

## 階段④確認意圖

> 由 interview-me 產出的確認意圖,2026-06-23。後續 spec / 規劃以此為準。
> 背景:查證揭露現有「語意泛化」是假象 —— 訓練資料 73.5% 重複 + 100% 模板合成、
> 正樣本自我參照、「語意」實為 102 條 `data/semantic_rules.json` 同義詞表、
> 評估集 selection bias(語意桶按定義篩「rule-based 必敗」的 query)。
> 真泛化的瓶頸在資料與評估,不在模型架構。

- **Outcome:** 用 Claude(session 內生成,免 API key)產出**多樣化口語/隱喻/跨域類比
  query**(破模板、破自我參照),補進訓練資料 → Colab 重訓 bi-encoder → 在一個
  **新建的、不偏袒的真評估集**上驗證「真泛化」。完整「資料 → 評估 → 模型」閉環。
- **User:** 維護者 —— 要能誠實證明「泛化有沒有真進步」的閉環,不再被 selection-bias
  數字自欺;受益是 demo 使用者(沒見過的說法也召得回對的房源)。
- **Why now:** 查證證明現有泛化是假象,要真泛化只能從資料與評估下手,模型架構解不了。
- **Success(數字 Δ + holdout 質性):** (1)新真評估集上重訓後 Recall@30 比重訓前明顯提升;
  (2)一批**生成時就隔離、風格刻意不同**的 holdout query,重訓後能召回對的房源
  (本機 preview 親驗實際案例)。
- **Constraint:** edge-first 不變(bi-encoder 仍 57MB INT8);Claude 只交付**能本機驗**的東西
  (訓練資料 JSON + 真評估集 + holdout + 本機可跑評估 harness);重訓在 Colab
  (不盲改訓練 cell);評估 ground-truth = Claude 生候選 + 用戶抽查,**同源 caveat 寫進 meta**;
  第一輪資料用 **append 求穩**。

### 階段④ Out of scope

- **部分替換 / 砍重複模板** — **後續必做**(第二輪),非本輪(第一輪只 append)
- 換更大 encoder / 離開 edge
- 真人寫評估 query(承認同源 caveat,不追完美 holdout)
- Claude 改 Colab 訓練 cell(分工:Claude 管資料+評估,用戶管重訓)
- 線上 LLM 推論

## Out of scope

- 嵌入式地圖(Leaflet)— 已砍,維持 Google Maps 連結預覽(佔空間、傷 UI)
- 後端服務
- 向量資料庫 / ANN 索引(HNSW)— 幾千~一萬筆規模,瀏覽器端暴力 cosine 即足夠
- 十萬+ 規模

## 規模假設(決定設計)

- 目標房源量級:**幾千 ~ 一萬筆**(單一城市租屋,爬蟲天花板)
- 此規模下:瀏覽器端純 JS 暴力 cosine 即時可行,**不需** 向量 DB / ANN / 後端

## 現況架構(問題根源)

- 推論在**瀏覽器端 ONNX edge** 跑(`frontend/models/*_quant.onnx`)
- CE 是 cross-encoder:每個候選都要 query+doc 一起過一次模型 → O(N) 推論,房源一多線性爆
- 向量檢索把它降到 O(1 次 query encode + 快速向量比對)
