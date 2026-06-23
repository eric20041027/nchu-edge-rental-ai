# 未來計劃 — 中長期路線(向量檢索 → 反饋微調)

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
2. **反饋微調**(用點擊/回饋線上重排)— 意圖見下方〈階段②確認意圖〉

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
