# Spec: 向量檢索召回(bi-encoder 取代 rule-based recall)

> 來源意圖:[docs/intent/vector-retrieval-roadmap.md](../intent/vector-retrieval-roadmap.md)
> 階段:中長期路線 ① 向量檢索(② 反饋微調為後續,不在本 spec)
> 狀態:**待審核**(spec-driven-development Phase 1)

## Objective

**做什麼:** 用「另訓的 bi-encoder + 離線預算房源 embedding + 線上暴力 cosine 召回」**取代**現有
`calculateRuleBasedScore` 的關鍵字 rule-based 召回,讓房源拓到 ~1萬筆時瀏覽器端檢索仍即時,
且召回不漏掉「語意相符但用詞不同」的房源。CE(cross-encoder)維持為精排(對召回 top-K 重排)。

**為什麼現在:** 現況 `recommend()`(`frontend/js/inference.js:1251`)流程為
`filterHardExclusions`(硬篩) → `calculateRuleBasedScore`(**每次 query 掃過每一筆房源**做關鍵字打分,
`.slice(0,15)`) → CE 對這 15 筆逐一 `scorePair`。CE 次數已 cap 在 15、不會線性爆;
**真正會隨房源線性爆 + 漏召回的是 rule-based keyword 階段**。本階段先把這段換成向量召回。

**使用者:** 你 + demo 使用者 —— 房源規模放大後仍即時拿到推薦。

**成功長什麼樣:** 見 Success Criteria。核心:「先試試看取代」必須有可量測的 A/B 結論,不是憑感覺。

## Tech Stack

- **訓練(離線,Colab)**:Python + PyTorch + sentence-transformers(bi-encoder 訓練),
  既有 `pipeline/model_training/` 與 `colab_train.ipynb` 風格;ONNX 匯出沿用 `dynamo=False` legacy tracer。
- **房源 embedding 預算(離線)**:Python 批次跑 bi-encoder → 房源向量,輸出靜態 JSON。
- **線上(瀏覽器 edge)**:既有 onnxruntime-web;query encode 走 ONNX,cosine 走純 JS
  (沿用現有 text2vec 原型向量同源做法:同 model / mean-pool / L2 norm,內積即 cosine)。
- 規模假設:**幾千~1萬筆** → 純 JS 暴力 cosine 即足夠,**不引入** 向量 DB / ANN / 後端。

## Commands

```
# 訓練 bi-encoder(Colab,離線)
# (新增 notebook 或 pipeline/model_training 腳本,沿用既有訓練習慣)

# 預算房源 embedding(離線,產靜態檔)
python pipeline/data_prep/build_property_embeddings.py \
    --model <bi_encoder_onnx_or_st> \
    --properties frontend/assets/property_data.json \
    --out frontend/assets/property_embeddings.json

# A/B 評估:向量召回 vs rule-based 召回(NDCG@5 / Recall@K)
python tests/eval_vector_vs_rulebased.py

# 前端本地預覽
# (沿用現有靜態檔服務方式,例如 python -m http.server 於 frontend/)

# 單元/整合測試
pytest tests/ -v
```

> 待補:上面新腳本路徑為 Plan 階段確定的目標,尚未存在。

## Project Structure

```
pipeline/model_training/      → bi-encoder 訓練腳本(沿用既有結構)
pipeline/data_prep/           → build_property_embeddings.py(離線預算房源向量)
frontend/assets/
  property_data.json          → 現有房源資料(704 筆,含 ce_text/text)
  property_embeddings.json    → 【新】離線預算的房源 embedding(Float32 + meta)
frontend/models/
  bi_encoder_dir/             → 【新】bi-encoder query 端 ONNX(量化)
frontend/js/inference.js      → recommend():召回階段改接向量召回
tests/
  eval_vector_vs_rulebased.py → 【新】A/B:向量召回 vs rule-based(NDCG@5/Recall@K)
docs/spec/vector-retrieval.md → 本 spec
```

## Code Style

沿用現有 text2vec 原型向量做法 —— embedding 存 dim + flat Float32,線上 reshape + 內積:

```javascript
// 沿用 inference.js 既有 query-expansion 的同源 cosine 寫法:
// query 向量 L2 normalize 後,與同樣 normalize 的房源向量內積即 cosine
function cosineTopK(queryVec, propVecs, k) {
    const scored = [];
    for (let i = 0; i < propVecs.length; i++) {
        let dot = 0;
        const v = propVecs[i];
        for (let d = 0; d < queryVec.length; d++) dot += queryVec[d] * v[d];
        scored.push([i, dot]);
    }
    scored.sort((a, b) => b[1] - a[1]);
    return scored.slice(0, k);   // 召回 top-K(K=30)→ 交給 CE 精排
}
```

慣例:embedding 同源(同 model / mean-pool / L2 norm),否則 cosine 無意義;
靜態 embedding 檔帶版本 query string(沿用 `?v=YYYYMMDD` 快取破壞慣例)。

## Testing Strategy

- **框架:** pytest(離線評估/資料);前端維持現有 worker/inference 手動驗證 + 既有測試。
- **A/B 評估(核心):** `tests/eval_vector_vs_rulebased.py` —— 用固定 query 集,
  比較「向量召回 top-K」與「rule-based 召回 top-K」的 **Recall@K** 與下游 **NDCG@5**
  (沿用 sleepy-sutherland 分支已有的 CE NDCG A/B 腳本風格)。
- **覆蓋:** 新增離線腳本(embedding 預算、評估)需有單元測試覆蓋核心函式;
  前端 cosine 召回函式需 happy-path + 邊界(空候選 / K 大於候選數)。
- **回歸:** 換召回後,現有否定意圖 / 字卡正確性等行為不可退化。

## Boundaries

- **Always:** embedding 同源(同 model/pool/norm);換召回前先跑 A/B 拿到數字;
  房源向量離線預算、線上只 encode query;守 edge-first(瀏覽器跑得動)。
- **Ask first:** 引入新訓練依賴(sentence-transformers 等);改 `property_data.json` schema;
  bi-encoder 模型大小 / 量化策略影響前端載入;CE 精排階段的 top-K 數值調整。
- **Never:** 為了向量檢索引入後端服務 / 向量 DB / ANN(超出本階段規模);
  在沒有 A/B 數字下就把 rule-based 召回刪掉(先並行可切換,確認再移除)。

## Success Criteria

具體、可測。**門檻已用 T0 現況基準定錨**(見 §T0 Baseline)。

1. **召回不退化:** 向量召回的 **Recall@K ≥ 現況**(同 query 集、同 K、同 harness):
   - Recall@15 ≥ **0.3846**(現況 production `.slice(0,15)`)
   - Recall@30 ≥ **0.4495**(規劃 K)
   且在「語意相符但用詞不同」的 query 上 **Recall 明顯較高**(A/B 分項量化)。
2. **下游品質不退化:** 召回階段 ranking **NDCG@5 ≥ 0.2469**(現況,同 harness、同 graded-relevance 慣例);
   端到端(+ CE 精排)NDCG@5 ≥ 現況(CE harness 另量)。
3. **可擴展性:** 在模擬 ~1萬筆房源下,瀏覽器端「query encode + cosine 召回」端到端
   **< 現有 rule-based 召回在同規模的耗時**,且無明顯卡頓(召回階段目標 < ~200ms)。
4. **載入可接受:** bi-encoder query 端 ONNX(量化)+ 房源 embedding 靜態檔
   加入後,首載總量相對現況 **74.85 MB**(CE 36.93 + NER 36.44 + property_data 1.49)
   增幅 **≤ ~5 MB**(embedding 檔走 float16/量化把關)。

> **T0 Baseline(2026-06-22 量測,harness:`tests/eval_rule_based_baseline.py`):**
> Recall@15 = 0.3846、Recall@30 = 0.4495、NDCG@5(召回階段)= 0.2469、首載 = 74.85 MB。
> **重要 caveat:** Recall/NDCG 在 fuzzy-join 後的 query 集上量(訓練資料 property 為舊版 blob,
> 與現行 704 筆 snapshot **join match-rate 僅 24.4%** —— 分佈 bimodal,2149 筆已不存在於現 snapshot、
> ~700 筆完美對上)。**絕對值偏低主因是 join 覆蓋稀疏 + 召回階段未含 CE;作為基準無妨,
> 因 T7 向量 A/B 會用同一 harness、同一 join、同一慣例對比**(相對差才是判準)。
> 召回 port 未含 NER 增強(瀏覽器專屬),為刻意離線省略,已記於 harness docstring。

5. **行為不回歸:** 否定意圖、字卡正確性等既有測試全綠。

## Resolved Decisions(2026-06-21 敲定)

1. **bi-encoder base model:** **沿用現有 CE 同源 base**(同一個中文 base 改成 bi-encoder)。
   理由:tokenizer/vocab 已驗證、與 CE 精排同源、量化管線現成、風險最低。embedding 維度多半 768。
2. **訓練資料:** **直接用 `data/processed/recommendation_train.json`**(29,443 對
   `{query, property, label(0/1), relevance, is_hard}`)做對比學習 —— label=1 為正樣本、
   is_hard=True 為硬負樣本,配 in-batch negatives。資料已現成,**不重組正負樣本**。
3. **召回 K:** 向量召回交給 CE 精排的 top-K = **30**(現況 rule-based 給 15 的 2 倍)。
   向量召回快,多召回幾筆讓 CE 有更大挑選空間(補召回、抗漏);CE 仍只跑 30 次,edge 端可接受。

## Open Questions(Plan 階段定錨,不阻塞方向)

1. **效能門檻定錨:** Success #3/#4 的具體 ms / MB 數字,需在現有裝置基準上量一次現況再定。
2. **A/B query 集:** 評估用的 query 集從哪來?(需含「同義/口語/語意」類 query 才測得出召回優勢)
