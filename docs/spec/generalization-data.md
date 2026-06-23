# Spec: 階段④ 泛化強化 — 多樣 query 生成 + 真評估集 + 重訓閉環(第一輪)

> 上游意圖:`docs/intent/vector-retrieval-roadmap.md`〈階段④確認意圖〉(interview-me, 2026-06-23)。
> 實作前需人工 review 通過。**本 spec 只涵蓋第一輪(append + 評估集 + holdout + 本機 harness)。**

## Objective

直面查證揭露的「假泛化」:用 Claude(session 內生成,免 API key)產出**多樣化口語/
隱喻/跨域類比 query**(破模板、破自我參照),補進訓練資料 → 用戶 Colab 重訓 bi-encoder
→ 在**新建的不偏袒真評估集**上驗證真泛化。

- **為何:** 現有泛化是假象(73.5% 重複模板 + 102 條同義詞表 + 評估集 selection bias)。
- **成功:** 數字 Δ(新評估集 Recall@30 重訓後提升)+ holdout 質性(隔離、風格不同的
  query 重訓後召回對的房源,本機 preview 親驗)。

**核心約束:edge-first(仍 57MB INT8);Claude 只交付能本機驗的(資料 JSON + 評估
harness);重訓在 Colab(不盲改訓練 cell);評估 GT = Claude 生 + 用戶抽查,同源 caveat 入 meta;
第一輪 append 求穩。**

## 資料契約(查證後,實況)

**訓練資料** `data/processed/recommendation_train.json`(train_bi_encoder `_load_pairs` 吃):
```json
{ "query": str, "property": str(房源文字), "label": 0|1, "relevance": int, "is_hard": bool }
```
- 正樣本 = `label==1` 的 {query, property};困難負樣本 = 同 query 的 `is_hard==true && label==0`。
- **property 是文字字串,非 idx。** 生成時需掌握「idx ↔ property 文字」對應
  (`frontend/assets/property_data.json` 每筆有 `idx` + `ce_text`/`text`)。

**評估集** `tests/fixtures/ab_eval_queries.json`(eval harness 吃):
```json
{ "query": str, "bucket": str, "n_relevant": int,
  "relevant_idxs": [property_data idx], "semantic_trigger": str }
```
- Recall@30 = binary relevance,用 `relevant_idxs`(property_data idx)。

## 產物(Claude 本機交付,全部能本機驗)

```
data/processed/generalization_queries.json   → 生成的多樣訓練 query(訓練 schema,label=1 正 + is_hard 負)
tests/fixtures/generalization_eval.json       → 新真評估集(評估 schema + meta 含同源 caveat)
tests/fixtures/generalization_holdout.json    → holdout(生成時隔離、風格刻意不同,絕不進訓練)
tests/eval_generalization.py                  → 本機可跑評估 harness(無 torch:吃 property_embeddings.json + 評估集算 Recall@30)
docs/spec/generalization-data.md              → 本檔
```

## 生成方法(Claude session 內,免 API key)

每筆生成帶三欄(試水溫已驗收格式),**零標點、空白分隔**:
```json
{ "query": "養狗的家庭 空間要夠大", "hit_features": ["可養寵物","整層"], "gen_type": "跨域類比", "src_idx": 450 }
```
- `gen_type` ∈ {口語, 隱喻, 跨域類比, 生活推理, negation, 多意圖}(難度分層用)。
- 轉訓練 schema:`{query, property: idx→ce_text, label:1, relevance:2, is_hard:false}`。
- 困難負樣本:對該 query 取「表面相似但硬衝突」房源(如「養狗」配「禁養寵物」房源),
  `label:0, is_hard:true`(沿用既有 mining 邏輯的精神,但由 Claude 生成配對)。

**holdout 隔離鐵則:** holdout query 在**生成時就抽出**,用獨立 `gen_type` 風格(刻意與訓練不同),
**永不進 `generalization_queries.json`**。spec 與 harness 都禁止 holdout idx 出現在訓練集。

## 評估集 ground-truth(Claude 生 + 用戶抽查)

- Claude 為每筆評估 query 標 `relevant_idxs`(認為相關的 property_data idx)。
- **用戶抽查**一部分(spec 建議 ≥20%)確認標註合理,抓系統性偏差。
- **同源 caveat 寫進 meta**:評估 query 與訓練 query 皆 Claude 生成,非完美 holdout;
  數字僅作相對 Δ 與趨勢判讀,不宣稱絕對泛化。

## Commands

```
# 本機(python3.12,無 torch):評估 harness 自我驗 + 跑分
python tests/eval_generalization.py --check        # 結構/隔離驗:holdout 不在訓練集、idx 對得上 property_data
python tests/eval_generalization.py                 # 算 Recall@30(需 property_embeddings.json;重訓前後各跑一次比 Δ)

# Colab(用戶,需 torch):重訓 bi-encoder(吃 append 後的 recommendation_train.json)
# 用既有 colab_train_bi_encoder.ipynb,資料路徑接 generalization_queries.json append 後的檔(用戶自接)
```

## Testing Strategy

**數字 Δ + holdout 質性,雙判準。**

1. **本機評估 harness 自我驗**(`--check`,無 torch):
   - holdout query 的 idx **絕不出現**在 `generalization_queries.json`(隔離鐵則)。
   - 評估集/holdout 的 `relevant_idxs` 都對得上 `property_data.json` 的 idx(無懸空 idx)。
   - 生成 query 全零標點(自我檢查正則)。
2. **Recall@30 Δ**(本機可跑,需 property_embeddings.json):
   - 重訓**前**跑一次(現有向量)→ baseline;重訓**後**跑一次(Colab 產的新向量)→ 比 Δ。
   - harness import 既有 T0 metric(`recall_at_k`),不重寫。
3. **holdout 質性親驗**(本機 preview):
   - 重訓後,把 holdout query 餵進前端 `recommend()`,看實際召回對不對(人眼判)。
   - 這是「真泛化」的試金石:holdout 風格刻意不同,召得回才算泛化。

## Boundaries

- **Always:** 生成 query 帶 hit_features+gen_type+src_idx;零標點;holdout 生成時隔離;
  評估 GT 用戶抽查;同源 caveat 入 meta;產物本機可驗(harness `--check` 綠)。
- **Ask first:** 改訓練 schema;砍既有模板資料(第二輪才做);改 bi-encoder 訓練超參;
  生成規模超過第一輪約定量。
- **Never:** 把 holdout 混進訓練;Claude 改 Colab 訓練 cell(盲改);宣稱絕對泛化數字
  (同源 caveat 在);離開 edge / 換大模型。

## Success Criteria(具體、可測)

- [ ] `generalization_queries.json` 產出,訓練 schema 正確,append 後 `recommendation_train.json` 可被 `_load_pairs` 正常讀(本機驗 schema)。
- [ ] `generalization_eval.json` + `generalization_holdout.json` 產出,評估 schema 正確,`relevant_idxs` 無懸空 idx。
- [ ] holdout idx **零交集**於訓練集(harness `--check` 把關)。
- [ ] 生成 query 全零標點。
- [ ] `tests/eval_generalization.py` 本機跑通,重訓前能算出 baseline Recall@30。
- [ ] meta 含同源 caveat。
- [ ] 用戶抽查 ≥20% 評估 GT 確認合理。

## Resolved Decisions(2026-06-23 人工確認,採建議規模)

1. **第一輪生成規模 ≈ 1000 筆正樣本 query**:覆蓋約 100–150 間代表性房源 × 每間 6–10 條
   多樣 query(相對 29k 是溫和 append,不爆量)。
2. **holdout 規模 50–80 條**:風格刻意與訓練不同(更口語/更隱喻),生成時隔離。
3. **評估集規模 100–150 條**:語意桶為主,**不按「rule-based 必敗」篩**(修掉舊 selection bias)。
4. **困難負樣本由 Claude 生成時順帶配**「硬衝突」負樣本(如「養狗」配「禁養寵物」房源),
   `label:0, is_hard:true`,本機驗 label/is_hard 正確。

> 執行注意:約 1000 筆 query 生成是長工作,**分批執行**(每批數百筆),逐批寫進 JSON。
