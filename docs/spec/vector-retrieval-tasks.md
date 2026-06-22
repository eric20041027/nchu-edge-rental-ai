# Tasks: 向量檢索召回(spec-driven Phase 3)

> Spec:[vector-retrieval.md](./vector-retrieval.md) · Plan:[vector-retrieval-plan.md](./vector-retrieval-plan.md)
> 狀態:**待審核**(Phase 3 — Tasks)
> 規則:每個 task 單一 session 可完成、≤5 檔、有 acceptance + verify。依相依排序。

---

## T0 — 量現況基準(CP0,gate:無基準不開工) ✅ 完成 2026-06-22

- [x] **Task:** 量 rule-based 召回的 Recall@K / NDCG@5 / 前端首載大小,定錨門檻寫回 spec。
  - **Acceptance:** ✅ spec Success #1/#2/#4 已填具體數字(見 spec §T0 Baseline)。
  - **Verify:** ✅ `python3 tests/eval_rule_based_baseline.py`(--sample 與全跑一致)。
  - **Files:** `docs/spec/vector-retrieval.md`、`tests/eval_rule_based_baseline.py`(新,930 行,
    含可複用 `ndcg_at_k`/`recall_at_k`/loaders 供 T7 A/B)。
  - **結果:** Recall@15=0.3846、Recall@30=0.4495、NDCG@5(召回階段)=0.2469、首載=74.85 MB。
    **Caveat:** fuzzy-join match-rate 24.4%(snapshot 漂移,bimodal,已驗證非調參問題);
    召回 port 未含 NER 增強(離線省略)。T7 用同 harness 對比,相對差為判準。
  - **效能基準待補:** 召回耗時(ms)為前端量測,留待 T5 接線後在瀏覽器量(離線 harness 不含 NER/onnx)。

## T1 — 準備 A/B query 集(可與 T2 並行,離線) ✅ 完成 2026-06-22

- [x] **Task:** 整理評估用 query 集,含語意型 + 關鍵字型兩 bucket,附 ground-truth(沿用 T0 join)。
  - **Acceptance:** ✅ `tests/fixtures/ab_eval_queries.json` —— 278 query(semantic 78 / keyword 200),
    每 query 有 `relevant_idxs`(T0 build_ground_truth 的 property_data idx,T7 免再 join)。
  - **Verify:** ✅ 可被未來 harness 載入;builder 確定性(md5 穩定);78/78 semantic 經獨立驗證
    在 K=30 確實 miss ≥1 相關(真實 blind spot)。
  - **Files:** `tests/fixtures/ab_eval_queries.json`、`tests/build_ab_eval_queries.py`(可重跑 builder)。
  - **關鍵發現(修正專案假設):** 召回階段**有** `expandQueryIntent`(INTENT_MAP,inference.js:1020
    `extractKeywords` 內呼叫),並非無語意擴展;只有 `semanticExpandQuery`(semantic_rules.json)
    是 CE 層才用。故 semantic bucket 用**經驗 blind-spot gate**:trigger 在 + INTENT_MAP 擴展後
    baseline 仍 miss(K=30),才算真 blind spot。單純 trigger membership 不夠(會被 INTENT_MAP 蓋掉)。
  - **Caveat:** semantic=78(達標下緣,未灌水);join match-rate 24.4%(沿用 T0);trigger 以「怕熱」為大宗。

---

## T2 — bi-encoder 訓練腳本(鏡像 trainer.py) ✅ 腳本完成 2026-06-22(GPU 訓練待你跑)

- [x] **Task:** 鏡像 `trainer.py` 寫 bi-encoder 對比學習:CE 同源 base(hfl/rbt6)、
  shared-weight encoder、mask-aware mean-pool + L2-norm(同源前端 cosine)、
  InfoNCE/MNRL loss(label=1 正 + is_hard 硬負 + in-batch negatives)。
  - **Acceptance:** ✅ 腳本 + Colab notebook 完成,可跑;`label`/`is_hard` 字串/原生型皆容錯
    (`_as_int`/`_as_bool`);資料計數正確(7022 正 / ~2130 硬負)。
  - **Verify(已做):** ✅ py_compile 通過;helper 邏輯獨立驗證;loss/pool 核心邏輯靜態審過
    (mean-pool div-zero 守衛、L2-norm、encode=forward 供 T3 匯出復用);Colab notebook 合法 JSON。
  - **Verify(待你在 GPU 跑):** 完整 train loss 下降 + dev 召回 margin + CP1(NDCG 不崩)。
    **本機無 torch,無法實跑訓練;agent 回報 sanity run loss 2.48→1.04、unit-norm OK、
    dev pos/neg cosine margin 0.15 —— 為 agent 環境結果,待 Colab 實證。**
  - **Files:** `pipeline/model_training/train_bi_encoder.py`(新,516 行)、`config.py`(additive:
    `bi_encoder_saved_dir` / `bi_encoder_temperature`)、`colab_train_bi_encoder.ipynb`(新,GPU 訓練)。
  - **負樣本構造:** anchor=有 label=1 房源的 query;positive=該房源;in-batch negatives=同批其他
    anchor 的 positive(MNRL);hard negatives=同 query 的 is_hard 房源(deduped、capped 2×B、批內共享)。
  - **T3 待辦:** docstring「NOTE FOR T3」已標明匯出 query 路徑(input_ids+attention_mask → mean-pool
    → L2-norm;dynamo=False;opset 15;先 `Exporter._apply_onnx_monkey_patch()`;pool+norm 進圖內)。
  - **Caveat:** 未加 sentence-transformers(守 spec boundary),bi-encoder 直接用 transformers 手刻。

## T3 — query 端 ONNX 匯出 + 量化(依賴 T2;可與 T4 並行)

- [ ] **Task:** 沿用 `exporter.py`(dynamo=False)匯出 bi-encoder query 端 ONNX,
  `quantize_model.py` 量化,放 `frontend/models/bi_encoder_dir/`。
  - **Acceptance:** 量化 ONNX 產出;query encode 輸出與 PyTorch 數值一致(cosine sanity)。
  - **Verify:** CP2 —— 同一 query 過 PyTorch vs ONNX,cosine ≈ 1。
  - **Files:** `pipeline/model_training/exporter.py`(沿用/小改)、`frontend/models/bi_encoder_dir/*`(產出)

## T4 — 房源 embedding 預算(依賴 T2;擴充 embedder.py)

- [ ] **Task:** 擴充既有 `pipeline/data_prep/embedder.py`:base 覆寫為訓好的 bi-encoder、
  輸出前端靜態檔(鏡像 `build_intent_prototypes.py` 的 Float32 + meta 格式)。
  - **Acceptance:** `frontend/assets/property_embeddings.json` 產出,含 dim + flat Float32 + 房源 id 對應;
    與 query 端**同源**(同 model/pool/L2 norm)。
  - **Verify:** CP3 —— 已知相似 query/房源 cosine 偏高;檔大小符合 T0 定的門檻。
  - **Files:** `pipeline/data_prep/embedder.py`(擴充)、`config.py`(小改)、
    `frontend/assets/property_embeddings.json`(產出)

---

## T5 — 前端召回函式 + 接線(依賴 T3 + T4)

- [ ] **Task:** `inference.js` 加 `cosineTopK`(鏡像現有同源 cosine),`recommend()` 召回階段
  用 query ONNX encode → `cosineTopK(K=30)` 取代 `calculateRuleBasedScore` 的召回。
  - **Acceptance:** 召回走向量;**rule-based 以 feature flag 保留可切換**(預設可先關)。
  - **Verify:** CP4 —— happy-path + 邊界(空候選 / K>候選數);手動跑幾條 query 看推薦合理。
  - **Files:** `frontend/js/inference.js`、(必要時)`frontend/js/inference-worker.js`

## T6 — 回歸驗證(依賴 T5)

- [ ] **Task:** 確認換召回後既有行為不退化:否定意圖、字卡價格/標題正確性等。
  - **Acceptance:** 既有相關測試全綠;手動驗證否定意圖 query 仍正確。
  - **Verify:** CP4 —— `pytest tests/ -v` 全綠 + 前端手動 spot check。
  - **Files:** (多為驗證,必要時補測試)`tests/*`

---

## T7 — A/B 評估(go/no-go gate,依賴 T5 + T1)

- [ ] **Task:** 寫 `tests/eval_vector_vs_rulebased.py`:用 T1 query 集比
  向量召回 vs rule-based 的 **Recall@K** + 端到端 **NDCG@5**;並模擬 ~1萬筆量可擴展性。
  - **Acceptance:** 產出 A/B 數字表(兩種召回 × Recall@K/NDCG@5),含語意類 query 分項。
  - **Verify:** CP5 —— 對照 Success #1/#2/#3/#4。**這是 go/no-go:**
    過 → 進「移除 rule-based 路徑」收尾;不過 → 保留切換、記錄結論、回頭調 T2/T4。
  - **Files:** `tests/eval_vector_vs_rulebased.py`(新)

---

## 收尾(僅在 T7 通過後)

- [ ] **Task:** A/B 過關後移除/封存 rule-based 召回路徑,更新 spec 狀態為已落地。
  - **Acceptance:** rule-based 召回路徑移除或明確封存;spec/CHANGELOG 同步。
  - **Verify:** 全測試綠;前端正常。
  - **Files:** `frontend/js/inference.js`、`docs/spec/vector-retrieval.md`、`CHANGELOG.md`

---

## 相依圖速覽

```
T0(gate) ─┐
T1 ───────┼─(並行)
T2 ───────┴─→ T3 ┐
              T4 ┴─→ T5 → T6
                         T5+T1 → T7(go/no-go)→ 收尾
```

- 並行可做:T1 ‖ T2;T3 ‖ T4。
- 順序硬相依:T2 先於 T3/T4;T5 在 T3+T4 後;T7 在 T5+T1 後。
- **T0 是 gate**,T7 是 **go/no-go gate**。
