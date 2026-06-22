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

## T5 — 前端召回函式 + 接線(依賴 T3 + T4) ✅ 完成 2026-06-22(瀏覽器實測過)

- [x] **Task:** `inference.js` 加 `cosineTopK` + bi-encoder worker,`recommend()` 召回階段
  用 query ONNX encode → `cosineTopK(30)` 取代 rule-based 召回,hard-exclusion 先過再取交集。
  - **Acceptance:** ✅ 召回走向量;`VECTOR_RECALL_ENABLED` flag 保留 rule-based 可切換(預設 true)。
  - **Verify:** ✅ CP4 瀏覽器實測 —— 語意 query「南區套房怕熱」→ `[vectorRecall] 30 candidates`
    → TOP1 有冷氣房源(怕熱→冷氣 召回成功);關鍵字 query「預算8000 套房 有電梯」→ 同樣走向量、
    81% top match;零 console error;fallback 路徑完整(worker 未就緒/逾時/flag off → rule-based)。
  - **Files:** `frontend/js/inference.js`、`frontend/js/bi-encoder-worker.js`(新)、`frontend/sw.js`
    (CACHE_VERSION bump + precache)、產物 `bi_encoder_dir/` + `property_embeddings.json`。
  - **關鍵修正:** embeddings 的 idxs 指原始 704 筆,但 propertyData 過濾成 701 → 用 idxToProp Map
    對映原始 prop 物件,hard-exclusion 交集用 reference identity,避開位置錯位 bug。
  - **效能:** 端到端(向量召回 + CE 精排 30 筆)~4.5-4.7s;召回耗時待 T7 細量(spec Open #1)。
  - **Caveat:** bi-encoder quant ONNX 57MB,首載 +61MB(超 Success #4 ≤5MB)。先接 T7 拿 A/B 數字,
    確認向量召回有贏再回頭瘦身(int4 / 共享 CE base)。

## T6 — 回歸驗證(依賴 T5) ✅ 完成 2026-06-22(瀏覽器實測過)

- [x] **Task:** 確認換向量召回後既有行為不退化:否定意圖等。
  - **關鍵發現:** 所有硬否定(rooftop/wooden/haunted/subsidy/**excludePet**)都在
    `filterHardExclusions` 強制執行,而**兩條召回路徑都先跑** filterHardExclusions ——
    向量召回只取代「scoring」階段,否定處理在它保留的「hard-filter」階段,故結構上不會退化。
  - **Verify(瀏覽器實測):** ✅「南區 套房 不要養寵物」→ `[vectorRecall] 30 candidates`
    → top-5 **零** 可養寵物/寵物友善房源(excludePet 在 filterHardExclusions 先濾掉);
    零 console error;30 結果正常渲染。code-trace + 實跑雙重確認。
  - **Scope 說明:** pytest 套件測 Python pipeline(crawlers/dataprep/training),T5 只動
    `frontend/js/`,**不在回歸面**;且本機無 pytest/pydantic/torch 無法跑(誠實標註)。
    相關回歸面是前端,已瀏覽器驗證。
  - **Files:** 無程式碼變更(純驗證 + 文件)。

---

## T7 — A/B 評估(go/no-go gate,依賴 T5 + T1) ✅ 完成 2026-06-22 — 判決 GO(本地實跑)

- [x] **Task:** `tests/eval_vector_vs_rulebased.py` —— 用 T1 query 集比向量召回 vs rule-based
  的 Recall@15/@30 + NDCG@5,per-bucket(semantic/keyword/all),印 GO/NO-GO verdict。
  - **Acceptance:** ✅ harness 完成 —— 複用 T0(metrics/loaders/rule_based_recall/hard-filter)+
    T1 fixture;vector 路徑**忠實鏡像 T5 production**(rbt6 tokenizer max64 單句、ONNX embedding、
    cosineTopK、filterHardExclusions 交集);GO/NO-GO 自動判決(exit 0 GO / 2 NO-GO)。
  - **Verify(已做,rule-based 半邊 + 內部一致性):** ✅ py_compile;`--check` 實跑 rule-based 欄:
    semantic Recall@30 = **0.007**、keyword = 0.077。**獨立驗證這不是 bug 是設計:**
    自跑確認 78 semantic 僅 2/78 在 rule-based top-30 有相關(T1 blind-spot gate 本就如此篩),
    故 rule-based 在 semantic 近零是預期floor —— vector 要打的就是這個。
  - **Verify(已本地實跑,GO ✅):** `python tests/eval_vector_vs_rulebased.py`(python3.12 venv +
    onnxruntime/transformers)。結果 **GO**(exit 0),三條件全 PASS:
    semantic Recall@30 **0.007 → 0.547**(+0.540)、Recall@15 0.000→0.506、NDCG@5 0.000→0.325;
    keyword Recall@30 0.077→0.359;all Recall@30 0.057→0.412。向量召回不只補語意洞,
    連 keyword 控制組也贏 —— 整體召回全面優於關鍵字。**專案前提被數字證實。**
  - **Files:** `tests/eval_vector_vs_rulebased.py`(新)。
  - **判決規則:** GO = vector Recall@K ≥ rule-based(overall + semantic)且 semantic@30 明顯更高。
    GO → 收尾(移除 rule-based + 57MB 瘦身);NO-GO → 保留 flag、記結論、回調 T2/T4。

---

## 收尾(T7=GO 後) ✅ A 完成 2026-06-22

- [x] **Task(A):** 封存 rule-based A/B 框架,向量召回轉正為 primary。
  - **決策:** rule-based **不刪** —— 它是 worker 未就緒/編碼逾時時的 fallback(刪了那些情況會零結果)。
    T7 證明向量贏的是「primary」角色;rule-based 從「對等 A/B 分支」降級為「安全網 fallback」。
  - **Acceptance:** ✅ `VECTOR_RECALL_ENABLED` 註解轉為 kill-switch 語義;recall/fallback 註解更新;
    spec 狀態 → 已落地。零行為變更(僅註解 + 文件)。
  - **Verify:** ✅ node --check 通過;邏輯未動(向量 primary + rule-based fallback 皆如 T5/T6 實測)。
  - **Files:** `frontend/js/inference.js`(註解)、`docs/spec/vector-retrieval.md`(狀態)。
- [x] **Task(B):** 57MB bi-encoder 瘦身評估 + repo dead-weight 清理。
  - **調查結論:** bi-encoder 57MB **已是最優 int8 量化**(= CE 同尺寸:40MB int8 + 16MB embedding
    uint8 + 0.4MB fp32,無漏量化層可撿)。**int4 是死路** —— NER int4 先例 76MB **比 int8 quant 36MB 還大**。
    要再降只剩傷準度的路(l2h128 蒸餾 / 共享 base),已 GO 不急,先不做。
  - **首載 +57MB:** 接受為語意召回大勝(Recall 0.007→0.547)的合理代價;SW 快取後僅首次下載。
  - **做了:** 刪除 repo 內唯一追蹤的 dead model `my_custom_model_quant.PREV-20260616.onnx`(57MB backup,
    0 引用)。其餘 fp32/int4/backup ~621MB 為 **gitignored 本機 clutter**(不在 repo/deploy,不影響首載),
    需要的話本機 `git gc` 或手動清。
  - **Files:** 刪 `frontend/models/custom_onnx_model_dir/my_custom_model_quant.PREV-20260616.onnx`。

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
