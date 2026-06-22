# Plan: 向量檢索召回(spec-driven Phase 2)

> Spec:[docs/spec/vector-retrieval.md](./vector-retrieval.md)
> 狀態:**待審核**(Phase 2 — Plan)
> 原則:reviewable。完成後才進 Phase 3 (Tasks)。

## 0. 既有基礎建設盤點(影響範圍:比 spec 假設的少很多)

讀 repo 後發現向量檢索其實**已被 scaffold、只是沒接前端**:

| 已存在 | 路徑 | 用途 | 本計畫怎麼用 |
|---|---|---|---|
| `EmbeddingPrecomputer` | `pipeline/data_prep/embedder.py` | 「precompute property embeddings for vector search」,有 `PropertyEmbedding`/`EmbeddingBatch` model | **擴充**:base 改成 CE 同源、輸出前端靜態 JSON(現預設 MiniLM,需覆寫) |
| CE 訓練器 | `pipeline/model_training/trainer.py`、`train_and_export_onnx.py` | cross-encoder 訓練 + ONNX 匯出 | **鏡像**寫 bi-encoder 訓練(對比學習) |
| ONNX 匯出 | `pipeline/model_training/exporter.py` | dynamo=False legacy tracer | **沿用**匯出 bi-encoder query 端 |
| 量化 | `pipeline/model_training/quantize_model.py` | 動態量化 | **沿用**量化 bi-encoder |
| 硬負樣本 | `pipeline/data_prep/mine_hard_negatives.py`、`model_training/mine_hard_examples.py` | hard negatives | 訓練資料已含 `is_hard`,**多半不需重跑** |
| NDCG 評估 | `model_training/ndcg_callback.py`、`evaluator.py` | 訓練/離線 NDCG | **沿用**做 A/B |
| 靜態向量檔先例 | `data_prep/build_intent_prototypes.py` | Float32 原型向量打包前端 | **鏡像**房源 embedding 打包格式 |
| 前端同源 cosine | `frontend/js/inference.js`(query expansion) | L2 norm + 內積即 cosine | **鏡像**召回階段 cosine |

→ 淨新增程式碼集中在:bi-encoder 訓練腳本、embedder 的輸出/base 覆寫、前端召回接線、A/B 腳本。

## 1. 主要元件與相依

```
[A] bi-encoder 訓練 (離線/Colab)
      └─ 產出:bi-encoder 權重
            │
            ├─→ [B] query 端 ONNX 匯出 + 量化 → frontend/models/bi_encoder_dir/
            │
            └─→ [C] 房源 embedding 預算 (擴充 embedder.py)
                      └─ 產出:frontend/assets/property_embeddings.json
                                  │
[B]+[C] ─────────────────────────┴─→ [D] 前端召回接線 (inference.js)
                                          recommend(): 向量召回 top-30 → CE 精排
                                                  │
[訓練資料 + B/C/D] ───────────────────────────────┴─→ [E] A/B 評估
                                                        向量召回 vs rule-based
                                                        (Recall@K / NDCG@5)
```

**相依鏈:** A → {B, C} → D → E。B 與 C 都只依賴 A,**可並行**。

## 2. 實作順序(依相依,非重要性)

1. **[A] bi-encoder 訓練** — 鏡像 `trainer.py`,用 `recommendation_train.json`
   (label=1 正 + is_hard 負)做對比學習(in-batch negatives)。CE 同源 base。
2. **[B] + [C] 並行**:
   - [B] query 端 ONNX 匯出(`exporter.py` dynamo=False)+ 量化(`quantize_model.py`)
   - [C] 擴充 `embedder.py`:base 覆寫為訓好的 bi-encoder、輸出 `property_embeddings.json`
     (鏡像 `build_intent_prototypes.py` 的 Float32 打包格式)
3. **[D] 前端召回接線** — `recommend()` 召回階段:用 query ONNX encode → `cosineTopK(K=30)`
   取代 `calculateRuleBasedScore` 的召回;**保留 rule-based 可切換**(feature flag),確認前不刪。
4. **[E] A/B 評估** — `tests/eval_vector_vs_rulebased.py`:固定 query 集比 Recall@K + 端到端 NDCG@5。

## 3. 風險與緩解

| 風險 | 影響 | 緩解 |
|---|---|---|
| bi-encoder 召回品質 < rule-based(「試試看」可能失敗) | 核心目標未達 | [E] A/B 先量化;**保留 rule-based 可切換**,數字不過關就回退,不刪舊路徑 |
| CE 同源 base 不適合 bi-encoder 對比學習 | 訓練不收斂/embedding 差 | 先小規模 sanity(少量 epoch 看 train loss / 驗證 NDCG);必要時退回 spec Open 的輕量 base |
| embedding 靜態檔太大,拖垮前端首載 | 違反 edge-first | 768 維 × ~1萬筆 ≈ 評估大小;必要時 float16/量化 embedding;Success #4 門檻把關 |
| query 端 ONNX 與房源 embedding 不同源 | cosine 無意義 | 同 model / mean-pool / L2 norm;[C] 與 [B] 用**同一份**訓好權重 |
| 效能/載入門檻沒先定錨 | 無法判定成功 | Plan 後第一步先量「現況基準」(見 §5 checkpoint 0) |

## 4. 並行 vs 順序

- **順序:** A 必須先(B/C 都依賴它);D 必須在 B+C 後;E 在 D 後。
- **並行:** [B] query ONNX 與 [C] 房源 embedding 預算,訓完即可同時做。
- A/B query 集(spec Open #2)可在 A 進行時**並行準備**(離線、不依賴模型)。

## 5. 驗證檢查點(checkpoint)

- **CP0(Plan 後立即):** 量現況基準 —— rule-based 召回耗時、端到端 NDCG@5、前端首載大小。
  **用來定錨 spec Open #1 的 ms/MB 門檻**。無基準不進 A。
- **CP1(A 後):** bi-encoder 訓練收斂(train loss 下降、離線驗證 NDCG 不崩)。
- **CP2(B 後):** query ONNX 量化後與 PyTorch 輸出 cosine 一致(數值 sanity)。
- **CP3(C 後):** 房源 embedding 與 query 同源驗證(已知相似 query/房源 cosine 偏高)。
- **CP4(D 後):** 前端召回 happy-path + 邊界(空候選 / K>候選數);既有否定意圖/字卡測試全綠。
- **CP5(E 後):** A/B 數字滿足 Success #1/#2;模擬 ~1萬筆滿足 #3/#4。**這是 go/no-go gate。**

## 6. 對應 Spec Success Criteria

| Success | 由哪個 checkpoint 驗 |
|---|---|
| #1 召回不退化(Recall@K) | CP5 (E) |
| #2 下游 NDCG@5 不退化 | CP5 (E) |
| #3 ~1萬筆可擴展性 | CP5(模擬規模) |
| #4 載入可接受 | CP0 定錨 → CP5 驗 |
| #5 行為不回歸 | CP4 |

## 7. 仍待定錨(不阻塞 Tasks 拆解)

- spec Open #1:效能 ms / MB 具體數字 → **CP0 量完現況再定**。
- spec Open #2:A/B query 集來源 → 需含同義/口語/語意類 query;可在 A 階段並行備。
