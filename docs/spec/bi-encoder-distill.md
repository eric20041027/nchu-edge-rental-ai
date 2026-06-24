# Bi-Encoder 蒸餾(rbt6 → rbt3)— Colab 執行手冊

> 目標:把現役 6 層 bi-encoder 蒸餾成 3 層 student,前端 `bi_encoder_quant.onnx`
> 由 ~57 MB 降到 ~38 MB(比照 Cross-Encoder 已做的 rbt6→rbt3),縮短冷載入。
> hidden_size 維持 768 → embedding 維度與 `property_embeddings.json` schema **不變**。

## 為什麼需要 GPU / Colab

- 本機無 torch/GPU,訓練跑不動;`build_property_embeddings` 在程式碼中明文標記為 Colab 段。
- **換模型 = 房源向量全部作廢**:student 是新的向量空間,現役 704 房源向量(用舊 rbt6 編)
  與 student 產生的 query 向量不可比。**必須用 student 重編房源向量**,否則 cosine 召回失效。

## 前置

- 現役 teacher 權重:`saved_models/rbt6_bi_encoder/`(config.json + 權重 + tokenizer)。
  這是**部署中**的 rbt6 bi-encoder;蒸餾對象是它,不是新下載的 hfl/rbt6。
  若本機/Colab 無此目錄,先取得 T2 訓練產物或從 T2 重訓。
- 訓練資料:`data/processed/recommendation_train.json`(label==1 共 7022 筆 anchor)。

## 流程(四步,全在 Colab GPU)

```bash
# 0. 環境
pip install torch transformers tokenizers onnx onnxruntime

# 1. 蒸餾:rbt6 teacher → rbt3 student
#    loss = α·cos-distill + (1−α)·MNRL,student = teacher embeddings + 前 3 層(截斷初始化)
python -m pipeline.model_training.distill_bi_encoder \
    --epochs 3 --batch-size 32 --lr 3e-5 --alpha 0.5 --bf16 --tf32
# → saved_models/rbt3_bi_encoder/   (student 權重 + tokenizer)
# 觀察 log:Last step loss < First、mean_student_teacher_cosine 高(≳0.9 = 沒塌)、
#         recall_at_1_vs_pool 不低於 teacher 太多。

# 2. Export student query encoder → ONNX + INT8(沿用既有 T3,不需改)
python -m pipeline.model_training.export_bi_encoder \
    --saved-dir saved_models/rbt3_bi_encoder
# → frontend/models/bi_encoder_dir/bi_encoder.onnx + bi_encoder_quant.onnx(~38 MB)
#   （pooling + L2-norm 已在圖內,與 student 同源）

# 3. 用 student 重編 704 房源向量(關鍵!不可省)
python -m pipeline.data_prep.build_property_embeddings \
    --saved-dir saved_models/rbt3_bi_encoder
# → frontend/assets/property_embeddings.json（dim 仍 768，count 不變）

# 4. Recall 守門:守住向量塌縮天花板(見 STAGE4_EVALUATION.md 教訓)
python -m pipeline.model_training.semantic_benchmark
# 比較 student vs teacher 的 recall@K / NDCG@5。
```

## Gate(過了才落地)

| 指標 | 門檻 | 出處 |
|---|---|---|
| `Last step loss < First step loss` | 必須 | 訓練 log(本機 smoke test 已驗證會降) |
| `mean_student_teacher_cosine` | ≳ 0.90 | distill eval(低 = 向量塌縮,student 偏離 teacher) |
| `recall_at_1_vs_pool` | ≥ teacher − 0.02 | distill eval vs T2 baseline |
| `semantic_benchmark` recall@K / NDCG@5 | 不低於 teacher 超過容忍 | step 4,守天花板 |
| `bi_encoder_quant.onnx` 大小 | ≈ 38 MB(較 57 MB ↓) | step 2 輸出 |

任一不過 → **不落地**,回退現役 rbt6(`git checkout` 那兩個檔)。

## 落地(gate 全過後)

需一起 commit 的**三個**前端產物(缺一前端就壞):

```
frontend/models/bi_encoder_dir/bi_encoder_quant.onnx   # 38 MB student
frontend/models/bi_encoder_dir/bi_encoder.onnx          # （非量化,若有 track）
frontend/assets/property_embeddings.json                # student 重編的房源向量
```

> ⚠️ ONNX 與 property_embeddings **必須同一次蒸餾產出**,版本錯配 = query/property 向量不同空間 = 召回壞掉。

## 本機可驗的部分(CPU smoke test)

無 GPU 時用 `--sample` 跑極小子集,只證明程式邏輯(loss 計算 + 下降 + student 單位範數):

```bash
python -m pipeline.model_training.distill_bi_encoder --sample 32 --epochs 1 --batch-size 8
```

訓練成果(recall）本機驗不了,需 Colab GPU 跑完整步驟。
