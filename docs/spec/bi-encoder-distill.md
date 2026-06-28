# Bi-Encoder 蒸餾(rbt6 → rbt3)— Colab 執行手冊

> 目標:把現役 6 層 bi-encoder 蒸餾成 3 層 student,前端 `bi_encoder_quant.onnx`
> 由 ~57 MB 降到 ~38 MB(比照 Cross-Encoder 已做的 rbt6→rbt3),縮短冷載入。
> hidden_size 維持 768 → embedding 維度與 `property_embeddings.json` schema **不變**。

## ✅ 已完成(2026-06-24,落地 main)

蒸餾跑完並上線。Colab 走完整鏈(重訓 rbt6 teacher → 蒸 rbt3 → export → 重編房源向量 → A/B gate)。

| | 結果 |
|---|---|
| `bi_encoder_quant.onnx` | 59.8 → **38.2 MB(−36%)** |
| Production wire(Vercel brotli) | 43.5 → **~27.9 MB(−36%)** |
| Recall@15 all | 0.2975 → 0.2883(−0.009,容忍內) |
| Recall@30 all | 0.3991 → **0.4000**(微升) |
| NDCG@5 all | 0.1769 → **0.1830**(升) |
| vs rule-based | 仍 **GO** |

benchmark.html 本機量測(localhost,非真實網速):bi-encoder 冷載入 9107(舊 rbt6,25Mbps 螢幕截圖)
→ 本機 rbt3 冷 ~3790ms;真正可比的是**位元組 −36%**(網速無關)。

---

## 為什麼需要 GPU / Colab

- 本機無 torch/GPU,訓練跑不動;`build_property_embeddings` 在程式碼中明文標記為 Colab 段。
- **換模型 = 房源向量全部作廢**:student 是新的向量空間,現役 974 房源向量(用舊 rbt6 編)
  與 student 產生的 query 向量不可比。**必須用 student 重編房源向量**,否則 cosine 召回失效。

## 前置

- **teacher 權重狀況(2026-06-24)**:bi-encoder 的 PyTorch teacher(`saved_models/rbt6_bi_encoder/`)
  **不在 repo / 本機**,只剩前端量化後的 onnx(不能當 teacher)。故流程從**重訓 rbt6 teacher**
  開始(step 1)。蒸餾對象是這個重訓的 6 層 bi-encoder,不是直接拿 hfl/rbt6 預訓練權重。
- 訓練資料:`data/processed/recommendation_train.json`(label==1 共 7022 筆 anchor)。

## 流程(全在 Colab GPU)

```bash
# 0. 環境
pip install torch transformers tokenizers onnx onnxruntime

# 1. 重訓 rbt6 teacher(從 hfl/rbt6;teacher 權重已遺失,必跑)
python -m pipeline.model_training.train_bi_encoder \
    --epochs 3 --batch-size 32 --lr 2e-5 --bf16 --tf32
# → saved_models/rbt6_bi_encoder/   (6 層 teacher 權重 + tokenizer)
# 記下這次 teacher 的 dev recall_at_1_vs_pool 當蒸餾 gate 基準。

# 2. 蒸餾:rbt6 teacher → rbt3 student
#    loss = α·cos-distill + (1−α)·MNRL,student = teacher embeddings + 前 3 層(截斷初始化)
python -m pipeline.model_training.distill_bi_encoder \
    --epochs 3 --batch-size 32 --lr 3e-5 --alpha 0.5 --bf16 --tf32
# → saved_models/rbt3_bi_encoder/   (student 權重 + tokenizer)
# 觀察 log:Last step loss < First、mean_student_teacher_cosine 高(≳0.9 = 沒塌)、
#         recall_at_1_vs_pool 不低於 step1 teacher 太多。

# 3. Export student query encoder → ONNX + INT8(沿用既有 T3,不需改)
python -m pipeline.model_training.export_bi_encoder \
    --saved-dir saved_models/rbt3_bi_encoder
# → frontend/models/bi_encoder_dir/bi_encoder.onnx + bi_encoder_quant.onnx(~38 MB)
#   （pooling + L2-norm 已在圖內,與 student 同源）

# 4. 用 student 重編 974 房源向量(關鍵!不可省)
python -m pipeline.data_prep.build_property_embeddings \
    --saved-dir saved_models/rbt3_bi_encoder
# → frontend/assets/property_embeddings.json（dim 仍 768，count 不變）

# 5. Recall 守門:vector vs rule-based A/B(守向量塌縮天花板,見 STAGE4_EVALUATION.md)
#    這支讀 step3 的 onnx + step4 的 property_embeddings,等同驗證落地鏈。
python tests/eval_vector_vs_rulebased.py
# 比較 student 向量召回 vs rule-based 的 recall@K / NDCG@5,per-bucket + overall。
# (此檔設計為直接執行,非 -m 模組;tests/ 無 __init__.py)
```

## Gate(過了才落地)

| 指標 | 門檻 | 出處 |
|---|---|---|
| `Last step loss < First step loss` | 必須 | step2 訓練 log(本機 smoke test 已驗證會降) |
| `mean_student_teacher_cosine` | ≳ 0.90 | step2 distill eval(低 = 向量塌縮,student 偏離 teacher) |
| `recall_at_1_vs_pool` | ≥ step1 teacher − 0.02 | step2 distill eval vs step1 teacher baseline |
| `eval_vector_vs_rulebased` recall@K / NDCG@5 | 不低於現役向量召回容忍內 | step 5,守天花板 |
| `bi_encoder_quant.onnx` 大小 | ≈ 38 MB(較 57 MB ↓) | step 3 輸出 |

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
