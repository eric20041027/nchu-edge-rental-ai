# HANDOFF — 新 session 接手 prompt

> 把下方分隔線內整段貼給新 session 即可完全接手當前進度。最後更新:2026-06-22。

---

我在接手 **NCHU edge-rental-ai**(中興大學 Edge AI 租屋推薦系統)專案。請先讀 `docs/spec/vector-retrieval.md`、`docs/spec/vector-retrieval-tasks.md`、`CHANGELOG.md` 與 README 掌握全貌,以下是當前進度摘要。

**專案本質:** 三模型瀏覽器端 ONNX 推論管線 —— NER(條件抽取)→ bi-encoder(向量召回)→ Cross-Encoder(語意精排),全部 INT8、在前端跑、零後端推論。repo 在 `/Users/smallfire/Desktop/nchu-edge-rental-ai`,以 `origin/main` 為準(多 worktree 環境,別被其他 worktree 的 stale 狀態誤導)。

**已完成(都在 main):** 中長期路線「向量檢索召回」**階段①完整落地**。召回從關鍵字 rule-based 升級為 bi-encoder 向量召回(primary),rule-based 降為 fallback(worker 未就緒/編碼逾時 800ms/`VECTOR_RECALL_ENABLED` kill-switch)。走完整個 spec-driven 流程 T0–T7:

- bi-encoder = CE 同源 hfl/rbt6、shared-weight、mean-pool+L2-norm(進 ONNX graph)、InfoNCE/MNRL temp 0.05、訓練資料用 `recommendation_train.json`(label=1 正 + is_hard 負)、召回 K=30。
- **T7 A/B 判決 GO**:語意 query Recall@30 **0.007→0.547**、全部 0.057→0.412。harness `tests/eval_vector_vs_rulebased.py`、查詢集 `tests/fixtures/ab_eval_queries.json`(278 query:78 語意 + 200 關鍵字)。caveat:評估集 fuzzy-join match-rate 24.4%,相對 Δ 為判準。
- 生產模型(git 實檔):bi-encoder `frontend/models/bi_encoder_dir/bi_encoder_quant.onnx` 57.0MB、CE `custom_onnx_model_dir/my_custom_model_quant.onnx` 38.7MB(rbt3)、NER 36.4MB;房源向量 `frontend/assets/property_embeddings.json`(704×768 float16)。
- 前端:`inference.js` 的 `recommend()` 召回段 = `filterHardExclusions`(否定意圖 excludePet 在此,兩召回路徑共用故回歸不退化)→ bi-encoder `cosineTopK(30)` 取交集 → CE 精排;worker `frontend/js/bi-encoder-worker.js`。
- 收尾 + 全文檔同步(README 重寫 + docs/ 7 份補 bi-encoder)都已 merge。

**下一步 = 路線圖階段②:反饋微調**(用 localStorage 累積的 👍/👎 做線上學習,**尚未開始**)。

**環境注意:** 本機是 CPU dev box,無 torch/onnxruntime;預設 python3 是 3.14(onnxruntime 無 wheel),但有 **python3.12** 可建 venv 跑 onnxruntime。模型訓練在 **Colab/A100**(`colab_train_bi_encoder.ipynb`)。

**工作偏好(重要):** ① 一條一條收尾 —— 每件事開分支 → PR → `gh pr merge --squash` → 刪分支;說「merge」就是合掉當前 PR。② 能本機驗的請親自驗(瀏覽器 preview 實跑、複跑 harness、`git cat-file -s` 核實大小),**別照單全收 sub-agent**(它們常在 stale worktree 給錯數字)。③ 不能驗的誠實標註。④ 字面指令與現實衝突先查證再用 AskUserQuestion 確認,別 silently 照做。⑤ 文檔零推測、全 git 查證。⑥ 用繁體中文。

請先確認讀完上述檔案、理解現狀,再等我指示下一步。

---

## 補充:關鍵檔案地圖(給接手者快速定位)

| 用途 | 路徑 |
|---|---|
| 向量檢索 spec / plan / tasks | `docs/spec/vector-retrieval*.md` |
| 路線意圖(階段①②) | `docs/intent/vector-retrieval-roadmap.md` |
| 前端推論主邏輯 | `frontend/js/inference.js`(`recommend()` 召回段)|
| bi-encoder worker | `frontend/js/bi-encoder-worker.js` |
| bi-encoder 訓練 / 匯出 / 房源向量 | `pipeline/model_training/train_bi_encoder.py`、`export_bi_encoder.py`、`pipeline/data_prep/build_property_embeddings.py` |
| Colab 訓練 notebook | `colab_train_bi_encoder.ipynb`(`--bf16 --tf32` A100 加速)|
| A/B harness / 查詢集 | `tests/eval_vector_vs_rulebased.py`、`tests/fixtures/ab_eval_queries.json` |
| rule-based 基準 harness | `tests/eval_rule_based_baseline.py` |
| CE 蒸餾訓練 | `pipeline/model_training/train_teacher.py`、`train_and_export_onnx.py` |
