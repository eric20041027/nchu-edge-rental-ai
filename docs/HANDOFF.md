# HANDOFF — 新 session 接手 prompt

> 把下方分隔線內整段貼給新 session 即可接手當前進度。最後更新:2026-06-28。

---

這是「興大 AI 租屋推薦系統」(nchu-edge-rental-ai) 的接續工作。請先讀 `docs/STAGE4_EVALUATION.md`、`docs/intent/vector-retrieval-roadmap.md`、README 與專案記憶(`~/.claude/projects/.../memory/MEMORY.md` 及相關檔)掌握全貌,別重做已完成的事。repo 在 `/Users/smallfire/Desktop/nchu-edge-rental-ai`,**以 `origin/main` 為準**(多 worktree 環境,別被其他 worktree 的 stale 狀態誤導)。

## 專案現況(2026-06-28,全在 main)
- Edge AI 租屋推薦:NER(條件抽取)→ bi-encoder(向量召回)→ Cross-encoder(語意精排),三個 **rbt3 INT8** 模型在瀏覽器 ONNX Runtime Web 跑,**零後端推論**。production 在 Vercel(`.onnx` 已 brotli 壓縮)。
- **房源 974 筆**(租租通 ddroom 665 + 興大官網 146 + 591 163)。bi-encoder rbt3 **38.2MB**、CE rbt3 38.7MB、NER 36.4MB。房源向量 `frontend/assets/property_embeddings.json`(974×768)。
- **CI 護欄常駐**:`.github/workflows/ci.yml` 兩 job(test 不裝 torch + recall-gate),PR 退步自動擋。

## 已完成的大階段(勿重做)
- **階段⑤ 載入優化**:bi-encoder 蒸餾 rbt6→rbt3,57→38MB 召回零損(#78)。INT4 是死路、brotli production 已生效。
- **階段⑥ 規模化**:CI 護欄(#80)+ 三來源擴量 704→974(#81/#83)。C 階段重訓評估後暫緩(同 stage4 天花板)。
- **架構精簡 5 批次(#85-89)**:死碼 −2000 行、`pipeline/crawlers/shared.py` + `frontend/js/worker-shared.js` 共用層、inference.js 拆 4 模組、eval 腳本移到 `pipeline/evaluation/`。
- **評估集更新到 974 + 修分桶 bug**(#90/#92)。
- **stage4 第二輪【已結案=破模板死路】**:三輪重訓實證最獨立的 `ab_eval_queries` all Recall@30 **越破模板越退**(現役 0.26 → v1 取代 0.073 → v2 合併+全跨域 0.047),`unified`/`holdout` 看似進步是**同源假象**。問題在 **encoder 容量非資料**。現役 rbt3 不動。詳見 `docs/STAGE4_EVALUATION.md`「第二輪實證」段。
- **語意天花板實證 + 設施隱喻結構化 boost(#97–#102,2026-06-28)**:
  - **#98 診斷根因=資料同質性**(95% 房源有陽台/88% 電梯)→ 12 口語意圖僅 4 個有可區辨答案。非模型/encoder。詳見 `docs/intent/semantic-understanding-roadmap.md`。
  - **#99 隱藏含義評估集**(`tests/fixtures/hidden_meaning_eval.json`):Precision@30 為主防 Recall 虛降,GT 客觀算 + 隨機基準對照。
  - **#100/#101 stage4 第三輪**:補特徵進召回 `text` + 設施 pair。實證補 text/重訓都守不住 ab_eval 鐵則(補 text 0.241、重訓 0.202 <現役 0.26);拆元兇=降採樣誤砍 50% semantic GT 正樣本。
  - **#102 設施隱喻結構化 boost【完成】**:口語設施隱喻(不想提水→飲水機、夏天怕電費→台電)走結構化過濾 union(`parseFacilityIntents` + `STRUCTURED_BOOST_ENABLED`),bi-encoder 不動、零模型風險。6 設施標靶 P@30 →1.0。
  - **#97 修 `evaluate_ce_quant.py`** 讀錯 logit(NOT_MATCH→MATCH);現役 CE NDCG@5=0.897。

## 關鍵約束 / 教訓
- **edge-first、零後端**:bi-encoder 召回品質已撞天花板,唯一出路是換更大 encoder(離 edge,屬重大決策,需先評估值不值得放棄零後端)。
- **評估鐵則**:獨立評估集是 `tests/fixtures/ab_eval_queries.json`(現役 all R@30=**0.26**);`unified`/`holdout` 與重訓資料同源會給**假進步**,別只看它們。
- **重訓需 Colab GPU**(本機無 torch);本機可建 python3.12 venv 跑 onnxruntime/eval。預設 python3 是 3.14(onnxruntime 無 wheel)。

## 現役檔案地圖
| 用途 | 路徑 |
|---|---|
| 路線意圖(階段①-⑥) | `docs/intent/vector-retrieval-roadmap.md` |
| stage4 評估結論 | `docs/STAGE4_EVALUATION.md` |
| 前端推論主邏輯 | `frontend/js/inference.js` + `constraint-parser.js` + `property-features.js` + `explainability.js` |
| 三 inference worker + 共用 | `frontend/js/{ner,bi-encoder,inference}-worker.js` + `worker-shared.js` |
| crawler(3 來源)+ 共用 | `pipeline/crawlers/crawler_{591,ddroom,nchu}.py` + `shared.py`;`runners.py` 一鍵 crawl |
| 資料管線(本機段) | `pipeline/build_frontend_data.py`(crawl→富化→property_data.json) |
| 向量重編(Colab) | `pipeline/data_prep/build_property_embeddings.py` |
| 訓練 / 蒸餾 / 匯出 | `pipeline/model_training/{train_bi_encoder,distill_bi_encoder,export_bi_encoder}.py` |
| 評估 harness(recall gate) | `pipeline/evaluation/eval_vector_vs_rulebased.py`(**直接跑非 -m**)+ `eval_generalization.py`(`--unified`/`--check`) |
| 泛化資料生成 | `pipeline/data_prep/gen_generalization.py`(免 API、純算,14 維度特徵驅動)|
| Colab notebooks | `notebooks/{bi_encoder_distill,expansion_reembed,stage4_retrain}_colab.ipynb` |

## 可能的下一步(未開始,擇一,等我指示)
1. 接更多租屋平台 crawler 衝房源量(沿用 `pipeline/crawlers/shared.py` + 階段③管線)
2. 評估「離 edge 換大 encoder」值不值得(突破召回天花板,但放棄零後端)
3. 產品功能(地圖標點:591 座標 100% 可得已驗 `crawler_591.extract_coords`,需擴 CSV schema + 前端)
4. 擴充結構化 boost 設施映射(目前 6 個有區辨力設施;`parseFacilityIntents`)
5. 其他我指定的方向

> 註:「設施隱喻召回」已由 #102 結構化 boost 解決(非重訓);語意理解天花板已實證根因=資料同質性(#98),補資料/重訓死路勿再試。

## 工作偏好(重要)
① **一條一條收尾** —— 每件事開分支 → PR → `gh pr merge --squash` → 刪分支;說「merge」就是合掉當前 PR;CI 綠才 merge。
② **能本機驗的請親自驗**(瀏覽器 preview 實跑、複跑 harness、`git cat-file -s`/`ls` 核實大小),**別照單全收 sub-agent / 好看數字 / agent 報告**(查證:grep 核引用、本機測試)。
③ 不能驗的誠實標註;文檔零推測、全 git 查證。
④ 字面指令與現實衝突先查證再用 AskUserQuestion 確認,別 silently 照做。
⑤ 用**繁體中文**。

請先確認讀完上述檔案 + 記憶、理解現狀,再等我指示下一步。

---
