# 階段④ 泛化評估方法論 + 本機跑法

> bi-encoder 召回效果的客觀衡量。重點:**指標要選對**,否則製造假象。

## 為什麼需要這份 doc

階段④四輪重訓中,一度因 holdout 手判「74%」誤判模型退步,差點不上線最佳版本。
真相是**用錯指標**:設施類單訴求是大桶(balcony 656 / elevator 596 間),硬套
Recall@K / TOP5 單欄位判斷會失真。**統一評估集**(指標選對)糾正了這個錯誤。

教訓:指標選錯會製造假象,比沒有指標更危險。

## 三種評估集(各有用途)

| 評估集 | fixture | query | GT | 指標 |
|---|---|---|---|---|
| 階段① A/B | `tests/fixtures/ab_eval_queries.json` | 真實混雜(278) | 少數強相關(人標) | Recall@K + NDCG(歷史對比) |
| 真 GT | `tests/fixtures/true_gt_eval.json` | 複合多訴求(12) | OSRM 交集小桶 | Recall@K |
| **統一**(主力) | `tests/fixtures/unified_eval.json` | 單訴求+複合混合(14) | property_data 客觀算 | **每筆標對的指標** |

### 統一評估集的核心:指標選對

每筆 query 標 `metric`,harness 按它算對的指標:

- **`metric=recall`**(小桶:距離/價格/複合交集,6 題)→ **Recall@K**(召回率)。
  GT 桶小(8-25 間),Recall@K 不會被數學上限壓失真。
- **`metric=precision`**(大桶設施:balcony/elevator/window/quiet/透天/冷氣/停車/便宜,8 題)
  → **Precision@K**(TOP K 命中該特徵的比例 = 純度)。設施桶大(數百間),
  Recall@K 失真,Precision@K 才是「TOP K 推得準不準」的直覺指標。

**總分 = (Recall 均值 + Precision 均值) / 2**,越高越平衡。這是**跨輪唯一可比的綜合指標**。

## 本機跑法(CPU venv,不需 torch / Colab)

bi-encoder query 編碼只需 onnxruntime + tokenizer(**不需 PyTorch**),故本機 CPU 可跑。
專案預設 python3.12 環境無這些套件 → 建一個臨時 venv。

```bash
# 1. 建 venv 裝 onnxruntime + transformers(CPU,約 1-2 分鐘)
python3.12 -m venv /tmp/eval_venv
/tmp/eval_venv/bin/pip install onnxruntime transformers tokenizers numpy

# 2. 確認前端有當前向量(frontend/assets/property_embeddings.json
#    + frontend/models/bi_encoder_dir/bi_encoder_quant.onnx)

# 3. 統一評估(從專案根的 worktree 目錄跑)
/tmp/eval_venv/bin/python tests/eval_generalization.py --unified --k 30

# 其他評估集:
/tmp/eval_venv/bin/python tests/eval_generalization.py --eval-set tests/fixtures/true_gt_eval.json --k 30
/tmp/eval_venv/bin/python tests/eval_vector_vs_rulebased.py          # 階段① A/B + GO/NO-GO
```

> `/tmp/eval_venv` 重開機會消失,重建即可(步驟 1)。venv 路徑可自訂。
> 結構自我驗(無模型,純 stdlib):`python3 tests/eval_generalization.py --check`

### 比多輪向量(選最佳上線版)

把各輪 `property_embeddings.json` + `bi_encoder_quant.onnx` 成對換進
`frontend/assets/` + `frontend/models/bi_encoder_dir/`,各跑一次 `--unified`,比總分。
跑完用 `git checkout` 還原成上線版,避免未上線檔殘留。

## 各輪完整結果(2026-06-24,K=30,本機 venv 一次性核實)

`兩兩cos` = 房源 embedding 前 100 筆兩兩 cosine 均值(向量塌縮指標,越低越分散越好)。

| 輪次 | emb | 兩兩cos↓ | Recall@30(小桶/複合) | Precision@30(設施) | 總分 | 上線 |
|---|---|---|---|---|---|---|
| 第二輪 | `8d7f447` | 0.7848 | 0.377 | 0.858 | 0.6175 | 曾上線 |
| 第三輪 | `e740203` | 0.7244 | 0.560 | 0.813 | 0.6861 | — |
| **第四輪** | **`f46fc64`** | **0.7033** | **0.624** | 0.854 | **0.7390** | ✅ **上線(最佳)** |
| 五輪 run1(加權+ep5) | `04b3860` | 0.7446 | 0.471 | 0.833 | 0.6523 | — |
| 五輪 run2 | `4495a2d` | 0.7185 | 0.525 | 0.829 | 0.6773 | — |
| 五輪 ep3+seed | `776e031` | 0.8079 | 0.525 | 0.854 | 0.6897 | — |
| 六輪(無加權+複合) | `1b72c64` | 0.7307 | 0.488 | 0.821 | 0.6542 | — |

## 結案:停在第四輪(0.7390)

**第四輪是天花板。** 二→三→四輪單調進步(0.618→0.686→0.739);第五輪起連續 4 次
嘗試(加權 / 降 epoch / 固定 seed / 移除加權+複合增強)**沒有一次追上第四輪**。

**根因:向量塌縮 ↔ 總分強負相關**(第四輪 cos 0.7033 最低 → 總分 0.739 最高;
其餘輪 cos 都 >0.71 → 總分都 <0.70)。塌縮 = 房源 embedding 擠成一團,cosine
檢索分不出遠近 → 召回變差。試過的對策與結果:

| 對策(第五/六輪) | 結果 |
|---|---|
| 維度加權 2.0-2.5(推弱項) | ❌ 加權正是塌縮元兇(cos 升到 0.745/0.808) |
| epoch 5→3(抑過擬合) | ❌ 反而更塌(0.808) |
| 固定 torch/cuda seed | ✅ 消除 run 間波動(可重現),但救不了塌縮 |
| 移除加權 + 補複合 query | ⚠️ 塌縮回 0.731(仍 >第四輪 0.703),總分 0.654 |

**結論:第四輪的「溫和加權 + 1266 pair」恰是向量分散度與訓練訊號的最佳平衡點。**
此資料規模 + 57MB INT8 模型容量下,0.739 是真實上限,繼續加料/調參為負報酬。
階段④目標(破假泛化、Recall@30 相對 +66%、holdout 隱喻召回正確、建立可信統一評估)
已全數達成。後續若要再推,需換更大 encoder(離開 edge,屬另一階段)或大幅擴真實房源。

## 第二輪實證(2026-06-27,擴量 974 後)— 確認 encoder 容量是瓶頸

第四輪結案時推論「需換更大 encoder」。擴量到 974 房源後重啟第二輪,用**獨立**評估集
(`tests/fixtures/ab_eval_queries.json`,278 query,非重訓同源)做最嚴格驗收,實證了那個推論:

| 重訓配置 | ab_eval all Recall@30 | 說明 |
|---|---|---|
| 現役 rbt3(未重訓) | **0.26** | 基準 |
| v1:只餵 generalization(取代訓練資料) | 0.073 | 過擬合生成模板 |
| v2:合併原+破模板 + 全 14 維度跨域類比 | **0.047** | 排除所有混淆變數仍退步 |

**趨勢:越破模板越退步。** 每次重訓都 semantic 進步、keyword 大退(v2 keyword R@30 −0.054)
→ 口語/跨域合成 query 把 bi-encoder 向量空間拉向「語意相似」、犧牲「字面區辨」
(向量塌縮的另一形式)。`unified`/`holdout` 看似進步(0.509 / NDCG 0.792)是**同源假象**
(那兩個評估集用 gen_generalization 維度生成,與重訓資料同源)。

**v2 已排除所有可疑變數**(合併不取代、全維度跨域、974 客觀 GT 都做對),結果仍退步 →
**問題不在資料/評估,在 encoder 容量。破模板資料增強這條路到此實證為死路,勿再試。**
唯一出路:換更大/各向同性 encoder(離開 edge,屬另一階段)。現役 rbt3 維持不動(ab_eval 0.26 最佳)。

## 同源 caveat

統一評估的 query 為手寫(非訓練同源,破 selection bias),但仍是少量人造樣本;
數字作**跨輪相對比較**與**趨勢判讀**,不宣稱絕對泛化。GT 全 `property_data` 欄位
/ OSRM 客觀算,非憑感覺標。
