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

## 四輪結果(2026-06-24,K=30)

| 輪次 | Recall@30(小桶/複合) | Precision@30(設施純度) | 總分 | 上線 |
|---|---|---|---|---|
| 第二輪 | 0.377 | 0.858 | 0.6175 | 曾上線 |
| 第三輪 | 0.560 | 0.813 | 0.6861 | — |
| **第四輪** | **0.624** | 0.854 | **0.7390** | ✅ 上線(`f46fc64`) |

穩定單調進步。第四輪複合/距離召回大勝,設施純度持平 → 總分最高。

## 同源 caveat

統一評估的 query 為手寫(非訓練同源,破 selection bias),但仍是少量人造樣本;
數字作**跨輪相對比較**與**趨勢判讀**,不宣稱絕對泛化。GT 全 `property_data` 欄位
/ OSRM 客觀算,非憑感覺標。
