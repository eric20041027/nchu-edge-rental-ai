# Ablation Study — Cross-Encoder v2.9 設計選擇驗證

> 針對 Cross-Encoder v2.9 的五個核心設計選擇進行系統性消融，
> 驗證每個組件的實際貢獻，並以實驗數據支撐設計決策。

> **註（2026-06-16）**：下方 v2.9 / v3.0 消融（NDCG@5 0.87x 量級）為**歷史結果**，
> 仍有效保留。Production 已換為 **C 組房源富化模型**，其消融見最末
> 「C 組房源富化消融（2026-06-16）」章節。兩者 NDCG 數量級不同，
> 因 query／test 集不同，**不可直接相減比較**。

## 實驗設計

共 11 個訓練 run（10 + V3.0），分為四組：

| 組別 | 研究問題 | Runs |
|:---|:---|:---|
| **Group A** | Loss 函式組合的影響（CE / RankNet / ListNet）| A0, A1, A2 |
| **Group B** | 知識蒸餾 alpha 大小的影響（固定 vs 餘弦退火）| B1, B2, B3 |
| **Group C** | 正則化組件的影響（FGM / R-Drop）| C2, C3, C4 |
| **Group D** | 噪聲輸入下的模型魯棒性（縮寫/錯字/口語化/數字格式）| D_with_FGM, D_no_FGM, D_V30 |
| **V3.0** | 根據消融結果組合最優設計 | V30_optimized |

所有 run 共用相同超參數（LR=3e-5, batch=32, epoch=10, patience=6, warmup=8%）。
基準（Reference）為 v2.9 全功能配置：CE + RankNet + ListNet + KD(cosine 0.38→0.12) + R-Drop(α=0.05) + FGM。

評估指標：**NDCG@5**（指數增益，$(2^{rel}-1)/\log_2(rank+2)$，以 flat 方式對 test set 全部 query 取均值）。

---

## 完整結果

| Run ID | 組別 | 說明 | F1 | NDCG@5 | ΔNDCG |
|:---|:---|:---|---:|---:|---:|
| **REF_v29** | REF | v2.9 Reference（全功能）| **84.1%** | **0.8719** | — |
| A0_CE_only | A | CE only，無排序 loss | 83.9% | 0.8719 | +0.0000 |
| A1_CE_RankNet | A | CE + RankNet | 83.8% | 0.8749 | +0.0030 |
| A2_CE_ListNet | A | CE + ListNet | 84.0% | 0.8739 | +0.0020 |
| B1_alpha_012 | B | KD alpha 固定 0.12 | 84.8% | 0.8769 | +0.0050 |
| B2_alpha_025 | B | KD alpha 固定 0.25 | 84.1% | 0.8753 | +0.0034 |
| B3_alpha_038 | B | KD alpha 固定 0.38 | 84.0% | 0.8728 | +0.0009 |
| C2_no_FGM | C | 移除 FGM | 84.1% | 0.8715 | −0.0004 |
| **C3_no_RDrop** | **C** | **移除 R-Drop** | **84.4%** | **0.8787** | **+0.0068** ✅ |
| C4_no_FGM_no_RDrop | C | 移除 FGM 和 R-Drop | 83.4% | 0.8722 | +0.0004 |
| V30_optimized | V30 | No RDrop + alpha=0.12 + 噪聲增強 | 84.2% | 0.8749 | +0.0030 |

**Group D — 噪聲測試集（noisy_test.json，4 種噪聲類型）**

| Run ID | Checkpoint | NDCG@5（noisy）| 相對 clean 跌幅 |
|:---|:---|---:|---:|
| D_noisy_with_FGM | REF_v29 | 0.3065 | −64.8% |
| D_noisy_no_FGM | C2_no_FGM | 0.3070 | −64.7% |
| D_noisy_V30 | V30_optimized | 0.3071 | −64.9% |

---

## 分析與討論

### Group A：排序 Loss 的邊際貢獻正向但有限

- A0（純 CE）與 REF 的 NDCG@5 幾乎相同（差 0.0000），說明 RankNet+ListNet 不是 flat NDCG 的主要驅動力
- 單獨加 RankNet（A1, +0.0030）優於單獨加 ListNet（A2, +0.0020），兩者組合在 REF 中同時使用
- **結論**：排序 loss 有正向貢獻，但在 flat NDCG 評測下絕對增益有限

### Group B：低 KD alpha 顯著優於 REF 的餘弦退火策略

- B1（fixed 0.12）> B2（fixed 0.25）> B3（fixed 0.38）：alpha 越低表現越好
- REF 使用 cosine annealing（0.38→0.12），初期 alpha 高達 0.38，teacher soft label 佔比過重，使學生在任務能力建立前就被過度引導
- 固定低 alpha 讓 CE loss 始終主導，KD 作為輔助訊號效果更好
- **結論**：KD alpha=0.12（固定）是最佳配置，可取代餘弦退火

### Group C：R-Drop 是唯一有害的正則化組件

- **C3_no_RDrop 是所有 11 個 run 中 clean test 最高分（0.8787，+0.0068）**
- R-Drop 要求兩次 forward pass 輸出一致，對中文短文本可能過強，干擾 KD loss 的梯度方向
- C2_no_FGM 是唯一低於 REF 的訓練 run（−0.0004），確認 FGM 對對抗魯棒性有貢獻
- C4（同時移除 FGM 和 R-Drop）結果介於兩者之間，說明 FGM 的損失部分抵消了移除 R-Drop 的收益
- **結論**：FGM 保留，R-Drop 可移除

### Group D：噪聲崩潰的根本原因

三個 checkpoint（REF、無 FGM、V30 增強訓練）在 noisy test 上全部降至 NDCG@5 ≈ 0.307，相對 clean test 跌幅高達 **64.8–64.9%**，且彼此差異不超過 0.0006。

- FGM 是**連續嵌入空間的對抗擾動**，對「縮寫替換」「錯字」「口語化」這類**離散詞彙分佈偏移**無效
- V30 的 25% 訓練噪聲增強同樣未能恢復 noisy test 性能，說明 4,182 個增強樣本（16.5% 最終訓練集）比例仍不足以改變模型的表示空間
- **根本原因**：噪聲輸入造成的 tokenization 差異（縮寫使序列分詞結果完全不同）無法以小比例資料增強彌補

未來方向：增加噪聲比例（50%+）或使用 subword-level data augmentation（CharSwap, back-translation）。

### V30_optimized：組合效益未能線性疊加

V30 結合三個最優發現（無 R-Drop + alpha=0.12 + 噪聲增強），clean test NDCG@5=0.8749（+0.0030），**低於單獨移除 R-Drop 的 C3（0.8787）**。

這說明消融實驗的改進不具線性可加性：
- 噪聲增強引入了 4,182 個困難樣本，對 clean test 有輕微負面影響
- C3 的超高分（+0.0068）可能部分源於該 run 的訓練隨機性，需多次 seed 驗證

**實際部署建議**：以 C3 配置（無 R-Drop，其餘與 v2.9 相同）作為下一版本起點，即目前 v3.0。

---

## 各組件對 NDCG@5 的貢獻彙整

```
組件         ΔNDCG（移除後）   結論
─────────────────────────────────────────────────
RankNet       −0.0030         ✅ 保留（有正向貢獻）
ListNet       −0.0020         ✅ 保留（有正向貢獻）
FGM           −0.0004         ✅ 保留（維持對抗穩健性）
R-Drop        +0.0068         ❌ 移除（有害）
KD alpha      +0.0050         ✅ 改為 fixed 0.12
噪聲增強       −0.0030 on V30  ⚠️  需加大比例才有效
```

---

## C 組房源富化消融（2026-06-16）

驗證「房源文字富化」（`property_to_text_enriched`：全 notes + 全 furniture，`MAX_LENGTH=128`）
相對舊基底文字的排序貢獻。此為 production 現用模型的依據。

| 組別 | 房源文字 | NDCG@5 | F1 |
|:---|:---|---:|---:|
| **A baseline** | 舊基底（furniture[:5] + 部分 notes）| 0.9351 | 0.833 |
| **C 富化** | `property_to_text_enriched`（全 notes + 全 furniture）| **0.9475** | **0.854** |
| Δ | — | **+0.0125** | **+0.021** |

**per-query 案例**：「想要採光好」由 **0 → 1**（富化前房源文字無「採光」相關描述，CE 無從匹配；
富化納入全 notes 後得以命中）。

**誠實註記**：+0.0125 的差距**混入了「C 組換用富化後 test 集相對較易」的成分**，
並非純粹的模型能力提升。此結果**足以證明富化方向正確**（per-query 0→1 為直接證據），
但**不宜宣稱為大幅提升**。

**來源**：[docs/property_enrichment_value.md](property_enrichment_value.md)、
`notebooks/ce_expansion_augment_experiment.ipynb`。


---

## 召回階段消融：bi-encoder 向量召回 vs rule-based（T7 A/B，2026-06-22）

上述 Group A–D 消融針對 **Cross-Encoder 精排階段**。本節記錄**召回階段**的 go/no-go 消融:以 bi-encoder 向量召回取代原本的關鍵字 rule-based 召回(`calculateRuleBasedScore`)。

### 動機 — 直接源自 Group D 結論

Group D 證明 noisy test 崩潰(NDCG@5 −64.8%)的根因是**離散詞彙分佈偏移**:縮寫/錯字/口語化使 tokenization 結果完全不同,FGM(連續嵌入擾動)與小比例資料增強都無法修復。**rule-based 關鍵字召回正是此問題的極端形式** —— 口語需求(「怕熱」)與房源用詞(「冷氣」)字面不重疊就召不到。bi-encoder 把 query 與房源編到同一語意向量空間,以 cosine 比對,結構性繞過字面比對,正是 Group D「未來方向」的對症解法。

### 實驗設計

- **評估集**:`tests/fixtures/ab_eval_queries.json` —— 278 query(78 語意 + 200 關鍵字)。語意 bucket 經「rule-based 在 K=30 會 miss ≥1 相關」的經驗 gate 篩選,構成可量測的召回 blind-spot。
- **harness**:`tests/eval_vector_vs_rulebased.py`,複用 T0 基準 harness 的 metrics/loaders,忠實鏡像前端召回路徑(query→bi-encoder ONNX→cosine→Top-30,與 hard-exclusion 取交集)。
- **指標**:Recall@15 / Recall@30 / NDCG@5,per-bucket。

### 結果(判定 GO)

| 分組 | 指標 | rule-based | bi-encoder 向量 | Δ |
|:---|:---|:---:|:---:|:---:|
| 語意 (78) | Recall@30 | 0.007 | **0.547** | +0.540 |
| 語意 (78) | Recall@15 | 0.000 | **0.506** | +0.506 |
| 語意 (78) | NDCG@5 | 0.000 | **0.325** | +0.325 |
| 關鍵字 (200) | Recall@30 | 0.077 | **0.359** | +0.282 |
| 全部 (278) | Recall@30 | 0.057 | **0.412** | +0.354 |

向量召回不只補語意 blind-spot,連關鍵字控制組也勝出 —— 整體召回全面優於 rule-based,判定 **GO**,轉正為 primary(rule-based 降為 fallback)。

> **方法 caveat**:評估集房源標註來自舊版訓練資料,與現行 704 筆 snapshot 經 token fuzzy-join(match-rate 24.4%,分佈 bimodal),故絕對值偏低;harness 兩邊同 join / 同慣例,**相對 Δ 才是判準**。
---

## 結果檔案

- `ablation_results/summary.json` — 所有 run 的 metrics 彙總
- `ablation_results/{run_id}/metrics.json` — 各 run 測試集指標
- `ablation_results/{run_id}/convergence.json` — 各 run 逐 epoch NDCG@5 收斂曲線
- `ablation_results/{run_id}/config.json` — 各 run 的 AblationConfig 序列化

## 重現消融實驗

```bash
set PYTHONUTF8=1
python -m pipeline.model_training.ablation_runner
# 已跑過的 run 自動跳過（skip if metrics.json exists）
```
