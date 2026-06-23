# CE 文字層偏誤修正 — 餵 enriched 文字給 cross-encoder (NO-GO — 已於 2026-06-16 被 C 組重訓方案取代)

**日期**: 2026-06-14
**結論**: ❌ **NO-GO**。cross-encoder 維持餵 `prop.text`,不改餵 buildCEText/enriched 文字。
**重跑**: `python pipeline/data_prep/eval_ce_text_enrichment.py [--sample N]`

> ⚠️ **此 NO-GO 已被取代** — 見下方「## 後記(2026-06-16):NO-GO 已被取代」。本 NO-GO 僅適用於「沿用舊 prop.text-trained 模型」的前提,該前提已不存在。以下原文保留作歷史記錄。

## 後記(2026-06-16):NO-GO 已被取代

本 doc 第 44-46 行自己預言的「唯一正解」——**重訓 CE on enriched 文字、訓練語料 property 端改用同款 enriched 文字、使線上線下一致**——已由 **C 組重訓方案實作落地**,當初的 NO-GO 因此解除。

**C 組做了什麼(訓練 + 推論皆改用富化文字,消除 OOD)**:
- 訓練與打分統一改用 `property_to_text_enriched`(納入全部 notes + 全部 furniture,不砍前 5),訓練語料 property 端與線上一致。
- 推論側不靠前端組裝,而是用 `pipeline/data_prep/precompute_ce_text.py` 在 Python 端 byte-exact 把 `ce_text` 預算進 `property_data.json`;前端 `scorePair(text, prop.ce_text || prop.text)` 直接讀預算欄。
- `MAX_LENGTH` 由 64 改 **128**(富化文字平均 ~98 token,64 會截斷)。
- 模型換為 C 組富化 rbt3 student(38.7 MB,舊模型曾備份為 `.PREV-20260616.onnx`,已於 dead-weight 清理(收尾 B)移除)。

**為什麼這次不再 OOD**:當初 NO-GO 的成因是「拿舊 prop.text-trained 模型去吃加長/改格式的文字」屬分佈外輸入,分數崩壞(「要有陽台」+7.9 → +0.5)。C 組訓練端與推論端**同款 enriched 文字**,線上線下一致,當初記錄的崩壞不復存在。

**結果**:NDCG@5 0.9351 → **0.9475**;「想要採光好」per-query 0 → **1**。

**結論**:此 NO-GO 只適用於「沿用舊 prop.text-trained 模型」前提,前提已不存在。原文(含 A/B 數據與 OOD 診斷)保留作歷史記錄,不刪除。

## 背景:想修什麼
`data_source_misalignment` 殘餘偏誤:最終分 = `rms*35 + CE*65`,CE(`scorePair`)只讀
`prop.text`(短結構字串)。興大 `text` 缺結構欄衍生的特徵詞(對外窗/保全/氣密窗/車位…),
使興大房在 65% 權重的文字層被系統性低估。P1 crawler 對齊 + P2 同義橋已把這些特徵補進
**結構欄**,但 CE 看不到——因為它讀的是 `prop.text` 而非 `buildPropText`。

**嘗試**:新增 `buildCEText`(buildPropText 去重版),把 `scorePair(text, prop.text)` 改成
`scorePair(text, buildCEText(prop))`,讓 CE 看到興大補齊的特徵詞。

## 離線 A/B 驗證(production CE `my_custom_model_quant.onnx` + 同 tokenizer)
8 條特徵 query × dd/nchu 各 80 房,OLD(prop.text)vs NEW(buildCEText):

| | old 均分 | new 均分 | Δ |
|---|---|---|---|
| dd | 0.765 | 0.681 | **−0.084** |
| nchu | 1.867 | 2.165 | +0.299 |

nchu 確實上升,但 **dd 下降**,且分項出現災難:

```
要有陽台   dd   +7.347 → -1.887  (-9.234)
要有陽台  nchu  +3.639 → -0.336  (-3.975)
```

## 為什麼 NO-GO:CE 對文字格式 OOD 敏感,非語意匹配
決定性診斷(「要有陽台」query):
- 房源 enriched 文字**明確含「有陽台」**,CE 卻從 **+7.9 掉到 +0.5**。
- raw text **根本沒提陽台**,反而拿 **+7.9**。
- 連最小擴充(raw + 只加「有陽台」)都掉到 **−1.8**。
- 全精度模型(非量化)重現同樣崩壞(7.87 → 0.49)→ **非 quant 假象**。

→ CE 是用短結構 `prop.text` 格式訓練的,**它在認格式/長度,不是在做關鍵字語意匹配**。
餵任何加長/改格式的文字都屬 OOD,分數變得不可靠甚至反向。option A(全 enriched)
與 option B(最小擴充)皆中此坑。

## 決策
- **scorePair 維持餵 `prop.text`。** inference.js 只留一段註解記錄此 NO-GO,無功能變更。
- 文字層根治的唯一正解是**重訓 CE on enriched 文字**——超出本次範圍,且既往重訓會回歸
  (見 [[retrain_jun13_result]]:704 房源重訓變差已 rollback)。若日後重訓,訓練語料的
  property 端應改用 buildCEText 同款 enriched 文字,使線上線下一致。
- 興大文字層偏誤因此**仍存在**,但已知成因與正解;結構欄/同義橋(P1/P2)已盡力在
  rule-based 35% 層補償。

## 殘留產物
- `pipeline/data_prep/eval_ce_text_enrichment.py` — A/B harness,保留供日後重訓前後比對。
