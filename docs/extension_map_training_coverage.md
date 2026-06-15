# Extension Map 融入訓練資料 — 覆蓋差距分析 (不值得做)

**日期**: 2026-06-15
**結論**: ❌ **不值得做**。extension map 的語意意圖訓練資料**已全部涵蓋**(用不同講法),融入訓練只是補同義字面變體,邊際收益,不值得冒重訓回歸風險(見 [retrain_jun13](../README.md) rollback 教訓)。
**重跑**: 分析腳本見下方「如何重現」;CE query 擴展 A/B 見 `python pipeline/data_prep/eval_ce_query_expansion.py`

## 背景:想做什麼

extension map(`data/semantic_rules.json` → `semanticExpandQuery`)是**推論時**前處理,把口語 query 翻成房源用詞再餵 CE。它**不參與 CE 訓練**(`generate_dataset.py` 沒 import 任何擴展表)。

**提問**:把 extension map 的內容也融入訓練/驗證/測試資料,能否提升 CE 的語意理解能力?

直覺上合理 —— 訓練時若 CE 就見過擴展格式,推論時再餵就不再是 OOD(對齊先前 query 擴展 A/B 發現的 train/inference 分佈偏移,見 [ce_query_expansion_ab 記憶] 與 `eval_ce_query_expansion.py`)。

## 關鍵區分:兩種「融入」做法

| 做法 | 效果 | 評價 |
|---|---|---|
| **A. 把擴展詞拼進 query 文字**(同推論) | CE 學「看到擴展詞→對應房源」 | ⚠️ 只是讓 CE **背下擴展表**,沒提升理解;表外講法(酷暑/悶到爆)照樣不懂 |
| **B. 用擴展表當標籤指引,生成多樣口語 query(不拼擴展詞)** | CE 從口語變體學到語意關聯 | ✅ 才是真理解 —— **但生成器 `generate_dataset.py` 已經在做** |

`Templates.FURNITURE` 已把口語(怕熱/夏天/想省伙食費)直接當「指向冷氣/廚房房源」的 query 生正樣本,這正是做法 B。所以問題收斂為:**extension map 有、但生成器沒涵蓋的口語,補進去值不值得?**

## 覆蓋差距分析(對照實際生成的 9,108 條 query)

把 extension map 105 個 key 依「指向特徵」歸類成語意 cluster,逐一查證訓練語料(實際生成樣本 query,非靜態解析腳本)是否涵蓋:

### 三類缺口,性質完全不同

| 類別 | 數量 | 該補進 CE 訓練? |
|---|---|---|
| **A. 距離類**(走路/騎車到學校→走路10分) | 15 | ❌ **絕對不要** — CE 不該學距離,走 OSRM 路網(見 `data_source_misalignment`、`inference.js` 註解) |
| **B. 疑似語意空白**(女生安全/合租/寵物/晚歸/床/瓦斯) | ~17 | ❌ **查證後不存在** — 見下 |
| **C. 口語字面變體**(語意已涵蓋,缺特定講法) | ~35 | 🔸 邊際收益 |

### B 類查證:疑似空白其實全部已涵蓋

初步靜態解析(只掃 `Templates.FURNITURE`)誤判女生安全/合租/寵物等為「語意空白」。對照**實際生成的 query** 後推翻 —— 生成器用 `extract_features` special 分支 + `build_queries` 的 `situational` 人設句涵蓋了這些語意,只是字面講法跟 extension key 不同:

| cluster | extension key | 訓練語料實際佐證 query |
|---|---|---|
| 女生安全 | 女生住/怕危險/治安 | `女生單獨住,門禁和管理員是必要條件` |
| 合租室友 | 找室友/想合租 | `要租想租雅房 打電腦`(雅房/分租/室友皆有) |
| 寵物口語 | 可貓/養貓/有毛孩 | `毛小孩要跟我一起住,找可養寵物的套房` |
| 瓦斯 | 有瓦斯 | `喜歡自炊,希望房間附廚房或瓦斯爐` |

**結論:extension map 的每個語意意圖,訓練資料都已用人設句/模板涵蓋。沒有任何 CE 完全沒學過的語意空白。**

## 最終結論:為什麼不值得做

1. **沒有真空白可填** — 每個語意意圖訓練已涵蓋,補進去只是 C 類同義字面變體,CE 早已從現有講法學到該語意,diminishing returns。
2. **做法 A 讓 CE 退化成背表;做法 B 生成器已在做。**
3. **重訓風險高** — 上次 704 房源重訓變差已 rollback(`retrain_jun13_result`)。為邊際收益冒回歸風險不划算。
   > 註:房源端富化(enrichment)的 NO-GO 已於 2026-06-16 由 C 組(訓練+推論一致)解除,該次重訓成功(NDCG@5 0.9351→0.9475)。但那是「房源文字富化」這條路;本 doc 討論的「extension map 融入 *query* 訓練」是另一回事,結論仍是不值得做。
4. **真正泛化瓶頸無法靠擴表解決** — 使用者講「酷暑」「悶到爆」這種**表內外都沒有**的講法,擴充字面表救不了。治本是 **bi-encoder fallback**(向量相似度接住表外講法,`ENCODER_FALLBACK_ENABLED` 骨架,見 `encoder_fallback_offline_decision.md`),那是另一條路。

## 如何重現

覆蓋差距用以下邏輯(對照 `data/semantic_rules.json` rules vs `data/processed/recommendation_*.json` 的 query 語料):
1. extension key 依指向特徵歸類成語意 cluster。
2. 對每 cluster,檢查訓練語料是否提到任一指向特徵詞或 key。
3. 距離類(走路10分/騎車10分)單獨歸 A 類排除。
4. 「疑似空白」必須對照**實際生成 query**(非靜態解析腳本)才準 — 生成器有 `Templates.FURNITURE` 外的 special/situational 路徑。

相關:`eval_ce_query_expansion.py`(query 端擴展 A/B,淨中性高方差)、`ce_text_layer_decision.md`(房源端 enrichment 原 NO-GO,已於 2026-06-16 由 C 組訓練+推論一致方案解除)。三者共同描繪 extension map 與 CE 的完整關係:**推論時前處理,不碰訓練,且融入(query)訓練不值得。**(注意:房源端富化 NO-GO 的解除不影響本結論——那是房源文字層的事,與 extension map 融入 query 訓練無關。)
