# Training Strategy

## 訓練策略總覽

### 關鍵設計一覽

| 技術 | 說明 |
|:---|:---|
| **FGM 對抗訓練** | embedding 注入梯度方向擾動，提升口語輸入泛化性 |
| ~~R-Drop~~ | v3.0 已移除；消融顯示 NDCG@5 +0.0068（C3_no_RDrop 為所有 run 最高分）|
| **metric = "loss"（teacher）** | 多任務損失在 F1 收斂後仍持續下降，loss metric 捕捉更好的排序校準點 |
| **隨機負樣本採樣** | `random.sample(neg_all, n)` 自然混合 ~69% rel=0（硬衝突）+ ~31% rel=-1（輕度不符），維持軟邊界學習信號 |
| **物件級切割** | Train/Dev/Test 按房源分割，測試集房源訓練期間完全未見 |
| **房源富化文字（C 組，2026-06-16）** | 訓練/打分文字來源改用 `property_to_text_enriched`（全 notes + 全 furniture），取代舊基底 `generate_dataset.py` 的 furniture[:5] + notes 只留含「寵物/限」 |

---

## 房源富化文字來源（C 組，2026-06-16）

訓練與打分使用的房源文字已由舊基底切換為 **`property_to_text_enriched`**：

- **全 notes**（不再只保留含「寵物/限」的片段）
- **全 furniture**（不再截斷為 furniture[:5]）

富化後文字更長（約 98 token），因此 `MAX_LENGTH` 由 64 提高至 **128**。
此切換讓「想安靜→隔音」「想要採光好→採光」這類需房源描述細節才學得到的語意得以保留。
富化腳本：`pipeline/data_prep/augment_with_expansion_map.py`。
C 組 A/B 評測結果見 [ABLATION_STUDY.md](ABLATION_STUDY.md)。

---

## FGM 對抗訓練（Fast Gradient Method）

**原理**：由 Goodfellow et al.（2014）提出，Zhu et al.（2019）將其應用於 NLP embedding 層。每個訓練步驟在正常反向傳播後，沿 embedding 梯度方向加入一個小擾動 $\delta$，再做一次前向+反向（不更新參數），讓模型同時對原始輸入和被擾動的輸入都有好的預測。

$$\delta = \varepsilon \cdot \frac{g}{\|g\|}, \quad g = \nabla_{\text{emb}}\mathcal{L}$$

$$\text{adversarial backward: } \mathcal{L}(\theta,\ \text{emb} + \delta)$$

**效果**：讓模型在 embedding 空間的鄰域內仍然穩健，有效提升對口語化、有錯字、非規範輸入的泛化性（這對租屋查詢尤為重要）。相比 PGD（多步對抗），FGM 只需一步，計算成本僅多一次 forward + backward。

**本專案**：ε = 1.0，作用於 `word_embeddings` 層；使用 `try/finally` 確保即使對抗 backward 拋出例外，embedding 仍會被還原。

---

## 餘弦退火（Cosine Annealing）

**原理**：讓某個超參數在訓練過程中以餘弦曲線平滑衰減，相比線性衰減更平緩，末期下降速度更慢，有助於在收斂末段微調。

$$v(t) = v_{\min} + (v_{\max} - v_{\min}) \cdot \frac{1 + \cos\left(\dfrac{\pi t}{T_{\text{total}}}\right)}{2}$$

**本專案用途：動態蒸餾權重 $\alpha$**

$$\alpha(t) = 0.12 + 0.26 \cdot \frac{1 + \cos\left(\dfrac{\pi t}{10}\right)}{2}$$

| epoch | $\cos(\pi t/10)$ | $\alpha$ | 訓練重心 |
|:---:|:---:|:---:|:---|
| 0 | +1.0 | 0.38 | teacher 引導為主（38% KD，62% task）|
| 5 | 0.0 | 0.25 | 均衡過渡 |
| 10 | −1.0 | 0.12 | task loss 主導（12% KD，88% task）|

初期高 $\alpha$ 讓 student 先從 teacher 學到語意空間的基本結構，防止 student 一開始就收斂到局部最差點（崩塌）；末期低 $\alpha$ 讓 task loss 精確優化當前任務的排序目標。

---

## R-Drop（正規化技術）

> **v3.0 student 已移除 R-Drop。** 以下保留技術說明，供理解 v2.9 及 teacher 訓練使用。

**原理**：由 Liang et al.（2021, Microsoft）提出。同一份輸入做兩次前向傳播，由於 Dropout 的隨機性，兩次會得到不同的 logits。最小化兩次輸出機率分佈的對稱 KL 散度，強制模型對 Dropout 擾動保持一致性，等效於對參數空間施加平滑正規化。

$$\mathcal{L}_{\text{R-Drop}} = \frac{1}{2}\left[D_{\mathrm{KL}}(P_1 \| P_2) + D_{\mathrm{KL}}(P_2 \| P_1)\right]$$

其中 $P_1 = \sigma(z^{(1)})$, $P_2 = \sigma(z^{(2)})$ 為同一輸入兩次 Dropout 前向的輸出機率。

**文獻效果**：在分類/NLU 任務上典型增益 +1–3% F1，且幾乎不增加推論成本（推論時不做第二次 forward）。

**本專案結論**：消融實驗（C3_no_RDrop）顯示，移除 R-Drop 後 NDCG@5 反而提升 +0.0068（0.8787 vs 0.8719）。推測原因：R-Drop 的對稱 KL 約束與 FGM 的對抗梯度方向衝突，在中文短文本租屋配對任務上造成干擾。v3.0 student 設定 $\alpha_{\text{rdrop}} = 0$；rbt6 teacher 訓練仍保留 R-Drop。

---

## Label Smoothing（標籤平滑）

**原理**：訓練時不使用 one-hot 標籤 $y \in \{0, 1\}$，而是以 $\varepsilon$ 的比例混入均勻分佈：

$$y_{\text{smooth}} = (1 - \varepsilon)\,y + \frac{\varepsilon}{K}$$

其中 $K=2$ (二元分類), $\varepsilon=0.05$。即正樣本標籤從 1.0 → 0.975，負樣本從 0.0 → 0.025。

**效果**：防止模型對訓練資料的標籤過度自信（logits 趨向 $\pm\infty$），改善信心校準（calibration），讓模型輸出的機率分數更能反映真實相關性，而非只追求分類邊界的最大化。對知識蒸餾尤為重要：teacher 的 soft label 如果本身校準不佳，傳遞給 student 的資訊也會有偏。

---

## 負樣本採樣策略（v2.4–v2.8 的 bug 根源）

| 策略 | rel=0（硬衝突）| rel=−1（輕度不符）| Teacher Prec | NDCG@5 |
|:---|:---|:---|:---|:---|
| Stratified hard-first（v2.4~v2.8 bug）| 100% | 0% | ~0.65 | ~0.76 |
| **Random mix（v2.9）** | **~69%** | **~31%** | **0.768** | **0.833** |

**根本原因**：`neg_hard`（rel=0，約 17,611 筆）> `target_neg`（約 9,590 筆），分層採樣 100% 取 rel=0，rel=−1 完全被排除。模型失去軟邊界學習信號，信心校準惡化。

---

## 超參數設定

| 參數 | 值 | 說明 |
|:---|:---|:---|
| learning_rate | 3e-5 | AdamW，cosine decay |
| batch_size | 32 | per device（FP16）|
| epochs | 10 | early stopping patience=6 |
| warmup_ratio | 0.08 | 約 1 epoch |
| weight_decay | 0.01 | L2 正則化 |
| max_grad_norm | 1.0 | 梯度裁剪 |
| hidden_dropout_prob | 0.15 | student Dropout |
| attention_dropout | 0.15 | student Attention Dropout |
| MAX_LENGTH | 128 | tokenizer 最大長度（C 組富化後文字較長，~98 token，由 64 提高）|
| DISTILL_TEMPERATURE | 4.0 | KD 溫度 |
| LABEL_SMOOTHING | 0.05 | CE 標籤平滑 |
| FGM ε | 1.0 | 對抗擾動幅度 |
