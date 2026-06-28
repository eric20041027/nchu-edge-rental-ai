# Model Architecture — Knowledge Distillation & Loss Functions

## 知識蒸餾架構（Knowledge Distillation）

### 為什麼使用蒸餾？

直接訓練 rbt3（3 層，38.7 MB INT8）排序上限約 NDCG@5 ≈ 0.72–0.75。由 rbt6（6 層）作 teacher 教導 rbt3，可讓小模型學到超越其容量限制的排序知識。

### 兩階段訓練

```
階段一：train_teacher.py  — rbt6 teacher
  資料  : 33,598 訓練樣本
  損失  : CE(ls=0.05) + RankNet×1.5 + ListNet + R-Drop + FGM
  存檔  : metric_for_best_model = "loss"（排序損失在 F1 峰值後仍持續下降）
  結果  : F1=0.859，Prec=0.768

階段二：train_and_export_onnx.py  — rbt3 student
  資料  : 同一份訓練資料 + 凍結的 rbt6 teacher
  損失  : (1-α)·L_task + α·T²·KL + FGM  ← v3.0: R-Drop 已移除（消融研究）
  存檔  : metric_for_best_model = "f1"
  結果  : v3.0 Dev F1=85.4%，預期 NDCG@5 ≈ 0.879（非富化基準）
  導出  : FP32 → Dynamic INT8 per_channel（38.7 MB）→ 同步至 frontend/
```

> **Production 現況（2026-06-16）**：部署模型已換為 **C 組房源富化 rbt3**。
> 模型基底改用富化文字訓練（`property_to_text_enriched`：全 notes + 全 furniture，
> `MAX_LENGTH=128`）。C 組 A/B 評測：**NDCG@5 = 0.9475 / F1 = 0.854**。
> 本檔下方各處 0.877（FP32）/ 0.809（INT8）/ 0.879 等數值為 **v3.0 非富化基準**，
> 保留作歷史對照（評測 query 集與富化版不同，NDCG 數量級不可直接比較）。
> 量化後體積由舊 57/60 MB 降至 **38.7 MB**（38,721,068 bytes）。
> 舊版曾備份為 `*.PREV-20260616.onnx`，已於 dead-weight 清理（收尾 B，PR #44）移除。

### 蒸餾損失

$$P_s = \sigma(z_s / T), \quad P_t = \sigma(z_t / T)$$

完整損失為：

$$\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\text{task}} + \alpha \cdot T^2 \cdot D_{\mathrm{KL}}(P_s \| P_t)$$

- $z_s$: student logits, $z_t$: teacher logits（凍結，純推論）
- $T = 4.0$：蒸餾溫度。原始 logits 差值 ≈ 6.4 → softmax ≈ [0.002, 0.998]（資訊量趨零）；縮放後差值 ≈ 1.6 → softmax ≈ [0.17, 0.83]（類別間序資訊可傳遞）
- $T^2$ 係數：抵消溫度縮放對梯度幅度的影響，確保 KL loss 與 task loss 在相同數量級

### 動態蒸餾權重 α（餘弦退火）

$$\alpha(t) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \frac{1 + \cos\left(\dfrac{\pi t}{T_{\text{epoch}}}\right)}{2}$$

$$\alpha_{\min} = 0.12, \quad \alpha_{\max} = 0.38, \quad T_{\text{epoch}} = 10$$

| 訓練階段 | $\alpha$ | 效果 |
|:---|:---|:---|
| 初期（t → 0）| 0.38 | teacher 主導，防止 student 初期崩塌 |
| 末期（t → 10）| 0.12 | task loss 主導，student 收斂至任務最優點 |

---

## 技術原理詳解

### 知識蒸餾（Knowledge Distillation）

**原理**：由 Hinton et al.（2015）提出。大模型（teacher）訓練完成後，其輸出的 softmax 機率分佈（稱為「soft label」）攜帶了比 one-hot 標籤更豐富的類別間關係。將 soft label 作為額外監督信號訓練小模型（student），可使 student 在參數量遠低於 teacher 的情況下達到接近甚至超越的效果。

**為何比直接訓練 student 好**：one-hot 標籤只告訴模型「這個是 Match」；而 teacher 的 soft label [0.02, 0.98] 還隱含了「這個配對雖然是 Match，但只有 98% 把握，有 2% 可能是邊界案例」，這個邊界資訊對排序任務尤為關鍵。

**本專案設定**：rbt6（6 層，228 MB FP32）→ rbt3（3 層，38.7 MB INT8，Dynamic per_channel），capacity 壓縮約 75%，NDCG@5 從 0.818 提升至 0.877（FP32）/ 0.809（INT8 部署）（student 超越 teacher，歸因於 v3.0 移除 R-Drop 與困難負樣本訓練）。以上為 v3.0 非富化基準；production 現為 C 組房源富化模型（NDCG@5 0.9475 / F1 0.854，見上方說明）。

---

### KL 散度（Kullback-Leibler Divergence）

**原理**：KL 散度衡量兩個機率分佈之間的差異，表示「用 Q 描述 P 時，相較於用 P 自身描述所多出的資訊量」。

$$D_{\mathrm{KL}}(P \| Q) = \sum_i P_i \log \frac{P_i}{Q_i}$$

- 非對稱: $D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P)$
- 當 $P = Q$ 時等於 0，兩分佈差異越大則值越大
- 在蒸餾中：令 $P = \sigma(z_t / T)$ 為 teacher 軟化分佈, $Q = \sigma(z_s / T)$ 為 student 軟化分佈，最小化 KL 即讓 student 逼近 teacher 的機率形狀

---

### 溫度縮放（Temperature Scaling）

**原理**：在 softmax 前將 logits 除以溫度 $T$，使輸出分佈趨於平滑 ($T > 1$) 或銳化 ($T < 1$)。

$$\sigma_T(z_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

**蒸餾溫度 $T_{\text{distill}} = 4.0$**：讓 teacher 的 soft label 不至於過度集中在最高類別，保留類別間的相對順序資訊：

| 情境 | logits | softmax | 資訊量 |
|:---|:---|:---|:---|
| $T=1.0$（原始）| $[-3.2,\ +3.2]$ | $[0.002,\ 0.998]$ | 近似 one-hot，邊界資訊幾乎消失 |
| $T=4.0$（蒸餾）| $[-0.8,\ +0.8]$ | $[0.31,\ 0.69]$ | 類別間距可傳遞 |

**$T^2$ 梯度補償**：溫度縮放會使梯度幅度縮小為原來的 $1/T^2$, 乘回 $T^2$ 確保蒸餾 loss 與 task loss 在相同數量級，不需要額外調整 learning rate。

---

### RankNet（配對排序損失）

**原理**：由 Burges et al.（2005, Microsoft Research）提出。從訓練集中萃取所有「i 應排在 j 前面」的配對 $(i, j)$，對每個配對最小化 sigmoid 交叉熵，要求分數差 $s_i - s_j > 0$。

$$\mathcal{L}_{\text{RankNet}} = \frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}} \log\left(1 + e^{-(s_i - s_j)}\right), \quad \mathcal{P} = \{(i,j) \mid r_i > r_j\}$$

**優於 CE 的地方**：CE 只知道「這個是 Match / 不是 Match」，無法利用 rel=1, 2, 3 之間的順序關係；RankNet 可直接利用 4 級相關性標籤的偏序資訊。

**任務溫度 $T_{\text{task}} = 2.0$**（梯度穩定性）

$$s_k = \frac{z_k^{(1)}}{T_{\text{task}}}$$

若模型已收斂, $s_i - s_j \approx 6.0$, 梯度 $\nabla \mathcal{L} \propto e^{-6} \approx 0.0025$ (幾乎消失). $T_{\text{task}}=2.0$ 縮至差值 3.0, 梯度 $\nabla \mathcal{L} \propto e^{-3} \approx 0.050$, 維持有效學習。

---

### ListNet（列表排序損失）

**原理**：由 Cao et al.（2007）提出。把整個 batch 的相關性分數視為一個機率分佈，要求模型輸出的分數分佈逼近真實相關性分佈。ListNet 同時考慮所有文件的相對順序，比 RankNet 的配對式方法捕捉更多全局排序資訊。

$$\mathcal{L}_{\text{ListNet}} = -\sum_{i} P_i^{*} \log P_i$$

$$P_i = \text{softmax}\left(\frac{s}{T_{\text{task}}}\right)_i \quad \text{(predicted dist.)}, \qquad P_i^{*} = \text{softmax}(r)_i \quad \text{(target dist.)}$$

**RankNet vs ListNet 的互補性**：RankNet 專注於兩兩之間誰該排前面（局部序）；ListNet 要求整體分佈形狀相似（全局序）。兩者合用可從不同角度優化排序品質。

---

### 損失函數組合

**Teacher（train_teacher.py）**

$$\mathcal{L}_{\text{teacher}} = \mathcal{L}_{\text{CE}} + 1.5\,\mathcal{L}_{\text{RankNet}} + \mathcal{L}_{\text{ListNet}} + \alpha_{\text{rdrop}}\,\mathcal{L}_{\text{R-Drop}}$$

**Student（train_and_export_onnx.py，v3.0）**

$$\mathcal{L}_{\text{student}} = (1-\alpha)\underbrace{\left(\mathcal{L}_{\text{CE}} + 1.5\,\mathcal{L}_{\text{RankNet}} + \mathcal{L}_{\text{ListNet}}\right)}_{\mathcal{L}_{\text{task}}} + \alpha\,T^2\,D_{\mathrm{KL}}$$

$$\mathcal{L}_{\text{CE}}:\text{ label smoothing},\quad \varepsilon=0.05$$

> v3.0 起移除 R-Drop（$\alpha_{\text{rdrop}}$ 設為 0）。消融實驗（C3_no_RDrop）顯示移除後 NDCG@5 提升 +0.0068，詳見 [ABLATION_STUDY.md](ABLATION_STUDY.md)。

---

## 向量召回 bi-encoder（Vector Recall）

上述 Cross-Encoder 為**精排**（re-rank）模型；在它之前先以 bi-encoder 做**向量召回**（vector recall），形成「召回 → 精排」兩段式檢索。bi-encoder 把 query 與房源各自獨立編碼成向量，用餘弦相似度快速取回 Top-K 候選，再交由 Cross-Encoder 對候選逐一精排。詳細規格見 [spec/vector-retrieval.md](spec/vector-retrieval.md)。

### 架構

- 基底：**rbt3（3 層，由 rbt6 蒸餾）**，shared-weight encoder（query 與房源共用同一組權重）。
  > 原為 `hfl/rbt6`（6 層）。2026-06-24 比照 Cross-Encoder 做 rbt6→rbt3 蒸餾,
  > 體積由 57→38 MB(−36%),召回幾乎零損(見下方蒸餾說明)。
- 輸出：768 維 embedding（hidden_size 蒸餾後不變），**mask-aware mean-pool + L2-normalize 已 baked 進 ONNX graph**（推論端不需再做 pooling / 正規化）。
- 量化：Dynamic INT8，**38.2 MB（38,214,155 bytes）**（原 rbt6 為 57.0 MB）。
- 檔案：`frontend/models/bi_encoder_dir/bi_encoder_quant.onnx`，由 `frontend/js/bi-encoder-worker.js` 載入。

### 訓練 / 蒸餾

- T2 teacher 訓練：`pipeline/model_training/train_bi_encoder.py`（rbt6 teacher）。
- **蒸餾**：`pipeline/model_training/distill_bi_encoder.py`，rbt6 teacher → rbt3 student。
  student = teacher embeddings + 前 3 層(截斷初始化);loss = α·cos-distill(student↔teacher 向量) + (1−α)·MNRL。
  流程見 [spec/bi-encoder-distill.md](spec/bi-encoder-distill.md)、`notebooks/bi_encoder_distill_colab.ipynb`。
- 損失：InfoNCE / MNRL（in-batch negatives + `is_hard` 困難負樣本，去重後上限 2×batch），temperature 0.05。
- 超參：teacher epochs 3 / lr 2e-5;蒸餾 epochs 3 / lr 3e-5 / α 0.5、batch 32、max_length 64、召回 K=30。

### 蒸餾 gate(rbt3 student vs rbt6 teacher,`tests/eval_vector_vs_rulebased.py` 278 query)

| 指標 | rbt6 (57MB) | rbt3 (38MB) | Δ |
|:---|:---|:---|:---|
| Recall@15 all | 0.2975 | 0.2883 | −0.009 |
| Recall@30 all | 0.3991 | 0.4000 | +0.001 |
| NDCG@5 all | 0.1769 | 0.1830 | +0.006 |
| semantic Recall@30 | 0.5808 | 0.5864 | +0.006 |

> 體積 −36%,召回零損(R@15 掉幅 < 容忍 0.02,其餘微升),仍 GO vs rule-based。

### 導出與數值驗證

- 腳本：`pipeline/model_training/export_bi_encoder.py`，`dynamo=False`、opset 15、Dynamic INT8。
- CP2 數值檢查（PyTorch vs ONNX 餘弦相似度）：fp32 = 1.000，int8 = 0.956。
- 房源 embedding 離線預先計算 → `frontend/assets/property_embeddings.json`（974×768 float16，已 L2-norm），推論端只需編碼 query 即可比對。

### A/B 評測（T7，判定 GO）

| 指標 | rule-based | bi-encoder |
|:---|:---|:---|
| semantic Recall@30 | 0.007 | 0.547 |
| semantic Recall@15 | 0.000 | 0.506 |
| semantic NDCG@5 | 0.000 | 0.325 |
| keyword Recall@30 | 0.077 | 0.359 |
| all Recall@30 | 0.057 | 0.412 |

> 評測使用 fuzzy-join 標註集（match-rate 24.4%），judgment 基礎為相對 delta 而非絕對值。harness：`tests/eval_vector_vs_rulebased.py`。

### 角色：primary recall + fallback

bi-encoder 已取代 rule-based 召回成為 **primary**；rule-based 保留為 **fallback**，於下列情況啟用：worker 尚未就緒、編碼逾時（800ms encode timeout）、或 `VECTOR_RECALL_ENABLED` kill-switch 關閉時。
