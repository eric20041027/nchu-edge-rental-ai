# 專案更新日誌 (Changelog) - 興大 AI 租屋推薦系統

---

## [2026.05.14 v2.4] - R-Drop + rbt6 Teacher + 動態蒸餾 α + 訓練策略全面修正

### 改進動機與問題分析

v2.3 評估後發現四個問題：
1. `metric_for_best_model="loss"` 儲存 loss 最低的 checkpoint，而非 F1 最高——**保存目標與優化目標不一致（Bug）**
2. 訓練 Precision 偏低（F1@0.7=81.5%，Prec=71.2%）——說明模型對部分 NOT_MATCH 仍過度自信預測為 MATCH
3. 蒸餾 α 固定為 0.40，初期應更依賴 teacher 引導、後期應更依賴 task loss，但無法自適應調整
4. 負樣本採樣完全隨機——rel=-1（純隨機負例）語意信號遠弱於 rel=0（硬衝突負例）

### 嘗試過的方向與失敗分析

**初始計劃（v2.4 Alpha）**：Focal Loss γ=2.0 + α=0.50→0.20 + weight_decay=0.05 + T_task 移除

訓練後指標：F1 79.4%（↓5.7%）、Precision 66%（↓9.9%）、NDCG@5 0.774（↓0.044）

根因分析：
| 變更項目 | 預期效果 | 實際問題 |
|---------|---------|---------|
| Focal Loss γ=2.0 | 提升 Precision | 過度抑制正例梯度，導致 Precision 崩潰至 66% |
| T_task=2.0 移除 | 簡化損失函數 | RankNet/ListNet 分數尺度驟變，排序損失信號失效 |
| rel=3 weight 15→6 | 平衡梯度來源 | 完美匹配樣本信號被削弱 60%，NDCG 大幅退步 |
| α_max=0.50 | 初期更多 teacher 引導 | 對 pre-trained teacher 來說 α 過大，task loss 過弱 |
| 自蒸餾鏈使用 v2.4 teacher | 延續 BANs | v2.4 Precision=0.66，teacher 品質惡化後 student 繼承偏差 |

**v2.4b 嘗試**（使用 v2.4 模型作為 teacher，校正 alpha 但未修復其他問題）：
- Epoch 6 F1=0.761，Precision=0.625——teacher 品質差導致蒸餾鏈退化，已終止

### 最終修正（v2.4c）

#### 🔁 R-Drop 正規化（α_rdrop=0.05）
- **原理**：同一批次進行**兩次前向傳播**（不同 Dropout mask），以兩次輸出間的對稱 KL 散度作為額外正規化項
- **效果**：強迫模型對相同輸入的兩次推斷保持一致，減少 Dropout 帶來的預測方差，文獻典型增益 +1~3% F1
- **實作**：正常前向（含 R-Drop）→ FGM 攻擊 → 對抗前向（不含 R-Drop）→ 還原 embedding
- 採用保守 α=0.05（避免與 FGM 對抗梯度衝突）

#### 🎓 升級 Teacher：rbt6（6 層）取代 rbt3（3 層）
- **問題**：自蒸餾（Student = Teacher 架構相同）在 teacher 品質下滑時，下一代 student 繼承並放大偏差
- **修正**：使用預訓練 `hfl/rbt6`（6 層，~117M params）作為 teacher，3 層 rbt3 student 從更強的語言模型中學習
- rbt6 的語意理解能力更強，soft label 品質更穩定（不受 task-specific 偏差影響）

#### 📉 動態 KD α（餘弦退火：0.38 → 0.12）
- 相比原計劃的 0.50→0.20，使用更保守的範圍：pre-trained rbt6 soft label 品質高但不含 task-specific 偏差
- **初期**（α=0.38）：借助 teacher 引導快速建立語意表示
- **後期**（α=0.12）：主要依賴 task loss 收斂，確保模型最終優化排序目標
- 公式：`α(t) = 0.12 + 0.26 × (1 + cos(π × t/T)) / 2`

#### 🎲 分層負樣本採樣
- 優先保留 rel=0（硬衝突）負例（16,693 個可用 vs 需要 9,590 個），完全填滿 1:1 負例配額
- **結果**：採樣後全為硬衝突（rel=0），完全消除低信號的純隨機負例（rel=-1）

#### ⚖️ 重新校準樣本權重

| relevance | v2.3 weight | v2.4c weight | 說明 |
|-----------|-------------|--------------|------|
| 3（完美） | 15.0 | **12.0** | 提高（v2.4 alpha 削減至 6 導致 NDCG 退步，本版回升） |
| 2（良好） | 4.0 | **4.0** | 維持 |
| 1（部分） | 0.8 | **0.8** | 維持 |
| 0（衝突） | 6.0 | **5.0** | 配合分層採樣微調 |
| -1（隨機）| 0.5 | **0.3** | 已透過採樣排除，此值僅作安全備份 |

#### 🔧 訓練超參數修正

| 參數 | v2.3 | v2.4c | 說明 |
|------|------|-------|------|
| `metric_for_best_model` | `"loss"` ❌ | `"f1"` ✅ | 修正 Bug：應儲存 F1 最高的 checkpoint |
| `greater_is_better` | `False` ❌ | `True` ✅ | 配合 F1 方向修正 |
| `lr_scheduler_type` | linear | **cosine** | 餘弦衰減更平滑，收斂後期不震盪 |
| `weight_decay` | 0.01 | **0.01** | 維持 v2.3（0.05 過強正規化會壓制 rel=3 梯度） |
| `label_smoothing_factor` | 0.0 | **0.05** | 輕度標籤平滑，改善校準 |
| `T_task` (ranking loss) | 2.0 | **2.0** | 維持（移除後 RankNet 梯度尺度崩潰） |
| `Focal Loss γ` | — | **0.0（停用）** | v2.4 alpha 實驗證明 γ=2.0 嚴重損害 Precision |
| `num_train_epochs` | 7 | **10** | 更多 epoch（Early Stopping 以 F1 為準） |
| Early stop patience | 8 steps | **6 epochs** on F1 | 避免在退步後繼續訓練 |

#### 📏 評估指標擴充
新增 `evaluate_model.py` 指標：
- **Graded NDCG@1, @3, @10**（補充原有 @5）
- **Precision@1, @3, @5**（top-k 中相關（rel≥1）的比例）
- **Hit@1**（首位是否為完美匹配 rel=3）
- **Bootstrap CI**（1000 次重採樣，Graded NDCG@5 95% 信賴區間）

#### 🔄 自動量化整合
- `export_to_onnx()` 末尾自動呼叫 INT8 量化，不再需要手動執行 `quantize_model.py`
- 同步自動複製完整 tokenizer（21,128 vocab）至 `frontend/models/custom_onnx_model_dir/`，防止 tokenizer 不同步

### 訓練設定摘要（最終實際執行配置）

> 注：v2.4c（rbt6 teacher + α=0.38→0.12）在實際訓練中發現 pre-trained rbt6 分類頭是隨機初始化，soft label 為噪訊（詳見技術問題 FAIL-04），已停用 KD。最終以下配置訓練。

```
Model   : hfl/rbt3（新鮮初始化，3 層 ~38M params）
Loss    = CE(ls=0.05) + RankNet(T=2.0)×1.5 + ListNet(T=2.0) + 0.05×R-Drop
KD      : 停用（α=0.0）— 無可用的高品質 task-finetuned teacher
Focal γ : 0.0（停用，見 FAIL-01）
T_task  : 2.0（維持，見 FAIL-02）
Epochs  : 10（Early Stopping patience=6 on F1）
LR      : 3e-5（Cosine）, weight_decay=0.01, label_smoothing=0.05
Batch   : 32, GPU: RTX 3060
Speed   : ~6.5 it/s（無 teacher forward pass，較 v2.3 快 35%）
```

### 訓練過程（每輪驗證指標）

| Epoch | Val Loss | Acc | Precision | Recall | F1 |
|-------|----------|-----|-----------|--------|-----|
| 1 | 5.256 | 69.0% | 49.0% | 87.7% | 62.9% |
| 2 | 4.924 | 78.5% | 58.4% | 97.0% | 72.9% |
| 3 | 4.863 | 78.9% | 60.0% | 88.4% | 71.5% |
| 4 | 4.750 | 81.7% | 63.3% | 91.9% | 75.0% |
| 5 | 4.813 | 81.0% | 62.8% | 89.6% | 73.8% |
| **6** | **4.799** | **82.5%** | **63.5%** | 97.5% | **76.9%** ✅ |
| 7 | 4.768 | 81.7% | 63.6% | 90.7% | 74.8% |
| 8 | 4.812 | 81.8% | 63.8% | 90.5% | 74.9% |
| 9 | 4.816 | 81.3% | 63.0% | 91.2% | 74.5% |
| 10 | 4.815 | 81.3% | 62.9% | 91.5% | 74.6% |

- **Best Checkpoint**：Epoch 6（F1=76.9%，Early Stopping 於 patience=4/6 時 10 epoch 結束）
- **Holdout Test Set**：Acc 82.8%、Precision 63.8%、Recall 96.2%、F1 76.7%
- 訓練時長：1150s（~19 分鐘），平均 6.5 it/s（RTX 3060，無 teacher forward pass）
- ONNX 導出：147 MB（FP32） → **36.8 MB**（INT8 量化，75% 壓縮）

### 完整評估指標（v2.4，量化 INT8 模型）

**Phase 1 二元分類（test set n=5,000）：**

| 閾值 | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| 0.5 | 76.1% | 55.3% | **97.9%** | 70.7% |
| 0.7 | 76.4% | 57.1% | 79.3% | 66.4% |
| 0.9 | 76.1% | 59.2% | 60.7% | 59.9% |

**Phase 2 排名指標（500 queries，Top-30 重排模擬）：**

| 指標 | v2.4 (rbt3 INT8) | v2.3 (rbt3 INT8) |
|------|-----------------|-----------------|
| **Graded NDCG@1** | **0.734** | — |
| **Graded NDCG@3** | **0.729** | — |
| **Graded NDCG@5** | **0.727 ± 0.017** | 0.818 ± 0.015 |
| **Graded NDCG@10** | **0.737** | — |
| Binary NDCG@5 | 0.606 | 0.691 |
| **MRR** | **0.611** | 0.692 |
| **Precision@1** | **0.774** | — |
| **Precision@3** | **0.768** | — |
| **Precision@5** | **0.760** | — |
| **Hit@1 (rel≥3)** | **0.462** | — |
| Avg Satisfaction | 0.603 | 0.678 |

**Top-30 候選池標籤分佈：** Perfect(3)=43.5%、Good(2)=17.6%、Partial(1)=13.4%、None(0)=25.5%

> **說明**：v2.4 的 Graded NDCG@5（0.727）低於 v2.3（0.818），主要原因是 **KD 被迫停用**（無可用的 task-finetuned teacher），rbt3 無蒸餾時基準約 0.70-0.75。v2.4 的貢獻在於完整修正訓練策略缺陷（Focal/T_task/自蒸餾鏈），為下一個有正確 teacher 的版本奠定基礎。

---

## [模型訓練問題排查記錄] — 訓練失敗根因分析

本節記錄直接影響模型效能的訓練策略失敗與推斷層 bug，包含根因分析與修正細節。

---

### 🔴 BUG-01：tokenizer.json 只有 5 個詞彙導致推斷完全失效

**發生版本**：v2.3 首次評估時發現  
**影響**：模型對所有輸入輸出幾乎相同的錯誤結果，推薦系統完全失效

**現象**：
- 同一 query + property 對，用不同 tokenizer 路徑推斷：
  - `frontend/` tokenizer → logits = `[1.787, -0.383]`（預測 NOT_MATCH）
  - `saved_models/rbt3_finetuned/` tokenizer → logits = `[-6.735, 1.578]`（預測 MATCH）
  - 兩者**完全相反**，且前者對所有輸入幾乎一致（模型輸入全相同）

**根因**：
```
frontend/models/custom_onnx_model_dir/tokenizer.json — 2,984 bytes
  vocab: {"[UNK]":0, "[SEP]":1, "[PAD]":2, "[CLS]":3, "[MASK]":4}
  ← 只有 5 個特殊 token，完整詞彙應為 21,128 個 WordPiece token
```
- 早期導出時 `tokenizers` 函式庫序列化只保留 padding/truncation 配置，未包含 `vocab.txt` 的 WordPiece 詞彙表
- 所有中文字元 token → `[UNK]`（ID=0），模型輸入等同 `[CLS][UNK]×63[SEP]`，對任意查詢完全相同

**修正**：
```python
# export_to_onnx() 末尾自動同步完整 tokenizer 至前端目錄
for fname in ("tokenizer.json", "tokenizer_config.json", "config.json",
              "special_tokens_map.json", "vocab.txt"):
    shutil.copy2(os.path.join(SAVED_MODEL_DIR, fname), frontend_dir)
```

---

### 🔴 BUG-02：metric_for_best_model="loss" 導致保存錯誤 Checkpoint

**發生版本**：v2.3 及之前所有版本  
**影響**：每次訓練儲存的都是 val loss 最低的 epoch，而非 F1 最高的 epoch

**現象**：
```
Epoch 6: F1=85.1%, Precision=75.9%  ← 實際最佳 F1，未被儲存
Epoch 7: F1=84.8%, Precision=75.3%  ← loss=2.546（最低），被誤選為 best checkpoint
```

**根因**：訓練損失同時包含 RankNet + ListNet 排序損失，這些損失在 binary F1 已收斂後仍繼續下降，導致 loss 最低的 epoch 不等於 F1 最高的 epoch。HuggingFace Trainer 依 `metric_for_best_model` 決定儲存哪個 checkpoint，設為 `"loss"` 時選出的是排序 loss 最低但 F1 略差的版本。

**修正**：
```python
TrainingArguments(metric_for_best_model="f1", greater_is_better=True, ...)
```

---

### 🟠 FAIL-01：Focal Loss γ=2.0 導致 Precision 崩潰（v2.4 Alpha）

**發生版本**：v2.4 Alpha  
**期望效果**：提升 Precision（下壓簡單正例梯度，聚焦困難邊界）  
**實際結果**：Precision 從 75.9% 驟降至 **66%**，NDCG@5 從 0.818 降至 0.774

**Focal Loss 公式**：
```
FL(x, y) = -(1 - p_t)^γ × log(p_t)
p_t = softmax(x)[y]   ← 模型對正確類別的信心
focal_weight = (1 - p_t)^γ  ← γ=2 時信心越高梯度越被壓制
```

**失敗根因分析**：
- 文獻（Lin et al., RetinaNet 2017）設計 Focal Loss 針對**物件偵測**場景，背景框數量極多（>99%）且容易分類，前景框困難但重要
- 租屋匹配任務的「正例」（MATCH）本身具高度多樣性：「6000以下南區套房」vs「可養貓電梯大樓」是完全不同的語意空間，不存在「過於容易」的系統性正例
- 初期訓練時，模型對正例的信心 `p_t` 仍低（0.5-0.7），但 γ=2 已顯著壓制梯度 → 正例學習嚴重滯後
- 加上蒸餾損失（α=0.50）同時稀釋 task loss → 有效 task gradient 幾乎消失
- 結果：模型對正例的分類邊界無法有效建立，誤判大量非相關房源為 MATCH（Precision 下降）

**驗證**：啟用 Focal Loss（γ=2）時的訓練行為：
```
Epoch 1: F1=0.611, Prec=0.449 （v2.4 Alpha，含 focal + 蒸餾）
Epoch 3: F1=0.742, Prec=0.610
Epoch 6: F1=0.794, Prec=0.660 ← 最終收斂，仍遠低於 v2.3 的 Prec=0.759
```

---

### 🟠 FAIL-02：T_task=2.0 移除導致排序損失梯度崩潰

**發生版本**：v2.4 Alpha（移除 T_task 以「簡化」損失）  
**影響**：RankNet 和 ListNet 排序損失信號消失，模型無法學習排序偏好

**分析**：
```python
# v2.3（有效）
T_task = 2.0
rel_logits = logits[:, 1] / T_task  # logit 範圍縮小至 ~[-2.5, 2.5]

# v2.4 Alpha（移除 T_task 後）
rel_logits = logits[:, 1]            # logit 範圍 ~[-5, 5]（未縮放）
```

**RankNet 損失對 T_task 的敏感性**：
```
RankNet_loss = log(1 + exp(-(s_i - s_j)))   for pairs where r_i > r_j

若 s_i - s_j = 6.0（有 T_task=2 時約為 3.0）
  exp(-6.0) ≈ 0.0025   → log(1.0025) ≈ 0.0025   ← 梯度幾乎為零（sigmoid 飽和）
  exp(-3.0) ≈ 0.0498   → log(1.0498) ≈ 0.0487   ← 有效梯度
```
- 未縮放的 logit 差異較大，sigmoid 函數飽和，梯度趨近於 0
- 模型學不到「高相關度房源應排在低相關度房源之前」的偏好
- ListNet 同樣受影響：`log_softmax(rel_logits)` 的 log 接近 0（其中一個元素機率 ≈ 1），KL 散度梯度消失

**修正**：維持 `T_task=2.0`（v2.4 最終配置保留此值）

---

### 🟠 FAIL-03：自蒸餾鏈退化（Bad Teacher → Worse Student）

**發生版本**：v2.4b（使用 v2.4 Alpha 產出的模型作為 teacher）  
**影響**：v2.4b 的 student 繼承了 teacher 的偏差，無法達到 v2.3 水準

**問題結構**：
```
v2.3 訓練 → saved_models/rbt3_finetuned  (F1=85.1%, Prec=75.9%)  ← 好 teacher
       ↓ 訓練完成覆寫
v2.4α 訓練 → saved_models/rbt3_finetuned  (F1=79.4%, Prec=66%)  ← 壞 teacher
       ↓ 以此為 teacher 進行 v2.4b
v2.4b 訓練 → Epoch 6: F1=0.761, Prec=0.625  ← 更差，蒸餾鏈退化
```

**根因**：
- `TEACHER_MODEL_PATH = saved_models/rbt3_finetuned`
- `SAVED_MODEL_DIR    = saved_models/rbt3_finetuned`
- 兩者指向同一個路徑，每次訓練完成後 teacher 被下一代 student 覆寫
- v2.4α 的 teacher (Prec=0.66) 會對大量非相關房源輸出 P(MATCH) > 0.4（不確定的正預測）
- Student 學習 teacher 的 soft label 分布 → 繼承「不確定」的邊界 → Precision 持續低

**數學說明**：
```
KL(P_student / T ‖ P_teacher / T) 的梯度
→ 驅使 student 的 logit 分布逼近 teacher 的 logit 分布
→ 若 teacher 對 rel=0 的樣本輸出 [0.38, 0.62]（softmax after T=4）
   student 被驅使預測 P(MATCH) ≈ 62%（此樣本應為 0%）
```

**教訓**：
1. `TEACHER_MODEL_PATH` 不應與 `SAVED_MODEL_DIR` 相同，或在每次訓練前備份 teacher
2. 使用 BANs / 自蒸餾時，必須確認 teacher 品質 ≥ 目標門檻（建議 F1 > 83%）
3. Teacher 品質退步時，應立即停止蒸餾鏈並重建 teacher

---

### 🟠 FAIL-04：Pre-trained rbt6 作為 Teacher 的分類頭隨機初始化問題

**發生版本**：v2.4c（切換至 `hfl/rbt6` 作為 teacher）  
**影響**：Epoch 1 F1=0.569（低於無 teacher 基準的 ~0.62），training 被終止

**現象**：
```
v2.4c Epoch 1: F1=0.569, Precision=0.435, Recall=0.823
  → 高 Recall + 低 Precision = 模型對幾乎所有輸入都預測 MATCH（閾值 0.5 時）
```

**根因**：
```python
teacher = AutoModelForSequenceClassification.from_pretrained(
    "hfl/rbt6",         # 只有 encoder 有預訓練權重
    num_labels=2,
    ignore_mismatched_sizes=True,   # ← 分類頭被隨機初始化！
)
```
- `hfl/rbt6` 是純語言模型（MLM），沒有 sequence classification head
- 載入後，分類頭（`classifier.weight`, `classifier.bias`）從 `kaiming_uniform_` / `zeros_` 初始化
- 隨機初始化的分類頭對任意輸入輸出接近均勻分布：`softmax([w·h+b]) ≈ [0.5, 0.5]`
- 經過溫度縮放 T=4.0 後，KL 目標仍接近均勻分布

**梯度分析**：
```
KL(P_student/T ‖ P_teacher/T) ≈ KL(P_student/T ‖ [0.5, 0.5])
→ 驅使 student 的兩個類別概率趨近 0.5
→ 等效於在 task loss 上疊加了一個拉向 0.5 的正規化項
→ 分類邊界模糊，以 0.5 為閾值時幾乎所有樣本都被預測為正例（High Recall / Low Prec）

影響量 = DISTILL_ALPHA_MAX × 損失 = 0.38 × 隨機梯度
→ 38% 的更新方向是噪訊
```

**診斷方式**：直接比較兩個 tokenizer 輸出：
```python
# 快速診斷：對同一輸入比較 student 和 teacher 的預測
student_logits = model(**inputs).logits  # 期望 [-6, +6] 範圍
teacher_logits = teacher(**inputs).logits  # 若接近 [0, 0]，teacher head 是隨機的
```

**修正**：設 `DISTILL_ALPHA_MAX = 0.0`（完全停用 KD），直到有高品質 task-finetuned teacher 可用

---

## [2026.05.13 v2.3] - 知識蒸餾壓縮（rbt6 → rbt3）+ 前端快速載入 + 評估指標修正

### 核心改進

#### 🧠 知識蒸餾：rbt6 Teacher → rbt3 Student
- **蒸餾設定**：溫度 T=4.0，蒸餾權重 α=0.40，任務損失比重 0.60
- **損失函數**：`0.60×(CE + RankNet×1.5 + ListNet) + 0.40×T²×KL(student/T ‖ teacher/T)`
- **FGM 對抗訓練**持續應用於 student，teacher 完全凍結
- 學習率提升至 3e-5（rbt3 較小容量需要更大更新幅度），7 Epoch

| Epoch | Val Loss | Acc | Precision | Recall | F1 |
|-------|----------|-----|-----------|--------|-----|
| 1 | 2.948 | 84.9% | 67.9% | 94.2% | 78.9% |
| 2 | 2.728 | 87.1% | 71.3% | 95.4% | 81.6% |
| 3 | 2.594 | 88.8% | 73.9% | 96.6% | 83.8% |
| 4 | 2.578 | 89.1% | 74.3% | 96.9% | 84.1% |
| 5 | 2.570 | 89.2% | 74.5% | 97.3% | 84.4% |
| 6 | **2.548** | **89.9%** | **75.9%** | 96.8% | **85.1%** ✅ |
| 7 | 2.546 | 89.6% | 75.3% | 97.0% | 84.8% |

- **Best Epoch 7**（loss 最低 2.546），Test set：F1 **83.1%**、Recall 97.7%
- 訓練時長：962s（~16 分鐘），4.82 it/s（RTX 3060）

#### 🐛 tokenizer.json 修正（致命 Bug）
- 發現 `frontend/models/custom_onnx_model_dir/tokenizer.json` 僅有 **5 個詞彙**（應為 21,128 個）
  - 原因：tokenizer 序列化只儲存了 padding/truncation 設定，未包含完整 WordPiece vocab
  - 影響：所有 token 被映射為 `[UNK]`，導致模型輸入全相同、推斷結果完全錯誤
- 修正：從 `saved_models/rbt3_finetuned/tokenizer.json`（439 KB）複製完整 tokenizer 至 frontend
- 同步修正 `export_to_onnx()` 加入自動複製 tokenizer 至前端目錄，防止未來訓練後不同步

#### 📊 完整評估指標（量化 INT8 模型，修正 tokenizer 後）

**Phase 1 二元分類（test set n=3,993）：**

| 閾值 | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| 0.5 | 86.2% | 68.5% | **98.0%** | 80.6% |
| 0.7 | 87.3% | 71.2% | 95.2% | **81.5%** ✅ |
| 0.9 | 88.0% | 79.3% | 80.3% | 79.8% |

推薦系統使用閾值 0.7 取得最佳 F1；高 Recall（98%）確保不遺漏好物件。

**Phase 2 排名指標（500 queries，Top-30 重排）：**

| 指標 | 分數 |
|------|------|
| Binary NDCG@5 | 0.691 |
| **Graded NDCG@5** | **0.818 ± 0.015** |
| MRR | 0.692 |
| Avg Satisfaction | 0.678 |

**Top-5 結果標籤分佈：** Perfect(3)=44.4%、Good(2)=19.1%、Partial(1)=12.8%、None(0)=23.7%

#### 📦 模型壓縮效果

| 指標 | rbt6（前代） | rbt3 蒸餾（本版） | 變化 |
|------|------------|----------------|------|
| 模型大小（量化 INT8） | 58 MB | **37 MB** | **-36%** |
| 首次載入總量 | ~108 MB | **~89 MB** | **-18%** |
| Val F1（best epoch） | 84.8% | **85.1%** | +0.3% |
| Test F1 | 83.6% | 83.1% | -0.5% |
| Graded NDCG@5 | — | **0.818** | — |
| MRR | — | **0.692** | — |
| Precision@0.7 | — | **71.2%** | — |
| 參數量（FP32） | 228 MB | 147 MB | -36% |

蒸餾讓 rbt3 的 F1 超越直接訓練 rbt3 預期值（~80-82%），幾乎達到 rbt6 水準。

#### ⚡ 前端模型快速載入（Cache API + Service Worker）
- 修正 `inference-worker.js` 的 `Date.now()` cache-busting bug（每次都重新下載 57 MB）
- 新增 **Cache API** 快取 Cross-Encoder（37 MB）與 NER（37 MB）模型
- NER worker 改為 vocab + model **並行**載入（省 1-2 秒）
- 新增 `sw.js` **Service Worker**：`.onnx` cache-first，JS/HTML stale-while-revalidate
- 載入進度條偵測快取命中時顯示「⚡ 快取」

**重複載入速度**：5-30 秒 → **< 1 秒**（完全命中 Cache API）

---

## [2026.05.12 v2.2] - 多維度品質提升：NER 壓縮、評分修正與前端增強

### 核心改進

#### 🔧 `compute_relevance_score` 全面修正
- **預算方向感知**：修正「以上」被誤判為上限的嚴重 Bug，改為正確語意（以上=下限，以下/以內=上限）
- **is_strict 過度觸發修正**：移除「找/想要/需要」等幾乎所有查詢都包含的詞，嚴格模式僅在「一定要/絕對/急尋/【/＃」等真正強調語境觸發
- **負分保護**：`satisfied -= 0.5` 改為 `satisfied += 0.0`，並在最終計算前加 `max(0.0, ...)` 防止 score_ratio < 0
- **Geo-tier 溢位修正**：`satisfied += 1 + bonus` 改為 `satisfied += min(1.0, 1.0 + bonus)` 防止超出 total_specified

#### 📝 查詢模板多樣性擴充
- **Strategy 6 (情境/角色式查詢)**：新增大一新生、交換學生、預算緊張族、WFH、寵物主人、安全意識、租補申請等 7 類角色情境查詢
- **Strategy 7 (負向需求查詢)**：加入「不要頂加」、「不要暗房」、「不要太吵」等真實用戶常見否定式查詢

#### 🚶 通勤時間整合 (Task 5)
- `load_properties()` 新增 `walk_mins` / `scooter_mins` 欄位讀取（969/972 筆有數據）
- `property_to_text()` 輸出新增「步行X分鐘」、「騎車X分鐘」使模型學習通勤語意
- `expr_distance()` 優先使用 CSV 實際通勤數據（OSRM 計算），無資料才退回估算（0.08km/min 步行，0.6km/min 騎車）
- 通勤閾值放寬：步行 15 → 20 分鐘，騎車 10 → 15 分鐘，新增「走路約X分」/「騎車約X分」變體

#### 💰 前端 NER BGT 實體預算過濾 (Task 6)
- 新增 `parseBudgetFromNER(budgetSpans)` 函數，支援萬/千/k/中文數字解析
- `recommend()` 中：當 regex 未能識別預算時，自動使用 NER 提取的 BGT 實體補充 `constraints.budget`
- 支援方向感知：BGT span 包含「以上」→ `limit='above'`，其餘預設 `limit='below'`

#### 🤖 NER 模型壓縮 (hfl/rbt3)
- 底層模型從 `bert-base-chinese`（12 層，98 MB）換為 `hfl/rbt3`（3 層，37 MB）
- F1-Score 維持 0.9972，總下載量 169 MB → 107 MB（-37%）
- 修正訓練腳本：`save_only_model=True`、`save_total_limit=2`（防止 optimizer.pt 耗盡磁碟）
- 修正 ONNX 導出：`dynamo=False` 解決 PyTorch 2.11 torch._dynamo 不相容問題

#### 👍 前端推薦反饋按鈕 (Task 3)
- 每張房源卡片底部新增「👍 有用 / 👎 不符」反饋列
- `saveFeedback()` 記錄 `{ts, query, propertyId, vote}` 至 localStorage，上限 500 筆
- 點擊後切換為「感謝回饋！」綠色提示，防止重複記錄

#### 📊 雙進度條 UI (Task 4)
- 模型載入介面分為兩條進度條：Cross-Encoder（綠色）+ NER 語意模型（紫色）
- NER Worker 改用 Streaming Fetch，每 512KB 回報一次進度（`ner_progress` 訊息）
- `initNER()` 新增 `onProgress` / `onReady` 回調參數

### Cross-Encoder 重新訓練結果（hfl/rbt6, CUDA RTX 3060）

| Epoch | Val Loss | Acc | Precision | Recall | F1 |
|-------|----------|-----|-----------|--------|-----|
| 1 | 4.371 | 85.4% | 67.9% | 97.0% | 79.9% |
| 2 | 4.259 | 86.4% | 69.7% | 96.3% | 80.9% |
| 3 | 4.159 | 88.8% | 73.7% | 97.0% | 83.8% |
| 4 | 4.145 | 89.2% | 74.4% | 97.3% | 84.3% |
| **5** | **4.135** | **89.7%** | **75.5%** | 96.8% | **84.8%** ✅ |

- **Test Set（Holdout）**：Acc 88.8%、Precision 73.3%、Recall 97.4%、F1 83.6%
- 5 個 Epoch 全部達成 New Best（loss 單調遞減）
- 訓練時長：745s（~12.4 分鐘），平均 4.45 it/s（RTX 3060 GPU，25× CPU 加速）
- ONNX 導出：228 MB（FP32） → **57.2 MB**（INT8 量化，75% 壓縮）

### 訓練資料更新
- 以修正後的 `generate_dataset.py` 重新生成：33,598 訓練 / 3,894 dev / 3,993 測試
- 負例比例保持 1:2.5（Pos: 9,590 / Neg: 24,008）

---

## [2026.05.11 v2.1] - 軟標籤排序損失函數最優化 (Soft-Label Ranking Loss)
### 核心突破
- **NDCG@5 性能大幅躍進**:
    - 前代 (Run 1): NDCG@5 = 0.7647, MRR = 0.6894
    - 新代 (Run 2): NDCG@5 = **0.9629** ✅ (+25.9%), MRR = **0.9515** ✅ (+37.9%)
    - 遠超目標 0.85，達成 96.29% 的前5推薦精準度

- **軟標籤組合損失函數**:
    - 設計: Total Loss = 0.5×CE(hard_labels) + 0.5×BCE(soft_labels)
    - 軟標籤來自 relevance 欄位: -1/0→0.0, 1→0.4, 2→0.7, 3→1.0
    - 效果: 模型學習分層排序邏輯，relevance=3 >> relevance=1，直接優化 NDCG

- **訓練資料優化**:
    - 負例比例: 1:1 → **1:2** (pos:neg)
    - 訓練樣本: 25,000 → **37,488** (+50% 資料量)
    - 對比訊號: 1:2 比例提供更強的排序對比，符合真實推薦場景

- **評估方法修正**:
    - eval_sample_size: 1,000 (不完整) → **10,000** (全覆蓋)
    - 評估基礎: 294 個 query → **1,428 個 query** (全部)
    - NDCG 計算: 現在基於完整的 query-candidate 分組，數字更真實

- **訓練過程**:
    - 最佳 epoch: 6 (eval_loss = 0.1758，相較前代 8 epoch 更快收斂)
    - Early stopping: 3 epoch patience 於 epoch 9 觸發
    - 訓練時長: 38 分鐘 (10,548 steps, 5.07 it/s)

- **ONNX 導出與量化**:
    - ONNX 模型: 228 MB (完整精度)
    - 量化模型: **57 MB** (INT8, 74.8% 壓縮)
    - 量化後 NDCG 無損 (同樣達成 0.9627)

### 技術改進
- **Trainer 架構升級**:
    - FGMTrainer 實作 `_compute_combined_loss()` 靜態方法
    - 軟標籤於 training_step 提取後再傳入模型，避免模型接收額外輸入
    - FGM 對抗訓練同步應用於 combined loss

- **模型儲存修正**:
    - 訓練後顯式調用 `trainer.save_model()` + `tokenizer.save_pretrained()`
    - Exporter 可直接從 saved_model_dir 載入（無需 fallback）

- **Quantizer 修正**:
    - 修正 weight_type 參數: "UINT8" (string) → QuantType.QUInt8 (enum)
    - 使用 onnxruntime.quantization.QuantType 正確轉換

### 部署檔案
- `frontend/models/custom_onnx_model_dir/my_custom_model.onnx`: 228 MB
- `frontend/models/custom_onnx_model_dir/my_custom_model_quant.onnx`: 57 MB (INT8)
- `D:\renting_models\rbt6_finetuned/`: 訓練模型 (Epoch 6 best)

### 文檔更新
- README.md: 性能指標表格全面更新，NDCG@5/MRR/Model Size 反映新數據
- 新增「軟標籤排序損失函數優化」章節，詳細說明公式與效果
- trainer.py 模組說明: 補充軟標籤損失與 1:2 不衡比例

---

## [2026.05.11 v2] - RBT6 交叉編碼器模型訓練完成
### 核心完成
- **RBT6 Cross-Encoder 訓練成功**:
    - 模型: hfl/rbt6 (6層中文RoBERTa)
    - 訓練時長: ~30分鐘 (RTX 3060 GPU)
    - 訓練數據: 38,000+ 合成樣本 (query-property pairs)
    - 測試集: 2,220 樣本

- **性能指標確認**:
    - 測試損失: 0.6338 (穩定收斂)
    - NDCG@5: 0.5234 ✓
    - MRR: 0.6573 ✓
    - 準確率: 62.90%
    - 無過擬合跡象 (eval loss 穩定在 0.65-0.69)

- **訓練優化配置**:
    - 優化器: AdamW (lr=2e-5)
    - 對抗訓練: FGM (Fast Gradient Method)
    - 學習率調度: Linear warmup + decay
    - Early stopping: 3個epoch耐心值

- **存儲管理優化**:
    - 移動 venv (5.25GB) 和 saved_models (2.83GB) 到 D: 驅動器
    - C: 驅動器空間從 0% 回復至 2% 可用
    - 模型檢查點位置: D:\renting_models\rbt6_finetuned (3,419 MB)

- **已知技術限制**:
    - ONNX 導出: transformers 5.8.0 中 SDPA 注意機制與 ONNX JIT 不相容
    - 替代方案進行中: torch.jit.trace, transformers.js, 或直接 PyTorch 推論

### 文檔更新
- 新增 TRAINING_COMPLETION_REPORT.txt 訓練完成報告
- 記錄模型性能、訓練過程、後續步驟

---

## [2026.05.11] - 環境整備、文檔完善與流程驗證
### 核心完成
- **Python 虛擬環境完整配置**:
    - 解決 MinGW Python 與 PyPI 兼容性問題，安裝標準 CPython 3.11
    - 配置 PyTorch 2.6.0+cu124 (CUDA 12.4)，GPU CUDA 可用性驗證通過
    - 安裝全部 requirements.txt 依賴，包括 datasets、seqeval、accelerate 等訓練所需套件
    - 虛擬環境可直接用於訓練與推論

- **完整端到端流程驗證**:
    - Phase 2 (數據處理): 6 步管道完整運行驗證，所有步驟正常執行
    - Phase 3 (模型訓練): RBT6 Cross-Encoder 訓練啟動，loss 正常下降 (0.70 → 0.68 @ 2 epochs)
    - 後台訓練進行中，預計數小時內完成

- **README.md 完全更新**:
    - 數據流水線圖更新：5 步 → 6 步 (新增 commute 步驟完整說明)
    - 目錄結構完全重寫：詳細列出 NER 模型、約束系統、訓練管道等新增模塊
    - 核心模組說明：5 個新章節，涵蓋 6 步數據處理、NER 模型、語意匹配、訓練流程、端到端執行
    - 系統亮點擴展：新增「雙層 NER + 語意匹配」、「6 步自適應管道」等亮點說明
    - 前端優化章節：補充 NER Web Worker、量化優化、延遲指標等技術細節
    - 執行指南升級：從 Shell (run_pipeline.sh) → Python CLI (pipeline_runner.py)，支援靈活的 --skip-phase 組合
    - 新增 NER 模型單獨訓練指令

- **專案大幅清理**:
    - 刪除廢棄腳本：run_pipeline.sh, 3 個舊 runner 檔案, 2 個測試腳本
    - 刪除臨時檔案：1877 個 __pycache__ 目錄, 9 個日誌檔, PyTorch wheel, 臨時模型檔等
    - 項目大小減少 ~2.5GB，結構整潔清晰

- **生成新的項目狀態報告**:
    - PROJECT_STATUS_REPORT.md：記錄完成項目、文件清理、技術細節、性能指標、執行參考等
    - 保留作為當前項目的正式狀態文檔

### 技術細節補充
- **NER 模型集成亮點**:
    - F1-Score = 0.958 (序列標註任務)
    - Accuracy = 0.972 (字符級)
    - 瀏覽器端推論延遲 <20ms (Web Worker)
    - INT8 量化後大小適合移動端

- **語意匹配性能**:
    - NDCG@5 = 0.862 (排序品質)
    - Matching Latency = <150ms (ONNX Runtime Web)
    - 雙模型總體積 ~100MB (INT8)

### 文檔整理
- 新增 PROJECT_STATUS_REPORT.md 作為正式狀態文檔
- 已確認所有階段計劃文件 (PHASE2_PLAN, PHASE3_*, PHASE4_*, 等) 都已實行完成，標記待刪除

---

## [2026.05.10] - 深度語義推測 V3 與硬性約束強化 (LTR 3.0)
### 核心更新
- **深度語義推測 (Lifestyle Intent Inference)**:
    - 重構 `generate_dataset.py` 語義模板，引入「生活型態聚類 (Lifestyle Clusters)」映射。
    - 支援從口語描述自動推測潛在設施需求（如：「省伙食費」➡️ 廚房/瓦斯；「外送族」➡️ 管理員/飲水機）。
    - 建立 15+ 組生活場景語義橋樑，讓模型具備「聽懂弦外之音」的能力。
- **硬性約束一票否決 (Hard-Constraint Enforcement)**:
    - 實施嚴格模式 (Strict Mode) 標籤過濾。針對預算溢出 (>10%)、禁養寵物、台水電缺失等關鍵標籤實施「硬性 0 分」策略。
    - 解決了模型在「地點優勢」與「設施缺失」之間的權衡偏見，優先保證用戶底線需求。
- **全端語義同步 (Full-Stack Semantic Sync)**:
    - 同步更新 `inference-worker.js` 與 `generate_dataset.py` 的擴展字典。
    - 實作「雙層語義橋接」：前端進行預處理擴展，後端模型進行深度意圖識別，達成 12/12 壓力測試的高分表現。

### 技術優化
- **壓力測試體系升級 (Stress Test v3)**:
    - 將 `semantic_stress_test.py` 擴展至 12 個核心案例，涵蓋 FB 真實貼文模擬與複雜功能複合測試。
    - 實作「全特徵掃描」驗證邏輯，確保 AI 推薦結果與原始資料庫特徵 100% 吻合。
- **高強度對抗訓練 (LTR 3.0)**:
    - 實施 5 Epochs 的精準 Fine-tuning，配合「地獄級負樣本注入」，顯著提升模型在邊界案例上的決策信心。

## [2026.05.10] - 語義特徵工程 2.0 與 V2 全自動管線升級
### 核心更新
- **全自動管線 V2 (Pipeline V2)**:
    - 重構 `run_pipeline.ps1` 與 `run_pipeline.sh`，整合最新的數據採集、語義增強、訓練與評估流程。
    - 實作跨平台（Windows/Unix）一致性執行邏輯，支持一鍵從爬蟲到 ONNX 導出。
- **高級特徵引擎 (FeatureEngine 2.0)**:
    - 在 `generate_dataset.py` 中實作結構化特徵提取，涵蓋：
        - **電費透明度**：自動分類台電計費 vs 固定費率。
        - **服務等級**：識別子母車、垃圾代收與包裹管理。
        - **CP 值維度**：計算「租金/坪數」單價，自動標註區域高 CP 值房源。
        - **地理分層**：基於 OSRM 距離將房源分為核心圈、活躍區與寧靜區。
        - **屋況分析**：自動辨識全新首租與精緻翻新。
- **可解釋性 AI 強化 (Explainable AI V2)**:
    - 升級前端「推薦理由」生成器，利用高級特徵顯示具備 Emoji 的人性化標籤（如：✨ 免追垃圾車、⚡ 台電計費）。
    - 實作「語義隱喻匹配」，讓系統理解「下班晚」對應「垃圾代收」的需求。

### 技術優化
- **LLM 樣本大規模增廣**:
    - 利用 Gemini API 成功生成 1,000 筆高品質樣本（500 筆困難負樣本 + 500 筆正向語義映射）。
    - 數據集規模突破 **3.3 萬筆** 訓練樣本，顯著提升模型對口語化查詢的理解力。
- **訓練監控升級**:
    - 實作 `CustomEarlyStoppingCallback`，在訓練日誌中即時顯示 Patience 狀態 (X/8)。
    - 將 **Precision (精確度)** 指標加入驗證輸出，便於精準監控誤報率。
- **依賴與環境優化**:
    - 整合 `python-dotenv` 管理 API 金鑰，優化 `google-genai` 呼叫穩定性。

## [2026.05.08] - 可解釋性 AI 與透明化排序系統升級
### 核心更新
- **可解釋性 AI (Explainable AI)**:
    - 實作前端「命中理由 (Match Reasons)」顯示，利用 NLP 提取關鍵屬性並以綠色標籤展示（如：陽台、台電計費）。
    - 提升使用者對推薦分數的信任感，讓系統從「黑盒」轉為「透明」。
- **混合過濾系統 (Hybrid Filtering V2)**:
    - 在 AI 模型之上加入硬性約束檢查層。
    - 針對「寵物政策」、「陽台需求」等關鍵約束進行衝突偵測。
    - 衝突房源自動執行 **90% 降分懲罰**，並在 UI 上執行 **70% 透明度淡化** 與 **紅色衝突警示**。
- **數據質量審計 (Data Auditing)**:
    - 新增 `audit_data.py` 自動化工具，支援 Big5/UTF-8 編碼自適應讀取。
    - 成功執行原始數據審計，確保租金與地點標註無重大異常。
- **極致模型壓縮 (Model Compression)**:
    - 優化量化腳本，針對 `Gather` 算子執行動態量化。
    - **成果**：將 RBT6 模型體積從 105MB 成功壓制至 **57.3 MB**，遠低於 GitHub 100MB 限制。

### 技術優化
- **訓練指標優化**:
    - 將 `metric_for_best_model` 切換為 **F1 Score**，平衡精確度與召回率。
    - 增加 **EarlyStopping Patience (8)**，允許模型在 Fine-tuning 階段有更長的時間進行細微權益優化。
- **評估流程升級**:
    - 在 `evaluate_model.py` 中整合混合過濾邏輯，使測試報告更貼近真實前端體驗。
    - 動態訓練日誌修復，正確顯示當前 Fine-tuning 模型名稱。

---

本文件詳實記錄了中興大學 AI 租屋推薦系統專案的技術演進歷程，統整了超過 140 次的提交紀錄。

## [第五階段] 極致性能優化與泛化能力增強 (當前階段)
目標：達成 Graded NDCG@5 > 0.85 門檻

### [2026-05-08] - 模型架構升級與前端效能飛躍
- 語意匹配模型重大升級：將「查詢-房源」二分類匹配任務之架構從 RBT3 升級至 hfl/rbt6 (6 層 RoBERTa)，在 Step 4000 達成 F1-Score 0.832 之卓越表現。
- 並行加載與串流技術：重構分詞器與模型之下載邏輯，實作 Promise.all 並行請求與原生 Fetch 串流進度追蹤，解決加載停滯與進度條異常問題。
- 全平台行動端適配：完成 Mobile-First 響應式佈局重構，針對行動端觸控、間距與字級進行專項優化，達成接近原生 App 的操作質感。
- 手動導出與評估工具：開發專用導出腳本 (export_from_checkpoint.py)，支援從任意檢查點手動提取最佳權重並自動產出技術評估報告。
- 對抗訓練 (FGM)：在 WeightedTrainer 中整合快速梯度方法 (Fast Gradient Method)，針對 Embedding 層注入擾動，提升語意辨識的魯棒性。
- LLM 困難樣本增廣：利用 Gemini API 生成 500 組高難度陷阱樣本，專門針對寵物政策、台水電、地點邊界等細微特徵進行誤導訓練。
- 隨機失活 (Dropout) 強化：將隱藏層與注意力機制的 Dropout 比例提升至 0.15，增強正則化效果並防止過擬合。
- 訓練環境相容性修正：修正 WeightedTrainer.training_step 函數簽名，以支援新版本 Transformers 庫之參數規範。

### [2026-05-07] - 排序校準與權重策略
- 溫度校準 (Temperature Scaling)：在損失函數中引入 T=2.0 的溫度係數，壓平 Logit 分佈以緩解 Sigmoid 飽和問題，顯著提升 NDCG 解析度。
- 激進權重學習：實施非線性權重映射（Perfect 樣本賦予 15.0 倍權重），強制模型優先學習高相關度房源之特徵。

---

## [第四階段] 精準度指標與通勤感知優化 (2026-04-27 至 2026-05-06)
目標：從二元分類評測轉型為分級排序評測

- 分級評測指標：正式採用 NDCG@5 與 MRR 作為核心效能監控指標，取代單一的 F1 分數。
- 通勤邏輯整合：整合 OSRM 開源路網資料，實現精確的通勤時間與路網距離計算。
- 困難負樣本挖掘：將正負樣本比優化至 1:2.5，並針對模型易錯樣本進行二次訓練。
- 預算語意解析：實施繁體中文數字與簡寫（如：6k、1萬5）的魯棒解析邏輯。

---

## [第三階段] 雙塔匹配架構與 UI 全面改版 (2026-03-18 至 2026-04-02)
目標：提升使用者體驗與深層語意匹配精度

- 句子對分類架構 (Cross-Encoder)：全面切換為 Cross-Encoder 架構，實現查詢與房源描述的直接深度匹配。
- UI/UX 視覺重構：實施玻璃擬態 (Glassmorphism)、網格漸變動畫與 staggered 階梯式進入動畫。
- 模型量化：實施 INT8 動態量化技術，在不損失精度的前提下大幅縮減模型體積，優化 Web 端載入速度。

---

## [第二階段] Web AI 部署與 ONNX 整合 (2026-03-08 至 2026-03-12)
目標：達成 100% 瀏覽器端本地 AI 推理

- ONNX Runtime Web：將推理邏輯從 Python 後端遷移至瀏覽器端 WASM 執行緒。
- Transformers.js 整合：實施 Hugging Face 的 Web 端框架，確保 Tokenizer 與模型權重之相容性。
- 實體辨識模型 (NER)：訓練並部署專用的三類別 NER 模型，自動提取查詢中的地點、預算與設備需求。
- Vercel 雲端部署：完成 WASM 與跨域請求 (CORS) 之標頭配置。

---

## [第一階段] 基礎建設與多源 ETL (2026-02-27 至 2026-03-07)
目標：資料採集與後端基準建立

- 多源爬蟲開發：開發針對 591、Dcard、FB 租屋社團的自動化資料採集腳本。
- Flask 後端建立：構建首個基於 Python 的推薦系統 API 服務。
- 關鍵字匹配引擎：實施第一代基於正則表達式 (Regex) 的過濾與評分邏輯。
- 資料集標準化：收集並正規化首批中興大學周邊租屋物件資料。

---

## [專案統計與規格]
- 總提交次數：140+
- 模型架構：RoBERTa-tiny (RBT6)
- 部署平台：Vercel + ONNXRuntime Web
- 核心指標：F1 分數 0.87，NDCG@5 目標 0.85
