# 消融實驗計畫 (Ablation Study Plan)

## 背景與目標

v2.9 使用了 6 種進階技術的組合：CE + RankNet + ListNet + KD（動態 α）+ R-Drop + FGM。  
目前 README 只列出最終 NDCG@5 = 0.833，但沒有任何實驗支撐「每個模組是否真的有貢獻」。

本計畫設計 **3 組系統性消融實驗** + **1 組泛化性測試**，目標是在 README 加入一張可以獨立說話的消融表格，讓技術選擇從「疊 Buff」變成「有工程證據的設計決策」。

---

## 實驗資源確認

- **訓練機**：i5-11600KF + GPU（已確認 CUDA 可用）
- **固定 Teacher**：`saved_models/rbt6_teacher/`（v2.9 訓練所用，所有 Student 消融共用同一 Teacher，確保比較公平）
- **固定資料**：`data/processed/recommendation_{train,dev,test}.json`（不重新 generate，保持一致）
- **每次訓練估計耗時**：約 30–60 分鐘（含 Early Stopping）
- **總實驗次數**：12 次訓練 run

---

## Group A：損失函數消融（最核心）

**問題**：RankNet 和 ListNet 各自貢獻了多少 NDCG@5？拿掉任一個會怎樣？

| Run | 設定 | 備注 |
|:---|:---|:---|
| A-0 | CE only | 純 Baseline，移除 RankNet + ListNet |
| A-1 | CE + RankNet | 只加配對排序損失 |
| A-2 | CE + ListNet | 只加列表排序損失 |
| A-3 | CE + RankNet + ListNet | **v2.9 現狀**（作為對照組） |

> **注意**：CE 指 `F.cross_entropy(label_smoothing=0.05)` + precision penalty，不含 Focal Loss（γ=0，已確認 disabled）。

### 需要修改的程式碼

**檔案**：`pipeline/model_training/train_and_export_onnx.py`  
在 `DistillTrainer.compute_loss()` 的 `# ── 2. RankNet` 和 `# ── 3. ListNet` 區塊外層加 flag 控制：

建議在頂部常數區新增：

```python
# ── Ablation flags (set before each experiment run) ──────────────────
ENABLE_RANKNET  = True   # A-0: False, A-1: True,  A-2: False, A-3: True
ENABLE_LISTNET  = True   # A-0: False, A-1: False, A-2: True,  A-3: True
```

然後在 `compute_loss()` 內：

```python
# ── 2. RankNet ──
if ENABLE_RANKNET and mask.sum() > 0:
    ranknet = F.softplus(-(s_i - s_j)) * mask
    task_loss = task_loss + (ranknet.sum() / mask.sum()) * 1.5

# ── 3. ListNet ──
if ENABLE_LISTNET:
    task_loss = task_loss + (-torch.sum(target_dist * pred_dist)) * 1.0
```

同樣的修改也套用到 `train_teacher.py` 的 `TeacherTrainer.compute_loss()`。  
→ Teacher 也需要跑 A-0/A-3 各一次，因為 Teacher 品質影響 KD 效果。

**現有版本歷史中的部分資料點**（可直接引用，不需重跑）：

- v2.6 Teacher（CE-only，loss metric）→ Teacher F1 = 77.2%，比 v2.9 Teacher F1 = 85.9% 低 8.7%
- v2.3 Student（CE + RankNet + ListNet + KD）→ NDCG@5 = 0.818

---

## Group B：KD 動態 α 消融

**問題**：餘弦退火 α 比固定 α 好多少？固定在哪個值效果最接近？

| Run | DISTILL_ALPHA_MAX | DISTILL_ALPHA_MIN | 說明 |
|:---|:---:|:---:|:---|
| B-1 | 0.12 | 0.12 | 固定低值（task-dominant 全程）|
| B-2 | 0.25 | 0.25 | 固定中值（教授最可能質疑的「你直接取中間值不就好了？」）|
| B-3 | 0.38 | 0.38 | 固定高值（KD-dominant 全程）|
| B-4 | 0.38 | 0.12 | **v2.9 現狀**（cosine 退火）|

### 需要修改的程式碼

**檔案**：`pipeline/model_training/train_and_export_onnx.py`

只需修改頂部兩行：

```python
DISTILL_ALPHA_MAX = 0.38   # ← 改這兩個值
DISTILL_ALPHA_MIN = 0.12
```

- B-1：`MAX = MIN = 0.12`
- B-2：`MAX = MIN = 0.25`
- B-3：`MAX = MIN = 0.38`
- B-4：`MAX = 0.38, MIN = 0.12`（不動）

### 額外收集：每個 epoch 的 validation NDCG@5

B 組最有說服力的輸出是**收斂曲線圖**，而非只有最終數字。  
`CleanLogCallback` 目前只輸出 loss/F1，需要在 evaluate 時額外計算並記錄每 epoch NDCG@5。

**建議**：在 `CustomEarlyStoppingCallback.on_evaluate()` 或新增一個 `NDCGLogCallback` 記錄每 epoch 的 NDCG@5 到 JSON 檔，跑完後用 matplotlib 出圖。

---

## Group C：正規化技術消融

**問題**：FGM 和 R-Drop 個別移除後損失多少？

| Run | FGM | R-Drop (α=0.05) | 說明 |
|:---|:---:|:---:|:---|
| C-1 | ✅ | ✅ | **v2.9 現狀** |
| C-2 | ❌ | ✅ | 只移除 FGM |
| C-3 | ✅ | ❌ | 只移除 R-Drop |
| C-4 | ❌ | ❌ | 兩者都移除（純 CE + Ranking + KD）|

### 需要修改的程式碼

**FGM 開關**（`train_and_export_onnx.py` 頂部）：

```python
ENABLE_FGM = True   # C-2 / C-4: False
```

`DistillTrainer.training_step()` 內：

```python
if ENABLE_FGM:
    fgm = FGM(model)
    fgm.attack()
    try:
        loss_adv = self.compute_loss(model, inputs, use_rdrop=False)
        self.accelerator.backward(loss_adv)
    finally:
        fgm.restore()
```

**R-Drop 開關**：頂部常數直接改 `RDROP_ALPHA = 0.0`（C-3/C-4 時）。`compute_loss()` 內已有 `if model.training and use_rdrop and RDROP_ALPHA > 0` 判斷，無需其他修改。

---

## Group D：FGM 泛化性測試（針對口語/非規範輸入）

**問題**：FGM 對「含錯字 / 非規範輸入」的泛化性提升了多少？

這組不需要重新訓練，而是對 **C-1（有 FGM）** 和 **C-2（無 FGM）** 的 checkpoint 分別在「正常測試集」和「噪音測試集」上評估。

### 噪音測試集設計

建立 `pipeline/data_prep/noise_generator.py`，對 `recommendation_test.json` 中的 `query` 欄位施加以下擾動：

| 噪音類型 | 規則 | 範例 |
|:---|:---|:---|
| **縮寫替換** | 中興大學→興大，台中→中市，套房→套 | `"近興大的套"` |
| **錯字注入** | 隨機替換 10% 字元為鄰近字（注音鄰近）| `"月租伍千"` → `"月租五仟"` |
| **口語化** | 加入語助詞、網路用語 | `"幫我找個這樣的ㄟ"` |
| **數字格式** | 全形數字、國字數字 | `"5000"` → `"五千"` / `"５０００"` |
| **標點省略** | 移除所有標點 | `"南區，套房，台電"` → `"南區套房台電"` |

> 噪音施加比例建議：從測試集 3993 筆中各取 200 筆，每筆套用 1–2 種噪音，共 1000 筆噪音測試集。

### 評估腳本修改

修改 `pipeline/model_training/evaluate_model.py`（或直接在 Trainer 的 evaluate 方法後加入），支援接受外部 test_path 參數：

```bash
python -m pipeline.model_training.evaluate_model --test_path data/processed/noisy_test.json
```

---

## 執行順序建議

```
1. Group A（4 runs）   ← 最核心，先做，報告時最有說服力
2. Group C（4 runs）   ← 和 A 共用部分 run（C-1 = v2.9 現狀，可直接使用現有結果）
3. Group B（4 runs）   ← 需要收斂曲線，額外工作量最高，最後做
4. Group D（0 run）    ← 只需在 C-1 / C-2 checkpoint 上評估，不需重訓
```

**總計**：11 次新訓練 run（C-1 共用 v2.9 現狀結果，Group D 無需重訓）

---

## 結果記錄格式

每次 run 結束後記錄以下數值（來自 `evaluate_model.py` 輸出）：

| Run | F1 | Precision | Recall | NDCG@5 | 備注 |
|:---|:---:|:---:|:---:|:---:|:---|
| A-0 CE only | | | | | |
| A-1 +RankNet | | | | | |
| A-2 +ListNet | | | | | |
| A-3 +RankNet+ListNet | **0.855** | **0.827** | — | **0.833** | v2.9 現狀 |
| B-1 α固定0.12 | | | | | |
| B-2 α固定0.25 | | | | | |
| B-3 α固定0.38 | | | | | |
| B-4 α cosine | **0.855** | **0.827** | — | **0.833** | v2.9 現狀 |
| C-2 no FGM | | | | | |
| C-3 no R-Drop | | | | | |
| C-4 no FGM no R-Drop | | | | | |
| D noisy / with FGM | | | | | | 噪音集 |
| D noisy / no FGM | | | | | | 噪音集 |

---

## 最終 README 輸出目標

實驗完成後，在 README 的「訓練策略」章節後插入「消融實驗」章節，格式如下：

```markdown
## 消融實驗（Ablation Study）

以下實驗固定 Teacher（rbt6 v2.9）、資料集、Tokenizer、超參數，
每次僅調整一個變數，在同一測試集（n=3,993）評估。

### A. 損失函數組合

| 設定 | F1 | NDCG@5 | Δ NDCG |
|:---|:---:|:---:|:---:|
| Baseline（CE only）| XX% | 0.XXX | — |
| + RankNet | XX% | 0.XXX | +0.0XX |
| + ListNet | XX% | 0.XXX | +0.0XX |
| **+ RankNet + ListNet（v2.9）** | **85.5%** | **0.833** | **+0.0XX** |

→ RankNet 和 ListNet 各自貢獻了 NDCG +X.X%，合用有 X% synergy（或接近線性疊加）。

### B. KD α 排程

| α 策略 | NDCG@5 |
| 固定 0.12 | 0.XXX |
| 固定 0.25 | 0.XXX |
| 固定 0.38 | 0.XXX |
| **餘弦退火 0.38→0.12（v2.9）** | **0.833** |

### C. 正規化技術

| 設定 | NDCG@5 | 噪音集 NDCG@5 |
| no FGM, no R-Drop | 0.XXX | 0.XXX |
| + R-Drop only | 0.XXX | 0.XXX |
| + FGM only | 0.XXX | **0.XXX** |
| **+ FGM + R-Drop（v2.9）**| **0.833** | **0.XXX** |

→ FGM 在乾淨測試集提升 +X.X%，在噪音測試集提升 +XX%（口語輸入泛化性）。
```

---

## 可調整的討論點

**你可以選擇跳過或縮減以下部分：**

1. **Group B 收斂曲線**：需要額外寫 NDCGLogCallback，工作量較高。如果時間有限，只跑 B-2（固定 0.25）作對比，省掉曲線圖，只呈現最終數字也足夠。

2. **Group D 噪音測試集**：需要寫 `noise_generator.py`。如果租屋查詢場景中口語泛化性不是評審關注點，可以降低優先級或只做 2 種噪音類型（縮寫 + 錯字）。

3. **Teacher 也做 A 組消融**：如果只想節省時間，Teacher 固定用 v2.9 Teacher，只做 Student 的損失消融。但如果要完整說明「為什麼 Teacher 也需要 RankNet/ListNet」，Teacher 消融能提供最強的理論支撐（已有 v2.6 CE-only 的歷史數據可部分替代）。

4. **總優先排序**：A → C-2（no FGM）→ B-2（α=0.25）三個 run 是最小可行集合，可以構成一張 3 行的消融表，已足以回應「這些機制是否必要」的質疑。
