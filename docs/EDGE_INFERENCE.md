# Edge Inference — 瀏覽器端推論效能分析

> **測量方式**：`frontend/benchmark.html` — 本地 HTTP server 啟動後開啟即可自動執行，輸出 P50/P95/P99 延遲與 heap 記憶體快照。

## 前端架構

1. **多 Web Worker 並行推論**：NER + Cross-Encoder + bi-encoder 各有獨立 Worker，主線程零阻塞
2. **Cache API + Service Worker**：`.onnx` cache-first；HTML/JS stale-while-revalidate；版本號控制快取失效
3. **串流進度追蹤**：Fetch API 監控資料流，精確顯示各模型各自的百分比進度
4. **向量召回（bi-encoder）**：query 經 bi-encoder ONNX 編碼後，於瀏覽器端對 974 筆房源向量做暴力 cosine，取 Top-30 作為主要召回（rule-based 關鍵字召回降為 fallback）
5. **NER BGT 預算過濾**：解析萬/千/k/中文數字，支援方向感知（「以上」= 下限，「以內」= 上限）
6. **推薦反饋**：每張卡片附 👍/👎，記錄至 localStorage（最多 500 筆）

---

## 召回階段：bi-encoder 向量召回

召回階段（產生 30 個精排候選）現以 **bi-encoder 向量 cosine** 為主，取代原先的 rule-based 關鍵字評分：

| 項目 | 數值 |
|:---|:---|
| 召回模型 | bi-encoder rbt3 Dynamic INT8（`bi_encoder_dir/bi_encoder_quant.onnx`，**38.2 MB**;原 rbt6 57 MB,#78 蒸餾）|
| Worker | `frontend/js/bi-encoder-worker.js`（獨立 Web Worker）|
| 輸出 | 768 維 embedding，graph 內含 mean-pool + L2-norm |
| 房源向量 | `frontend/assets/property_embeddings.json`，974 × 768 float16（~6.0 MB），已 L2-normalized |
| 召回流程 | query → bi-encoder 編碼 → mean-pool/L2-norm → 對 974 筆房源向量暴力 cosine → Top-30（與 hard-exclusion allowed set 取交集）|
| cosine 成本 | 純 JS 迴圈，974 × 768 規模極小，可忽略 |
| fallback | rule-based 關鍵字召回（Worker 未就緒 / 800ms 編碼逾時 / `VECTOR_RECALL_ENABLED` kill-switch）|

> **A/B 結果（T7，go/no-go gate）**：semantic Recall@30 0.007 → 0.547；all Recall@30 0.057 → 0.412。判定 **GO**。
> bi-encoder 尚未單獨做 per-pass latency / 載入時間 benchmark；此處僅列已驗證的檔案大小（38.2 MB，#78 蒸餾後）作為新增首次載入權重。

---

## 推論任務規格

| 項目 | 數值 |
|:---|:---|
| 模型 | rbt3 Dynamic INT8 per_channel（`my_custom_model_quant.onnx`，**38.7 MB**，C 組富化 student）|
| 每次查詢 | 30 個候選房源，30 次獨立 forward pass |
| 輸入長度 | `MAX_LENGTH = 128` tokens（query + property text pair；富化文字平均 ~98 token 故需 128）|
| 執行環境 | ONNX Runtime Web + WASM SIMD，最多 4 執行緒 |
| 主線程影響 | **零**（所有推論在 Web Worker 內進行）|

---

## 理論計算量分析

rbt3 每次 forward pass 的主要計算（128 tokens）：

| 元件 | FLOP 估算 |
|:---|:---|
| Self-Attention × 3 層 | 3 × 4 × 128² × 768 ≈ 302M |
| FFN（768→3072→768）× 3 層 | 3 × 2 × 128 × 768 × 3072 ≈ 1,811M |
| 合計（INT8 等效）| **~2.1G INT8 ops / pass** |

> MAX_LENGTH 由 64 → 128 後，Self-Attention 隨序列長度平方成長（64² → 128²），FFN 線性成長，合計計算量約翻倍。

INT8 WASM SIMD 在現代 x86 CPU 上吞吐量約 100–400 GOPS，理論下限 ~2.5ms/pass；實際加上 JS 開銷、tensor 分配與 WASM 呼叫邊界約 **10–60ms/pass**，視裝置而定。

---

## 量化策略評估（Cross-Encoder）

三種量化方案對 4,386 筆測試樣本的對比（batch=1，模擬瀏覽器逐筆推論）：

| 策略 | 大小 | Accuracy | F1 | P50 延遲 | P95 延遲 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Dynamic INT8 per_tensor（舊）| 57.2 MB | 0.7832 | 0.6335 | 18.7 ms | 41.0 ms |
| **Dynamic INT8 per_channel（現用，C 組富化 student）** | **38.7 MB** | **0.8568** | **0.7191** | **14.2 ms** | **24.8 ms** |
| Static INT8 QDQ（已棄用）| 228 MB | 0.7563 | 0.0000 | — | — |

> 現用模型為 2026-06-16 C 組富化 rbt3 student，大小由舊版 57.4 MB 降至 38.7 MB（舊版曾備份為 `*.PREV-20260616.onnx`，已於 dead-weight 清理（收尾 B，PR #44）移除）。Accuracy/F1 為原 per_channel 評估值，富化重訓後 NDCG@5 0.9351 → 0.9475。延遲欄為舊 MAX_LENGTH=64 基準，128 token 實機數字見下方「實測延遲」。

**per_channel 說明**：對每個輸出 channel 獨立計算量化 scale，比全局 per_tensor scale 精度更高，避免不同 channel 動態範圍差異過大導致截斷誤差。batch=1 時 ORT kernel 也跑得比 per_tensor 更快（P95 −40%）。

**Static QDQ 棄用原因**：activation 校準需要代表性樣本，現有 dev set 正負比例不均，導致 activation scale 偏移，所有輸出預測為 negative（F1=0）。ONNX Runtime Web 對 QDQ 格式支援也有限制。

---

## 量化指標評估（NER）

NER 模型（rbt6 INT8 per_channel，`ner_model_quant.onnx`，37 MB）在 dev set 160 筆樣本上的實測結果：

| 指標 | FP32（訓練後）| INT8（部署版）|
|:---|:---:|:---:|
| Precision | — | 0.9985 |
| Recall | — | 0.9897 |
| **F1** | **0.9779** | **0.9941** |

INT8 per_channel 量化後 F1 微幅提升（+0.0162），在 dev set 樣本數較小（160 筆）的條件下屬正常統計波動，無精度退化。

**NER INT8 Latency（batch=1，本機 Mac）：**

| 百分位 | 延遲 |
|:---:|:---:|
| P50 | 5.82 ms |
| P75 | 6.53 ms |
| **P95** | **6.97 ms** |
| P99 | 8.05 ms |

---

## 實測延遲（Windows 11，Intel Core i5-11600KF，HW concurrency 12，4 WASM threads）

> 測試環境：i5-11600KF（6C12T，Rocket Lake）；benchmark.html 5 暖機 + 10 計時，82 Mbps 網路

**模型載入時間（首次 vs 快取後）：**

| 模型 | 首次載入（無快取）| 快取後載入（SW Cache）| 節省 |
|:---|:---:|:---:|:---:|
| NER（37 MB INT8）| **6,501 ms** | **770 ms** | ↓ 88% |
| Cross-Encoder（**舊** 57 MB INT8）| **9,810 ms** | **871 ms** | ↓ 91% |
| **總計** | **16.31 s** | **1.64 s** | ↓ **90%** |

> **上表 Cross-Encoder 載入數字為舊 57 MB（pre-enriched）CE 的 benchmark，非現行 production CE。** production CE 已換 C 組富化 rbt3 student（**38.7 MB**），檔案更小，首次載入應更快（尚未重測；此處保留舊實測為歷史對照）。
>
> **首次載入現為三個模型**：NER + Cross-Encoder + bi-encoder。新增的 bi-encoder（**38.2 MB** INT8，#78 蒸餾後;原 57 MB）外加 `property_embeddings.json`（~6.0 MB），是疊加在原 NER+CE 之上的新首次載入權重；bi-encoder 尚未單獨 benchmark 載入時間，上表未含其數字。Service Worker 於首次載入後快取所有模型。

> 快取後載入：主頁面從 Cache Storage 讀取 ArrayBuffer 並 transfer 給 Worker，無網路請求，時間為純 WASM session 初始化耗時。

**Per-pass latency（單次推論，benchmark.html 10 計時，舊 MAX_LENGTH=64 基準）：**

| 百分位 | 延遲 |
|:---:|:---:|
| avg | 246 ms |
| P50 | 246 ms |
| **P95** | **248 ms** |

> 首次與快取後推論延遲差異 ≤ 23 ms（WASM 本地計算，與網路無關）。
> 上表為舊 MAX_LENGTH=64 模型的高階機（i5-11600KF）數字。

**MAX_LENGTH=128 實機實測（C 組富化模型，單執行緒）：**

| 項目 | 數值 |
|:---|:---|
| per-pair 延遲 | **~152–168 ms/pair** |
| 測試條件 | MAX_LENGTH=128（128 token），單執行緒 |
| 舊 baseline 對照 | ~137 ms/pair（MAX_LENGTH=64） |

> 測試條件已由 64 token 改為 128 token（富化文字需 128），計算量約翻倍（見「理論計算量分析」）；單執行緒下 per-pair 延遲由舊 ~137ms@64 升至 ~152–168ms@128。上方 248ms P95 為高階機 4 執行緒下的舊 64-token 數字，測試環境不同，不可直接相比。

---

## 各裝置延遲推算

以實測高階機（P95 per-pass = 248 ms）為基準，依各裝置 INT8 WASM 吞吐量比例外插：

| 裝置類型 | 代表機型 | per-pass P95 | **30-pass 估算** | 主觀感受 |
|:---|:---|:---:|:---:|:---|
| 高階開發機（已實測）| i5-11600KF，12 執行緒 | **248 ms** | **~7,440 ms** ✅ | 約 7 秒等待 |
| 中階學生筆電 | i5-10th, Ryzen 5 4500U | ~330 ms | ~9,900 ms | 約 10 秒 |
| 中階手機 | Snapdragon 778G, Dimensity 900 | ~600 ms | ~18,000 ms | 約 18 秒 |
| 低階預算手機 | Snapdragon 460, Helio G85 | ~1,350 ms | ~40,000 ms | 明顯等待 |

> benchmark.html 量測為單對推論（1 × forward pass）；30-pass 估算為線性外插，實際受 WASM JIT 預熱影響。手機端數字需實機驗證。

---

## 記憶體實測與崩潰風險

**實測數據（同一環境）：**

```
Heap after model load  : 56.6 MB
Heap after 5th run     : 55.2 MB
Inference delta        : −1.4 MB  ← GC 自然釋放，無洩漏
Session init (一次性)  : 249.9 ms
Model buffer size      : 36.8 MB（ArrayBuffer）
```

**Heap 組成分析：**

```
WASM runtime + ORT kernels : ~10 MB
Transformers.js + vocab    : ~12 MB
ONNX model buffer          : 36.8 MB（heap 外 ArrayBuffer，不觸發 GC）
其他 runtime overhead      : ~5 MB
─────────────────────────────────
實測穩定 heap              : 56.6 MB

推論期間 per-pass tensor spike：
  input_ids / token_type_ids / attention_mask [1×128] int64 ≈ 3 KB
  output logits [1×2] float32                               ≈ 8 B
  ─────────────────────────────
  單次 spike                 : < 4 KB（即時釋放，MAX_LENGTH 128 後約翻倍仍可忽略）
```

**結論：**
- **不會崩潰**：56.6 MB 遠低於 iOS Safari（~1.4 GB）和 Android Chrome（512 MB–1 GB）的 JS heap 上限
- **無記憶體洩漏**：5 組 30-pass 後 heap 反降 1.4 MB，GC 正常運作
- **不會嚴重耗電**：2 秒的一次性 INT8 WASM 計算，非持續佔用；實測無明顯發熱

---

## 執行 Benchmark

```bash
cd frontend && python -m http.server 8000
# 開啟 http://localhost:8000/benchmark.html
# 點擊 "Start Benchmark"，約 30–90 秒後輸出完整報告
```

輸出包含：P50 / P75 / P95 / P99 per-pass latency、30-pass 總延遲、heap before/after delta、執行緒數與 SIMD 環境資訊。
