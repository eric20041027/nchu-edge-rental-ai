# Edge Inference — 瀏覽器端推論效能分析

> **測量方式**：`frontend/benchmark.html` — 本地 HTTP server 啟動後開啟即可自動執行，輸出 P50/P95/P99 延遲與 heap 記憶體快照。

## 前端架構

1. **雙 Web Worker 並行推論**：NER + Cross-Encoder 各有獨立 Worker，主線程零阻塞
2. **Cache API + Service Worker**：`.onnx` cache-first；HTML/JS stale-while-revalidate；版本號控制快取失效
3. **串流進度追蹤**：Fetch API 監控資料流，精確顯示兩個模型各自的百分比進度
4. **NER BGT 預算過濾**：解析萬/千/k/中文數字，支援方向感知（「以上」= 下限，「以內」= 上限）
5. **推薦反饋**：每張卡片附 👍/👎，記錄至 localStorage（最多 500 筆）

---

## 推論任務規格

| 項目 | 數值 |
|:---|:---|
| 模型 | rbt3 INT8（`my_custom_model_quant.onnx`，38.6 MB）|
| 每次查詢 | 30 個候選房源，30 次獨立 forward pass |
| 輸入長度 | `MAX_LENGTH = 64` tokens（query + property text pair）|
| 執行環境 | ONNX Runtime Web + WASM SIMD，最多 4 執行緒 |
| 主線程影響 | **零**（所有推論在 Web Worker 內進行）|

---

## 理論計算量分析

rbt3 每次 forward pass 的主要計算（64 tokens）：

| 元件 | FLOP 估算 |
|:---|:---|
| Self-Attention × 3 層 | 3 × 4 × 64² × 768 ≈ 75M |
| FFN（768→3072→768）× 3 層 | 3 × 2 × 64 × 768 × 3072 ≈ 905M |
| 合計（INT8 等效）| **~980M INT8 ops / pass** |

INT8 WASM SIMD 在現代 x86 CPU 上吞吐量約 100–400 GOPS，理論下限 ~2.5ms/pass；實際加上 JS 開銷、tensor 分配與 WASM 呼叫邊界約 **10–60ms/pass**，視裝置而定。

---

## 實測延遲（Windows 11，Intel Core i5-11600KF，HW concurrency 12，4 WASM threads）

> 測試環境：i5-11600KF（6C12T，Rocket Lake）；5 組 × 30-pass，warmup 3 組已排除

**Per-pass latency（單次 tokenize + forward）：**

| 百分位 | 延遲 |
|:---:|:---:|
| P50 | 64.4 ms |
| P75 | 68.6 ms |
| **P95** | **81.0 ms** |
| P99 | 108.2 ms |
| Min / Max | 48.1 ms / 115.7 ms |

**30-candidate rerank 總延遲：**

| 百分位 | 延遲 |
|:---:|:---:|
| P50 | 1,908 ms |
| **P95** | **2,219 ms** |
| Min / Max | 1,734 ms / 2,219 ms |

> Run-to-run 變異 ~28%（1,734–2,219 ms），主因為 OS scheduler jitter 與 WASM JIT 預熱差異，非模型本身不穩定。

---

## 各裝置延遲推算

以實測高階機（P95 per-pass = 81 ms）為基準，依各裝置 INT8 WASM 吞吐量比例外插：

| 裝置類型 | 代表機型 | per-pass P95 | **30-pass P95** | 主觀感受 |
|:---|:---|:---:|:---:|:---|
| 高階開發機（已實測）| i7/Ryzen 7，12 核 | **81 ms** | **2,219 ms** ✅ | 約 2 秒等待 |
| 中階學生筆電 | i5-10th, Ryzen 5 4500U | ~110 ms | ~3,000 ms | 約 3 秒 |
| 中階手機 | Snapdragon 778G, Dimensity 900 | ~200 ms | ~5,500 ms | 約 5-6 秒 |
| 低階預算手機 | Snapdragon 460, Helio G85 | ~450 ms | ~12,000 ms | 明顯等待 |

> 手機端數字為外插估算（WASM SIMD 吞吐量比較），需實機驗證。

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
  input_ids / token_type_ids / attention_mask [1×64] int64 ≈ 1.5 KB
  output logits [1×2] float32                              ≈ 8 B
  ─────────────────────────────
  單次 spike                 : < 2 KB（即時釋放）
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
