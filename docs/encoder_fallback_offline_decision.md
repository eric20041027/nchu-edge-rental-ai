# Bi-encoder 意圖層 fallback — 離線覆蓋率天花板驗證 (P0 上線決策)

> **2026-06-27 更新**:前端 no-op 骨架(`ENCODER_FALLBACK_ENABLED`、`initEncoderFallback`、
> `encoderFallbackExpand`、狀態變數)已於架構精簡批次1移除(本決策 NO-GO、骨架長期未動)。
> **要救此功能**:依本文檔「決策」段重建 —— 字面表擴列舉(便宜路徑)優先,或換各向同性編碼器;
> 離線評估腳本 `pipeline/data_prep/eval_encoder_fallback_offline.py` 仍保留可重跑。

**日期**: 2026-06-14
**結論**: ❌ **NO-GO**(目前形式)。不投入 transformers.js / 不灌 205MB、不翻 `ENCODER_FALLBACK_ENABLED`。
**重跑**: `python pipeline/data_prep/eval_encoder_fallback_offline.py [--sweep] [--dump-misroutes N]`

## 為什麼做這個驗證
骨架已合 main(PR#4,`071ae0d`,flag 關 = no-op)。上線前要先用**純 Python 離線**證明
fallback 值得 205MB 前端成本——而不是直接燒 transformers.js 才發現不行。

- 編碼器:`shibing624/text2vec-base-chinese`(mean-pool + L2,max_len 64),與
  `build_intent_prototypes.py` 同源。
- corpus:`data/raw/llm_queries.json` + `data/raw/hard_traps.json`,去重 **2093** 條口語 query
  (含品種詞「黃金獵犬/柴犬/柯基」、「興大正門」等規則表沒列舉的講法)。
- 判對錯:**類別啟發式自動判**。每條 query 帶 category(寵物衝突/設備衝突/噪音衝突…),
  把 132 規則用 expansion token 映射到同一組 topic;fallback top-1 routed rule 的 topic
  == query category topic → CORRECT,落到別 topic → MISROUTE,thr 以下沒命中 → MISS。
  *(rule→topic 映射是手工近似,故 headline 是天花板估計,非定論。)*

## 數字(typed literal-miss queries,n=947)

字面層覆蓋率:caught **37.5%** / literal-miss **62.5%**(1308/2093)。這份 corpus 比舊
1500 集更難,所以漏接率比 memory 記的 33.5%/665 高——正是壓力測試要的。

thr=0.55 / top-3(骨架提案值):

| 指標 | 值 |
|---|---|
| CORRECT(路由到對的 topic) | **49.0%** |
| MISROUTE(路由到錯的 topic) | **28.3%** |
| MISS(fallback 靜默) | 22.7% |
| precision-of-fired(correct / 有開火) | **63.4%** |

### 門檻敏感度掃描(`--sweep`)
| thr | correct | misroute | miss | precision |
|---|---|---|---|---|
| 0.50 | 56.1% | 40.5% | 3.4% | 58.0% |
| **0.55** | **49.0%** | **28.3%** | 22.7% | **63.4%** |
| 0.60 | 39.5% | 9.6% | 50.9% | 80.4% |
| 0.65 | 26.4% | 2.3% | 71.3% | 91.9% |
| 0.70 | 12.2% | 0.3% | 87.4% | 97.5% |

`top_k=1` 與 `top_k=3` 數字完全相同 → 第2/3命中從不超過 thr,top-k 無作用。

## 判讀
1. **沒有任何門檻同時高覆蓋 + 高精準。** 要 precision≥80% 得拉到 thr0.60,correct 掉到
   39.5%(且 51% 靜默);要 precision≥90%(thr0.65)correct 崩到 26.4%。這正是
   text2vec **各向異性 0.294** 天花板在咬:thr0.55 附近 cosine 訊號僅略高於噪聲,
   近門檻命中近乎隨機路由。
2. **28.3% misroute 是主動傷害,不只是沒幫上忙。** 它把 query 的擴展詞灌到**錯的特徵
   topic**(地點 query→「只租幾個月/頂樓加蓋」、寵物 query→「不想跟人共用」),
   直接踩中「擴展詞路由錯會傷 NDCG」的風險(任務警示#2)。fallback 開火時有 1/4 以上是錯的。
3. **misroute 樣態 = 表層詞共現,非意圖。** 大量「興大正門」地點句飄到隨機短規則
   (沒有「興大正門」原型,query 浮到 thr0.55 附近任何泛用短句)。品種詞那類**有**被
   接住(柴犬/柯基/拉布拉多→養狗/可貓 cos 0.55–0.66,見 --dump-correct)——但這只佔
   pet topic 的一半(pet correct 50.8%),且夾帶 16% misroute。

CP 值:205MB 前端成本(對比專案一路在做的 INT8/WASM 加速)換來「開火即 1/4 錯、且拉精準就
近乎不開火」的 fallback,**不值得**。

## 決策
- **不接 transformers.js、不翻 flag。** 骨架維持 main 上 no-op(零風險),保留結構待日後。
- **若要救**,優先順序:
  1. 字面表擴列舉(便宜):品種詞、「興大正門/正門口」地點別名直接進 `semantic_rules.json`
     字面層——這些是 misroute 重災區且枚舉成本低,直接砍掉 fallback 要扛的部分。
  2. 各向異性才是根因——蒸餾 student(memory 記 0.382,比 FP16 0.294 還差)救不了精準度
     問題,**先別投入**。
  3. 真要做語意 fallback,需換更各向同性的編碼器或加 whitening,且 thr 要 ≥0.60 配高精準,
     接受低覆蓋(只接住高信心句)——但那時覆蓋率 39.5%,收益已小。

## 重跑指令
```bash
python pipeline/data_prep/eval_encoder_fallback_offline.py                  # headline + per-topic
python pipeline/data_prep/eval_encoder_fallback_offline.py --sweep          # thr x top_k 掃描
python pipeline/data_prep/eval_encoder_fallback_offline.py --dump-misroutes 25 --dump-correct 15
```
