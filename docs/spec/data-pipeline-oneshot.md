# Spec: 階段③ 資料管線一鍵化(兩段)+ 砍 FB

> 上游意圖:`docs/intent/vector-retrieval-roadmap.md`〈階段③確認意圖〉(interview-me, 2026-06-23,
> 含 code 查證後的「兩段」現實修正)。實作前需人工 review 通過。

## Objective

把「爬蟲 → 富化 → 重算向量 → 上線」從散落的手動 script 串成**兩段可重複管線**,
段內零手動。先打通管線,之後再接新平台 crawler。

- **為何:** 擴充房源從「手動跑一串 script」變「跑一條命令」;流程手動是房源量上不去的根因。
- **使用者:** 維護者。受益是 demo 使用者(房源更多)。
- **成功長相:** 本機段一條命令產出 `property_data.json`(本機親驗);Colab 段一條命令產出
  `property_embeddings.json`。砍掉 FB 來源。

**核心約束:中興主軸(geo_tier/distance 相對中興,不動)、不跨城市、不引入 city、
edge-first 不變、先管線後平台、不重訓(向量重算用已訓練權重)。**

## 現實:為何分兩段(code 查證)

| 前端產物 | 生成 script | 本機(python3.12, onnxruntime, 無 torch)能跑? |
|---|---|---|
| `property_data.json` | `precompute_embeddings.py` | ✅ 純文本解析,CPU 快 |
| `property_embeddings.json` | `build_property_embeddings.py` | ❌ build 需 **torch** forward;但 `--check` mode 無需 torch |

→ **本機段** = crawl(可選)→ `precompute_embeddings.py`(基本富化,**無 ce_text**)
  → `precompute_ce_text.py --write`(補 `ce_text`,**CE 精排必需**;缺它前端 CE 會 OOD 退化,
  見 `inference.js:1470`)→ `build_property_embeddings.py --check` 驗。**三步,純 CPU 可跑。**
  - 查證(2026-06-23 本機實跑):precompute + ce_text 兩步跑完,產物與現存 `property_data.json`
    **逐筆完全一致**(identical content: True)→ 證明未改邏輯、本機段步驟正確。
→ **Colab 段** = `property_data.json` + 已訓練 bi-encoder 權重 → `build_property_embeddings.py`(torch)→ `property_embeddings.json`。

## Tech Stack

- Python 3.12(本機 venv,有 onnxruntime,無 torch);Colab/A100(有 torch)跑向量重算。
- 既有 script(都在 origin/main 追蹤,已查證):
  - `pipeline/data_prep/precompute_embeddings.py` — 讀 `data/raw/nchu_rental_info.csv` → 寫
    `frontend/assets/property_data.json`(`main()`,L226;out L282)。純 CPU。
  - `pipeline/data_prep/build_property_embeddings.py` — 讀 `property_data.json`(+ 權重)→ 寫
    `property_embeddings.json` float16(L73-74);`--check` mode 無 torch(L101-113);`build` 需 torch(L135+)。
- 既有 crawler:`pipeline/crawlers/`(ddroom/nchu 結構化;**fb 待砍**)。
- 既有 orchestrator `pipeline/orchestrator.py` 的 Phase 1-2-3 是 crawl→data_prep→**模型訓練**,
  **不是**階段③要的路徑(它的 Phase 3 是重訓,本案 out-of-scope)→ **本案不改 orchestrator**,
  另寫薄入口。

## Commands

```
# 本機段(純 CPU,段內零手動):
python -m pipeline.build_frontend_data            # 新薄入口:precompute → property_data.json → --check 驗
#   (可選 --crawl 先跑 ddroom/nchu 爬蟲更新 CSV;預設用既有 CSV)

# Colab 段(需 torch + 已訓練權重):
python -m pipeline.data_prep.build_property_embeddings   # → property_embeddings.json

# self-check(本機,無 torch):
node /dev/null  # n/a;本案 self-check 為 python:
python -m pipeline.data_prep.build_property_embeddings --check   # 驗 property_data.json 記錄數/欄位
```

## Project Structure(本 spec 觸及)

```
pipeline/build_frontend_data.py        → 新薄入口:串本機段三步(precompute → ce_text --write → --check)
pipeline/__init__.py                   → 改 lazy import(PEP 562):重依賴(crawlers→pydantic)延遲,本機可跑
pipeline/data_prep/__init__.py         → 同上 lazy 化(models→pydantic 延遲)
pipeline/data_prep/precompute_embeddings.py  → 既有,不改邏輯(本機段步驟1)
pipeline/data_prep/precompute_ce_text.py     → 既有,不改邏輯(本機段步驟2,補 ce_text)
pipeline/data_prep/build_property_embeddings.py → 既有,不改邏輯(Colab 段)
pipeline/crawlers/fb_crawler.py        → 刪除(砍 FB)
pipeline/crawlers/config.py            → 移除 fb_output_json 定義(L20 附近)
data/raw/fb_queries.json               → 刪除(連 FB 訓練查詢來源一起砍)
pipeline/data_prep/generate_dataset.py → 移除 external_query_files 的 fb_queries 行(L971)
docs/spec/data-pipeline-oneshot.md     → 本檔
```

> 為什麼新寫薄入口而非改 orchestrator:orchestrator 的 Phase 3 是模型重訓(out-of-scope),
> 串它會把重訓拖進來。本機段只需 precompute + check 兩步,一個薄 main 即可,YAGNI。

## Code Style

薄入口(目標形狀):

```python
# pipeline/build_frontend_data.py
"""本機段一鍵:crawl(可選)→ precompute → property_data.json → --check 驗。需 torch 的向量重算在 Colab 段。"""
import argparse, subprocess, sys

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", action="store_true", help="先跑 ddroom/nchu 爬蟲更新 CSV(需網路)")
    args = ap.parse_args()

    if args.crawl:
        from pipeline.runners import run_crawlers
        from pipeline.crawlers import CrawlerConfig
        run_crawlers(CrawlerConfig())          # 砍 FB 後只剩 ddroom/nchu

    # 富化:CSV → property_data.json(純 CPU)
    from pipeline.data_prep import precompute_embeddings
    precompute_embeddings.main()

    # 段尾驗證:確認 property_data.json 記錄數/欄位 OK(無 torch)
    rc = subprocess.call([sys.executable, "-m", "pipeline.data_prep.build_property_embeddings", "--check"])
    if rc != 0:
        print("[build_frontend_data] --check 失敗:property_data.json 異常", file=sys.stderr)
        return rc
    print("[build_frontend_data] 本機段完成。下一步 Colab 段重算向量。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

## Testing Strategy

**行為判準為主。**

1. **本機段端到端親驗**(python3.12 venv,純 CPU):
   - 跑 `python -m pipeline.build_frontend_data`(不帶 --crawl,用既有 CSV)。
   - 斷言:`property_data.json` 被重新產出、記錄數 == 既有 704(或爬蟲更新後的新數)、
     `--check` exit 0、富化欄位(geo_tier/has_*)齊全。
   - 與既有 `property_data.json` diff:邏輯未改 → 產物應逐欄一致(除非換了 CSV)。
2. **砍 FB 不破壞既有流程:**
   - `precompute_embeddings.py` + `generate_dataset.py` 跑得過(fb_queries.json 缺失已 graceful skip,
     報告 Q6 查證)。grep 確認無殘留 fb import 報錯。
3. **Colab 段不在本機驗**(需 torch)→ 誠實標註;以 `--check` 作為本機能做的最大驗證
   (確認 property_data.json 對 build 而言格式正確)。

## Boundaries

- **Always:** 串既有 script、不改其富化/向量邏輯;本機段純 CPU 可跑可驗;
  砍 FB 後 grep 確認無殘留報錯;改完本機段親跑一次。
- **Ask first:** 改 precompute/build 的富化或向量邏輯;改 orchestrator;動 geo_tier/distance 地理語境;
  接新平台 crawler(那是「先管線後平台」的後續,另案)。
- **Never:** 重訓模型;引入 city 欄位/跨城市;後端服務;假裝向量重算能本機跑(torch 依賴誠實標註)。

## Success Criteria(具體、可測)

- [ ] `python -m pipeline.build_frontend_data` 一條命令本機跑通,零手動產出 `property_data.json`。
- [ ] 段尾 `--check` exit 0,記錄數/富化欄位驗證通過。
- [ ] 不帶 --crawl 用既有 CSV 時,產物與當前 `property_data.json` 邏輯一致(未改邏輯)。
- [ ] `fb_crawler.py` + `data/raw/fb_queries.json` 刪除;`crawlers/config.py` 無 `fb_output_json` 殘留;
      `generate_dataset.py:971` 的 fb_queries 行移除;`run_crawlers` 只剩 ddroom/nchu;
      重跑 `generate_dataset.py` 不報錯(其餘 5 個 external query 來源不受影響)。
- [ ] Colab 段命令 `python -m pipeline.data_prep.build_property_embeddings` 文檔化(README/spec),
      本機以 `--check` 為最大驗證,torch 依賴誠實標註。
- [ ] orchestrator.py 未被改動(本案不碰)。

## Resolved Decisions(2026-06-23 人工確認)

1. 本機段**預設不爬**(用既有 CSV,純離線可驗);`--crawl` 為 opt-in(需網路)。
2. `build_frontend_data.py` 放 **`pipeline/` 根**(與 orchestrator.py 同層,跨 crawl + data_prep 兩域)。
3. **不保留 `fb_queries.json`** —— 連同 FB 訓練查詢來源一起砍(徹底擺脫 FB,與意圖一致)。
   - **連帶影響(查證 `generate_dataset.py:970-977`):** `fb_queries.json` 是 6 個 external query
     來源之一,被注入 `recommendation_train.json`。砍掉後,**未來重跑 data_prep 時訓練資料不再含
     FB 查詢樣本**(現存 `recommendation_train.json` 不受影響,除非重跑)。
   - **動作:** 刪 `data/raw/fb_queries.json` + 從 `generate_dataset.py:971` external_query_files
     移除該行 + 刪 `crawlers/config.py:20` 的 `fb_output_json` + 刪 `fb_crawler.py`。
   - 此項已超出「只砍 crawler」,進入「砍訓練資料來源」,屬 **Ask first** 範疇,已確認。
