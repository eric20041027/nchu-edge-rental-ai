# Spec: 階段② 反饋微調 — CE 精排後的回饋重排層

> 上游意圖:`docs/intent/vector-retrieval-roadmap.md`〈階段②確認意圖〉(interview-me, 2026-06-23)。
> 本 spec 把該意圖展開成可驗收規格。實作前需人工 review 通過。

## Objective

在現有三階段管線(NER → bi-encoder 召回 → CE 精排)的**最末端**,加一層
**純後處理回饋重排**:用 localStorage 累積的 per-propertyId 👍/👎,對 CE 算完的
分數加固定調整量(👍 輕推、👎 重罰但不消失),再重排。

- **為何:** 讓推薦隨使用者回饋越用越貼個人,且 per-browser、跨 query/session 累積。
- **使用者:** 這個瀏覽器的使用者 / demo 操作者。回饋只在本機,不上雲。
- **成功長相:** 👎 某房源 → 重跑同/異 query 它明顯下沉;👍 → 上升;跨 session 仍生效;
  kill-switch 關掉 → 行為完全回到階段①(逐位元組等價的排序)。

**核心約束:不碰任何模型權重、零後端、零重訓、純前端、獨立可關、零侵入既有召回+CE。**

## Tech Stack

- 前端純 JS(ES modules),無新依賴。
- 既有資料源:`localStorage['renting_feedback_log']`,schema(實況,見 `frontend/js/app.js:554`):
  ```js
  { ts: ISOString, query: string, propertyId: string, vote: 1 | -1 }  // 上限 500 筆
  ```
- propertyId 公式(實況,見 `frontend/js/app.js:514`):`property.id || property.url`。
  **本層必須復用同一公式**,否則記分對不上回饋。

## Commands

```
# 無 build step(靜態前端)。本機預覽:
python3 -m http.server 8000 --directory frontend   # 或既有 preview 工具

# self-check(確定性、無框架):
node tests/feedback_rerank.test.mjs
```

## Project Structure(本 spec 觸及)

```
frontend/js/inference.js   → recommend():在 1498(CE 算完、sort 前)掛 applyFeedbackRerank()
frontend/js/feedback.js    → 新檔:loadFeedbackScores() + applyFeedbackRerank()(純函式、可單測)
frontend/js/app.js         → 既有 saveFeedback();本 spec 不改寫入端,只讀
tests/feedback_rerank.test.mjs → 新檔:確定性 self-check
docs/spec/feedback-rerank.md   → 本檔
```

> 為什麼抽 `feedback.js`:`applyFeedbackRerank` 要能脫離瀏覽器在 node 下單測,
> 故做成不依賴 DOM/localStorage 的純函式(localStorage 讀取由薄 wrapper 注入)。

## Code Style

純函式、immutable、early return。範例(目標形狀,非最終值):

```js
// feedback.js
export const FEEDBACK_LOG_KEY = 'renting_feedback_log';  // 與 app.js 同源(app.js 之後可 import 收斂)
export const FEEDBACK_BONUS  = 8;   // 👍 升權(疊在 0–100 的 CE score 上)
export const FEEDBACK_PENALTY = 25; // 👎 降權,PENALTY > BONUS(👎 重罰但不消失)

// propertyId 公式必須與 app.js 一致
export const propertyIdOf = (p) => p.id || p.url;

// 薄 reader:唯一碰 localStorage 之處(副作用隔離於此,不進測試路徑)
export function readFeedbackLog() {
    try { return JSON.parse(localStorage.getItem(FEEDBACK_LOG_KEY) || '[]'); }
    catch { return []; }
}

// 把回饋 log 壓成 { propertyId -> 最後一筆 vote }(最近意圖優先,非加總)
export function loadFeedbackScores(log) {
    const latest = {};
    for (const e of log) latest[e.propertyId] = e.vote;  // log 時序遞增 → 後者覆蓋
    return latest;
}

// 純後處理:回傳新陣列,不 mutate 入參;clamp 0–100
export function applyFeedbackRerank(scoredResults, feedbackScores) {
    const adjusted = scoredResults.map(r => {
        const vote = feedbackScores[propertyIdOf(r.property)];
        if (!vote) return r;
        const delta = vote === 1 ? FEEDBACK_BONUS : -FEEDBACK_PENALTY;
        const score = Math.max(0, Math.min(100, r.score + delta));
        return { ...r, score, feedbackAdjusted: vote };  // 標記供 UI/debug
    });
    return adjusted.sort((a, b) => b.score - a.score);
}
```

掛載(`inference.js`,kill-switch 比照 `VECTOR_RECALL_ENABLED`):

```js
// 檔頂:import { readFeedbackLog, loadFeedbackScores, applyFeedbackRerank } from './feedback.js';
const FEEDBACK_RERANK_ENABLED = true;  // 檔頂常數,kill-switch

// recommend() 內,1498 isCancelled() 之後、原 1499 sort 之前:
// inference 只 import feedback.js 的 reader,完全不碰 app.js(R2 消除)。
if (FEEDBACK_RERANK_ENABLED) {
    scoredResults = applyFeedbackRerank(scoredResults, loadFeedbackScores(readFeedbackLog()));
} else {
    scoredResults.sort((a, b) => b.score - a.score);  // 原行為,逐位元組等價
}
```

> kill-switch ON 但無任何回饋時:`feedbackScores` 為空 → 每筆 early return →
> 等同原 sort。即「空回饋 == 階段①行為」,這是驗收條件之一。

## Testing Strategy

**行為判準為主,不做離線 Recall@K A/B。** 回饋即 ground-truth,離線指標會循環論證。

1. **確定性 self-check**(`tests/feedback_rerank.test.mjs`,node、assert、無框架):
   - 給定假 `scoredResults` + 假 `feedbackScores`,斷言重排後順序符合預期。
   - 涵蓋:👎 把高分房源壓到低分房源之下;👍 把低分推上去;
     空回饋 → 順序與輸入 sort 完全一致;clamp 不越界(score 不 <0 或 >100);
     同 propertyId 多筆回饋 → 取最後一筆;不 mutate 入參。
2. **瀏覽器 preview 親自驗**(實跑,不靠口述):
   - 跑一次 query → 對某房源按 👎 → 重跑同 query → 它明顯下沉。
   - 👍 另一房源 → 上升。
   - 換一個 query → 該 👎 房源若出現仍被壓低(跨 query 生效)。
   - reload 頁面(模擬跨 session)→ 回饋仍生效(localStorage 持久)。
   - 設 `FEEDBACK_RERANK_ENABLED = false` → 行為回到階段①。

## Boundaries

- **Always:** propertyId 用 `p.id || p.url`(與 app.js 同);純函式不 mutate;
  clamp 0–100;kill-switch OFF 時逐位元組等價於階段①;改完跑 self-check + preview 親驗。
- **Ask first:** 動 `BONUS`/`PENALTY` 以外的排序公式;改 CE 既有計分(`finalPercentage`);
  改回饋寫入端 schema;加任何依賴。
- **Never:** 重訓/重匯出模型;把 👎 泛化到向量鄰居;硬隱藏房源(👎 只下沉不消失);
  上雲/跨裝置同步;離線 A/B 數字。

## Success Criteria(具體、可測)

- [ ] `FEEDBACK_RERANK_ENABLED = false` 時,`recommend()` 輸出排序與本 PR 前**完全一致**。
- [ ] 空 `renting_feedback_log` 時,ON/OFF 輸出一致(空回饋 == 階段①)。
- [ ] 對房源 X 按 👎 後重跑同 query,X 的最終 score 下降 `PENALTY`(clamp 後),排名下沉。
- [ ] 按 👍 後 score 上升 `BONUS`,排名上升。
- [ ] 換不同 query,X 仍被壓低(跨 query)。reload 後仍生效(跨 session)。
- [ ] self-check 全綠;`applyFeedbackRerank` 不 mutate 入參。
- [ ] 無新依賴;`recommend()` 召回與 CE 段未被改寫(僅在末端加掛點)。

## Resolved Decisions(2026-06-23 人工確認)

1. `BONUS` / `PENALTY` 初值 = **8 / 25**(👎 重罰),為佔位值;**實作時在 preview 用真資料觀察微調**。
   這是「現實需 tuning」的旋鈕,留參數不寫死語意。
2. 被回饋調整的房源**不在 UI 標記**。`feedbackAdjusted` 欄位仍預留供 debug,但不呈現。
3. CE 因 `isCancelled()` 提早 return null 的路徑不經過掛點 → 不受影響,**可接受**。
