// 階段② 反饋重排層 — CE 精排後的純後處理。
// spec: docs/spec/feedback-rerank.md。不碰模型權重,純前端,可關。
//
// 唯一的副作用(readFeedbackLog 讀 localStorage)隔離在本檔頂部;
// 其餘全為純函式,可在 node 下確定性單測。

export const FEEDBACK_LOG_KEY = 'renting_feedback_log';  // 與 app.js 寫入端同源
export const FEEDBACK_BONUS   = 8;   // 👍 升權(疊在 0–100 的 CE score 上)
export const FEEDBACK_PENALTY = 25;  // 👎 降權,PENALTY > BONUS(👎 重罰但不消失)

// propertyId 公式必須與 app.js:514 的 data-id 一致,否則記分對不上回饋。
export const propertyIdOf = (p) => p.id || p.url;

// 薄 reader:唯一碰 localStorage 之處(副作用隔離,不進測試路徑)。
export function readFeedbackLog() {
    try { return JSON.parse(localStorage.getItem(FEEDBACK_LOG_KEY) || '[]'); }
    catch { return []; }
}

// 把回饋 log 壓成 { propertyId -> 最後一筆 vote }。
// log 時序遞增(app.js 用 push 追加),後者覆蓋前者 → 最近意圖優先,非加總。
export function loadFeedbackScores(log) {
    const latest = {};
    for (const e of log) latest[e.propertyId] = e.vote;
    return latest;
}

// 純後處理重排:回傳新陣列,不 mutate 入參;score clamp 0–100。
// scoredResults: [{ property, score, ... }],feedbackScores: { propertyId -> vote }。
export function applyFeedbackRerank(scoredResults, feedbackScores) {
    const adjusted = scoredResults.map(r => {
        const vote = feedbackScores[propertyIdOf(r.property)];
        if (!vote) return r;  // 無回饋 → 原樣(空回饋 == 階段①)
        const delta = vote === 1 ? FEEDBACK_BONUS : -FEEDBACK_PENALTY;
        const score = Math.max(0, Math.min(100, r.score + delta));
        return { ...r, score, feedbackAdjusted: vote };  // feedbackAdjusted 預留 debug,UI 不呈現
    });
    return adjusted.sort((a, b) => b.score - a.score);
}
