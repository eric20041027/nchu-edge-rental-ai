// 階段② 反饋重排 self-check — 確定性、node、assert、無框架。
// 跑:node tests/feedback_rerank.test.mjs
// spec: docs/spec/feedback-rerank.md「Testing Strategy」六 case。

import assert from 'node:assert/strict';
import {
    FEEDBACK_BONUS, FEEDBACK_PENALTY,
    propertyIdOf, loadFeedbackScores, applyFeedbackRerank,
} from '../frontend/js/feedback.js';

const P = (id, score) => ({ property: { id }, score });          // 假房源結果
const ids = (arr) => arr.map(r => r.property.id);                 // 取排序後 id 序列

let n = 0;
const test = (name, fn) => { fn(); n++; console.log(`  ✓ ${name}`); };

// ① 👎 把高分房源壓到低分房源之下
test('👎 sinks a high-scored property below a lower one', () => {
    const input = [P('A', 90), P('B', 70)];           // A 本來在前
    const out = applyFeedbackRerank(input, { A: -1 }); // 👎 A → 90-25=65 < 70
    assert.deepEqual(ids(out), ['B', 'A']);
    assert.equal(out.find(r => r.property.id === 'A').score, 90 - FEEDBACK_PENALTY);
});

// ② 👍 把低分房源推上去
test('👍 lifts a low-scored property above a higher one', () => {
    const input = [P('A', 80), P('B', 75)];
    const out = applyFeedbackRerank(input, { B: 1 });  // 👍 B → 75+8=83 > 80
    assert.deepEqual(ids(out), ['B', 'A']);
    assert.equal(out.find(r => r.property.id === 'B').score, 75 + FEEDBACK_BONUS);
});

// ③ 空回饋 → 順序與輸入「依 score 降冪 sort」完全一致(空回饋 == 階段①)
test('empty feedback == stage①: pure score sort, no change', () => {
    const input = [P('A', 60), P('B', 90), P('C', 75)];
    const out = applyFeedbackRerank(input, {});
    assert.deepEqual(ids(out), ['B', 'C', 'A']);       // 純 score 降冪
    assert.deepEqual(out.map(r => r.score), [90, 75, 60]);
});

// ④ clamp 不越界:👍 觸頂 100、👎 觸底 0
test('clamp keeps score within 0–100', () => {
    const out = applyFeedbackRerank([P('hi', 98), P('lo', 10)], { hi: 1, lo: -1 });
    assert.equal(out.find(r => r.property.id === 'hi').score, 100);  // 98+8 → clamp 100
    assert.equal(out.find(r => r.property.id === 'lo').score, 0);    // 10-25 → clamp 0
});

// ⑤ 同 propertyId 多筆回饋 → 取最後一筆(最近意圖優先)
test('multiple votes for same id → last one wins', () => {
    const log = [
        { propertyId: 'A', vote: 1 },
        { propertyId: 'A', vote: -1 },  // 最後是 👎
        { propertyId: 'A', vote: 1 },   // …又改 👍 → 最終 👍
    ];
    const scores = loadFeedbackScores(log);
    assert.equal(scores.A, 1);
    const onlyDown = loadFeedbackScores([{ propertyId: 'A', vote: 1 }, { propertyId: 'A', vote: -1 }]);
    assert.equal(onlyDown.A, -1);  // 最後是 👎 → 👎
});

// ⑥ applyFeedbackRerank 不 mutate 入參(深比對前後)
test('does not mutate input array or objects', () => {
    const input = [P('A', 90), P('B', 70)];
    const snapshot = JSON.parse(JSON.stringify(input));
    applyFeedbackRerank(input, { A: -1 });
    assert.deepEqual(input, snapshot);  // 入參一字未動
});

// 補:propertyId fallback id→url
test('propertyIdOf falls back to url when id missing', () => {
    assert.equal(propertyIdOf({ id: 'x', url: 'u' }), 'x');
    assert.equal(propertyIdOf({ url: 'u' }), 'u');
});

console.log(`\n${n} checks passed.`);
