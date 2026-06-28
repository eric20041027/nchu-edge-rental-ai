/**
 * 結構化設施 boost 確定性 self-check(node 直跑,非 CI):
 *   node frontend/js/structured-boost.selfcheck.mjs
 *
 * 守:① parser 口語隱喻 → 設施意圖映射正確 ② 假房源 + 假意圖 → boost 命中判定正確。
 * 前端專案無 JS 測試框架,故用無依賴 assert 腳本(對齊 ponytail: 留最小可跑檢查)。
 */
import assert from 'node:assert';
import { parseFacilityIntents } from './constraint-parser.js';
import { boolFieldState } from './property-features.js';

// ① 口語隱喻 → 設施意圖(bi-encoder 召回弱的那些)
const intentCases = [
    ['不想每天提水上樓', 'water_dispenser'],
    ['懶得一直出門買水', 'water_dispenser'],
    ['我有機車要停', 'has_parking'],
    ['夏天怕被收很貴的電費', 'is_taipower'],
    ['想把棉被拿出去曬太陽', 'dryingArea'],
    ['租金想拿來報稅扣抵', 'taxDeductible'],
    ['不想另外付水費', 'waterIncluded'],
];
for (const [q, field] of intentCases) {
    assert.strictEqual(parseFacilityIntents(q)[field], true, `parser 漏認: ${q} → ${field}`);
}

// 一般 query 不該誤觸發設施意圖
const neg = parseFacilityIntents('找便宜的套房');
assert.ok(!Object.values(neg).some(Boolean), '一般 query 誤觸發設施意圖');

// ② propMatchesFacility 邏輯(複製判定,驗布林欄 + ce_text 文字回退)
function propMatchesFacility(prop, intents) {
    const ceText = String(prop.ce_text || prop.text || '');
    if (intents.water_dispenser && boolFieldState(prop, 'water_dispenser') === 'yes') return true;
    if (intents.has_parking &&
        (boolFieldState(prop, 'has_parking') === 'yes' || ceText.includes('車位') || ceText.includes('停車'))) return true;
    if (intents.is_taipower && boolFieldState(prop, 'is_taipower') === 'yes') return true;
    if (intents.dryingArea && ceText.includes('曬衣')) return true;
    if (intents.taxDeductible && ceText.includes('可報稅')) return true;
    if (intents.waterIncluded && ceText.includes('含水')) return true;
    return false;
}
// 飲水機房源命中飲水機意圖
assert.ok(propMatchesFacility({ water_dispenser: true }, { water_dispenser: true }),
    '飲水機房源未命中飲水機意圖');
// 無飲水機房源不命中
assert.ok(!propMatchesFacility({ water_dispenser: false }, { water_dispenser: true }),
    '無飲水機房源誤命中');
// 台電意圖 + is_taipower 房源
assert.ok(propMatchesFacility({ is_taipower: true }, { is_taipower: true }), '台電房源未命中');
// 曬衣場走 ce_text 文字
assert.ok(propMatchesFacility({ ce_text: '套房 曬衣場 陽台' }, { dryingArea: true }), '曬衣場文字未命中');

console.log('PASS: 結構化 boost self-check (parser 映射 + 命中判定)');
