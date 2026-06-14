/**
 * inference.js — Sentence-Pair Classification Recommendation Engine
 *
 * Uses the fine-tuned ALBERT model for sentence-pair classification:
 * Input:  [CLS] user_query [SEP] property_description [SEP]
 * Output: logits → softmax → match probability
 *
 * Sequential inference to avoid ONNX "Session already started" error.
 */
import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

let worker = null;
let propertyData = [];
let pendingInference = new Map();
let inferenceIdCounter = 0;

// NER worker state
let nerWorker   = null;
let pendingNER  = new Map();
let nerIdCounter = 0;
let nerReady    = false;

// --- Bi-encoder intent fallback (SKELETON, default OFF) ----------------------
// 字面意圖表 (expandQueryIntent) 只認列舉的口語講法;規則表沒有的講法 (例
// 「黃金獵犬」「想養柯基」) 會字面零命中、整句不擴展。bi-encoder fallback 用
// text2vec 編碼單句 query,與離線預算的 132 原型向量做 cosine,thr 以上 top-k
// 路由到對應規則的擴展詞,接住這類漏接。
//
// 設計 (見 semantic_expansion_overhaul / open_todos P0):
//   - 原型離線算好 (build_intent_prototypes.py → data/intent_prototypes.json),
//     前端只編碼 query 一次,不在前端算 132 原型。
//   - 原型與 query 編碼器「同源」才能比 cosine:同 model / mean-pool / L2 norm。
//   - expandQueryIntent 維持同步;編碼器與原型 async 預載,未載入則 fallback skip
//     (happy path 零影響)。
//   - ⚠️ 預設關閉。上線需 (1) 產出 intent_prototypes.json (2) 接 transformers.js
//     AutoModel 真正載入編碼器 (3) 翻開此旗標。骨架階段保留介接點不灌 205MB 模型。
const ENCODER_FALLBACK_ENABLED = false;

// 預載狀態 (initEncoderFallback 填入)。null = 未載入 → encoderFallbackExpand skip。
let intentEncoder    = null;   // 同源 query 編碼器 (上線時 = transformers.js AutoModel)
let intentPrototypes = null;   // { dim, thr, top_k, rules:{k:expansion}, vecs:{k:Float32Array} }

// --- Source-aware boolean reliability (bool 空值≠False) ---------------------
// 兩來源爬蟲抓取的欄位子集不同，部分 bool 欄在某來源「整欄崩塌」(≈0% true)，
// 那是「爬蟲沒抓該欄」而非「房子真的沒有」。對崩塌欄的 false 必須視為「未知」
// (回退文字判斷 / 交 AI)，不可當「明確無」硬判，否則系統性誤殺該來源房源。
//
// 崩塌判定來自現有 704 筆 property_data.json 實測 true 比率(<5% 視為崩塌)：
//   has_parking      nchu 61% / dd  0%   → dd 崩塌
//   water_dispenser  nchu 61% / dd  0%   → dd 崩塌
//   has_waste_disposal nchu 0% / dd 99%  → nchu 崩塌(興大頁面真無此欄)
//   has_subsidy      nchu  0% / dd 100%  → nchu 崩塌(興大真無租屋補助欄)
//   has_window       nchu 70% / dd 100%  → 兩來源皆可信(2026-06-14 補抓安全管理表後脫離崩塌)
//   has_elevator / has_balcony 兩來源皆有訊號 → 皆可信
// 2026-06-14 興大 crawler 補抓 租金包含/安全管理/消防逃生 二級表格 + 衍生特色標籤後，
// has_window 0%→70%、safety_level high 0%→94%、特色 avg 1.6→5.3。
// 資料換版時需依新統計更新此表 (見 data_source_misalignment 記憶)。
const COLLAPSED_BOOL_FIELDS = {
    nchu: new Set(['has_waste_disposal', 'has_subsidy']),
    dd:   new Set(['has_parking', 'water_dispenser']),
};

function propSource(prop) {
    return (prop.url || '').includes('nchu') ? 'nchu' : 'dd';
}

// 三態解讀一個 bool 設施欄：
//   'yes'     — has_xxx===true，明確有
//   'no'      — has_xxx===false 且該欄對此來源可信，明確無
//   'unknown' — has_xxx===false 但該欄對此來源崩塌，未知(交文字/AI 判)
function boolFieldState(prop, field) {
    if (prop[field] === true) return 'yes';
    if (COLLAPSED_BOOL_FIELDS[propSource(prop)]?.has(field)) return 'unknown';
    return 'no';
}

// --- Property Data Synchronization ---
export async function initData() {
    const response = await fetch('assets/property_data.json?v=20260614i');
    propertyData = await response.json();
    console.log(`Loaded ${propertyData.length} property descriptions`);
}

// --- NLP Engine Initialization via Web Worker ---
export async function initNLP(onProgress) {
    // 背景預載 bi-encoder fallback (flag 關時為 no-op,不阻塞主初始化)。
    initEncoderFallback();
    if (!worker) {
        return new Promise((resolve, reject) => {
            console.log("Initializing Inference Web Worker...");
            worker = new Worker('js/inference-worker.js', { type: 'module' });

            worker.onmessage = (e) => {
                const { type, message, score, id, error, loaded, total } = e.data;
                if (type === 'status' && onProgress) {
                    onProgress({ status: 'progress', message, loaded, total });
                } else if (type === 'ready') {
                    console.log('Inference Worker Ready');
                    if (onProgress) onProgress({ status: 'ready' });
                    resolve();
                } else if (type === 'scoreResult') {
                    const callback = pendingInference.get(id);
                    if (callback) {
                        callback(score);
                        pendingInference.delete(id);
                    }
                } else if (type === 'error') {
                    console.error('Worker error:', message);
                    if (reject) reject(new Error(message));
                }
            };

            worker.postMessage({ 
                type: 'init', 
                data: { origin: window.location.origin } 
            });
        });
    }
}

// --- NER Worker Initialization ---
export async function initNER(onProgress = null, onReady = null) {
    if (nerWorker) return;
    return new Promise((resolve) => {
        nerWorker = new Worker('js/ner-worker.js', { type: 'module' });
        nerWorker.onmessage = (e) => {
            const { type, entities, id } = e.data;
            if (type === 'ner_ready') {
                nerReady = true;
                console.log('NER Worker Ready');
                if (onReady) onReady();
                resolve();
            } else if (type === 'ner_result') {
                const cb = pendingNER.get(id);
                if (cb) { cb(entities); pendingNER.delete(id); }
            } else if (type === 'ner_error') {
                console.warn('NER worker error:', e.data.message);
                resolve();  // non-fatal — app continues without NER
            } else if (type === 'ner_status') {
                console.log('[NER]', e.data.message);
            } else if (type === 'ner_progress') {
                if (onProgress) onProgress({ loaded: e.data.loaded, total: e.data.total });
            }
        };
        nerWorker.postMessage({ type: 'ner_init', data: { origin: window.location.origin } });
    });
}

// --- NER Entity Extraction (with 800ms timeout) ---
async function nerExtract(query) {
    if (!nerWorker || !nerReady) return { locations: [], budgets: [], features: [] };
    return new Promise((resolve) => {
        const id = nerIdCounter++;
        const timer = setTimeout(() => {
            pendingNER.delete(id);
            resolve({ locations: [], budgets: [], features: [] });
        }, 800);
        pendingNER.set(id, (entities) => {
            clearTimeout(timer);
            resolve(entities);
        });
        nerWorker.postMessage({ type: 'ner_extract', data: { query, id } });
    });
}

// --- Proxy to Worker for Scoring ---
async function scorePair(query, propertyText) {
    return new Promise((resolve) => {
        const id = inferenceIdCounter++;
        pendingInference.set(id, resolve);
        worker.postMessage({
            type: 'score',
            data: { query, propertyText, id }
        });
    });
}

// --- Constraint Parsing & Normalization ---
function parseConstraintsFromText(text) {
    let budget = null, limit = null;
    let minBudget = null, maxBudget = null;
    let genderUnrestricted = false, hasGenderMention = false, hasBudgetMention = false, hasRoomTypeMention = false;
    let wantsUtilityBilling = false, maxElectricityPrice = null;
    let requireBalcony = false, requireWindow = false, requireParking = false, requireWaste = false;
    let requireSubsidy = false, isSocialHousing = false;
    let excludeRooftop = false, excludeWooden = false, excludeHaunted = false;

    if (text.includes('不限女') || text.includes('不限性別') || text.includes('男生') || text.includes('男士')) {
        genderUnrestricted = true;
        hasGenderMention = true;
    } else if (text.includes('限女') || text.includes('限男')) {
        hasGenderMention = true;
    }

    // Exclusions (Hard Filtering)
    const negativeWords = "(謝絕|不要|拒絕|禁|❌|不接受|不想|討厭|避免|不要有|不要找)";
    if (text.match(new RegExp(`${negativeWords}[^。！？\\n]*(頂加|加蓋|頂樓)`))) excludeRooftop = true;
    if (text.match(new RegExp(`${negativeWords}[^。！？\\n]*木板`))) excludeWooden = true;
    if (text.match(new RegExp(`${negativeWords}[^。！？\\n]*凶宅`))) excludeHaunted = true;

    // Explicit Requirements
    if (text.match(/(要有|必須|希望|想找)[^。！？\n]*陽台/)) requireBalcony = true;
    else if (text.includes('陽台')) requireBalcony = true; // Soft requirement
    
    if (text.match(/(要有|必須|希望|想找)[^。！？\n]*窗/)) requireWindow = true;
    else if (text.includes('窗')) requireWindow = true;

    if (text.includes('車位') || text.includes('停車')) requireParking = true;
    if (text.includes('子母車') || text.includes('垃圾')) requireWaste = true;
    
    if (text.includes('補助') || text.includes('補貼') || text.includes('報稅') || text.includes('入籍')) requireSubsidy = true;
    if (text.includes('社宅') || text.includes('社會住宅')) isSocialHousing = true;

    if (text.includes('以上')) limit = 'above';
    else if (text.includes('以下') || text.includes('以內') || text.includes('內')) limit = 'below';

    // Parse Utility Billing (台水台電)
    if (text.includes('台水') || text.includes('台電') || text.includes('獨立電錶') || text.includes('獨立電表')) {
        wantsUtilityBilling = true;
    }
    const elecMatch = text.match(/度\s*(\d+(?:\.\d+)?)\s*[元塊]/);
    if (elecMatch) {
        maxElectricityPrice = parseFloat(elecMatch[1]);
    }

    let rt = text.replace(/一/g, '1').replace(/二/g, '2').replace(/兩/g, '2').replace(/三/g, '3')
        .replace(/四/g, '4').replace(/五/g, '5').replace(/六/g, '6').replace(/七/g, '7')
        .replace(/八/g, '8').replace(/九/g, '9').replace(/十/g, '10').replace(/半/g, '30');

    let maxWalkMins = null;
    let walkMatch = rt.match(/(?:走路|步行)[^\d]*(\d+)[^\d]*(?:分鐘|分)/);
    if (walkMatch) maxWalkMins = parseInt(walkMatch[1]);

    let maxScooterMins = null;
    let scooterMatch = rt.match(/(?:機車|騎車)[^\d]*(\d+)[^\d]*(?:分鐘|分)/);
    if (scooterMatch) maxScooterMins = parseInt(scooterMatch[1]);


    // Handle Range Budget (e.g., 6000-12000, 6000~12000, 6千-1萬2)
    let rt_range = rt.replace(/(\d+(?:\.\d+)?)萬(\d*)/g, (m, p1, p2) => {
        let val = parseFloat(p1) * 10000;
        if (p2) val += parseInt(p2) * 1000;
        return val;
    }).replace(/(\d+)千/g, (m, p1) => parseInt(p1) * 1000);
    
    let rangeMatch = rt_range.match(/(\d{3,})\s*[-~～至到]\s*(\d{3,})/);
    if (rangeMatch) {
        minBudget = parseInt(rangeMatch[1]);
        maxBudget = parseInt(rangeMatch[2]);
        hasBudgetMention = true;
    }

    if (!hasBudgetMention) {
        if (rt.includes('萬')) {
            let m = rt.match(/(\d+(?:\.\d+)?)萬(\d*)/);
            if (m) {
                budget = parseFloat(m[1]) * 10000 + (m[2] ? parseInt(m[2]) * 1000 : 0);
                hasBudgetMention = true;
            }
        }
        if (!budget) {
            rt = rt.replace(/千/g, '000').replace(/[kK]/g, '000');
            let m2 = rt.match(/(\d{4,})/);
            if (m2) {
                budget = parseInt(m2[1]);
                hasBudgetMention = true;
            }
        }
        if (!budget) {
            let m3 = rt.match(/(\d+)/);
            if (m3) {
                let val = parseInt(m3[1]);
                if (val < 100) budget = val * 1000;
                else if (val >= 1000) budget = val;
                hasBudgetMention = true;
            }
        }
    }

    let wantsRoomType = null;
    if (text.includes('套房')) { hasRoomTypeMention = true; wantsRoomType = '套房'; }
    else if (text.includes('雅房')) { hasRoomTypeMention = true; wantsRoomType = '雅房'; }
    else if (text.includes('工作室')) { hasRoomTypeMention = true; wantsRoomType = '工作室'; }

    return {
        budget, minBudget, maxBudget, limit, genderUnrestricted, hasGenderMention, hasBudgetMention, hasRoomTypeMention, wantsRoomType,
        wantsUtilityBilling, maxElectricityPrice, requireBalcony, requireWindow, requireParking, requireWaste,
        requireSubsidy, isSocialHousing,
        excludeRooftop, excludeWooden, excludeHaunted, maxWalkMins, maxScooterMins,
        wantsPet: (text.includes('養貓') || text.includes('養狗') || text.includes('寵物')),
        requireElevator: (text.includes('電梯') || text.includes('升降梯') || text.includes('不爬樓') || text.includes('不用爬') || text.includes('不想爬') || text.includes('不要爬') || text.includes('腿不好') || text.includes('膝蓋不好')),
        requireCooking: (text.includes('開伙') || text.includes('開火') || text.includes('自炊') || text.includes('煮飯') || text.includes('炒菜') || text.includes('在家煮') || text.includes('自己煮')),
        requireWaterDispenser: (text.includes('飲水機')),
        requirePrivateWasher: (text.includes('獨洗') || text.includes('個人洗衣機')),
        requireGuard: (text.includes('代收') || text.includes('包裹') || text.includes('管理員') || text.includes('警衛')),
        originalText: text
    };
}

// --- Explainability: Match Reasons & Conflict Detection ---
function explainMatch(query, prop, constraints) {
    const reasons = [];
    const pText = buildPropText(prop).toLowerCase();
    const q = query.toLowerCase();
    
    // 1. Budget & CP Value
    if (constraints.hasBudgetMention) {
        if (prop.rent <= (constraints.maxBudget || constraints.budget)) {
            if (prop.cp_tag === "high_cp") {
                reasons.push("💎 區域高 CP 值首選");
            } else {
                reasons.push("💰 符合您的預算範圍");
            }
        }
    }

    // 2. Billing & Electricity
    if (q.includes('省錢') || q.includes('台電') || q.includes('怕熱')) {
        if (prop.billing_type === "taipower") {
            reasons.push("⚡ 台電計費，省電好幫手");
        }
    }

    // 3. Service & Convenience (Garbage/Parcels)
    if (q.includes('垃圾') || q.includes('追車') || q.includes('子母車') || q.includes('方便')) {
        if (prop.service_level === "five_star") {
            reasons.push("✨ 免追垃圾車 + 代收包裹");
        } else if (prop.service_level === "basic" || prop.has_waste_disposal) {
            reasons.push("🧹 設有子母車，丟垃圾免煩惱");
        }
    }

    // 4. Distance & Geo Tier
    // 移除:geo_tier 在現有爬蟲資料退化(701/704=core,3=active),無法據以生成可信標籤。
    // 距離訊號改由 OSRM 通勤距離(distance / walk_mins)在排序層處理,不在此產生臆測標籤。

    // 5. Condition & Aesthetics
    if (q.includes('漂亮') || q.includes('質感') || q.includes('新') || q.includes('裝潢')) {
        if (prop.condition === "new") {
            reasons.push("🏠 全新首租，質感第一手");
        } else if (prop.condition === "renovated") {
            reasons.push("🎨 精緻裝潢，充滿設計感");
        }
    }

    // 6. Semantic rules — multi-trigger + explicit property check
    // Each rule: triggers[] (any match in query activates), check(pText, prop) verifies house has it
    const semanticRules = [
        {
            triggers: ['冰箱'],
            check: p => p.includes('冰箱'),
            label: '🧊 附有冰箱'
        },
        {
            triggers: ['洗衣機', '獨洗', '洗衣', '獨立洗'],
            check: p => p.includes('洗衣機') || p.includes('獨立洗'),
            label: '🫧 附獨立洗衣機'
        },
        {
            triggers: ['冷氣', '空調', '怕熱', '夏天熱'],
            check: p => p.includes('冷氣') || p.includes('空調'),
            label: '❄️ 房內附冷氣'
        },
        {
            triggers: ['電視', '看電視', '電視機', 'tv', '液晶', '追劇'],
            check: p => p.includes('電視') || p.includes('液晶'),
            label: '📺 房內附電視'
        },
        {
            triggers: ['廚房', '開伙', '煮飯', '瓦斯', '自炊', '炒菜', '料理', '在家煮', '自己煮', '開火', '煮東西'],
            check: p => p.includes('廚房') || p.includes('開伙') || p.includes('瓦斯') || p.includes('可自炊') || p.includes('電磁爐') || p.includes('排油煙') || p.includes('流理台') || p.includes('爐'),
            label: '🍳 可開伙自炊'
        },
        {
            triggers: ['電梯', '升降梯', '不爬樓', '不用爬', '不想爬', '不要爬', '腿不好', '膝蓋不好'],
            check: p => p.includes('電梯'),
            label: '🛗 有電梯，行動便利'
        },
        {
            triggers: ['貓', '狗', '養貓', '養狗', '帶貓', '帶狗', '毛小孩', '寵物', '貓咪', '狗狗', '可養'],
            check: (p, prop) => !p.includes('禁養') && !p.includes('不可養') && (
                p.includes('可養') || p.includes('寵物') || p.includes('友善') ||
                p.includes('可貓') || p.includes('可狗') || prop.has_pet
            ),
            label: '🐱 友善毛小孩環境'
        },
        {
            triggers: ['補助', '補貼', '租金補貼', '報稅', '入籍', '租屋補助'],
            check: (p, prop) => !p.includes('不可補助') && !p.includes('不可報稅') && !p.includes('不可入籍') && (
                prop.has_subsidy || p.includes('可補助') || p.includes('可報稅') || p.includes('可入籍') || p.includes('補助')
            ),
            label: '📑 可申請政府租金補貼'
        },
        {
            triggers: ['陽台', '晾衣', '晾曬', '晾衫'],
            check: p => p.includes('陽台') || p.includes('晾衣'),
            label: '☀️ 有私人陽台可晾衣'
        },
        {
            triggers: ['窗', '採光', '通風', '光線', '明亮'],
            check: p => p.includes('窗') || p.includes('採光') || p.includes('通風'),
            label: '🪟 採光通風對外窗'
        },
        {
            triggers: ['停車', '車位', '機車位', '腳踏車', '單車', '自行車', '機車'],
            check: p => p.includes('停車') || p.includes('車位') || p.includes('機車'),
            label: '🛵 附有機車停放空間'
        },
        {
            triggers: ['飲水機', '飲水', '開水'],
            check: p => p.includes('飲水機') || p.includes('飲水'),
            label: '💧 配有公共飲水機'
        },
        {
            triggers: ['門禁', '保全', '安全', '管理員', '管理室'],
            check: p => p.includes('門禁') || p.includes('保全') || p.includes('管理員'),
            label: '🔒 有門禁管理，安全有保障'
        },
        {
            triggers: ['網路', 'wifi', 'wi-fi', '無線', '寬頻', '含網路', '附網路'],
            check: p => p.includes('網路') || p.includes('wifi') || p.includes('寬頻'),
            label: '📶 含網路費用'
        },
        {
            triggers: ['熱水', '熱水器', '獨立熱水', '不搶熱水'],
            check: p => p.includes('熱水器') || p.includes('獨立熱水') || p.includes('瓦斯熱水'),
            label: '🚿 獨立熱水，不搶澡'
        },
        {
            triggers: ['全配', '家具', '家電', '附家電', '附家具'],
            check: p => (p.includes('冰箱') || p.includes('冷氣')) && (p.includes('床') || p.includes('桌')),
            label: '🛋️ 家具家電全配'
        },
        // ── 女生安全 ──────────────────────────────────
        {
            triggers: ['女生獨居', '獨居女', '女生住', '女生安全', '怕危險', '治安', '監視器', '女性友善'],
            check: p => p.includes('監視器') || p.includes('女性') || p.includes('門禁') || p.includes('管理員'),
            label: '🛡️ 女性友善 / 安全管理'
        },
        // ── 衛浴獨立 ──────────────────────────────────
        {
            triggers: ['不想共用廁所', '不想共廁', '個人衛浴', '獨立衛浴', '獨衛', '想泡澡', '浴缸'],
            check: p => p.includes('獨衛') || p.includes('獨立衛浴') || p.includes('浴缸') || p.includes('套房'),
            label: '🚿 獨立衛浴不共用'
        },
        // ── 租期彈性 ──────────────────────────────────
        {
            triggers: ['短租', '只租幾個月', '不確定租多久', '剛畢業', '工作不穩定', '彈性租期'],
            check: p => p.includes('短租') || p.includes('彈性') || p.includes('不限租期') || p.includes('月租'),
            label: '📅 租期彈性不限長短'
        },
        // ── 合租 / 室友 ───────────────────────────────
        {
            triggers: ['找室友', '想合租', '不想一個人住', '合租', '分租'],
            check: p => p.includes('室友') || p.includes('合租') || p.includes('分租') || p.includes('雅房'),
            label: '👥 可合租 / 室友同住'
        },
        {
            triggers: ['一個人住', '不想跟人共用', '獨住'],
            check: p => p.includes('套房') || p.includes('獨衛') || p.includes('獨立'),
            label: '🏠 獨立套房不共用'
        },
        // ── 交通通勤 ── 移除:check 的 公車/捷運/交通/生活機能 在爬蟲資料 0 命中,
        //    此規則永遠無法 truthy(dead code)。通勤訊號改由 OSRM distance 處理。
        // ── 在家工作 / WFH ────────────────────────────
        {
            triggers: ['在家工作', 'WFH', '遠距工作', '居家辦公', '書桌', '打報告', '念書', '讀書'],
            check: p => p.includes('書桌') || p.includes('寬頻') || p.includes('網路') || p.includes('安靜'),
            label: '💻 適合居家辦公 / 讀書'
        },
        // ── 預算暗示 ──────────────────────────────────
        {
            triggers: ['學生', '剛出社會', '薪水不多', '不要太貴', '便宜', '省錢', '實惠'],
            check: (p, prop) => prop.cp_tag === 'high_cp' || p.includes('學生') || p.includes('實惠') || p.includes('經濟'),
            label: '💰 經濟實惠 / 學生友善'
        },
        // ── 採光朝向 ──────────────────────────────────
        {
            triggers: ['不要西曬', '採光', '東向', '南向', '對外窗', '明亮'],
            check: p => p.includes('採光') || p.includes('對外窗') || p.includes('東向') || p.includes('南向'),
            label: '🌤️ 採光佳 / 無西曬'
        },
        // ── 安靜 / 隔音 ───────────────────────────────
        {
            triggers: ['怕吵', '安靜', '隔音', '靜巷'],
            check: p => p.includes('隔音') || p.includes('靜巷') || p.includes('氣密') || p.includes('安靜'),
            label: '🔇 安靜隔音佳'
        },
        // ── 夜貓子 / 無門禁 ───────────────────────────
        {
            triggers: ['夜貓子', '作息晚', '晚歸', '無門禁', '24小時'],
            check: p => p.includes('無門禁') || p.includes('24小時') || p.includes('自由進出') || p.includes('不限'),
            label: '🌙 無門禁限制 / 自由進出'
        }
    ];

    semanticRules.forEach(rule => {
        if (reasons.length >= 3) return;
        if (reasons.includes(rule.label)) return;
        const triggered = rule.triggers.some(t => q.includes(t));
        if (!triggered) return;
        if (rule.check(pText, prop)) {
            reasons.push(rule.label);
        }
    });

    // Default highlights if empty
    if (reasons.length === 0) {
        if (prop.cp_tag === "high_cp") reasons.push("💎 區域高 CP 值選");
        if (prop.service_level === "five_star") reasons.push("✨ 高品質社區管理");
        if (prop.billing_type === "taipower") reasons.push("⚡ 電費照台電撥款");
    }

    return [...new Set(reasons)].slice(0, 3);
}

function checkConflicts(prop, constraints) {
    const { wantsPet, wantsRoomType } = constraints;
    const pText = buildPropText(prop);

    // 1. Room Type Mismatch
    if (wantsRoomType && prop.room_type && prop.room_type !== wantsRoomType) {
        return `此房源為${prop.room_type}，您指定的是${wantsRoomType}`;
    }

    // 2. Pet Conflict
    if (wantsPet && (pText.includes('禁養') || pText.includes('不可養'))) {
        return "此房源禁養寵物";
    }

    // 3. Gender Conflict
    if (constraints.hasGenderMention && constraints.originalText) {
        const orig = constraints.originalText;
        const isMale = orig.includes('男生') || orig.includes('男士') || orig.includes('男性');
        const isFemale = orig.includes('女生') || orig.includes('女士') || orig.includes('女性');
        const isFemaleOnly = pText.includes('限女');
        const isMaleOnly = pText.includes('限男');
        if (isMale && isFemaleOnly) return "此房源僅限女性";
        if (isFemale && isMaleOnly) return "此房源僅限男性";
    }

    // 4. Smoking
    if (constraints.originalText?.includes('抽菸') && (pText.includes('禁菸') || pText.includes('禁止吸菸'))) {
        return "此房源禁止吸菸";
    }

    return null;
}

// --- Hard Exclusion Filtering ---
function filterHardExclusions(properties, constraints) {
    const { 
        budget, minBudget, maxBudget, limit, genderUnrestricted, hasGenderMention, hasBudgetMention,
        excludeRooftop, excludeWooden, maxElectricityPrice, wantsUtilityBilling,
        maxWalkMins, maxScooterMins,
        requireSubsidy, isSocialHousing, requireBalcony, requireWindow, requireParking, requireWaste,
        wantsPet, requireElevator, requireCooking
    } = constraints;
    const candidates = [];

    for (const prop of properties) {
        // 1. Core Policy Exclusions (Keep these hard)
        if (excludeRooftop && (prop.is_rooftop || prop.text.includes('頂加'))) continue;
        if (excludeWooden && prop.is_wooden_partition) continue;
        if (requireSubsidy && (prop.text.includes('不可補助') || prop.text.includes('不可報稅') || prop.text.includes('不可入籍'))) continue;
        if (isSocialHousing && !prop.text.includes('社會住宅') && !prop.text.includes('社宅')) continue;

        // 1b. Documented hard constraints (一票否決): exclude only EXPLICIT conflicts,
        // leave unstated properties for AI to judge so the candidate pool isn't over-pruned.
        if (wantsPet && (prop.text.includes('禁養') || prop.text.includes('不可養') || prop.text.includes('不可寵') || prop.text.includes('謝絕寵物'))) continue;
        // 電梯：只信「文字明確寫無電梯」。has_elevator===false 在興大來源不可靠（爬蟲常未抓到，
        // false 可能代表「未知」而非「真的沒有」），不可作為硬篩依據，否則誤殺興大房源。
        if (requireElevator && (prop.text.includes('無電梯') || prop.text.includes('沒有電梯') || prop.text.includes('沒電梯'))) continue;
        if (requireCooking && (prop.text.includes('禁開伙') || prop.text.includes('不可開伙') || prop.text.includes('不可開火') || prop.text.includes('禁炊'))) continue;

        // 2. Soft Amenities (REMOVED HARD CONTINUES)
        // We no longer 'continue' here. We let these be handled by Rule-Based and AI scoring.
        // This ensures semantic matches for things like "不想追垃圾車" are found even if keywords differ.

        // Commute time filtering
        let dist = parseFloat(prop.distance);
        if (!isNaN(dist) && dist > 0) {
            if (maxWalkMins !== null) {
                let walkMins = Math.round(dist / 0.075);
                if (walkMins > maxWalkMins + 3) continue; // +3 mins grace period
            }
            if (maxScooterMins !== null) {
                let scooterMins = Math.max(1, Math.round(dist / 0.417));
                if (scooterMins > maxScooterMins + 2) continue; // +2 mins grace period
            }
        }

        
        if (maxElectricityPrice) {
            // "5元/度"
            const billing = prop.electricity_billing || "";
            const match = billing.match(/(\d+(?:\.\d+)?)/);
            if (match && parseFloat(match[1]) > maxElectricityPrice) continue;
        }

        // If user specifically asks for Taishui Taipower and NOT maxElectricityPrice,
        // we can filter out properties that are explicitly > 5 NTD, though we handle this softly in scoring too.
        if (wantsUtilityBilling) {
            const billing = prop.electricity_billing || "";
            if (billing.includes("度")) {
                const match = billing.match(/(\d+(?:\.\d+)?)/);
                if (match && parseFloat(match[1]) >= 5) {
                    // If they explicitly want Taishui Taipower, properties charging >= 5/kwh are generally rejected
                    continue;
                }
            }
        }

        if (hasGenderMention && genderUnrestricted) {
            const isFemaleOnly = prop.text.includes('限女') || (prop.furniture && prop.furniture.includes('限女'));
            if (isFemaleOnly) continue;
        }
        if (hasBudgetMention) {
            if (maxBudget !== null && prop.rent > maxBudget) continue;
            if (budget !== null) {
                const effectiveLimit = limit || 'below';
                if (effectiveLimit === 'below' && prop.rent > budget) continue;
                if (effectiveLimit === 'above' && prop.rent < budget) continue;
            }
        }
        candidates.push(prop);
    }
    return candidates;
}


// --- NER BGT Entity Budget Parsing ---
function parseBudgetFromNER(budgetSpans) {
    if (!budgetSpans || budgetSpans.length === 0) return null;
    let budget = null;
    let limit = null;

    for (const span of budgetSpans) {
        // Detect direction from original span
        if (span.includes('以上')) limit = 'above';
        else if (span.includes('以下') || span.includes('以內') || span.includes('內')) limit = limit || 'below';

        let s = span
            .replace(/[一１]/g, '1').replace(/[二２兩]/g, '2').replace(/[三３]/g, '3')
            .replace(/[四４]/g, '4').replace(/[五５]/g, '5').replace(/[六６]/g, '6')
            .replace(/[七７]/g, '7').replace(/[八８]/g, '8').replace(/[九９]/g, '9')
            .replace(/十/g, '10');

        // Handle 萬 notation first (e.g., 1萬2 → 12000)
        const wanMatch = s.match(/(\d+(?:\.\d+)?)萬(\d*)/);
        if (wanMatch) {
            const candidate = parseFloat(wanMatch[1]) * 10000 + (wanMatch[2] ? parseInt(wanMatch[2]) * 1000 : 0);
            if (candidate > 0) { budget = candidate; continue; }
        }
        // Handle 千 / k / K
        s = s.replace(/千/g, '000').replace(/[kK]/g, '000');
        const numMatch = s.match(/(\d{3,})/);
        if (numMatch) {
            const candidate = parseInt(numMatch[1]);
            if (candidate >= 1000) budget = candidate;
        }
    }
    return budget ? { budget, limit: limit || 'below' } : null;
}

// --- Property Feature Normalization (房源端特徵正規化) ---
// 把房源的結構化欄位(furniture/features/notes)+ bool 設施欄一起納入可比對文字,
// 並做同義詞歸一,讓查詢擴展詞(可寵/廚房/獨衛…)能對上房源實際用詞(可養貓/可開伙/獨立衛浴…)。
// 落地率實測:只看 text 17.6% → +結構欄+bool 30.5% → +同義歸一 45.8%。
const PROP_SYNONYMS = {
    "可寵":["可養貓","可養狗","可養寵物","可養其他寵物"],"寵物友善":["可養貓","可養狗","可養寵物"],
    "廚房":["可開伙","流理台"],"開火":["可開伙","瓦斯","電磁爐"],"自炊":["可開伙"],"可伙":["可開伙"],
    "抽油煙機":["排油煙"],"獨衛":["獨立衛浴","專用衛浴"],"獨立衛浴":["獨衛"],"獨廁":["獨立衛浴","獨衛"],
    "變頻":["冷氣"],"變頻冷氣":["冷氣"],"吹冷氣":["冷氣"],"全新":["新裝潢","新成屋"],
    "管理員":["保全","警衛"],"監視器":["保全","監視"],"門禁":["保全","刷卡"],
    "床架":["床"],"床墊":["床"],"書桌椅":["桌子","書桌","椅子"],
    "天然瓦斯熱水器":["熱水器","瓦斯"],"電熱水器":["熱水器"],
    "全配":["家具","家電"],"全家具":["家具"],"全家電":["家電"],"家具齊全":["家具"],
    "子母車":["垃圾"],"垃圾代收":["垃圾"],"獨立洗衣機":["洗衣機"],"獨洗":["洗衣機"],
    // P2 救援橋 (token 對房源自由描述端 0 命中,但有未架橋的真實同義詞/結構欄):
    // 詳見 data/UNVERIFIABLE_TOKENS_AUDIT.md 與 pipeline/data_prep/audit_expansion_tokens.py。
    "禁菸":["無菸"],
    "採光":["對外窗","窗"],"通風":["對外窗","窗"],
    "安全":["保全"],"刷卡":["保全","門禁"],"女性友善":["限女","女性"],
    "租補":["租金補貼","補貼"],"室友":["雅房","分租"],"合租":["雅房","分租"],
    "隔音":["氣密窗","氣密"],
    // 台水/電費族:靠 buildPropText 納入 electricity_billing 欄(排除「不明」)後可命中。
    "台水":["水費"],"帳單":["台電","台水","電費"],"自繳":["台電","台水"],"標準電費":["台電"],
};
const BOOL_FIELD_FEATURES = {
    has_elevator:"電梯", has_window:"對外窗", has_balcony:"陽台",
    has_parking:"車位 停車場", has_waste_disposal:"垃圾處理", is_rooftop:"頂樓",
    water_dispenser:"飲水機", private_washer:"獨洗", has_subsidy:"補助", is_taipower:"台電",
};

// 產生房源「完整可比對文字」: text + 結構化欄位 + bool 設施詞。所有房源關鍵字比對統一使用。
function buildPropText(prop) {
    const parts = [prop.text || ""];
    for (const f of ["furniture", "features", "building_type", "room_type"]) {
        if (prop[f]) parts.push(String(prop[f]).replace(/\//g, " "));
    }
    for (const f of ["notes", "other_fees"]) {
        if (Array.isArray(prop[f])) parts.push(prop[f].join(" "));
    }
    for (const [bk, wd] of Object.entries(BOOL_FIELD_FEATURES)) {
        if (prop[bk] === true) parts.push(wd);
    }
    // 電費/水費計費方式 (台電/台水/獨立電錶) 落在結構欄,非自由文字。納入可比對文字以
    // 支援「台水/標準電費/帳單自繳」等查詢;「不明」是未知值非訊號,排除避免假命中。
    if (prop.electricity_billing && prop.electricity_billing !== "不明") {
        parts.push(String(prop.electricity_billing));
    }
    return parts.join(" ");
}
// 註:曾嘗試把 buildPropText(去重後)餵給 cross-encoder 以補興大文字層偏誤,經離線
// A/B 驗證為 NO-GO(見 pipeline/data_prep/eval_ce_text_enrichment.py 與
// docs/ce_text_layer_decision.md):CE 只認訓練時的短結構 prop.text 格式,餵較長的
// enriched 文字屬 OOD,分數崩壞(「要有陽台」query:含「有陽台」的 enriched 文字反而
// 從 +7.9 掉到 +0.5,連 raw+「有陽台」都掉到 −1.8)。故 scorePair 維持餵 prop.text。
// 文字層根治需重訓 CE(超出本次範圍,且既往重訓會回歸 — 見 retrain_jun13_result)。

// 房源是否含某特徵詞(含同義歸一): 直接命中, 或任一同義詞命中。
function propHasFeature(propText, feature) {
    if (propText.includes(feature)) return true;
    const syns = PROP_SYNONYMS[feature];
    if (syns) for (const s of syns) if (propText.includes(s)) return true;
    return false;
}

// --- Semantic Query Expansion ---
/**
 * 預載 bi-encoder fallback 的編碼器與原型向量 (async, 啟動時呼叫一次)。
 * flag 關閉時直接 return,不引入任何成本。骨架階段:結構完整、實際 model load
 * 與 prototypes fetch 標為 TODO,上線時補。
 */
async function initEncoderFallback() {
    if (!ENCODER_FALLBACK_ENABLED) return;
    if (intentEncoder && intentPrototypes) return;  // 已載入
    try {
        // 1) 載入離線預算的原型向量 (build_intent_prototypes.py 產出)。
        const resp = await fetch('assets/intent_prototypes.json');
        if (!resp.ok) throw new Error(`prototypes HTTP ${resp.status}`);
        const raw = await resp.json();
        // 預轉 Float32Array,query 比對時零額外配置。
        const vecs = {};
        for (const [k, v] of Object.entries(raw.prototypes)) {
            vecs[k] = Float32Array.from(v);
        }
        intentPrototypes = {
            dim: raw.dim, thr: raw.thr, top_k: raw.top_k,
            rules: raw.rules, vecs,
        };

        // 2) 載入同源 query 編碼器。必須與 build_intent_prototypes.py 同 model /
        //    mean-pool / L2 norm,否則 cosine 無意義。
        // TODO(上線): 接 transformers.js AutoModel。
        //   const { AutoModel } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
        //   const tok   = await AutoTokenizer.from_pretrained(raw.model);
        //   const model = await AutoModel.from_pretrained(raw.model, { quantized: true });
        //   intentEncoder = async (text) => { ...mean-pool over attention_mask → L2 norm... };
        // 骨架階段不灌模型 → 維持 null,encoderFallbackExpand 會自動 skip。
        console.info('[encoderFallback] prototypes 已載入;編碼器尚未接 (骨架階段)');
    } catch (err) {
        console.warn('[encoderFallback] 預載失敗,fallback 停用:', err);
        intentEncoder = null;
        intentPrototypes = null;
    }
}

/**
 * 字面意圖表零命中時的 bi-encoder fallback (同步)。
 * 編碼 query → 與 132 原型 cosine → thr 以上取 top-k → 回傳要 append 的擴展詞字串
 * (空字串表示無命中)。編碼器/原型未預載時直接回 ''(skip),不阻塞 happy path。
 *
 * @param {string} query 已知字面表零命中的原始查詢
 * @returns {string} 以空白接合的擴展詞,或 '' 表示 fallback 也無命中
 */
function encoderFallbackExpand(query) {
    if (!ENCODER_FALLBACK_ENABLED) return '';
    if (!intentEncoder || !intentPrototypes) return '';  // 未預載 → skip

    // query 向量已 L2 normalize → 與同樣 normalize 的原型內積即 cosine。
    const qv = intentEncoder(query);                 // 同源編碼器 (上線時填入)
    const { thr, top_k, rules, vecs } = intentPrototypes;

    const scored = [];
    for (const [intent, pv] of Object.entries(vecs)) {
        let dot = 0;
        for (let i = 0; i < pv.length; i++) dot += qv[i] * pv[i];
        if (dot >= thr) scored.push([intent, dot]);
    }
    if (scored.length === 0) return '';

    scored.sort((a, b) => b[1] - a[1]);
    const hits = scored.slice(0, top_k);
    // 否定守衛:fallback 命中的 intent 若在 query 中被否定詞緊鄰修飾,略過
    // (與字面層同邏輯,避免「不想養黃金獵犬」誤路由到可寵)。語意命中無確切
    // 字串位置,故對該 intent 在 query 任一出現位置檢查;查無字串則視為語意命中保留。
    const NEGATORS = '不沒無非免勿';
    const expansions = [];
    for (const [intent] of hits) {
        const idx = query.indexOf(intent);
        const negated = idx > 0 && NEGATORS.includes(query[idx - 1]);
        if (!negated) expansions.push(rules[intent]);
    }
    return expansions.join(' ');
}

function expandQueryIntent(query) {
    let expanded = query;
    const intentMap = {
        // >>> GENERATED: semantic rules (sync_semantic_rules.py) >>>
        "潔癖":      "獨洗 禁菸",
        "愛乾淨":     "獨洗 禁菸",
        "稍微潔癖":    "獨洗 禁菸",
        "想在家煮飯":   "可伙 廚房 流理台 瓦斯爐 電磁爐 開火",
        "想自己煮飯":   "可伙 廚房 流理台 瓦斯爐 開火",
        "在家開伙":    "可伙 廚房 抽油煙機 流理台 瓦斯 開火 自炊 電磁爐 排油煙機",
        "想下廚":     "可伙 廚房 抽油煙機 瓦斯爐",
        "要下廚":     "可伙 廚房 抽油煙機 瓦斯爐",
        "喜歡下廚":    "可伙 廚房 抽油煙機 瓦斯爐 流理台",
        "喜歡自己煮":   "可伙 廚房 流理台 瓦斯爐",
        "自己煮":     "廚房 瓦斯 開火 流理台 可伙 自炊 電磁爐 排油煙機",
        "自炊":      "可伙 廚房 流理台 電磁爐 開火 瓦斯 自炊 排油煙機",
        "省伙食費":    "廚房 瓦斯 開火 流理台",
        "省餐費":     "可伙 廚房 流理台",
        "不想外食":    "可伙 廚房 流理台 電磁爐",
        "不吃外食":    "可伙 廚房 流理台 瓦斯爐",
        "可以煮東西":   "可伙 廚房",
        "要能煮飯":    "可伙 廚房 流理台 電磁爐",
        "煮飯":      "可伙 廚房 流理台",
        "開火":      "可伙 廚房 瓦斯爐 電磁爐",
        "要有廚房":    "廚房 流理台 可伙",
        "有瓦斯":     "天然瓦斯 瓦斯爐 可伙",
        "天然瓦斯":    "天然瓦斯 瓦斯爐 可伙 廚房",
        "怕熱":      "冷氣 變頻 吹冷氣 變頻冷氣",
        "夏天":      "冷氣",
        "怕悶熱":     "陽台 採光 通風 對外窗",
        "採光好":     "採光 對外窗",
        "網美":      "採光",
        "獨洗獨曬":    "洗衣機 陽台 曬衣 獨洗",
        "有車":      "車位 停車場",
        "開車":      "車位 停車場",
        "可貓":      "可寵 養寵 寵物友善 可養貓",
        "可狗":      "可寵 養寵 寵物友善 可養狗",
        "有毛孩":     "可寵 寵物",
        "台水電":     "台電 台水 帳單 自繳",
        "省電費":     "變頻 台電",
        "懶人":      "電梯 子母車 垃圾處理 飲水機",
        "外送族":     "管理員 飲水機 子母車",
        "不想出門":    "管理員 飲水機 子母車",
        "不想追垃圾車":  "子母車 垃圾處理 垃圾代收",
        "怕吵":      "隔音 氣密窗 禁菸",
        "安靜":      "隔音 氣密窗 禁菸",
        "晚歸":      "門禁 管理員 安全 刷卡",
        "女生獨居":    "管理員 門禁 監視器 女性友善 安全",
        "女生住":     "管理員 門禁 監視器 安全",
        "獨居女":     "管理員 門禁 監視器 女性友善",
        "女生安全":    "管理員 門禁 監視器 安全",
        "怕危險":     "管理員 門禁 監視器 安全",
        "治安":      "管理員 門禁 監視器 安全",
        "拎包入住":    "冰箱 洗衣機 床",
        "什麼都有":    "冰箱 洗衣機",
        "家電齊全":    "冰箱 洗衣機 冷氣",
        "要有冰箱":    "冰箱",
        "要有書桌":    "書桌 書桌椅",
        "要有床":     "床架 床墊",
        "要有熱水":    "熱水器 天然瓦斯熱水器 電熱水器",
        "找室友":     "雅房 分租 室友 合租",
        "想合租":     "雅房 分租 室友 合租",
        "不想一個人住":  "雅房 分租 室友",
        "騎車上班":    "機車停車位 停車",
        "不要西曬":    "採光",
        "要有陽台":    "陽台 曬衣 採光 通風",
        "在家工作":    "網路 寬頻 書桌",
        "WFH":     "網路 寬頻 書桌",
        "遠距工作":    "網路 寬頻 書桌",
        "居家辦公":    "網路 寬頻 書桌",
        "打報告":     "寬頻 網路 書桌",
        "上網":      "寬頻 網路",
        "念書":      "書桌 書桌椅 寬頻",
        "讀書":      "書桌 書桌椅 寬頻",
        "不想爬樓梯":   "電梯 大樓 華廈",
        "搬東西":     "電梯",
        "膝蓋不好":    "電梯 大樓 華廈",
        "機車":      "機車停車位",
        "高品質":     "管理員 電梯",
        "不想去自助洗":  "洗衣機 獨立洗衣機",
        "不想共用洗衣機": "洗衣機 獨立洗衣機",
        "養貓":      "可養貓 寵物友善 可寵",
        "養狗":      "可養狗 寵物友善 可寵",
        "台電":      "台電 台水 標準電費",
        "獨立電表":    "獨立電錶 台電",
        "不爬樓梯":    "電梯 華廈 大樓",
        "不要爬樓梯":   "電梯 華廈 大樓",
        "腿不好":     "電梯 華廈 大樓",
        "在家煮":     "廚房 瓦斯 開火 自炊 電磁爐 排油煙機 流理台",
        "想煮飯":     "廚房 瓦斯 開火 自炊 電磁爐 排油煙機 流理台",
        "希望煮飯":    "廚房 瓦斯 開火 自炊 電磁爐 排油煙機 流理台",
        "下班晚":     "子母車 垃圾代收 門禁 管理員 安全",
        "省錢":      "台電 台水 補助 租補",
        "生活便利":    "興大路",
        "走路到學校":   "走路10分",
        "走路去學校":   "走路10分",
        "走路可以到":   "走路10分",
        "走路就可以":   "走路10分",
        "走路過去":    "走路10分",
        "步行到學校":   "走路10分",
        "步行去學校":   "走路10分",
        "步行可以到":   "走路10分",
        "騎車到學校":   "騎車10分",
        "騎車去學校":   "騎車10分",
        "騎車可以到":   "騎車10分",
        "騎車就可以":   "騎車10分",
        "騎車過去":    "騎車10分",
        "騎機車到學校":  "騎車10分",
        "騎機車去學校":  "騎車10分",
    // <<< GENERATED <<<
    };

    // 否定守衛:略過「被否定詞緊鄰修飾」的命中,避免子字串碰撞把反義句帶偏。
    // 例:「沒有車」含「有車」、「不開車」含「開車」→ 否則會誤擴展成「車位 停車場」,
    // 把無車使用者推去停車位房源。比對 intent 前一字是否為 不/沒/無/非/免/勿。
    const NEGATORS = '不沒無非免勿';
    for (const [intent, expansion] of Object.entries(intentMap)) {
        let from = 0, idx;
        while ((idx = query.indexOf(intent, from)) !== -1) {
            // 注意:''.includes 對空字串恆為 true,故句首(idx===0)須明確視為「無否定詞」。
            const negated = idx > 0 && NEGATORS.includes(query[idx - 1]);
            if (!negated) {
                expanded += " " + expansion;
                break;  // 命中一次即擴展,與原行為一致
            }
            from = idx + 1;  // 此處被否定,繼續找下一個非否定出現位置
        }
    }

    // 字面表零命中 (整輪沒 append 任何擴展) → 交 bi-encoder fallback 接住規則表
    // 沒列舉的口語講法。flag 關 / 編碼器未預載時 encoderFallbackExpand 回 '',
    // expanded 維持原值 → 對現有 happy path 零影響。
    if (expanded === query) {
        const fb = encoderFallbackExpand(query);
        if (fb) expanded += " " + fb;
    }
    return expanded;
}

// --- Keyword Extraction ---
function extractKeywords(text) {
    const expandedText = expandQueryIntent(text);
    const stopWords = ['近', '靠近', '想找', '尋找', '住在', '一間', '想要', '預算', '大約', '希望', '位於', '位在', '位處', '在', '含', '有', '附', '座落於', '座落'];
    const locSuffixes = ['路', '街', '大道', '區'];

    return expandedText.split(/\s+|[,，、。]/)
        .filter(k => k.length > 1 && !k.match(/^\d+$/))
        .map(k => {
            let clean = k;
            stopWords.forEach(sw => { if (clean.startsWith(sw)) clean = clean.substring(sw.length); });
            locSuffixes.forEach(suffix => {
                if (clean.endsWith(suffix) && clean.length > suffix.length) {
                    const locPrefixes = ['位', '於', '在', '處'];
                    locPrefixes.forEach(p => { if (clean.startsWith(p)) clean = clean.substring(p.length); });
                }
            });
            return clean;
        })
        .filter(k => k.length > 1);
}

// --- Rule-based Pre-Scoring ---
function calculateRuleBasedScore(candidates, queryKeywords, text, constraints) {
    const { 
        budget: userBudget, minBudget, maxBudget, hasBudgetMention, hasRoomTypeMention, wantsUtilityBilling,
        requireBalcony, requireWindow, requireParking, requireWaste, maxWalkMins, maxScooterMins
    } = constraints;


    const hasLocationMention = queryKeywords.some(kw =>
        kw.endsWith('路') || kw.endsWith('街') || kw.endsWith('大道') ||
        kw.includes('區') || kw.includes('正門') || kw.includes('側門') || kw.includes('男宿')
    );

    // const queryKeywords was already declared at the top of the function
    const ignoreList = ['房', '推薦', '附近', '一下', '預算', '大概', '想要', '需求', '尋找'];
    const semanticMap = {
        '垃圾': ['子母車', '代收垃圾', '垃圾處理', '垃圾子車'],
        '電費': ['台電', '獨立電錶', '台水台電'],
        '陽台': ['陽台', '露台'],
        '電梯': ['電梯', '華廈', '大樓'],
        '車位': ['停車', '車位', '車庫']
    };

    const preScored = candidates.map(prop => {
        let kScore = 0, matchCount = 0, totalRequirements = 0;
        const pText = buildPropText(prop).toLowerCase();

        // 1. Semantic Amenity Scoring (The "Option A" logic)
        queryKeywords.forEach(kw => {
            if (kw.length < 2 || ignoreList.includes(kw)) return;
            
            totalRequirements++;
            // 含同義歸一：擴展詞(可寵/廚房/獨衛…)對上房源實際用詞(可養貓/可開伙/獨立衛浴…)。
            let isMatch = propHasFeature(pText, kw);

            // --- Special Case: Intent-Based Mapping + Boolean Flags ---
            // bool 設施欄一律走 boolFieldState 三態：yes→命中；no→信任文字回退；
            // unknown(該來源此欄崩塌)→純看文字，不因假性 false 而判定無 (待辦1)。
            if (kw.includes('樓梯') || kw.includes('電梯')) {
                const elevatorKws = ['電梯', '華廈', '大樓'];
                isMatch = boolFieldState(prop, 'has_elevator') === 'yes' || elevatorKws.some(alt => pText.includes(alt));
            }
            else if (kw.includes('垃圾') || kw.includes('追車')) {
                const wasteKws = ['子母車', '代收垃圾', '垃圾處理', '垃圾子車'];
                isMatch = boolFieldState(prop, 'has_waste_disposal') === 'yes' || wasteKws.some(alt => pText.includes(alt));
            }
            else if (kw.includes('陽台')) {
                isMatch = boolFieldState(prop, 'has_balcony') === 'yes' || pText.includes('陽台');
            }
            else if (kw.includes('窗')) {
                isMatch = boolFieldState(prop, 'has_window') === 'yes' || pText.includes('窗');
            }
            else if (kw.includes('車位') || kw.includes('停車')) {
                isMatch = boolFieldState(prop, 'has_parking') === 'yes' || pText.includes('車位') || pText.includes('停車');
            }
            else if (kw.includes('電') || kw.includes('錢') || kw.includes('省')) {
                if (kw.includes('電費') || kw.includes('台電') || kw.includes('省')) {
                    const powerKws = ['台電', '獨立電錶', '台水台電'];
                    isMatch = (prop.electricity_billing && prop.electricity_billing.includes('台電')) || 
                              (prop.notes && prop.notes.some(n => n.includes('台電'))) ||
                              powerKws.some(alt => pText.includes(alt));
                }
            }
            
            // Generic semantic expansion for other groups
            if (!isMatch) {
                for (const [group, alternates] of Object.entries(semanticMap)) {
                    if (kw.includes(group) || group.includes(kw)) {
                        if (alternates.some(alt => pText.includes(alt))) {
                            isMatch = true;
                            break;
                        }
                    }
                }
            }

            if (isMatch) {
                matchCount++;
                kScore += 15;
            }
        });

        // 2. Commute Time Scoring
        const isCommuteExplicit = text.includes('近') || text.includes('走') || text.includes('分鐘') || text.includes('公里');
        // distance===0 代表 geocode 失敗(地址爬壞)→ 視為「未知」不加分,
        // 不可當成 0 公里超近校(否則 Math.ceil(0/0.08)=0 會誤加分)。
        const hasCommuteSignal = (prop.walk_mins > 0) || (prop.distance > 0);
        if (maxWalkMins !== null && isCommuteExplicit && hasCommuteSignal) {
            totalRequirements++;
            const propWalk = prop.walk_mins || Math.ceil(prop.distance / 0.08);
            if (propWalk <= maxWalkMins) {
                matchCount++;
                kScore += 20;
            }
        }

        if (maxScooterMins !== null && isCommuteExplicit && hasCommuteSignal) {
            totalRequirements++;
            const propScooter = prop.scooter_mins || Math.max(1, Math.ceil(prop.distance / 0.5));
            if (propScooter <= maxScooterMins) {
                matchCount++;
                kScore += 15;
            }
        }
        
        // Amenity scoring is now handled in Step 1 (Semantic Amenity Scoring)
        // Step 3. Special Contextual Scoring (Location, Room Type, Budget)

        if (hasLocationMention) {
            totalRequirements++;
            let locMatch = false;
            queryKeywords.forEach(kw => {
                if (prop.text.includes(kw)) {
                    if (kw.endsWith('路') || kw.endsWith('街') || kw.endsWith('大道')) kScore += 15, locMatch = true;
                    if (kw.includes('區')) kScore += 5, locMatch = true;
                    if (kw.includes('正門') || kw.includes('側門')) kScore += 10, locMatch = true;
                }
            });
            if (locMatch) matchCount++;
        }

        if (hasRoomTypeMention) {
            totalRequirements++;
            let rtMatch = false;
            ['套房', '雅房', '工作室'].forEach(rt => {
                if (text.includes(rt) && prop.text.includes(rt)) rtMatch = true;
            });
            if (rtMatch) matchCount++, kScore += 10;
        }

        if (hasBudgetMention) {
            totalRequirements += 2;
            if (minBudget !== null && maxBudget !== null) {
                if (prop.rent >= minBudget && prop.rent <= maxBudget) {
                    matchCount += 2;
                    kScore += 10;
                } else if (prop.rent < minBudget) {
                    matchCount += 1.5;
                    kScore += 5;
                } else {
                    const diff = prop.rent - maxBudget;
                    if (diff <= 1000) {
                        matchCount += 0.5;
                        kScore += 1;
                    }
                }
            } else if (userBudget !== null) {
                const diff = prop.rent - userBudget;
                if (Math.abs(diff) <= 500) {
                    matchCount += 2;
                    kScore += 10;
                } else if (prop.rent < userBudget) {
                    matchCount += 1.5;
                    kScore += 3;
                } else if (diff <= 1500) {
                    matchCount += 0.5;
                    kScore += 1;
                }
            }
        }

        if (wantsUtilityBilling) {
            totalRequirements++;
            let utilityMatch = prop.electricity_billing && (
                prop.electricity_billing.includes("台電") ||
                prop.electricity_billing.includes("台水") ||
                prop.electricity_billing === "含電費" ||
                prop.electricity_billing === "獨立電錶"
            );
            if (utilityMatch) {
                matchCount++;
                kScore += 10;
            }
        }

        const rms = totalRequirements > 0 ? (matchCount / totalRequirements) : 1.0;
        return { prop, kScore, rms };
    });

    preScored.sort((a, b) => (b.kScore + b.rms * 20) - (a.kScore + a.rms * 20));
    return preScored.slice(0, 15);  // Reduced from 30→15: fewer AI calls = faster response
}

// --- Response Formatting ---
function formatResponse(scoredResults, top_k) {
    return scoredResults.slice(0, top_k).map(item => ({
        id: item.property.url,
        title: `${item.property.room_type} | ${item.property.address}`,
        price_str: item.property.rent_str,
        url: item.property.url,
        imgUrl: item.property.img || null,
        score: item.score,
        match_reasons: item.match_reasons || [],
        conflict_reason: item.conflict_reason || null,
        size: item.property.size || "坪數未提供",
        floor: item.property.floor || "樓層未提供",
        furniture: item.property.furniture || "無特殊設施提供",
        distance: item.property.distance,
        address: item.property.address,
        contact: item.property.contact || "不具名",
        phone: item.property.phone || "無資料",
        features: item.property.features || "",
        deposit: item.property.deposit ?? null,
        deposit_str: item.property.deposit_str || "",
    }));
}

let currentQueryId = 0;

// --- Main Recommendation Pipeline ---
// onPartialResult(results): optional callback called immediately with rule-based results
export async function recommend(text, top_k = 20, onPartialResult = null) {
    // Increment the query ID — any in-progress inference with an older ID will detect
    // the mismatch and exit early, allowing this new query to proceed immediately.
    const myQueryId = ++currentQueryId;

    const isCancelled = () => myQueryId !== currentQueryId;

    try {
        console.log("User Query:", text);
        const startTime = performance.now();

        // 1. Data Parsing & Filtering
        const constraints = parseConstraintsFromText(text);

        // Re-parse walk/scooter limits from expanded text (e.g. "走路可以到" → "走路10分")
        if (constraints.maxWalkMins === null || constraints.maxScooterMins === null) {
            const expandedForConstraints = expandQueryIntent(text);
            const rtExp = expandedForConstraints.replace(/一/g,'1').replace(/二/g,'2').replace(/兩/g,'2').replace(/三/g,'3')
                .replace(/四/g,'4').replace(/五/g,'5').replace(/六/g,'6').replace(/七/g,'7')
                .replace(/八/g,'8').replace(/九/g,'9').replace(/十/g,'10').replace(/半/g,'30');
            if (constraints.maxWalkMins === null) {
                const wm = rtExp.match(/(?:走路|步行)[^\d]*(\d+)[^\d]*(?:分鐘|分)/);
                if (wm) constraints.maxWalkMins = parseInt(wm[1]);
            }
            if (constraints.maxScooterMins === null) {
                const sm = rtExp.match(/(?:機車|騎車)[^\d]*(\d+)[^\d]*(?:分鐘|分)/);
                if (sm) constraints.maxScooterMins = parseInt(sm[1]);
            }
        }

        // 1.5 NER entity extraction — runs in parallel with hard filtering
        const nerEntities = await nerExtract(text);
        if (nerEntities.locations.length > 0) {
            constraints.nerLocations = nerEntities.locations;
        }

        // Augment budget constraints with NER-detected BGT entities when regex missed them
        if (nerEntities.budgets && nerEntities.budgets.length > 0 && !constraints.hasBudgetMention) {
            const nerBudget = parseBudgetFromNER(nerEntities.budgets);
            if (nerBudget) {
                constraints.budget = nerBudget.budget;
                constraints.limit  = nerBudget.limit;
                constraints.hasBudgetMention = true;
                console.log('[NER] Budget extracted from BGT entity:', nerBudget);
            }
        }

        const candidates = filterHardExclusions(propertyData, constraints);

        // 2. Keyword & Rule-based Pre-scoring
        const queryKeywords = extractKeywords(text);

        // Augment keywords with NER-detected features and locations
        [...nerEntities.features, ...nerEntities.locations].forEach(k => {
            if (k && k.length > 1 && !queryKeywords.includes(k)) queryKeywords.push(k);
        });
        const topCandidates = calculateRuleBasedScore(candidates, queryKeywords, text, constraints);

        // 2.5 Progressive: Immediately yield rule-based top results so UI feels instant
        if (onPartialResult && topCandidates.length > 0) {
            const quickResults = topCandidates.slice(0, top_k).map(({ prop, rms }) => ({
                property: prop,
                score: Math.round(rms * 75), // Rule-based estimate
                match_reasons: explainMatch(text, prop, constraints),
                conflict_reason: checkConflicts(prop, constraints)
            }));
            quickResults.sort((a, b) => b.score - a.score);
            onPartialResult(formatResponse(quickResults, top_k));
        }

        // 2.6 Yield to UI thread again before starting expensive AI inference
        await new Promise(resolve => setTimeout(resolve, 50));

        // 3. AI Re-ranking (runs after partial results are shown)
        const scoredResults = [];
        for (let i = 0; i < topCandidates.length; i++) {
            // If a newer query has arrived, abort this one immediately
            if (isCancelled()) return null;

            const { prop, rms } = topCandidates[i];
            try {
                const aiScore = await scorePair(text, prop.text);
                
                // RoBERTa scores are well-calibrated (0.0 ~ 1.0), apply light rescaling
                const normalizedAiScore = Math.max(0, Math.min(1.0, (aiScore - 0.01) / 0.89));
                
                let finalPercentage = Math.round((rms * 35) + (normalizedAiScore * 65));
                if (rms === 1.0 && finalPercentage < 80) finalPercentage = 80 + Math.round(normalizedAiScore * 15);
                
                // --- Explainability & Hybrid Filtering (Option 1) ---
                const match_reasons = explainMatch(text, prop, constraints);
                const conflict_reason = checkConflicts(prop, constraints);
                
                if (conflict_reason) {
                    finalPercentage *= 0.1; // Aggressive reduction for conflicts
                }
                
                scoredResults.push({ 
                    property: prop, 
                    score: Math.min(100, Math.round(finalPercentage)),
                    match_reasons,
                    conflict_reason
                });
            } catch (err) {
                console.error(`AI scoring error for property ${i}:`, err);
            }
        }

        // 4. Return final AI-ranked results
        if (isCancelled()) return null;
        scoredResults.sort((a, b) => b.score - a.score);
        console.log(`Inference complete: ${scoredResults.length} results in ${(performance.now() - startTime).toFixed(0)}ms`);

        if (scoredResults.length > 0) {
            console.log("Top Match:", { query: text, property: scoredResults[0].property.text, score: scoredResults[0].score + "%" });
        }

        return formatResponse(scoredResults, top_k);
    } catch (err) {
        throw err;
    }
}
