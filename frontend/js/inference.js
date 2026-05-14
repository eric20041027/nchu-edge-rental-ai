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

// --- Property Data Synchronization ---
export async function initData() {
    const response = await fetch('assets/property_data.json?v=20260310');
    propertyData = await response.json();
    console.log(`Loaded ${propertyData.length} property descriptions`);
}

// --- NLP Engine Initialization via Web Worker ---
export async function initNLP(onProgress) {
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

    if (text.includes('套房') || text.includes('雅房') || text.includes('工作室')) {
        hasRoomTypeMention = true;
    }

    return { 
        budget, minBudget, maxBudget, limit, genderUnrestricted, hasGenderMention, hasBudgetMention, hasRoomTypeMention, 
        wantsUtilityBilling, maxElectricityPrice, requireBalcony, requireWindow, requireParking, requireWaste, 
        requireSubsidy, isSocialHousing,
        excludeRooftop, excludeWooden, excludeHaunted, maxWalkMins, maxScooterMins,
        wantsPet: (text.includes('養貓') || text.includes('養狗') || text.includes('寵物')),
        requireWaterDispenser: (text.includes('飲水機')),
        requirePrivateWasher: (text.includes('獨洗') || text.includes('個人洗衣機')),
        requireGuard: (text.includes('代收') || text.includes('包裹') || text.includes('管理員') || text.includes('警衛')),
        originalText: text // Added to fix property access in checkConflicts
    };
}

// --- Explainability: Match Reasons & Conflict Detection ---
function explainMatch(query, prop, constraints) {
    const reasons = [];
    const pText = (prop.text + (prop.furniture || "") + (prop.notes ? prop.notes.join(" ") : "")).toLowerCase();
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
    if (q.includes('走路') || q.includes('近') || q.includes('步行')) {
        if (prop.geo_tier === "core") {
            reasons.push("🚀 步行核心區，下課就到家");
        } else if (prop.geo_tier === "active") {
            reasons.push("📍 位於熱鬧商圈，生活機能優");
        }
    }

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
            triggers: ['廚房', '開伙', '煮飯', '瓦斯', '自炊', '炒菜', '料理'],
            check: p => p.includes('廚房') || p.includes('開伙') || p.includes('瓦斯') || p.includes('可自炊'),
            label: '🍳 可開伙自炊'
        },
        {
            triggers: ['電梯', '升降梯', '不爬樓', '不用爬'],
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
    const { wantsPet } = constraints;
    const pText = prop.text + (prop.notes ? prop.notes.join(" ") : "");
    
    // 1. Pet Conflict
    if (wantsPet && (pText.includes('禁養') || pText.includes('不可養'))) {
        return "此房源禁養寵物";
    }

    // 2. Gender Conflict (Simplified detection)
    if (constraints.hasGenderMention && constraints.originalText) {
        if (constraints.genderUnrestricted === false) {
             if (pText.includes('限女性') && constraints.originalText.includes('男')) return "此房源僅限女性";
             if (pText.includes('限男性') && constraints.originalText.includes('女')) return "此房源僅限男性";
        }
    }

    // 3. Smoking
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
        requireSubsidy, isSocialHousing, requireBalcony, requireWindow, requireParking, requireWaste
    } = constraints;
    const candidates = [];
    
    for (const prop of properties) {
        // 1. Core Policy Exclusions (Keep these hard)
        if (excludeRooftop && (prop.is_rooftop || prop.text.includes('頂加'))) continue;
        if (excludeWooden && prop.is_wooden_partition) continue;
        if (requireSubsidy && (prop.text.includes('不可補助') || prop.text.includes('不可報稅') || prop.text.includes('不可入籍'))) continue;
        if (isSocialHousing && !prop.text.includes('社會住宅') && !prop.text.includes('社宅')) continue;

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
            const match = prop.electricity_billing.match(/(\d+(?:\.\d+)?)/);
            if (match && parseFloat(match[1]) > maxElectricityPrice) continue;
        }

        // If user specifically asks for Taishui Taipower and NOT maxElectricityPrice, 
        // we can filter out properties that are explicitly > 5 NTD, though we handle this softly in scoring too.
        if (wantsUtilityBilling && prop.electricity_billing && prop.electricity_billing.includes("度")) {
            const match = prop.electricity_billing.match(/(\d+(?:\.\d+)?)/);
            if (match && parseFloat(match[1]) >= 5) {
                // If they explicitly want Taishui Taipower, properties charging >= 5/kwh are generally rejected
                continue; 
            }
        }

        if (hasGenderMention && genderUnrestricted) {
            const isFemaleOnly = prop.text.includes('限女') || (prop.furniture && prop.furniture.includes('限女'));
            if (isFemaleOnly) continue;
        }
        if (hasBudgetMention) {
            if (maxBudget !== null && prop.rent > maxBudget) continue;
            if (limit && budget !== null) {
                if (limit === 'below' && prop.rent > budget) continue;
                if (limit === 'above' && prop.rent < budget) continue;
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

// --- Semantic Query Expansion ---
function expandQueryIntent(query) {
    let expanded = query;
    const intentMap = {
        '潔癖': '全新 獨洗 禁菸 質感 裝潢 漂亮',
        '下班晚': '子母車 垃圾代收 門禁 管理員 安全',
        '省錢': '台電 台水 便宜 補助 租補',
        '怕熱': '台電 變頻冷氣 西曬隔熱',
        '怕吵': '水泥隔間 巷弄 靜巷 頂樓',
        '高品質': '五星級 管理員 電梯 漂亮 全新 質感',
        '生活便利': '超商 興大路 核心區 核心 核心圈'
    };

    for (const [intent, expansion] of Object.entries(intentMap)) {
        if (query.includes(intent)) {
            expanded += " " + expansion;
        }
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
        const pText = (prop.text + (prop.furniture || "") + (prop.notes ? prop.notes.join(" ") : "")).toLowerCase();

        // 1. Semantic Amenity Scoring (The "Option A" logic)
        queryKeywords.forEach(kw => {
            if (kw.length < 2 || ignoreList.includes(kw)) return;
            
            totalRequirements++;
            let isMatch = pText.includes(kw);
            
            // --- Special Case: Intent-Based Mapping + Boolean Flags ---
            if (kw.includes('樓梯') || kw.includes('電梯')) {
                const elevatorKws = ['電梯', '華廈', '大樓'];
                isMatch = prop.has_elevator || elevatorKws.some(alt => pText.includes(alt));
            } 
            else if (kw.includes('垃圾') || kw.includes('追車')) {
                const wasteKws = ['子母車', '代收垃圾', '垃圾處理', '垃圾子車'];
                isMatch = prop.has_waste_disposal || wasteKws.some(alt => pText.includes(alt));
            }
            else if (kw.includes('陽台')) {
                isMatch = prop.has_balcony || pText.includes('陽台');
            }
            else if (kw.includes('窗')) {
                isMatch = prop.has_window || pText.includes('窗');
            }
            else if (kw.includes('車位') || kw.includes('停車')) {
                isMatch = prop.has_parking || pText.includes('車位') || pText.includes('停車');
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
        if (maxWalkMins !== null && isCommuteExplicit) {
            totalRequirements++;
            const propWalk = prop.walk_mins || Math.ceil(prop.distance / 0.08);
            if (propWalk <= maxWalkMins) {
                matchCount++;
                kScore += 20; 
            }
        }
        
        if (maxScooterMins !== null && isCommuteExplicit) {
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
            let utilityMatch = prop.electricity_billing === "台水台電" || prop.electricity_billing === "含電費";
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
