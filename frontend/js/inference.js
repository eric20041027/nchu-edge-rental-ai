/**
 * inference.js — Sentence-Pair Classification Recommendation Engine
 *
 * Uses the fine-tuned ALBERT model for sentence-pair classification:
 * Input:  [CLS] user_query [SEP] property_description [SEP]
 * Output: logits → softmax → match probability
 *
 * Sequential inference to avoid ONNX "Session already started" error.
 */
import { parseConstraintsFromText, parseBudgetFromNER } from './constraint-parser.js';
import { boolFieldState, buildPropText, propHasFeature } from './property-features.js';
import { explainMatch, checkConflicts } from './explainability.js';
// 註:主執行緒不直接用 transformers.js — 推論在 inference-worker.js / ner-worker.js(各自 import)。

// 階段② 反饋重排(CE 精排後的純後處理,可關)。spec: docs/spec/feedback-rerank.md
import { readFeedbackLog, loadFeedbackScores, applyFeedbackRerank } from './feedback.js';

let worker = null;
let propertyData = [];
let pendingInference = new Map();
let inferenceIdCounter = 0;

// NER worker state
let nerWorker   = null;
let pendingNER  = new Map();
let nerIdCounter = 0;
let nerReady    = false;

// --- Vector recall (T5;T7 A/B 已採用為 primary) ------------------------------
// 向量召回已是正式召回路徑(T7 A/B GO:semantic Recall@30 0.007→0.547、整體皆贏)。
// 此旗標保留為 kill-switch:出事翻 false 可秒回退關鍵字 rule-based recall。
// rule-based 不再是對等 A/B 分支,但仍當 worker 未就緒/編碼逾時時的 fallback(見下方)。
const VECTOR_RECALL_ENABLED = true;

// 階段② kill-switch:翻 false 即停用回饋重排,排序逐位元組回到階段①。
const FEEDBACK_RERANK_ENABLED = true;

// bi-encoder worker 狀態 (鏡像 NER worker)
let biEncoderWorker = null;
let biEncoderReady  = false;
let pendingBiEnc    = new Map();
let biEncIdCounter  = 0;

// 離線預算的房源向量 (loadPropertyData 載入):
//   { dim, idxs:[704 個 property_data idx], vecs:Float32Array(704*768),
//     byIdx:Map<property_data idx → row 起始 offset> }
let propertyEmbeddings = null;


// --- Property Data Synchronization ---
// Fetch with retry + backoff. Mobile reloads often abort in-flight requests or
// hit transient 5G drops, which previously failed the whole init with no recovery.
async function fetchWithRetry(url, { retries = 3, backoff = 600 } = {}) {
    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const response = await fetch(url, { cache: 'default' });
            if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
            return response;
        } catch (e) {
            lastErr = e;
            if (attempt < retries) {
                await new Promise(r => setTimeout(r, backoff * (attempt + 1)));
            }
        }
    }
    throw lastErr;
}

export async function initData() {
    const response = await fetchWithRetry('assets/property_data.json?v=20260616e');
    const raw = await response.json();
    // 過濾爬蟲空殼房源:來源網站有少數房源 address/rent 全空(只有 url+img,連價格地址都
    // 沒有),對使用者完全無用,且軟性 query(如「希望房間明亮一點」)下無條件扣分反而會被
    // 排成 TOP 1 → 卡片價格/標題空白(NT$ 與標題皆空)。在載入階段剔除根治。
    propertyData = raw.filter(p => p && p.address && p.rent > 0);
    const dropped = raw.length - propertyData.length;
    console.log(`Loaded ${propertyData.length} property descriptions${dropped ? ` (剔除 ${dropped} 筆無效房源)` : ''}`);

    // 向量召回:載入離線預算的房源向量 (flag 開時才載,省流量)。
    // 傳入未過濾的 raw:embeddings 的 idxs 是針對「原始 704 筆 property_data.json」
    // 的索引,而 propertyData 已剔除 3 筆空殼 → 不能用 propertyData[idx] 位置對映。
    if (VECTOR_RECALL_ENABLED) {
        loadPropertyEmbeddings(raw);
    }
}

// 載入 property_embeddings.json,把 vecs 攤平成一條 Float32Array。
// rawData = 原始(未過濾)property_data 陣列,embeddings 的 idxs 即其索引。
// 建 idx → prop 物件的 Map (idxToProp) 直接對映,避開過濾後位置錯位。
// 失敗為非致命:propertyEmbeddings 維持 null → recommend 自動 fallback 到 rule-based。
async function loadPropertyEmbeddings(rawData) {
    try {
        const resp = await fetchWithRetry('assets/property_embeddings.json?v=20260624');
        const emb = await resp.json();
        const vecs = Float32Array.from(emb.vecs);   // 704*768 flat row-major,每列已 L2 norm
        const idxToProp = new Map();
        for (const idx of emb.idxs) {
            if (rawData[idx]) idxToProp.set(idx, rawData[idx]);  // 對映到原始 prop 物件
        }
        propertyEmbeddings = { dim: emb.dim, idxs: emb.idxs, vecs, idxToProp };
        console.log(`Loaded ${emb.count} property embeddings (dim ${emb.dim})`);
    } catch (err) {
        console.warn('[vectorRecall] 房源向量載入失敗,回退 rule-based recall:', err);
        propertyEmbeddings = null;
    }
}

// --- NLP Engine Initialization via Web Worker ---
export async function initNLP(onProgress) {
    // 向量召回 bi-encoder 由 app.js 帶進度 callback 啟動 (見 initBiEncoder export);
    // 此處不再自行 init,避免無 callback 的 worker 搶先建立。
    if (!worker) {
        return new Promise((resolve, reject) => {
            console.log("Initializing Inference Web Worker...");
            worker = new Worker('js/inference-worker.js', { type: 'module' });

            worker.onmessage = (e) => {
                const { type, message, score, id, error, loaded, total } = e.data;
                if (type === 'status' && onProgress) {
                    onProgress({ status: 'progress', message, loaded, total, init: e.data.init });
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
                if (onProgress) onProgress({ loaded: e.data.loaded, total: e.data.total, init: e.data.init });
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

// --- Bi-encoder Worker Initialization (向量召回) ---
// 鏡像 initNER:flag 關閉時直接 no-op,不建立 worker。非阻塞、非致命。
// onProgress({loaded,total,init})、onReady() 供載入畫面顯示第三條進度條。
// 回傳 true 表示已啟動 worker;false 表示 flag 關閉(no-op,呼叫端應視為「無此模型」)。
export function initBiEncoder(onProgress = null, onReady = null) {
    if (!VECTOR_RECALL_ENABLED) return false;
    if (biEncoderWorker) { if (onReady && biEncoderReady) onReady(); return true; }
    biEncoderWorker = new Worker('js/bi-encoder-worker.js', { type: 'module' });
    biEncoderWorker.onmessage = (e) => {
        const { type, embedding, id, loaded, total } = e.data;
        if (type === 'bienc_ready') {
            biEncoderReady = true;
            console.log('Bi-encoder Worker Ready');
            if (onReady) onReady();
        } else if (type === 'bienc_progress') {
            if (onProgress) onProgress({ loaded, total });
        } else if (type === 'encodeResult') {
            const cb = pendingBiEnc.get(id);
            if (cb) { cb(embedding); pendingBiEnc.delete(id); }
        } else if (type === 'bienc_error') {
            console.warn('Bi-encoder worker error:', e.data.message);  // 非致命 → 回退 rule-based
        } else if (type === 'bienc_status') {
            // 下載結束後的 WASM session 編譯期 → 回報 init 旗標,UI 顯示「初始化中…」
            if (onProgress && e.data.init) onProgress({ init: true });
            console.log('[BiEnc]', e.data.message);
        }
    };
    biEncoderWorker.postMessage({ type: 'bienc_init', data: { origin: window.location.origin } });
    return true;
}

// --- Query 向量編碼 (800ms timeout,逾時/未就緒回 null → caller fallback) ---
async function encodeQuery(text) {
    if (!VECTOR_RECALL_ENABLED) return null;
    if (!biEncoderWorker || !biEncoderReady) return null;
    return new Promise((resolve) => {
        const id = biEncIdCounter++;
        const timer = setTimeout(() => {
            pendingBiEnc.delete(id);
            resolve(null);
        }, 800);
        pendingBiEnc.set(id, (embedding) => {
            clearTimeout(timer);
            resolve(embedding ? Float32Array.from(embedding) : null);
        });
        biEncoderWorker.postMessage({ type: 'encode', data: { query: text, id } });
    });
}

// --- cosine top-k:queryVec 與房源向量皆 unit-norm → dot = cosine ---
// 回傳 [{idx, score}] (idx = property_data idx),score 由高到低排序。
// 704*768 純 JS 迴圈即可,無需向量化。
function cosineTopK(queryVec, k) {
    if (!propertyEmbeddings || !queryVec) return [];
    const { dim, idxs, vecs } = propertyEmbeddings;
    const rows = idxs.length;
    const scored = new Array(rows);
    for (let row = 0; row < rows; row++) {
        const base = row * dim;
        let dot = 0;
        for (let j = 0; j < dim; j++) dot += queryVec[j] * vecs[base + j];
        scored[row] = { idx: idxs[row], score: dot };
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, Math.max(0, Math.min(k, rows)));
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

// --- Hard Exclusion Filtering ---
function filterHardExclusions(properties, constraints) {
    const {
        budget, maxBudget, limit, genderUnrestricted, hasGenderMention, hasBudgetMention,
        excludeRooftop, excludeWooden, maxElectricityPrice, wantsUtilityBilling,
        maxWalkMins, maxScooterMins,
        requireSubsidy, isSocialHousing,
        wantsPet, excludePet, requireElevator, requireCooking
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
        // 「不要養寵物」:排除明確可養寵物的房源(notes/text 含「可養」)。使用者要避開寵物房,
        // 把可養寵物者一票否決;未提及寵物政策的房源留給 AI 判斷(不過度刪減候選池)。
        if (excludePet) {
            const petText = buildPropText(prop);
            if (petText.includes('可養') || petText.includes('寵物友善')) continue;
        }
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




// --- Semantic Query Expansion ---
function expandQueryIntent(query) {
    let expanded = query;
    const intentMap = {
        // >>> GENERATED: semantic rules (sync_semantic_rules.py) >>>
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
        maxWalkMins, maxScooterMins
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

        // 2. Recall stage — 產出 topCandidates: [{prop, rms}],供 step-3 CE rerank 消費。
        //    向量召回為 primary(T7 採用);worker 未就緒/編碼逾時/kill-switch 關 → fallback rule-based。
        let topCandidates = null;

        if (VECTOR_RECALL_ENABLED && propertyEmbeddings && biEncoderReady) {
            const qVec = await encodeQuery(text);  // 逾時/失敗回 null → 下方回退
            if (qVec) {
                // hard exclusion 仍須遵守:先取得允許集 (頂加/木板/凶宅/預算硬篩),
                // 再從 cosine top-k 取「在允許集內」的命中,維持 cosine 序,取前 30。
                // 即:用向量相似度取代關鍵字 SCORING,但硬約束照樣 honor。
                // candidates 與 idxToProp 皆指向同批 raw prop 物件 → 用物件參考做交集。
                const allowed = new Set(candidates);
                const hits = cosineTopK(qVec, propertyEmbeddings.idxs.length);
                topCandidates = [];
                for (const { idx, score } of hits) {
                    const prop = propertyEmbeddings.idxToProp.get(idx);
                    if (!prop || !allowed.has(prop)) continue;
                    // rms (下游 rule-based blend + rms===1.0 boost 用) = cosine 夾到 0..1。
                    const rms = Math.max(0, Math.min(1, score));
                    topCandidates.push({ prop, rms });
                    if (topCandidates.length >= 30) break;
                }
                console.log(`[vectorRecall] ${topCandidates.length} candidates via bi-encoder cosine`);
            }
        }

        // Fallback (非對等 A/B 分支,僅安全網):kill-switch 關 / 向量未就緒 / 編碼逾時
        //   → 原本的關鍵字 rule-based recall,避免零結果。T7=GO 後保留此路徑當 degradation。
        if (topCandidates === null) {
            const queryKeywords = extractKeywords(text);
            // Augment keywords with NER-detected features and locations
            [...nerEntities.features, ...nerEntities.locations].forEach(k => {
                if (k && k.length > 1 && !queryKeywords.includes(k)) queryKeywords.push(k);
            });
            topCandidates = calculateRuleBasedScore(candidates, queryKeywords, text, constraints);
        }

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
        let scoredResults = [];  // let:階段② ON 時 applyFeedbackRerank 回新陣列需重新賦值
        for (let i = 0; i < topCandidates.length; i++) {
            // If a newer query has arrived, abort this one immediately
            if (isCancelled()) return null;

            const { prop, rms } = topCandidates[i];
            try {
                // C 組 cross-encoder 用富化房源文字訓練 → 必須餵 prop.ce_text(預算於
                // property_data.json,= property_to_text_enriched:全 notes + 全 furniture)。
                // 餵舊的短 prop.text 會 OOD(訓練/推論不一致)。ce_text 缺失時 fallback
                // prop.text 以防舊快取 JSON,但正常情況 704 筆皆有。
                const aiScore = await scorePair(text, prop.ce_text || prop.text);
                
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
        // 階段②:CE 精排後的回饋重排(可關)。OFF 時為原地 sort,逐位元組回到階段①。
        if (FEEDBACK_RERANK_ENABLED) {
            scoredResults = applyFeedbackRerank(scoredResults, loadFeedbackScores(readFeedbackLog()));
        } else {
            scoredResults.sort((a, b) => b.score - a.score);
        }
        console.log(`Inference complete: ${scoredResults.length} results in ${(performance.now() - startTime).toFixed(0)}ms`);

        if (scoredResults.length > 0) {
            console.log("Top Match:", { query: text, property: scoredResults[0].property.text, score: scoredResults[0].score + "%" });
        }

        return formatResponse(scoredResults, top_k);
    } catch (err) {
        throw err;
    }
}
