/**
 * inference-worker.js - Off-main-thread ONNX Inference Worker
 *
 * Handles loading of the 84MB ONNX model and all semantic scoring.
 * Runs as an ES Module Worker ({type: 'module'}) to support top-level imports.
 */

import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// ONNX Runtime Web must be loaded via dynamic import inside the worker
let ort = null;

let tokenizer = null;
let session = null;
const MAX_LENGTH = 64;

async function init(localOrigin, noCache = false) {
    try {
        // Dynamically load ORT inside the worker context
        const ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs');
        ort = ortModule.default ?? ortModule;

        // 1. Configure Transformers.js
        env.allowRemoteModels = false;
        env.allowLocalModels = true;
        env.useBrowserCache = !noCache;   // cold benchmark: bypass Transformers.js cache
        env.localModelPath = localOrigin + '/';

        // 2. Start Tokenizer and Model download in parallel
        // Progress mapping: tokenizer = 0–5%, model download = 5–100%
        const MODEL_SIZE = 59_000_000; // ~57 MB, used when Content-Length missing
        const tokenizerPromise = AutoTokenizer.from_pretrained('models/custom_onnx_model_dir', {
            progress_callback: (p) => {
                if (p.status === 'progress') {
                    // Map tokenizer progress to 0–5% of the total bar
                    const pct = p.total > 0 ? (p.loaded / p.total) : 0;
                    postMessage({ type: 'status', message: '正在加載分詞器...', loaded: Math.round(pct * 0.05 * MODEL_SIZE), total: MODEL_SIZE });
                }
            }
        });

        const modelUrl = localOrigin + '/models/custom_onnx_model_dir/my_custom_model_quant.onnx';
        const modelFetchPromise = (async () => {
            const response = await fetch(modelUrl, { cache: noCache ? 'no-store' : 'force-cache' });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const reader = response.body.getReader();
            const contentLength = +response.headers.get('Content-Length') || MODEL_SIZE;

            let receivedLength = 0;
            let chunks = [];
            let lastUpdate = 0;

            while(true) {
                const {done, value} = await reader.read();
                if (done) break;
                chunks.push(value);
                receivedLength += value.length;

                // Throttled UI update (every 512KB) to save CPU
                // Map model download to 5–100% of the total bar
                if (receivedLength - lastUpdate > 512 * 1024) {
                    const modelPct = Math.min(receivedLength / contentLength, 1);
                    const mappedLoaded = Math.round((0.05 + modelPct * 0.95) * contentLength);
                    postMessage({
                        type: 'status',
                        message: '正在下載 AI 模型...',
                        loaded: mappedLoaded,
                        total: contentLength
                    });
                    lastUpdate = receivedLength;
                }
            }
            
            const modelBuffer = new Uint8Array(receivedLength);
            let position = 0;
            for(let chunk of chunks) {
                modelBuffer.set(chunk, position);
                position += chunk.length;
            }
            return modelBuffer;
        })();

        // Wait for both to complete
        const [loadedTokenizer, loadedModelBuffer] = await Promise.all([
            tokenizerPromise,
            modelFetchPromise
        ]);

        tokenizer = loadedTokenizer;

        // 2. Create Session
        session = await ort.InferenceSession.create(loadedModelBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            sessionOptions: { numThreads: 4 }
        });

        postMessage({ type: 'ready' });
    } catch (err) {
        postMessage({ type: 'error', message: err.message });
    }
}

async function scorePair(query, propertyText) {
    const encoded = await tokenizer(query, {
        text_pair: propertyText,
        padding: 'max_length',
        truncation: true,
        max_length: MAX_LENGTH,
        return_tensors: 'np',
        return_token_type_ids: true,
    });

    const inputs = {};
    for (const key of session.inputNames) {
        if (encoded[key]) {
            inputs[key] = new ort.Tensor('int64',
                BigInt64Array.from(encoded[key].data.map(v => BigInt(v))),
                encoded[key].dims
            );
        }
    }

    const results = await session.run(inputs);
    const logits = results.logits.data;
    const maxL = Math.max(logits[0], logits[1]);
    const exp0 = Math.exp(logits[0] - maxL);
    const exp1 = Math.exp(logits[1] - maxL);
    return exp1 / (exp0 + exp1);
}

/**
 * semanticExpandQuery - Maps colloquial intentions to specific property features.
 */
function semanticExpandQuery(query) {
    // 與 pipeline/data_prep/lifestyle_mapper.py LIFESTYLE_CLUSTERS 保持同步
    const expansionMap = {
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

    let expanded = query;

    // 否定守衛:略過「被否定詞緊鄰修飾」的命中,避免子字串碰撞把反義句帶偏。
    // 例:「不開車」含「開車」、「不怕熱」含「怕熱」→ 否則會把使用者明確不要的設施
    // (車位 / 冷氣)擴展進去餵 CE,反而把那類房源推高。比對 key 前一字是否為否定詞。
    // 與 inference.js 的 expandQueryIntent 守衛保持同款邏輯(markers 外,sync 腳本不覆寫)。
    // 限制:只擋「否定詞緊貼 key」,隔字否定(如「不想養貓」)仍漏接,需 bi-encoder fallback 根治。
    const NEGATORS = '不沒無非免勿';
    for (const [key, expansion] of Object.entries(expansionMap)) {
        let from = 0, idx;
        while ((idx = query.indexOf(key, from)) !== -1) {
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

// Buffer-based init: warm benchmark passes pre-fetched ArrayBuffer from main thread Cache Storage
async function initFromBuffer(localOrigin, modelBuffer) {
    try {
        const ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs');
        ort = ortModule.default ?? ortModule;

        env.allowRemoteModels = false;
        env.allowLocalModels = true;
        env.useBrowserCache = true;
        env.localModelPath = localOrigin + '/';

        tokenizer = await AutoTokenizer.from_pretrained('models/custom_onnx_model_dir');

        session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            sessionOptions: { numThreads: 4 }
        });
        postMessage({ type: 'ready' });
    } catch (err) {
        postMessage({ type: 'error', message: err.message });
    }
}

onmessage = async (e) => {
    const { type, data } = e.data;

    if (type === 'init') {
        await init(data.origin, data.noCache ?? false);
    } else if (type === 'init_buffer') {
        await initFromBuffer(data.origin, data.modelBuffer);
    } else if (type === 'score') {
        const { query, propertyText, id } = data;
        
        // --- [NEW] Semantic Query Expansion ---
        const expandedQuery = semanticExpandQuery(query);
        
        const score = await scorePair(expandedQuery, propertyText);
        postMessage({ type: 'scoreResult', score, id, expandedQuery });
    }
};
