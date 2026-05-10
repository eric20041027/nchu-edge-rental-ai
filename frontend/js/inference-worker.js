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

async function init(localOrigin) {
    try {
        // Dynamically load ORT inside the worker context
        const ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs');
        ort = ortModule.default ?? ortModule;

        // 1. Configure Transformers.js
        env.allowRemoteModels = false;
        env.allowLocalModels = true;
        env.useBrowserCache = true;
        env.localModelPath = localOrigin + '/';

        // 2. Start Tokenizer and Model download in parallel
        const tokenizerPromise = AutoTokenizer.from_pretrained('models/custom_onnx_model_dir', {
            progress_callback: (p) => {
                if (p.status === 'progress') {
                    postMessage({ type: 'status', message: '正在加載分詞器...', loaded: p.loaded, total: p.total });
                }
            }
        });

        const modelUrl = localOrigin + '/models/custom_onnx_model_dir/my_custom_model_quant.onnx?v=' + Date.now();
        const modelFetchPromise = (async () => {
            const response = await fetch(modelUrl, { cache: 'force-cache' });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const reader = response.body.getReader();
            const contentLength = +response.headers.get('Content-Length');
            
            let receivedLength = 0;
            let chunks = [];
            let lastUpdate = 0;

            while(true) {
                const {done, value} = await reader.read();
                if (done) break;
                chunks.push(value);
                receivedLength += value.length;
                
                // Throttled UI update (every 512KB) to save CPU
                if (receivedLength - lastUpdate > 512 * 1024) {
                    postMessage({ 
                        type: 'status', 
                        message: '正在下載 AI 模型...', 
                        loaded: receivedLength, 
                        total: contentLength || 88000000 
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
    const expansionMap = {
        "潔癖": "全新 獨洗 禁菸 乾淨",
        "愛乾淨": "全新 獨洗 禁菸 乾淨",
        "想下廚": "可伙 廚房 抽油煙機 瓦斯爐",
        "要下廚": "可伙 廚房 抽油煙機 瓦斯爐",
        "可以煮東西": "可伙 廚房",
        "不想爬樓梯": "電梯 大樓 華廈",
        "懶人": "電梯 子母車 垃圾處理 飲水機",
        "搬東西": "電梯 搬家",
        "怕熱": "冷氣 吹冷氣 變頻",
        "夏天": "冷氣",
        "打報告": "寬頻 網路 上網 書桌",
        "上網": "寬頻 網路",
        "獨洗獨曬": "洗衣機 陽台 曬衣 獨洗",
        "可貓": "可寵 養寵 寵物友善",
        "可狗": "可寵 養寵 寵物友善",
        "有毛孩": "可寵 寵物",
        "台水電": "台電 台水 帳單 自繳",
        "省電費": "變頻 台電",
        "自己煮": "廚房 瓦斯 開火 自炊 省錢",
        "省伙食費": "廚房 瓦斯 開火",
        "外送族": "管理員 飲水機 子母車",
        "不想出門": "管理員 飲水機 子母車",
        "潔癖": "全新 獨洗 禁菸 乾淨",
        "稍微潔癖": "全新 獨洗 禁菸 乾淨",
        "怕悶熱": "陽台 採光 通風 對外窗",
        "有車": "車位 停車場",
        "開車": "車位 停車場",
        "怕吵": "隔音 氣密窗 禁菸 寧靜",
        "首選": "全新",
        "不想追垃圾車": "子母車 垃圾處理",
        "網美": "裝潢 採光 漂亮 落地窗",
        "採光好": "落地窗 採光",
    };

    let expanded = query;
    for (const [key, expansion] of Object.entries(expansionMap)) {
        if (query.includes(key)) {
            expanded += " " + expansion;
        }
    }
    return expanded;
}

onmessage = async (e) => {
    const { type, data } = e.data;

    if (type === 'init') {
        await init(data.origin);
    } else if (type === 'score') {
        const { query, propertyText, id } = data;
        
        // --- [NEW] Semantic Query Expansion ---
        const expandedQuery = semanticExpandQuery(query);
        
        const score = await scorePair(expandedQuery, propertyText);
        postMessage({ type: 'scoreResult', score, id, expandedQuery });
    }
};
