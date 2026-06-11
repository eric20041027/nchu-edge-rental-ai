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
        // ── 清潔 / 潔癖 ──────────────────────────────
        "潔癖":           "全新 獨洗 禁菸 乾淨 裝潢",
        "稍微潔癖":       "全新 獨洗 禁菸",
        "愛乾淨":         "全新 獨洗 禁菸 乾淨",

        // ── 煮飯 / 自炊（完整口語覆蓋）──────────────
        "想在家煮飯":     "可伙 廚房 流理台 瓦斯爐 電磁爐 開火",
        "想自己煮飯":     "可伙 廚房 流理台 瓦斯爐 開火",
        "在家開伙":       "可伙 廚房 抽油煙機 流理台",
        "想下廚":         "可伙 廚房 抽油煙機 瓦斯爐",
        "要下廚":         "可伙 廚房 抽油煙機 瓦斯爐",
        "喜歡下廚":       "可伙 廚房 抽油煙機 瓦斯爐 流理台",
        "喜歡自己煮":     "可伙 廚房 流理台 瓦斯爐",
        "自己煮":         "可伙 廚房 流理台 瓦斯 開火",
        "自炊":           "可伙 廚房 流理台 電磁爐 開火",
        "省伙食費":       "廚房 瓦斯 開火 流理台",
        "省餐費":         "可伙 廚房 流理台",
        "不想外食":       "可伙 廚房 流理台 電磁爐",
        "不吃外食":       "可伙 廚房 流理台 瓦斯爐",
        "可以煮東西":     "可伙 廚房",
        "要能煮飯":       "可伙 廚房 流理台 電磁爐",
        "煮飯":           "可伙 廚房 流理台",
        "開火":           "可伙 廚房 瓦斯爐 電磁爐",
        "要有廚房":       "廚房 流理台 可伙",
        "有瓦斯":         "天然瓦斯 瓦斯爐 可伙",
        "天然瓦斯":       "天然瓦斯 瓦斯爐 可伙 廚房",

        // ── 冷氣 / 溫度 ──────────────────────────────
        "怕熱":           "冷氣 變頻 吹冷氣",
        "夏天":           "冷氣",
        "西曬":           "遮陽 窗簾 隔熱",

        // ── 採光 / 通風 ──────────────────────────────
        "怕悶熱":         "陽台 採光 通風 對外窗",
        "採光好":         "落地窗 採光 對外窗",
        "網美":           "裝潢 採光 漂亮 落地窗",

        // ── 洗衣 ─────────────────────────────────────
        "獨洗獨曬":       "洗衣機 陽台 曬衣 獨洗",
        "不想去自助洗":   "洗衣機 獨立洗衣機",
        "不想共用洗衣機": "洗衣機 獨立洗衣機",

        // ── 電梯 / 行動 ──────────────────────────────
        "不想爬樓梯":     "電梯 大樓 華廈",
        "懶人":           "電梯 子母車 垃圾處理 飲水機",
        "搬東西":         "電梯",
        "膝蓋不好":       "電梯 大樓",

        // ── 停車 ─────────────────────────────────────
        "有車":           "車位 停車場",
        "開車":           "車位 停車場",
        "機車":           "機車停車位",

        // ── 寵物 ─────────────────────────────────────
        "可貓":           "可寵 養寵 寵物友善 可養貓",
        "可狗":           "可寵 養寵 寵物友善 可養狗",
        "有毛孩":         "可寵 寵物",
        "養貓":           "可養貓 寵物友善 可寵",
        "養狗":           "可養狗 寵物友善 可寵",

        // ── 電費 / 計費 ──────────────────────────────
        "台水電":         "台電 台水 帳單 自繳",
        "省電費":         "變頻 台電",
        "台電":           "台電 台水 標準電費",
        "獨立電表":       "獨立電錶 台電",

        // ── 網路 / 讀書 ──────────────────────────────
        "打報告":         "寬頻 網路 書桌",
        "上網":           "寬頻 網路",
        "念書":           "書桌 書桌椅 安靜",
        "讀書":           "書桌 書桌椅 安靜",

        // ── 垃圾 / 便利 ──────────────────────────────
        "不想追垃圾車":   "子母車 垃圾處理 垃圾代收",
        "外送族":         "管理員 飲水機 子母車",
        "不想出門":       "管理員 飲水機 子母車",

        // ── 安靜 / 隔音 ──────────────────────────────
        "怕吵":           "隔音 氣密窗 禁菸 靜巷",
        "安靜":           "靜巷 隔音",
        "夜貓子":         "無門禁 24小時 自由進出",
        "作息晚":         "無門禁 24小時 自由進出",
        "晚歸":           "門禁 管理員 安全 刷卡",

        // ── 女生安全 ──────────────────────────────────
        "女生獨居":       "管理員 門禁 監視器 女性友善 安全",
        "女生住":         "管理員 門禁 監視器 安全",
        "獨居女":         "管理員 門禁 監視器 女性友善",
        "女生安全":       "管理員 門禁 監視器 安全",
        "怕危險":         "管理員 門禁 監視器 安全",
        "治安":           "管理員 門禁 監視器 靜巷 安全",

        // ── 拎包入住 / 家具家電 ───────────────────────
        "拎包入住":       "全配 全家具 全家電 冰箱 洗衣機 床",
        "不想買家具":     "全配 全家具 家具齊全",
        "什麼都有":       "全配 全家具 全家電 冰箱 洗衣機",
        "家電齊全":       "冰箱 洗衣機 冷氣 全家電",
        "要有冰箱":       "冰箱 全配",
        "要有書桌":       "書桌 書桌椅",
        "要有床":         "床架 床墊 全配",
        "空屋":           "空屋 自備家具",

        // ── 衛浴獨立 ──────────────────────────────────
        "不想共用廁所":   "獨衛 獨立衛浴 套房",
        "不想共廁":       "獨衛 獨立衛浴 套房",
        "個人衛浴":       "獨衛 獨立衛浴",
        "獨立衛浴":       "獨衛 套房",
        "想泡澡":         "浴缸 獨衛",
        "要有熱水":       "熱水器 天然瓦斯熱水器 電熱水器",

        // ── 租期彈性 ──────────────────────────────────
        "短租":           "短期 彈性租期 月租 不限租期",
        "只租幾個月":     "短租 彈性租期 不限租期",
        "不確定租多久":   "彈性租期 短租 月租",
        "剛畢業":         "短租 彈性 經濟實惠",
        "工作不穩定":     "彈性租期 短租",

        // ── 合租 / 室友 ───────────────────────────────
        "找室友":         "雅房 分租 室友 合租",
        "想合租":         "雅房 分租 室友 合租",
        "不想一個人住":   "雅房 分租 室友",
        "一個人住":       "獨立套房 獨衛 獨廁 套房",
        "不想跟人共用":   "獨立套房 獨衛 套房",

        // ── 交通 ──────────────────────────────────────
        "騎車上班":       "機車停車位 停車",
        "通勤":           "近公車 近捷運 交通便利",
        "沒有車":         "近公車 生活機能 便利商店 交通便利",
        "不開車":         "近公車 近捷運 生活機能",
        "上班方便":       "交通便利 近公車 近捷運",

        // ── 採光朝向 ──────────────────────────────────
        "不要西曬":       "非西向 東向 北向 採光",
        "要有陽台":       "陽台 曬衣 採光 通風",
        "不要頂樓":       "非頂樓 非頂加",
        "頂樓加蓋":       "頂加",

        // ── 在家工作 ──────────────────────────────────
        "在家工作":       "網路 寬頻 書桌 安靜",
        "WFH":            "網路 寬頻 書桌 安靜",
        "遠距工作":       "網路 寬頻 書桌 安靜",
        "居家辦公":       "網路 寬頻 書桌 安靜",

        // ── 預算暗示 ──────────────────────────────────
        "學生":           "學生套房 經濟實惠 低價",
        "剛出社會":       "經濟實惠 低價 套房",
        "薪水不多":       "經濟實惠 低租金 實惠",
        "不要太貴":       "實惠 低租金 經濟",
        "便宜":           "低租金 經濟實惠",

        // ── 其他 ─────────────────────────────────────
        "首租":           "全新",
        "首選":           "全新",
        "高品質":         "管理員 電梯 漂亮 全新",
        "健身":           "健身房 交誼廳",
        "念書":           "書桌 書桌椅 安靜 寬頻",
        "讀書":           "書桌 書桌椅 安靜 寬頻",
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
        await init(data.origin, data.noCache ?? false);
    } else if (type === 'score') {
        const { query, propertyText, id } = data;
        
        // --- [NEW] Semantic Query Expansion ---
        const expandedQuery = semanticExpandQuery(query);
        
        const score = await scorePair(expandedQuery, propertyText);
        postMessage({ type: 'scoreResult', score, id, expandedQuery });
    }
};
