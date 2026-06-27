/**
 * bi-encoder-worker.js - 單句 query 向量編碼 Worker (rbt6 bi-encoder, INT8 ONNX)
 *
 * 載入量化後的 bi-encoder,把使用者單句 query 編成 768 維向量,供主執行緒
 * 與離線預算的 704 房源向量做 cosine 召回 (取代 rule-based recall scoring)。
 * 跑在 Web Worker 避免阻塞 UI。鏡像 ner-worker.js 結構。
 *
 * 模型輸入: "input_ids","attention_mask" (int64, 無 token_type_ids)
 * 模型輸出: "embedding" (1,768),已在圖內 mean-pool + L2 normalize。
 */

import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';
import { loadORT, streamingFetch, createSession } from './worker-shared.js';

const MAX_LEN = 64;

// ONNX Runtime Web 在 worker 內動態 import
let ort = null;
let tokenizer = null;
let session = null;

// ── Initialization ──────────────────────────────────────────────────────────

async function init(origin, noCache = false) {
    try {
        const cacheMode = noCache ? 'no-store' : 'default';
        postMessage({ type: 'bienc_status', message: '載入向量編碼模組...' });
        ort = await loadORT();

        // 1. Transformers.js 設定 (與 inference-worker.js 同款,僅吃本地模型)
        env.allowRemoteModels = false;
        env.allowLocalModels = true;
        env.useBrowserCache = !noCache;
        env.localModelPath = origin + '/';

        // 2. 載入 tokenizer (hfl/rbt6,與 CE 同源)
        postMessage({ type: 'bienc_status', message: '載入向量編碼分詞器...' });
        tokenizer = await AutoTokenizer.from_pretrained('models/bi_encoder_dir');

        // 3. 下載 + 載入量化模型 (streaming 進度回報)
        postMessage({ type: 'bienc_status', message: '載入向量編碼模型...' });
        const modelBuffer = await streamingFetch(
            origin + '/models/bi_encoder_dir/bi_encoder_quant.onnx', cacheMode,
            (loaded, total) => postMessage({ type: 'bienc_progress', loaded, total }),
        );

        // 下載完成但 WASM session 編譯仍需數秒 → 通知 UI 進入「初始化中」,
        // 否則進度條卡在 100% 看似當掉。
        postMessage({ type: 'bienc_status', message: '初始化向量編碼 Session...', init: true });
        session = await createSession(ort, modelBuffer);

        postMessage({ type: 'bienc_ready' });
    } catch (err) {
        postMessage({ type: 'bienc_error', message: err.message });
    }
}

// Buffer-based init: warm benchmark 把 Cache Storage 預取的 ArrayBuffer 傳進來,跳過網路。
async function initFromBuffer(origin, modelBuffer) {
    try {
        postMessage({ type: 'bienc_status', message: '載入向量編碼模組...' });
        ort = await loadORT();

        env.allowRemoteModels = false;
        env.allowLocalModels = true;
        env.useBrowserCache = true;
        env.localModelPath = origin + '/';

        postMessage({ type: 'bienc_status', message: '載入向量編碼分詞器...' });
        tokenizer = await AutoTokenizer.from_pretrained('models/bi_encoder_dir');

        postMessage({ type: 'bienc_status', message: '初始化向量編碼 Session...', init: true });
        session = await createSession(ort, modelBuffer);
        postMessage({ type: 'bienc_ready' });
    } catch (err) {
        postMessage({ type: 'bienc_error', message: err.message });
    }
}

// ── Inference ───────────────────────────────────────────────────────────────

async function encode(query) {
    if (!session || !tokenizer) return null;

    // 單句編碼:無 text_pair、不要 token_type_ids (模型圖沒有此輸入)。
    const encoded = await tokenizer(query, {
        padding: 'max_length',
        truncation: true,
        max_length: MAX_LEN,
        return_tensors: 'np',
        return_token_type_ids: false,
    });

    // 只餵模型實際需要的輸入 (input_ids + attention_mask),iterate inputNames
    // 以免送進模型沒有的 token_type_ids 而報錯。
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
    // 唯一輸出 "embedding" (1,768),已 mean-pool + L2 norm。
    const out = results.embedding ?? results[session.outputNames[0]];
    return Float32Array.from(out.data);
}

// ── Message handler ─────────────────────────────────────────────────────────

onmessage = async (e) => {
    const { type, data } = e.data;

    if (type === 'bienc_init') {
        await init(data.origin, data.noCache ?? false);
    } else if (type === 'bienc_init_buffer') {
        await initFromBuffer(data.origin, data.modelBuffer);
    } else if (type === 'encode') {
        const { query, id } = data;
        try {
            const embedding = await encode(query);
            postMessage({ type: 'encodeResult', id, embedding: embedding ? Array.from(embedding) : null });
        } catch (err) {
            // 非致命:回 null 讓主執行緒 fallback 到 rule-based recall
            postMessage({ type: 'encodeResult', id, embedding: null });
        }
    }
};
