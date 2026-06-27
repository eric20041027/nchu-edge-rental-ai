/**
 * worker-shared.js — 三 inference worker(ner / bi-encoder / inference)共用工具。
 *
 * 架構精簡批次3 抽出。三 worker 原本各自重複:ORT 動態 import、streaming fetch +
 * 進度回報迴圈、InferenceSession.create。此處統一;各 worker 傳自己的 onProgress
 * callback(保留各自的 postMessage type 前綴)。
 *
 * worker 以 { type: 'module' } 建立(見 inference.js),故可 import 本檔。
 */

/** 動態載入 ONNX Runtime Web,回傳 ort 命名空間。 */
export async function loadORT() {
    const ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs');
    return ortModule.default ?? ortModule;
}

/**
 * Streaming 下載模型並回報進度。
 * @param {string} url 模型 URL
 * @param {string} cacheMode fetch cache 模式('default' / 'no-store')
 * @param {(loaded:number, total:number) => void} onProgress 每 chunk 回報(total>0 才回報)
 * @returns {Promise<Uint8Array>} 完整模型 buffer
 */
export async function streamingFetch(url, cacheMode, onProgress) {
    const res = await fetch(url, { cache: cacheMode });
    if (!res.ok) throw new Error(`model fetch failed: ${res.status}`);
    const contentLength = parseInt(res.headers.get('Content-Length') || '0', 10);
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.byteLength;
        if (contentLength > 0 && onProgress) onProgress(received, contentLength);
    }
    const buffer = new Uint8Array(received);
    let pos = 0;
    for (const chunk of chunks) { buffer.set(chunk, pos); pos += chunk.byteLength; }
    return buffer;
}

/**
 * 從 model buffer 建 WASM InferenceSession。
 * @param {object} sessionOptions 可選(inference-worker 傳 {numThreads:4};ner/bi 省略)
 */
export async function createSession(ort, modelBuffer, sessionOptions = null) {
    const opts = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
    };
    if (sessionOptions) opts.sessionOptions = sessionOptions;
    return ort.InferenceSession.create(modelBuffer, opts);
}
