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

onmessage = async (e) => {
    const { type, data } = e.data;

    if (type === 'init') {
        await init(data.origin);
    } else if (type === 'score') {
        const { query, propertyText, id } = data;
        const score = await scorePair(query, propertyText);
        postMessage({ type: 'scoreResult', score, id });
    }
};
