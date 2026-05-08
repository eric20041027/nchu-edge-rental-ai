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

        // 2. Load Tokenizer with progress
        tokenizer = await AutoTokenizer.from_pretrained('models/custom_onnx_model_dir', {
            progress_callback: (p) => {
                if (p.status === 'progress') {
                    postMessage({ type: 'status', message: '正在加載分詞器...', loaded: p.loaded, total: p.total });
                }
            }
        });

        // 3. Manually fetch the 84MB model to track progress accurately
        const modelUrl = localOrigin + '/models/custom_onnx_model_dir/my_custom_model_quant.onnx';
        postMessage({ type: 'status', message: '準備下載 AI 模型...', loaded: 0, total: 88000000 });

        const response = await fetch(modelUrl);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const reader = response.body.getReader();
        const contentLength = +response.headers.get('Content-Length');
        
        let receivedLength = 0;
        let chunks = [];
        while(true) {
            const {done, value} = await reader.read();
            if (done) break;
            chunks.push(value);
            receivedLength += value.length;
            postMessage({ 
                type: 'status', 
                message: '正在下載 AI 模型...', 
                loaded: receivedLength, 
                total: contentLength || 88000000 
            });
        }

        // 4. Create Session from the downloaded buffer
        const modelBuffer = new Uint8Array(receivedLength);
        let position = 0;
        for(let chunk of chunks) {
            modelBuffer.set(chunk, position);
            position += chunk.length;
        }

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
