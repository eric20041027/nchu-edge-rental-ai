/**
 * inference-worker.js - Off-main-thread ONNX Inference Worker
 * 
 * Handles loading of the 84MB model and execution of semantic scoring.
 */

// Import ONNX Runtime and Transformers (Tokenizer) inside Worker
import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// Import ONNX Runtime Web via script tag in worker is tricky, 
// using CDN import instead.
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

let tokenizer = null;
let session = null;
const MAX_LENGTH = 64;

// Configure Xenova/Transformers for local models
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.useBrowserCache = true;

/**
 * Initialization function
 */
async function init(localOrigin) {
    try {
        env.localModelPath = localOrigin + '/';
        
        postMessage({ type: 'status', message: 'Loading Tokenizer...' });
        tokenizer = await AutoTokenizer.from_pretrained('models/custom_onnx_model_dir');
        
        postMessage({ type: 'status', message: 'Loading 84MB AI Model...' });
        const modelUrl = localOrigin + '/models/custom_onnx_model_dir/my_custom_model_quant.onnx';
        
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            sessionOptions: { numThreads: 4 }
        });

        postMessage({ type: 'ready' });
    } catch (err) {
        postMessage({ type: 'error', message: err.message });
    }
}

/**
 * Single-pair scoring
 */
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
 * Message Handler
 */
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
