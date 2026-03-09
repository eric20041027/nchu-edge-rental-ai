/**
 * inference.js — Sentence-Pair Classification Recommendation Engine
 *
 * Uses the fine-tuned ALBERT model for sentence-pair classification:
 * Input:  [CLS] user_query [SEP] property_description [SEP]
 * Output: logits → softmax → match probability
 *
 * Optimized with batch inference to minimize browser latency.
 */
import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

let tokenizer = null;
let session = null;
let propertyData = [];

const MAX_LENGTH = 128;
const BATCH_SIZE = 10; // Process 10 properties per ONNX call

// ============================================================
// Data Loading
// ============================================================

export async function initData() {
    const response = await fetch('/property_data.json');
    propertyData = await response.json();
    console.log(`Loaded ${propertyData.length} property descriptions`);
}

// ============================================================
// Model Loading
// ============================================================

export async function initNLP(onProgress) {
    if (!tokenizer || !session) {
        env.allowRemoteModels = false;
        env.allowLocalModels = true;
        env.useBrowserCache = true;
        env.localModelPath = window.location.origin + '/';

        try {
            if (onProgress) onProgress({ status: 'progress', file: 'tokenizer.json', loaded: 10, total: 100 });
            tokenizer = await AutoTokenizer.from_pretrained('custom_onnx_model_dir');
            if (onProgress) onProgress({ status: 'progress', file: 'tokenizer.json', loaded: 50, total: 100 });

            if (onProgress) onProgress({ status: 'progress', file: 'model.onnx', loaded: 50, total: 100 });
            session = await window.ort.InferenceSession.create(
                window.location.origin + '/custom_onnx_model_dir/model.onnx',
                {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all',
                    externalData: [{
                        path: 'my_custom_model.onnx.data',
                        data: window.location.origin + '/custom_onnx_model_dir/model.onnx.data'
                    }]
                }
            );
            if (onProgress) onProgress({ status: 'ready', file: 'model.onnx', loaded: 100, total: 100 });
            console.log('ONNX Sentence-Pair model loaded successfully');
        } catch (err) {
            console.error('Model loading error:', err);
            throw err;
        }
    }
}

// ============================================================
// Batch Inference: Score multiple query-property pairs at once
// ============================================================

async function scoreBatch(query, propertyTexts) {
    const batchSize = propertyTexts.length;
    if (batchSize === 0) return [];

    // Tokenize all pairs
    const allInputIds = [];
    const allAttentionMask = [];
    const allTokenTypeIds = [];

    for (const propText of propertyTexts) {
        const encoded = await tokenizer(query, {
            text_pair: propText,
            padding: 'max_length',
            truncation: true,
            max_length: MAX_LENGTH,
            return_tensors: 'np',
        });
        allInputIds.push(...encoded.input_ids.data);
        allAttentionMask.push(...encoded.attention_mask.data);
        allTokenTypeIds.push(...encoded.token_type_ids.data);
    }

    // Create batch tensors
    const inputIds = new ort.Tensor('int64',
        BigInt64Array.from(allInputIds.map(v => BigInt(v))),
        [batchSize, MAX_LENGTH]
    );
    const attentionMask = new ort.Tensor('int64',
        BigInt64Array.from(allAttentionMask.map(v => BigInt(v))),
        [batchSize, MAX_LENGTH]
    );
    const tokenTypeIds = new ort.Tensor('int64',
        BigInt64Array.from(allTokenTypeIds.map(v => BigInt(v))),
        [batchSize, MAX_LENGTH]
    );

    const feeds = {
        input_ids: inputIds,
        attention_mask: attentionMask,
        token_type_ids: tokenTypeIds,
    };

    const results = await session.run(feeds);
    const logits = results.logits.data; // [batch_size * 2] flat array

    // Extract match probabilities via softmax
    const probs = [];
    for (let i = 0; i < batchSize; i++) {
        const l0 = logits[i * 2];     // not-match logit
        const l1 = logits[i * 2 + 1]; // match logit
        const maxL = Math.max(l0, l1);
        const exp0 = Math.exp(l0 - maxL);
        const exp1 = Math.exp(l1 - maxL);
        probs.push(exp1 / (exp0 + exp1));
    }

    return probs;
}

// ============================================================
// Budget Parsing (for hard exclusion only)
// ============================================================

function parseBudgetFromText(text) {
    let budget = null;
    let limit = null;

    if (text.includes('以上')) limit = 'above';
    else if (text.includes('以下') || text.includes('以內') || text.includes('內')) limit = 'below';

    let rt = text.replace(/一/g, '1').replace(/二/g, '2').replace(/兩/g, '2').replace(/三/g, '3')
        .replace(/四/g, '4').replace(/五/g, '5').replace(/六/g, '6').replace(/七/g, '7')
        .replace(/八/g, '8').replace(/九/g, '9');

    if (rt.includes('萬')) {
        let m = rt.match(/(\d+)萬(\d*)/);
        if (m) budget = parseInt(m[1]) * 10000 + (m[2] ? parseInt(m[2]) * 1000 : 0);
    }
    if (!budget) {
        rt = rt.replace(/千/g, '000').replace(/[kK]/g, '000');
        let m2 = rt.match(/(\d{4,})/);
        if (m2) budget = parseInt(m2[1]);
    }
    if (!budget) {
        let m3 = rt.match(/(\d+)/);
        if (m3) {
            let val = parseInt(m3[1]);
            if (val < 100) budget = val * 1000;
            else if (val >= 1000) budget = val;
        }
    }

    return { budget, limit };
}

// ============================================================
// Main Recommendation Function (Batch Optimized)
// ============================================================

export async function recommend(text, top_k = 5) {
    console.log("User Query:", text);
    const startTime = performance.now();

    const { budget: userBudget, limit: budgetLimit } = parseBudgetFromText(text);
    console.log("Parsed Budget:", userBudget, "Limit:", budgetLimit);

    // Step 1: Apply hard exclusions first (no model needed)
    const candidates = [];
    for (let i = 0; i < propertyData.length; i++) {
        const prop = propertyData[i];
        if (userBudget && budgetLimit) {
            if (budgetLimit === 'below' && prop.rent > userBudget) continue;
            if (budgetLimit === 'above' && prop.rent < userBudget) continue;
        }
        candidates.push({ index: i, prop });
    }
    console.log(`After hard exclusion: ${candidates.length} / ${propertyData.length} candidates`);

    // Step 2: Batch score all remaining candidates
    const allProbs = [];
    for (let bStart = 0; bStart < candidates.length; bStart += BATCH_SIZE) {
        const batch = candidates.slice(bStart, bStart + BATCH_SIZE);
        const propTexts = batch.map(c => c.prop.text);
        const batchProbs = await scoreBatch(text, propTexts);
        allProbs.push(...batchProbs);
    }

    // Step 3: Collect results
    let results = [];
    for (let i = 0; i < candidates.length; i++) {
        const percentage = Math.round(allProbs[i] * 100);
        if (percentage <= 5) continue;

        results.push({
            property: candidates[i].prop,
            score: percentage,
        });
    }

    // Step 4: Sort and return
    results.sort((a, b) => b.score - a.score);

    const elapsed = (performance.now() - startTime).toFixed(0);
    console.log(`Inference complete: ${results.length} results in ${elapsed}ms`);

    return results.slice(0, top_k).map(item => ({
        id: item.property.url,
        title: `${item.property.room_type} | ${item.property.address}`,
        price_str: item.property.rent_str,
        url: item.property.url,
        imgUrl: item.property.img || null,
        score: item.score,
        match_details: `AI 模型配對分數: ${item.score}%`,
        size: item.property.size || "坪數未提供",
        floor: item.property.floor || "樓層未提供",
        furniture: item.property.furniture || "無特殊設施提供",
        distance: item.property.distance,
        address: item.property.address,
    }));
}
