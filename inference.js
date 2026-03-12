/**
 * inference.js — Sentence-Pair Classification Recommendation Engine
 *
 * Uses the fine-tuned ALBERT model for sentence-pair classification:
 * Input:  [CLS] user_query [SEP] property_description [SEP]
 * Output: logits → softmax → match probability
 *
 * Sequential inference to avoid ONNX "Session already started" error.
 */
import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

let tokenizer = null;
let session = null;
let propertyData = [];

const MAX_LENGTH = 64;

// ============================================================
// Data Loading
// ============================================================

export async function initData() {
    const response = await fetch('/property_data.json?v=20260310');
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
            const modelUrl = window.location.origin + '/custom_onnx_model_dir/model.onnx?v=20260311_v1';
            session = await window.ort.InferenceSession.create(
                modelUrl,
                {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all',
                    externalData: [{
                        path: 'my_custom_model.onnx.data',
                        data: window.location.origin + '/custom_onnx_model_dir/my_custom_model.onnx.data?v=20260311_v1'
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
// Single Pair Inference
// ============================================================

async function scorePair(query, propertyText) {
    const encoded = await tokenizer(query, {
        text_pair: propertyText,
        padding: 'max_length',
        truncation: true,
        max_length: MAX_LENGTH,
        return_tensors: 'np',
        return_token_type_ids: true, // 重要：必須有這個，模型才能區分 query 與 property
    });

    const inputs = {};
    const modelInputs = session.inputNames; // ["input_ids", "attention_mask", "token_type_ids"]

    for (const key of modelInputs) {
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

// ============================================================
// Constraint Parsing (for hard exclusion only)
// ============================================================

function parseConstraintsFromText(text) {
    let budget = null;
    let limit = null;
    let genderUnrestricted = false; // true if user specifically says "不限女"

    // Detection for "unrestricted gender"
    if (text.includes('不限女') || text.includes('不限性別') || text.includes('男生') || text.includes('男士')) {
        genderUnrestricted = true;
    }

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

    return { budget, limit, genderUnrestricted };
}

let inferenceLock = false;

// ============================================================
// Main Recommendation Function
// ============================================================

export async function recommend(text, top_k = 5) {
    if (inferenceLock) {
        console.warn("New recommendation ignored: Previous inference still in progress.");
        return null; // app.js handles null/empty
    }
    inferenceLock = true;

    try {
        console.log("User Query:", text);
        const startTime = performance.now();

        const { budget: userBudget, limit: budgetLimit, genderUnrestricted } = parseConstraintsFromText(text);
        console.log("Parsed constraints:", { userBudget, budgetLimit, genderUnrestricted });

        // Step 1: Apply hard exclusions first
        const candidates = [];
        for (let i = 0; i < propertyData.length; i++) {
            const prop = propertyData[i];

            // Gender exclusion: If user says "男生" or "不限女", skip female-only rooms
            if (genderUnrestricted) {
                const isFemaleOnly = prop.text.includes('限女') || (prop.furniture && prop.furniture.includes('限女'));
                if (isFemaleOnly) continue;
            }

            // Budget exclusion (only if limit word like "以下" is present)
            if (userBudget && budgetLimit) {
                if (budgetLimit === 'below' && prop.rent > userBudget) continue;
                if (budgetLimit === 'above' && prop.rent < userBudget) continue;
            }
            candidates.push(prop);
        }
        console.log(`After hard exclusion: ${candidates.length} / ${propertyData.length} candidates`);

        // Step 2: Two-Stage Retrieval
        // Stage 2.1: Simple Keyword-Overlap + Price Proximity Retrieval
        console.log(`Stage 1: Keyword-overlap + Price proximity retrieval...`);
        const queryKeywords = text.split(/\s+|[,，、。]/).filter(k => k.length > 1 && !k.match(/^\d+$/)); // 數字不當關鍵字

        const preScoredCandidates = candidates.map(prop => {
            let kScore = 0;
            // 關鍵字匹配
            queryKeywords.forEach(kw => {
                if (prop.text.includes(kw)) kScore += 2; // AI 模型更注重關鍵字
            });

            // 價格接近程度匹配 (如果您輸入的是 6000 而沒有 "以下")
            if (userBudget && !budgetLimit) {
                const diff = Math.abs(prop.rent - userBudget);
                // 擴散公式：越接近分數越高，差 500 元以內得分較高
                const priceSimilarity = 10.0 / (1.0 + diff / 500.0);
                kScore += priceSimilarity;
            }

            return { prop, kScore };
        });

        // Sort by keyword score and take top 30
        preScoredCandidates.sort((a, b) => b.kScore - a.kScore);
        const topCandidates = preScoredCandidates.slice(0, 30).map(c => c.prop);
        console.log(`Stage 1 complete: Selected top ${topCandidates.length} candidates for AI re-ranking.`);

        // Stage 2.2: AI Re-ranking (Cross-Encoder)
        const scoredResults = [];
        console.log(`Stage 2: Starting AI re-ranking for ${topCandidates.length} candidates...`);

        for (let i = 0; i < topCandidates.length; i++) {
            const prop = topCandidates[i];
            try {
                const aiScore = await scorePair(text, prop.text); // 0.0 ~ 1.0
                let finalScore = aiScore;

                // 如果是純數字預算（無以上/以下），結合價格接近程度
                if (userBudget && !budgetLimit) {
                    const diff = Math.abs(prop.rent - userBudget);
                    const priceSimilarity = 1.0 / (1.0 + diff / 500.0); // 0.0 ~ 1.0

                    // 混和評分：AI (70%) + 價格 (30%)
                    finalScore = (aiScore * 0.7) + (priceSimilarity * 0.3);
                }

                const percentage = Math.round(finalScore * 100);
                if (percentage > 5) {
                    scoredResults.push({ property: prop, score: percentage });
                }
            } catch (err) {
                console.error(`AI scoring error for property ${i}:`, err);
            }
        }

        // Step 3: Sort and return
        scoredResults.sort((a, b) => b.score - a.score);

        // Debug: Log top result detail
        if (scoredResults.length > 0) {
            console.log("Top Match Details:", {
                query: text,
                property: scoredResults[0].property.text,
                score: scoredResults[0].score + "%"
            });
        }

        const elapsed = (performance.now() - startTime).toFixed(0);
        console.log(`Inference complete: ${scoredResults.length} results in ${elapsed}ms`);

        return scoredResults.slice(0, top_k).map(item => ({
            id: item.property.url,
            title: `${item.property.room_type} | ${item.property.address}`,
            price_str: item.property.rent_str,
            url: item.property.url,
            imgUrl: item.property.img || null,
            score: item.score,
            match_details: "",
            size: item.property.size || "坪數未提供",
            floor: item.property.floor || "樓層未提供",
            furniture: item.property.furniture || "無特殊設施提供",
            distance: item.property.distance,
            address: item.property.address,
        }));
    } finally {
        inferenceLock = false;
    }
}
