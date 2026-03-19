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
        // 配置 ONNX Runtime WASM 路徑 (若從 CDN 載入且有報錯才需手動指定)
        // 此處移除手動指定，讓 CDN 版本的 ort.min.js 自動尋找相應版本的 .wasm
        
        env.allowRemoteModels = false;
        env.allowLocalModels = true;
        env.useBrowserCache = true;
        env.localModelPath = window.location.origin + '/';

        try {
            console.log("Starting model initialization...");
            if (onProgress) onProgress({ status: 'progress', file: 'tokenizer.json', loaded: 10, total: 100 });
            
            // 1. 加載 Tokenizer
            tokenizer = await AutoTokenizer.from_pretrained('custom_onnx_model_dir');
            console.log("Tokenizer loaded.");
            if (onProgress) onProgress({ status: 'progress', file: 'tokenizer.json', loaded: 30, total: 100 });

            // 2. 加載 ONNX 模型
            if (onProgress) onProgress({ status: 'progress', file: 'model.onnx', loaded: 40, total: 100 });
            const modelUrl = window.location.origin + '/custom_onnx_model_dir/model.onnx?v=20260318_v1';
            
            console.log("Creating InferenceSession for:", modelUrl);
            
            // 簡化 Session 創建：如果模型 >= 7MB 通常已經包含權重，
            // 除非它特別是 split format。我們先嘗試不帶 externalData 載入。
            session = await window.ort.InferenceSession.create(modelUrl, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            if (onProgress) onProgress({ status: 'ready', file: 'model.onnx', loaded: 100, total: 100 });
            console.log('ONNX Sentence-Pair model loaded successfully');
        } catch (err) {
            console.error('Model loading error details:', err);
            // 如果第一次失敗，嘗試帶上 externalData (以防萬一)
            try {
                console.log("Retrying with externalData...");
                const modelUrl = window.location.origin + '/custom_onnx_model_dir/model.onnx?v=20260318_v1';
                session = await window.ort.InferenceSession.create(modelUrl, {
                    executionProviders: ['wasm'],
                    externalData: [{
                        path: 'model.onnx.data', // 統一名稱為 model.onnx.data
                        data: window.location.origin + '/custom_onnx_model_dir/model.onnx.data?v=20260318_v1'
                    }]
                });
                if (onProgress) onProgress({ status: 'ready', file: 'model.onnx', loaded: 100, total: 100 });
                console.log('ONNX model loaded on retry with externalData');
            } catch (retryErr) {
                console.error('Model loading retry failed:', retryErr);
                throw retryErr;
            }
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
    let genderUnrestricted = false;
    let hasGenderMention = false;
    let hasBudgetMention = false;
    let hasRoomTypeMention = false;

    // Detection for "unrestricted gender"
    if (text.includes('不限女') || text.includes('不限性別') || text.includes('男生') || text.includes('男士')) {
        genderUnrestricted = true;
        hasGenderMention = true;
    } else if (text.includes('限女') || text.includes('限男')) {
        hasGenderMention = true;
    }

    if (text.includes('以上')) limit = 'above';
    else if (text.includes('以下') || text.includes('以內') || text.includes('內')) limit = 'below';

    let rt = text.replace(/一/g, '1').replace(/二/g, '2').replace(/兩/g, '2').replace(/三/g, '3')
        .replace(/四/g, '4').replace(/五/g, '5').replace(/六/g, '6').replace(/七/g, '7')
        .replace(/八/g, '8').replace(/九/g, '9');

    if (rt.includes('萬')) {
        let m = rt.match(/(\d+)萬(\d*)/);
        if (m) {
            budget = parseInt(m[1]) * 10000 + (m[2] ? parseInt(m[2]) * 1000 : 0);
            hasBudgetMention = true;
        }
    }
    if (!budget) {
        rt = rt.replace(/千/g, '000').replace(/[kK]/g, '000');
        let m2 = rt.match(/(\d{4,})/);
        if (m2) {
            budget = parseInt(m2[1]);
            hasBudgetMention = true;
        }
    }
    if (!budget) {
        let m3 = rt.match(/(\d+)/);
        if (m3) {
            let val = parseInt(m3[1]);
            if (val < 100) budget = val * 1000;
            else if (val >= 1000) budget = val;
            hasBudgetMention = true;
        }
    }

    if (text.includes('套房') || text.includes('雅房') || text.includes('工作室')) {
        hasRoomTypeMention = true;
    }

    return { budget, limit, genderUnrestricted, hasGenderMention, hasBudgetMention, hasRoomTypeMention };
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

        const { budget: userBudget, limit: budgetLimit, genderUnrestricted, hasGenderMention, hasBudgetMention, hasRoomTypeMention } = parseConstraintsFromText(text);
        
        // Step 1: Apply hard exclusions (Only for explicit constraints)
        const candidates = [];
        for (let i = 0; i < propertyData.length; i++) {
            const prop = propertyData[i];

            if (hasGenderMention && genderUnrestricted) {
                const isFemaleOnly = prop.text.includes('限女') || (prop.furniture && prop.furniture.includes('限女'));
                if (isFemaleOnly) continue;
            }

            if (hasBudgetMention && budgetLimit) {
                if (budgetLimit === 'below' && prop.rent > userBudget) continue;
                if (budgetLimit === 'above' && prop.rent < userBudget) continue;
            }
            candidates.push(prop);
        }

        // Step 2: Two-Stage Retrieval
        // 提取關鍵字並過濾掉常見動詞/介詞
        const stopWords = [
            '近', '靠近', '想找', '尋找', '住在', '一間', '想要', '預算', '大約', '希望',
            '位於', '位在', '位處', '在', '含', '有', '附', '座落於', '座落'
        ];
        
        // 額外定義地址後綴，用於提取核心地址
        const locSuffixes = ['路', '街', '大道', '區'];

        let queryKeywords = text.split(/\s+|[,，、。]/)
            .filter(k => k.length > 1 && !k.match(/^\d+$/))
            .map(k => {
                let clean = k;
                // 1. 移除開頭的 stopWords
                stopWords.forEach(sw => { if (clean.startsWith(sw)) clean = clean.substring(sw.length); });
                
                // 2. 如果是地址類關鍵字（以路、街等結尾），嘗試進一步清理開頭
                // 例如：從 "位於國光路" 提取出 "國光路"
                locSuffixes.forEach(suffix => {
                    if (clean.endsWith(suffix) && clean.length > suffix.length) {
                        // 移除常見的方位或連字
                        const locPrefixes = ['位', '於', '在', '處'];
                        locPrefixes.forEach(p => { if (clean.startsWith(p)) clean = clean.substring(p.length); });
                    }
                });
                return clean;
            })
            .filter(k => k.length > 1);

        const hasLocationMention = queryKeywords.some(kw => 
            kw.endsWith('路') || kw.endsWith('街') || kw.endsWith('大道') || 
            kw.includes('區') || kw.includes('正門') || kw.includes('側門') || kw.includes('男宿')
        );

        const preScoredCandidates = candidates.map(prop => {
            let kScore = 0;
            let matchCount = 0;
            let totalRequirements = 0;

            // 1. Location Matching
            if (hasLocationMention) {
                totalRequirements++;
                let locMatch = false;
                queryKeywords.forEach(kw => {
                    if (prop.text.includes(kw)) {
                        if (kw.endsWith('路') || kw.endsWith('街') || kw.endsWith('大道')) kScore += 15, locMatch = true;
                        if (kw.includes('區')) kScore += 5, locMatch = true;
                        if (kw.includes('正門') || kw.includes('側門')) kScore += 10, locMatch = true;
                    }
                });
                if (locMatch) matchCount++;
            }

            // 2. Room Type Matching
            if (hasRoomTypeMention) {
                totalRequirements++;
                let rtMatch = false;
                ['套房', '雅房', '工作室'].forEach(rt => {
                    if (text.includes(rt) && prop.text.includes(rt)) rtMatch = true;
                });
                if (rtMatch) matchCount++, kScore += 10;
            }

            // 3. Budget Matching
            if (hasBudgetMention) {
                totalRequirements++;
                const diff = Math.abs(prop.rent - userBudget);
                if (diff < 500) matchCount++, kScore += 5;
                else if (prop.rent <= userBudget) matchCount += 0.5, kScore += 2;
            }

            // Calculate Requirement Match Score (RMS)
            // Rule: "Not mentioned = Match"
            let rms = totalRequirements > 0 ? (matchCount / totalRequirements) : 1.0;

            return { prop, kScore, rms };
        });

        // Take top 30
        preScoredCandidates.sort((a, b) => (b.kScore + b.rms * 20) - (a.kScore + a.rms * 20));
        const topCandidates = preScoredCandidates.slice(0, 30);

        // Stage 2: AI Re-ranking
        const scoredResults = [];
        for (let i = 0; i < topCandidates.length; i++) {
            const { prop, rms } = topCandidates[i];
            try {
                const aiScore = await scorePair(text, prop.text); 
                
                // Final Score Blend: Bias towards 100 if RMS is high
                // Score = (RMS * 40) + (AI * 60)
                let finalPercentage = Math.round((rms * 40) + (aiScore * 60));
                
                // Fine-tune: If it's a perfect requirement match (RMS=1), ensure a high floor
                if (rms === 1.0 && finalPercentage < 85) finalPercentage = 85 + (aiScore * 15);

                scoredResults.push({ property: prop, score: Math.min(100, Math.round(finalPercentage)) });
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
