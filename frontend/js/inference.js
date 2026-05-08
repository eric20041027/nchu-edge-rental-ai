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

let worker = null;
let propertyData = [];
let pendingInference = new Map();
let inferenceIdCounter = 0;

// --- Property Data Synchronization ---
export async function initData() {
    const response = await fetch('assets/property_data.json?v=20260310');
    propertyData = await response.json();
    console.log(`Loaded ${propertyData.length} property descriptions`);
}

// --- NLP Engine Initialization via Web Worker ---
export async function initNLP(onProgress) {
    if (!worker) {
        return new Promise((resolve, reject) => {
            console.log("Initializing Inference Web Worker...");
            worker = new Worker('js/inference-worker.js', { type: 'module' });

            worker.onmessage = (e) => {
                const { type, message, score, id, error, loaded, total } = e.data;
                if (type === 'status' && onProgress) {
                    onProgress({ status: 'progress', message, loaded, total });
                } else if (type === 'ready') {
                    console.log('Inference Worker Ready');
                    if (onProgress) onProgress({ status: 'ready' });
                    resolve();
                } else if (type === 'scoreResult') {
                    const callback = pendingInference.get(id);
                    if (callback) {
                        callback(score);
                        pendingInference.delete(id);
                    }
                } else if (type === 'error') {
                    console.error('Worker error:', message);
                    if (reject) reject(new Error(message));
                }
            };

            worker.postMessage({ 
                type: 'init', 
                data: { origin: window.location.origin } 
            });
        });
    }
}

// --- Proxy to Worker for Scoring ---
async function scorePair(query, propertyText) {
    return new Promise((resolve) => {
        const id = inferenceIdCounter++;
        pendingInference.set(id, resolve);
        worker.postMessage({
            type: 'score',
            data: { query, propertyText, id }
        });
    });
}

// --- Constraint Parsing & Normalization ---
function parseConstraintsFromText(text) {
    let budget = null, limit = null;
    let minBudget = null, maxBudget = null;
    let genderUnrestricted = false, hasGenderMention = false, hasBudgetMention = false, hasRoomTypeMention = false;
    let wantsUtilityBilling = false, maxElectricityPrice = null;
    let requireBalcony = false, requireWindow = false, requireParking = false, requireWaste = false;
    let requireSubsidy = false, isSocialHousing = false;
    let excludeRooftop = false, excludeWooden = false, excludeHaunted = false;

    if (text.includes('不限女') || text.includes('不限性別') || text.includes('男生') || text.includes('男士')) {
        genderUnrestricted = true;
        hasGenderMention = true;
    } else if (text.includes('限女') || text.includes('限男')) {
        hasGenderMention = true;
    }

    // Exclusions (Hard Filtering)
    if (text.match(/(謝絕|不要|拒絕|禁|❌|不接受)[^。！？\n]*(頂加|加蓋|頂樓)/)) excludeRooftop = true;
    if (text.match(/(謝絕|不要|拒絕|禁|❌|不接受)[^。！？\n]*木板/)) excludeWooden = true;
    if (text.match(/(謝絕|不要|拒絕|禁|❌|不接受)[^。！？\n]*凶宅/)) excludeHaunted = true;

    // Explicit Requirements
    if (text.match(/(要有|必須|希望|想找)[^。！？\n]*陽台/)) requireBalcony = true;
    else if (text.includes('陽台')) requireBalcony = true; // Soft requirement
    
    if (text.match(/(要有|必須|希望|想找)[^。！？\n]*窗/)) requireWindow = true;
    else if (text.includes('窗')) requireWindow = true;

    if (text.includes('車位') || text.includes('停車')) requireParking = true;
    if (text.includes('子母車') || text.includes('垃圾')) requireWaste = true;
    
    if (text.includes('補助') || text.includes('補貼') || text.includes('報稅') || text.includes('入籍')) requireSubsidy = true;
    if (text.includes('社宅') || text.includes('社會住宅')) isSocialHousing = true;

    if (text.includes('以上')) limit = 'above';
    else if (text.includes('以下') || text.includes('以內') || text.includes('內')) limit = 'below';

    // Parse Utility Billing (台水台電)
    if (text.includes('台水') || text.includes('台電') || text.includes('獨立電錶') || text.includes('獨立電表')) {
        wantsUtilityBilling = true;
    }
    const elecMatch = text.match(/度\s*(\d+(?:\.\d+)?)\s*[元塊]/);
    if (elecMatch) {
        maxElectricityPrice = parseFloat(elecMatch[1]);
    }

    let rt = text.replace(/一/g, '1').replace(/二/g, '2').replace(/兩/g, '2').replace(/三/g, '3')
        .replace(/四/g, '4').replace(/五/g, '5').replace(/六/g, '6').replace(/七/g, '7')
        .replace(/八/g, '8').replace(/九/g, '9').replace(/十/g, '10').replace(/半/g, '30');

    let maxWalkMins = null;
    let walkMatch = rt.match(/(?:走路|步行)[^\d]*(\d+)[^\d]*(?:分鐘|分)/);
    if (walkMatch) maxWalkMins = parseInt(walkMatch[1]);

    let maxScooterMins = null;
    let scooterMatch = rt.match(/(?:機車|騎車)[^\d]*(\d+)[^\d]*(?:分鐘|分)/);
    if (scooterMatch) maxScooterMins = parseInt(scooterMatch[1]);


    // Handle Range Budget (e.g., 6000-12000, 6000~12000, 6千-1萬2)
    let rt_range = rt.replace(/(\d+(?:\.\d+)?)萬(\d*)/g, (m, p1, p2) => {
        let val = parseFloat(p1) * 10000;
        if (p2) val += parseInt(p2) * 1000;
        return val;
    }).replace(/(\d+)千/g, (m, p1) => parseInt(p1) * 1000);
    
    let rangeMatch = rt_range.match(/(\d{3,})\s*[-~～至到]\s*(\d{3,})/);
    if (rangeMatch) {
        minBudget = parseInt(rangeMatch[1]);
        maxBudget = parseInt(rangeMatch[2]);
        hasBudgetMention = true;
    }

    if (!hasBudgetMention) {
        if (rt.includes('萬')) {
            let m = rt.match(/(\d+(?:\.\d+)?)萬(\d*)/);
            if (m) {
                budget = parseFloat(m[1]) * 10000 + (m[2] ? parseInt(m[2]) * 1000 : 0);
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
    }

    if (text.includes('套房') || text.includes('雅房') || text.includes('工作室')) {
        hasRoomTypeMention = true;
    }

    return { 
        budget, minBudget, maxBudget, limit, genderUnrestricted, hasGenderMention, hasBudgetMention, hasRoomTypeMention, 
        wantsUtilityBilling, maxElectricityPrice, requireBalcony, requireWindow, requireParking, requireWaste, 
        requireSubsidy, isSocialHousing,
        excludeRooftop, excludeWooden, excludeHaunted, maxWalkMins, maxScooterMins,
        wantsPet: (text.includes('養貓') || text.includes('養狗') || text.includes('寵物')),
        originalText: text // Added to fix property access in checkConflicts
    };
}

// --- Explainability: Match Reasons & Conflict Detection ---
function explainMatch(query, prop, constraints) {
    const reasons = [];
    const pText = prop.text + (prop.furniture || "") + (prop.notes ? prop.notes.join(" ") : "");
    const q = query.toLowerCase();

    // 1. Financial & Budget Matches (High value)
    if (prop.text.includes('台水台電') || prop.notes?.includes('獨立電錶')) {
        if (q.includes('省') || q.includes('錢') || q.includes('便宜') || q.includes('台電')) {
            reasons.push('台電計費');
        }
    }
    if (pText.includes('租補') || pText.includes('補助')) reasons.push('可申請租補');

    // 2. Convenience & Amenities (Quality of life)
    if (pText.includes('子母車') || pText.includes('垃圾處理')) reasons.push('免追垃圾車');
    if (pText.includes('飲水機')) reasons.push('有飲水機');
    if (pText.includes('電梯')) reasons.push('有電梯');
    if (pText.includes('獨立洗衣機') || (pText.includes('洗衣機') && !pText.includes('共用'))) {
        reasons.push('個人洗衣機');
    }
    if (pText.includes('陽台') || prop.has_balcony) reasons.push('有陽台');
    if (pText.includes('採光') || pText.includes('大窗')) reasons.push('採光佳');
    if (pText.includes('機車') || pText.includes('車位')) reasons.push('有機車位');

    // 3. Location & Keyword matches
    const keywords = extractKeywords(query);
    keywords.forEach(kw => {
        if (pText.includes(kw) && (kw.endsWith('路') || kw.endsWith('街') || kw.includes('區') || kw.includes('興大'))) {
            reasons.push(kw);
        }
    });

    // Limit to top 3 most relevant reasons to keep UI clean
    return [...new Set(reasons)].slice(0, 3);
}

function checkConflicts(prop, constraints) {
    const { wantsPet } = constraints;
    const pText = prop.text + (prop.notes ? prop.notes.join(" ") : "");
    
    // 1. Pet Conflict
    if (wantsPet && (pText.includes('禁養') || pText.includes('不可養'))) {
        return "此房源禁養寵物";
    }

    // 2. Gender Conflict (Simplified detection)
    if (constraints.hasGenderMention && constraints.originalText) {
        if (constraints.genderUnrestricted === false) {
             if (pText.includes('限女性') && constraints.originalText.includes('男')) return "此房源僅限女性";
             if (pText.includes('限男性') && constraints.originalText.includes('女')) return "此房源僅限男性";
        }
    }

    // 3. Smoking
    if (constraints.originalText?.includes('抽菸') && (pText.includes('禁菸') || pText.includes('禁止吸菸'))) {
        return "此房源禁止吸菸";
    }
    
    return null;
}

// --- Hard Exclusion Filtering ---
function filterHardExclusions(properties, constraints) {
    const { 
        budget, minBudget, maxBudget, limit, genderUnrestricted, hasGenderMention, hasBudgetMention,
        excludeRooftop, excludeWooden, maxElectricityPrice, wantsUtilityBilling,
        maxWalkMins, maxScooterMins,
        requireSubsidy, isSocialHousing, requireBalcony, requireWindow, requireParking, requireWaste
    } = constraints;
    const candidates = [];
    
    for (const prop of properties) {
        if (excludeRooftop && (prop.is_rooftop || prop.text.includes('頂加'))) continue;
        if (excludeWooden && prop.is_wooden_partition) continue;
        
        // Subsidy Exclusion
        if (requireSubsidy && (prop.text.includes('不可補助') || prop.text.includes('不可報稅') || prop.text.includes('不可入籍'))) continue;
        if (isSocialHousing && !prop.text.includes('社會住宅') && !prop.text.includes('社宅')) continue;

        // Hard Amenity Exclusions
        if (requireBalcony && !prop.text.includes('陽台') && !(prop.furniture && prop.furniture.includes('陽台'))) continue;
        if (requireWindow && !prop.text.includes('對外窗') && !prop.text.includes('採光')) continue;
        if (requireParking && !prop.text.includes('車位') && !prop.text.includes('停車')) continue;
        if (requireWaste && !prop.text.includes('子母車') && !prop.text.includes('代收垃圾') && !prop.text.includes('垃圾處理')) continue;

        // Commute time filtering
        let dist = parseFloat(prop.distance);
        if (!isNaN(dist) && dist > 0) {
            if (maxWalkMins !== null) {
                let walkMins = Math.round(dist / 0.075);
                if (walkMins > maxWalkMins + 3) continue; // +3 mins grace period
            }
            if (maxScooterMins !== null) {
                let scooterMins = Math.max(1, Math.round(dist / 0.417));
                if (scooterMins > maxScooterMins + 2) continue; // +2 mins grace period
            }
        }

        
        if (maxElectricityPrice) {
            // "5元/度"
            const match = prop.electricity_billing.match(/(\d+(?:\.\d+)?)/);
            if (match && parseFloat(match[1]) > maxElectricityPrice) continue;
        }

        // If user specifically asks for Taishui Taipower and NOT maxElectricityPrice, 
        // we can filter out properties that are explicitly > 5 NTD, though we handle this softly in scoring too.
        if (wantsUtilityBilling && prop.electricity_billing && prop.electricity_billing.includes("度")) {
            const match = prop.electricity_billing.match(/(\d+(?:\.\d+)?)/);
            if (match && parseFloat(match[1]) >= 5) {
                // If they explicitly want Taishui Taipower, properties charging >= 5/kwh are generally rejected
                continue; 
            }
        }

        if (hasGenderMention && genderUnrestricted) {
            const isFemaleOnly = prop.text.includes('限女') || (prop.furniture && prop.furniture.includes('限女'));
            if (isFemaleOnly) continue;
        }
        if (hasBudgetMention) {
            if (maxBudget !== null && prop.rent > maxBudget) continue;
            if (limit && budget !== null) {
                if (limit === 'below' && prop.rent > budget) continue;
                if (limit === 'above' && prop.rent < budget) continue;
            }
        }
        candidates.push(prop);
    }
    return candidates;
}


// --- Keyword Extraction ---
function extractKeywords(text) {
    const stopWords = ['近', '靠近', '想找', '尋找', '住在', '一間', '想要', '預算', '大約', '希望', '位於', '位在', '位處', '在', '含', '有', '附', '座落於', '座落'];
    const locSuffixes = ['路', '街', '大道', '區'];

    return text.split(/\s+|[,，、。]/)
        .filter(k => k.length > 1 && !k.match(/^\d+$/))
        .map(k => {
            let clean = k;
            stopWords.forEach(sw => { if (clean.startsWith(sw)) clean = clean.substring(sw.length); });
            locSuffixes.forEach(suffix => {
                if (clean.endsWith(suffix) && clean.length > suffix.length) {
                    const locPrefixes = ['位', '於', '在', '處'];
                    locPrefixes.forEach(p => { if (clean.startsWith(p)) clean = clean.substring(p.length); });
                }
            });
            return clean;
        })
        .filter(k => k.length > 1);
}

// --- Rule-based Pre-Scoring ---
function calculateRuleBasedScore(candidates, queryKeywords, text, constraints) {
    const { 
        budget: userBudget, minBudget, maxBudget, hasBudgetMention, hasRoomTypeMention, wantsUtilityBilling,
        requireBalcony, requireWindow, requireParking, requireWaste, maxWalkMins, maxScooterMins
    } = constraints;


    const hasLocationMention = queryKeywords.some(kw =>
        kw.endsWith('路') || kw.endsWith('街') || kw.endsWith('大道') ||
        kw.includes('區') || kw.includes('正門') || kw.includes('側門') || kw.includes('男宿')
    );

    const preScored = candidates.map(prop => {
        let kScore = 0, matchCount = 0, totalRequirements = 0;

        // Commute Time Scoring
        if (maxWalkMins !== null) {
            totalRequirements++;
            const propWalk = prop.walk_mins || Math.ceil(prop.distance / 0.08);
            if (propWalk <= maxWalkMins) {
                matchCount++;
                kScore += 20; // Walking time is usually a very strong intent
            } else if (propWalk <= maxWalkMins + 3) {
                matchCount += 0.5;
                kScore += 5;
            }
        }

        if (maxScooterMins !== null) {
            totalRequirements++;
            const propScooter = prop.scooter_mins || Math.max(1, Math.ceil(prop.distance / 0.5));
            if (propScooter <= maxScooterMins) {
                matchCount++;
                kScore += 15;
            } else if (propScooter <= maxScooterMins + 2) {
                matchCount += 0.5;
                kScore += 5;
            }
        }


        if (requireBalcony) {
            totalRequirements++;
            if (prop.has_balcony) { matchCount++; kScore += 15; }
        }
        if (requireWindow) {
            totalRequirements++;
            if (prop.has_window) { matchCount++; kScore += 15; }
        }
        if (requireParking) {
            totalRequirements++;
            if (prop.has_parking) { matchCount++; kScore += 10; }
        }
        if (requireWaste) {
            totalRequirements++;
            if (prop.has_waste_disposal) { matchCount++; kScore += 10; }
        }

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

        if (hasRoomTypeMention) {
            totalRequirements++;
            let rtMatch = false;
            ['套房', '雅房', '工作室'].forEach(rt => {
                if (text.includes(rt) && prop.text.includes(rt)) rtMatch = true;
            });
            if (rtMatch) matchCount++, kScore += 10;
        }

        if (hasBudgetMention) {
            totalRequirements += 2;
            if (minBudget !== null && maxBudget !== null) {
                if (prop.rent >= minBudget && prop.rent <= maxBudget) {
                    matchCount += 2;
                    kScore += 10;
                } else if (prop.rent < minBudget) {
                    matchCount += 1.5;
                    kScore += 5;
                } else {
                    const diff = prop.rent - maxBudget;
                    if (diff <= 1000) {
                        matchCount += 0.5;
                        kScore += 1;
                    }
                }
            } else if (userBudget !== null) {
                const diff = prop.rent - userBudget;
                if (Math.abs(diff) <= 500) {
                    matchCount += 2;
                    kScore += 10;
                } else if (prop.rent < userBudget) {
                    matchCount += 1.5;
                    kScore += 3;
                } else if (diff <= 1500) {
                    matchCount += 0.5;
                    kScore += 1;
                }
            }
        }

        if (wantsUtilityBilling) {
            totalRequirements++;
            let utilityMatch = prop.electricity_billing === "台水台電" || prop.electricity_billing === "含電費";
            if (utilityMatch) {
                matchCount++;
                kScore += 10;
            }
        }

        const rms = totalRequirements > 0 ? (matchCount / totalRequirements) : 1.0;
        return { prop, kScore, rms };
    });

    preScored.sort((a, b) => (b.kScore + b.rms * 20) - (a.kScore + a.rms * 20));
    return preScored.slice(0, 15);  // Reduced from 30→15: fewer AI calls = faster response
}

// --- Response Formatting ---
function formatResponse(scoredResults, top_k) {
    return scoredResults.slice(0, top_k).map(item => ({
        id: item.property.url,
        title: `${item.property.room_type} | ${item.property.address}`,
        price_str: item.property.rent_str,
        url: item.property.url,
        imgUrl: item.property.img || null,
        score: item.score,
        match_reasons: item.match_reasons || [],
        conflict_reason: item.conflict_reason || null,
        size: item.property.size || "坪數未提供",
        floor: item.property.floor || "樓層未提供",
        furniture: item.property.furniture || "無特殊設施提供",
        distance: item.property.distance,
        address: item.property.address,
        contact: item.property.contact || "不具名",
        phone: item.property.phone || "無資料",
    }));
}

let currentQueryId = 0;

// --- Main Recommendation Pipeline ---
// onPartialResult(results): optional callback called immediately with rule-based results
export async function recommend(text, top_k = 20, onPartialResult = null) {
    // Increment the query ID — any in-progress inference with an older ID will detect
    // the mismatch and exit early, allowing this new query to proceed immediately.
    const myQueryId = ++currentQueryId;

    const isCancelled = () => myQueryId !== currentQueryId;

    try {
        console.log("User Query:", text);
        const startTime = performance.now();

        // 1. Data Parsing & Filtering
        const constraints = parseConstraintsFromText(text);
        const candidates = filterHardExclusions(propertyData, constraints);
        
        // 2. Keyword & Rule-based Pre-scoring
        const queryKeywords = extractKeywords(text);
        const topCandidates = calculateRuleBasedScore(candidates, queryKeywords, text, constraints);

        // 2.5 Progressive: Immediately yield rule-based top results so UI feels instant
        if (onPartialResult && topCandidates.length > 0) {
            const quickResults = topCandidates.slice(0, top_k).map(({ prop, rms }) => ({
                property: prop,
                score: Math.round(rms * 75), // Rule-based estimate
                match_reasons: explainMatch(text, prop, constraints),
                conflict_reason: checkConflicts(prop, constraints)
            }));
            quickResults.sort((a, b) => b.score - a.score);
            onPartialResult(formatResponse(quickResults, top_k));
        }

        // 2.6 Yield to UI thread again before starting expensive AI inference
        await new Promise(resolve => setTimeout(resolve, 50));

        // 3. AI Re-ranking (runs after partial results are shown)
        const scoredResults = [];
        for (let i = 0; i < topCandidates.length; i++) {
            // If a newer query has arrived, abort this one immediately
            if (isCancelled()) return null;

            const { prop, rms } = topCandidates[i];
            try {
                const aiScore = await scorePair(text, prop.text);
                
                // RoBERTa scores are well-calibrated (0.0 ~ 1.0), apply light rescaling
                const normalizedAiScore = Math.max(0, Math.min(1.0, (aiScore - 0.01) / 0.89));
                
                let finalPercentage = Math.round((rms * 35) + (normalizedAiScore * 65));
                if (rms === 1.0 && finalPercentage < 80) finalPercentage = 80 + Math.round(normalizedAiScore * 15);
                
                // --- Explainability & Hybrid Filtering (Option 1) ---
                const match_reasons = explainMatch(text, prop, constraints);
                const conflict_reason = checkConflicts(prop, constraints);
                
                if (conflict_reason) {
                    finalPercentage *= 0.1; // Aggressive reduction for conflicts
                }
                
                scoredResults.push({ 
                    property: prop, 
                    score: Math.min(100, Math.round(finalPercentage)),
                    match_reasons,
                    conflict_reason
                });
            } catch (err) {
                console.error(`AI scoring error for property ${i}:`, err);
            }
        }

        // 4. Return final AI-ranked results
        if (isCancelled()) return null;
        scoredResults.sort((a, b) => b.score - a.score);
        console.log(`Inference complete: ${scoredResults.length} results in ${(performance.now() - startTime).toFixed(0)}ms`);

        if (scoredResults.length > 0) {
            console.log("Top Match:", { query: text, property: scoredResults[0].property.text, score: scoredResults[0].score + "%" });
        }

        return formatResponse(scoredResults, top_k);
    } catch (err) {
        throw err;
    }
}
