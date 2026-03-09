import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

let tokenizer = null;
let session = null;
let rentalData = [];

// Initialize CSV data
export async function initData() {
    return new Promise((resolve, reject) => {
        window.Papa.parse('/nchu_rental_info.csv', {
            download: true,
            header: true,
            complete: function (results) {
                rentalData = results.data.filter(row => row['網址']);
                rentalData.forEach(row => {
                    let rentStr = row['租金'];
                    if (!rentStr) {
                        row.Rent_Num = 999999;
                    } else {
                        let match = String(rentStr).replace(/,/g, '').match(/(\d+)/);
                        row.Rent_Num = match ? parseInt(match[1]) : 999999;
                    }

                    const splitList = (str) => str ? String(str).split('/').map(s => s.trim()).filter(Boolean) : [];
                    row.Furniture_List = splitList(row['家具設施']);
                    row.Included_List = splitList(row['租金包含']);
                    row.Security_List = splitList(row['安全管理']);
                    row.Note_List = splitList(row['備註']);
                    row.distance = parseFloat(row['距離(km)']) || 0;
                });
                resolve();
            },
            error: reject
        });
    });
}

// Initialize NLP model using ALBERT via ONNXRuntime Web
export async function initNLP(onProgress) {
    if (!tokenizer || !session) {
        env.allowRemoteModels = false;
        env.allowLocalModels = true;
        env.useBrowserCache = true;

        // Point Transformers.js to the root path of the server
        env.localModelPath = window.location.origin + '/';

        try {
            // 1. Load Tokenizer using relative directory name
            if (onProgress) onProgress({ status: 'progress', file: 'tokenizer.json', loaded: 10, total: 100 });
            tokenizer = await AutoTokenizer.from_pretrained('custom_onnx_model_dir');

            // 2. Setup ONNXRuntime Web paths
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

            // 3. Load ONNX model session directly via URI with externalData linking
            if (onProgress) onProgress({ status: 'progress', file: 'model.onnx', loaded: 50, total: 100 });
            session = await ort.InferenceSession.create(
                window.location.origin + '/custom_onnx_model_dir/model.onnx',
                {
                    executionProviders: ['wasm'],
                    externalData: [
                        {
                            path: 'my_custom_model.onnx.data', // 原本匯出時寫死在 ONNX 裡面的黨名
                            data: window.location.origin + '/custom_onnx_model_dir/model.onnx.data'
                        }
                    ]
                }
            );
            if (onProgress) onProgress({ status: 'progress', file: 'model.onnx', loaded: 100, total: 100 });

        } catch (e) {
            console.error("NLP Init Error:", e);
            throw e;
        }
    }
}

async function segmentText(text) {
    if (!text || !tokenizer || !session) return [];

    // 1. Tokenize Text
    const tokens = await tokenizer(text, { return_tensor: false });

    // 2. Prepare Tensors
    const input_ids = new ort.Tensor('int64', BigInt64Array.from(tokens.input_ids.map(BigInt)), [1, tokens.input_ids.length]);
    const attention_mask = new ort.Tensor('int64', BigInt64Array.from(tokens.attention_mask.map(BigInt)), [1, tokens.attention_mask.length]);
    const token_type_ids = new ort.Tensor('int64', BigInt64Array.from(tokens.token_type_ids.map(BigInt)), [1, tokens.token_type_ids.length]);

    const feeds = {
        input_ids: input_ids,
        attention_mask: attention_mask,
        token_type_ids: token_type_ids
    };

    // 3. Run ONNX Inference
    const results = await session.run(feeds);
    const logits = results.logits.data; // Float32Array format [batch, seq_len, num_labels]
    const num_labels = results.logits.dims[2];

    // 4. Argmax and Decode B/I tags
    let words = [];
    let current_word = '';

    for (let i = 0; i < tokens.input_ids.length; i++) {
        const token_id = tokens.input_ids[i];
        // Skip Special Tokens (CLS=101, SEP=102, PAD=0)
        if (token_id === 101 || token_id === 102 || token_id === 0) continue;

        let max_val = -Infinity;
        let max_idx = 0;
        for (let j = 0; j < num_labels; j++) {
            let val = logits[i * num_labels + j];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }

        // Custom model uses 3 labels: 0="O", 1="B-Target", 2="I-Target"
        const label = (max_idx === 1) ? 'B' : (max_idx === 2 ? 'I' : 'O');

        // Decode token to string (Albert format adds ' ', replacing it)
        let char = tokenizer.decode([token_id]).replace(/ /g, '').trim();
        if (!char) continue;

        if (label === 'B') {
            if (current_word) words.push(current_word);
            current_word = char;
        } else if (label === 'I') {
            current_word += char.replace(/^##/, '');
        } else if (label === 'O') {
            if (current_word) {
                words.push(current_word);
                current_word = ''; // Reset after pushing
            }
        }
    }
    if (current_word) words.push(current_word);

    return words;
}

function tagFeatures(words) {
    const features = {
        "地址(區域)": null,
        "格局(房型)": null,
        "類型(建築)": null,
        "預算": null,
        "預算限制": null,
        "家具設施": [],
        "租金包含": [],
        "安全管理與消防": [],
        "寵物": null,
        "性別限制": null,
        "距離需求_km": null,
        "備註": []
    };

    const furniture_keywords = ["床", "衣櫃", "電話", "網路", "寬頻", "冰箱", "洗衣機", "脫水機", "電視", "第四台", "書桌", "熱水器", "冷氣", "穿衣鏡", "電梯", "車位", "機車", "汽車", "飲水機", "陽台", "曬衣", "獨洗", "獨曬", "開伙", "廚房", "烘衣", "沙發", "對外窗"];
    const included_keywords = ["水費", "電費", "網路費", "管理費", "清潔費", "瓦斯"];
    const security_keywords = ["監視器", "監視系統", "攝影", "感應", "滅火器", "警報", "照明", "逃生", "防盜"];

    for (let i = 0; i < words.length; i++) {
        let word = words[i];

        const parseBudget = (str) => {
            let text = str.replace(/以[上下內]/g, '').replace(/內/g, '');
            text = text.replace(/一/g, '1').replace(/二/g, '2').replace(/兩/g, '2').replace(/三/g, '3')
                .replace(/四/g, '4').replace(/五/g, '5').replace(/六/g, '6').replace(/七/g, '7')
                .replace(/八/g, '8').replace(/九/g, '9');
            if (text.includes("萬")) {
                let v = parseFloat(text.replace("萬", "."));
                if (!isNaN(v)) return Math.floor(v * 10000);
            }
            text = text.replace(/千/g, '000').replace(/[kK]/g, '000').replace(/百/g, '00');
            const match = text.match(/\d+/);
            if (match) return parseInt(match[0], 10);
            return null;
        };

        if (word === "預算" && i + 1 < words.length) {
            let pb = parseBudget(words[i + 1]);
            if (pb) features["預算"] = pb < 100 ? pb * 1000 : pb;
        } else if (!features["預算"]) {
            let pb = parseBudget(word);
            if (pb) {
                if (i + 1 < words.length && ["元", "塊", "千", "萬", "k", "K"].includes(words[i + 1])) {
                    let nextW = words[i + 1].toLowerCase();
                    if (nextW === "千" || nextW === "k") pb *= 1000;
                    if (nextW === "萬") pb *= 10000;
                }
                features["預算"] = pb < 100 ? pb * 1000 : pb;
            }
        }

        if (word.includes("以上") || word.includes("上")) features["預算限制"] = "above";
        if (word.includes("以下") || word.includes("以內") || word.includes("內") || word.includes("下")) features["預算限制"] = "below";

        // 地區匹配：使用 includes 避免 NER 合併詞漏掉
        const region_keywords = ["南區", "西區", "東區", "北區", "中區", "大里", "大里區", "烏日", "市區", "校區", "學校"];
        for (let rk of region_keywords) {
            if (word.includes(rk)) { features["地址(區域)"] = rk; break; }
        }

        // 房型匹配：使用 includes 處理 NER 合併詞（如「獨洗獨曬套房」）
        const room_keywords = ["套房", "雅房", "整層", "家庭式", "住家"];
        for (let rk of room_keywords) {
            if (word.includes(rk)) { features["格局(房型)"] = rk; break; }
        }

        // 建築類型匹配
        const building_keywords = ["透天", "透天厝", "公寓", "電梯大樓", "別墅"];
        for (let bk of building_keywords) {
            if (word.includes(bk)) { features["類型(建築)"] = bk; break; }
        }

        for (let f_kw of furniture_keywords) {
            if (word.includes(f_kw) && !features["家具設施"].includes(f_kw)) {
                features["家具設施"].push(f_kw);
            }
        }

        for (let inc_kw of included_keywords) {
            if (word.includes(inc_kw)) {
                let is_included = false;
                for (let j = Math.max(0, i - 2); j < i; j++) {
                    if (["包", "包含", "含"].includes(words[j])) is_included = true;
                }
                if (is_included && !features["租金包含"].includes(inc_kw)) {
                    features["租金包含"].push(inc_kw);
                }
            }
        }

        for (let sec_kw of security_keywords) {
            if (word.includes(sec_kw) && !features["安全管理與消防"].includes(sec_kw)) {
                features["安全管理與消防"].push(sec_kw);
            }
        }

        if (word.includes("寵物") || word.includes("貓") || word.includes("狗")) {
            let is_allowed = true;
            for (let j = Math.max(0, i - 2); j < i; j++) {
                if (["不", "禁", "不可", "不能"].includes(words[j])) is_allowed = false;
            }
            features["寵物"] = is_allowed ? "可養寵物" : "禁養寵物";
        }

        if (word.includes("男") || word.includes("女")) {
            for (let j = Math.max(0, i - 2); j < i; j++) {
                if (["限", "只"].includes(words[j])) features["性別限制"] = `限${word}生`;
            }
        }

        // --- 距離特徵解析邏輯 ---
        if (["近", "正門", "男宿", "女宿", "旁邊", "附近"].includes(word) || word.includes("正門") || word.includes("近")) {
            if (features["距離需求_km"] === null || features["距離需求_km"] > 1.0) {
                features["距離需求_km"] = 1.0; // 預設「近」或「正門」代表1公里內
            }
        }

        if (word === "騎車" || word === "騎機車") {
            let mins = 5; // 預設 5 分鐘
            for (let j = i; j < Math.min(words.length, i + 3); j++) {
                let parsed = parseInt(words[j]);
                if (!isNaN(parsed)) mins = parsed;
            }
            // 騎車時速抓約 40km/h => 1分鐘大約移動 0.6 km
            let km = mins * 0.6;
            if (features["距離需求_km"] === null || features["距離需求_km"] > km) {
                features["距離需求_km"] = km;
            }
        }

        if (word === "走路" || word === "步行") {
            let mins = 5;
            for (let j = i; j < Math.min(words.length, i + 3); j++) {
                let parsed = parseInt(words[j]);
                if (!isNaN(parsed)) mins = parsed;
            }
            // 走路時速抓約 4.8km/h => 1分鐘大約移動 0.08 km
            let km = mins * 0.08;
            if (features["距離需求_km"] === null || features["距離需求_km"] > km) {
                features["距離需求_km"] = km;
            }
        }
    }

    return features;
}

// ============================================================
// Cosine Similarity Recommendation Engine
// ============================================================

// Shared vector schema — every vector uses the same dimension layout
const REGION_KEYS = ["南區", "西區", "東區", "北區", "中區", "大里", "烏日"];
const ROOM_KEYS = ["套房", "雅房", "整層", "住家"];
const BUILDING_KEYS = ["公寓", "透天", "電梯大樓", "別墅"];
const FURNITURE_KEYS = ["床", "衣櫃", "電話", "網路", "寬頻", "冰箱", "洗衣機", "脫水機", "電視", "第四台", "書桌", "熱水器", "冷氣", "穿衣鏡", "電梯", "車位", "機車", "汽車", "飲水機", "陽台", "曬衣", "獨洗", "獨曬", "開伙", "廚房", "烘衣", "沙發", "對外窗"];
const INCLUDED_KEYS = ["水費", "電費", "網路費", "管理費", "清潔費", "瓦斯"];
const SECURITY_KEYS = ["監視器", "攝影", "感應", "滅火器", "警報", "照明", "逃生", "防盜"];

// Dimension index calculation
// [0]      budget_sim          (Gaussian similarity)
// [1]      distance_sim        (Gaussian similarity)
// [2..8]   region one-hot      (7 dims)
// [9..12]  room one-hot        (4 dims)
// [13..16] building one-hot    (4 dims)
// [17..44] furniture multi-hot (28 dims)
// [45..50] included multi-hot  (6 dims)
// [51..58] security multi-hot  (8 dims)
// [59]     pet_friendly        (0 or 1)
// Total: 60 dimensions

const VECTOR_LENGTH = 60;
const IDX_BUDGET = 0;
const IDX_DISTANCE = 1;
const IDX_REGION_START = 2;
const IDX_ROOM_START = IDX_REGION_START + REGION_KEYS.length;       // 9
const IDX_BUILDING_START = IDX_ROOM_START + ROOM_KEYS.length;       // 13
const IDX_FURNITURE_START = IDX_BUILDING_START + BUILDING_KEYS.length; // 17
const IDX_INCLUDED_START = IDX_FURNITURE_START + FURNITURE_KEYS.length; // 45
const IDX_SECURITY_START = IDX_INCLUDED_START + INCLUDED_KEYS.length;  // 51
const IDX_PET = IDX_SECURITY_START + SECURITY_KEYS.length;            // 59

// Human-readable names for each dimension (used in match_details)
function getDimensionName(idx) {
    if (idx === IDX_BUDGET) return "預算";
    if (idx === IDX_DISTANCE) return "距離";
    if (idx >= IDX_REGION_START && idx < IDX_ROOM_START) return REGION_KEYS[idx - IDX_REGION_START];
    if (idx >= IDX_ROOM_START && idx < IDX_BUILDING_START) return ROOM_KEYS[idx - IDX_ROOM_START];
    if (idx >= IDX_BUILDING_START && idx < IDX_FURNITURE_START) return BUILDING_KEYS[idx - IDX_BUILDING_START];
    if (idx >= IDX_FURNITURE_START && idx < IDX_INCLUDED_START) return FURNITURE_KEYS[idx - IDX_FURNITURE_START];
    if (idx >= IDX_INCLUDED_START && idx < IDX_SECURITY_START) return INCLUDED_KEYS[idx - IDX_INCLUDED_START];
    if (idx >= IDX_SECURITY_START && idx < IDX_PET) return SECURITY_KEYS[idx - IDX_SECURITY_START];
    if (idx === IDX_PET) return "可養寵物";
    return `dim${idx}`;
}

// Masked cosine similarity — only considers dimensions where the user expressed a preference
// Unmentioned features (userVec[i] === 0) are ignored, not penalized
function cosineSimilarity(userVec, houseVec) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < userVec.length; i++) {
        if (userVec[i] === 0) continue; // Skip dimensions the user didn't mention
        dot += userVec[i] * houseVec[i];
        normA += userVec[i] * userVec[i];
        normB += houseVec[i] * houseVec[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Gaussian similarity: returns 1.0 when values are identical, decays towards 0
function gaussianSim(a, b, sigma) {
    let diff = a - b;
    return Math.exp(-(diff * diff) / (2 * sigma * sigma));
}

// Build a feature vector from the user's extracted features
function buildUserVector(features, rawText) {
    let vec = new Float32Array(VECTOR_LENGTH);

    // Budget — use raw budget value normalized for Gaussian comparison
    let budget = features["預算"];
    // Fallback: parse budget from raw text if NER missed it
    if (!budget) {
        let rt = rawText.replace(/一/g, '1').replace(/二/g, '2').replace(/兩/g, '2').replace(/三/g, '3')
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
    }
    if (budget) vec[IDX_BUDGET] = budget / 30000.0; // Normalize to ~0-1 range

    // Distance
    let dist = features["距離需求_km"];
    if (dist) vec[IDX_DISTANCE] = 1.0 - Math.min(dist / 5.0, 1.0); // closer = higher

    // Region
    let region = features["地址(區域)"];
    if (region) {
        for (let i = 0; i < REGION_KEYS.length; i++) {
            if (region.includes(REGION_KEYS[i])) { vec[IDX_REGION_START + i] = 1.0; break; }
        }
    }

    // Room type
    let room = features["格局(房型)"];
    if (room) {
        for (let i = 0; i < ROOM_KEYS.length; i++) {
            if (room.includes(ROOM_KEYS[i])) { vec[IDX_ROOM_START + i] = 1.0; break; }
        }
    }

    // Building type
    let building = features["類型(建築)"];
    if (building) {
        for (let i = 0; i < BUILDING_KEYS.length; i++) {
            if (building.includes(BUILDING_KEYS[i])) { vec[IDX_BUILDING_START + i] = 1.0; break; }
        }
    }

    // Furniture multi-hot
    for (let f of features["家具設施"]) {
        for (let i = 0; i < FURNITURE_KEYS.length; i++) {
            if (f.includes(FURNITURE_KEYS[i])) vec[IDX_FURNITURE_START + i] = 1.0;
        }
    }

    // Included fees multi-hot
    for (let f of features["租金包含"]) {
        for (let i = 0; i < INCLUDED_KEYS.length; i++) {
            if (f.includes(INCLUDED_KEYS[i])) vec[IDX_INCLUDED_START + i] = 1.0;
        }
    }

    // Security multi-hot
    for (let f of features["安全管理與消防"]) {
        for (let i = 0; i < SECURITY_KEYS.length; i++) {
            if (f.includes(SECURITY_KEYS[i])) vec[IDX_SECURITY_START + i] = 1.0;
        }
    }

    // Pet
    if (features["寵物"] === "可養寵物") vec[IDX_PET] = 1.0;

    return vec;
}

// Build a feature vector from a rental property row (CSV data)
function buildHouseVector(row) {
    let vec = new Float32Array(VECTOR_LENGTH);

    // Budget normalized
    if (row.Rent_Num && row.Rent_Num < 900000) {
        vec[IDX_BUDGET] = row.Rent_Num / 30000.0;
    }

    // Distance — closer = higher value
    let dist = parseFloat(row['距離(km)']);
    if (!isNaN(dist)) {
        vec[IDX_DISTANCE] = 1.0 - Math.min(dist / 5.0, 1.0);
    }

    // Region
    let addr = String(row['地址'] || '');
    for (let i = 0; i < REGION_KEYS.length; i++) {
        if (addr.includes(REGION_KEYS[i])) { vec[IDX_REGION_START + i] = 1.0; break; }
    }

    // Room type
    let roomStr = String(row['格局'] || '');
    for (let i = 0; i < ROOM_KEYS.length; i++) {
        if (roomStr.includes(ROOM_KEYS[i])) { vec[IDX_ROOM_START + i] = 1.0; break; }
    }

    // Building type
    let buildStr = String(row['類型'] || '');
    for (let i = 0; i < BUILDING_KEYS.length; i++) {
        if (buildStr.includes(BUILDING_KEYS[i])) { vec[IDX_BUILDING_START + i] = 1.0; break; }
    }

    // Furniture multi-hot
    for (let item of row.Furniture_List) {
        for (let i = 0; i < FURNITURE_KEYS.length; i++) {
            if (item.includes(FURNITURE_KEYS[i])) vec[IDX_FURNITURE_START + i] = 1.0;
        }
    }

    // Included fees
    let includedStr = String(row['租金包含'] || '');
    for (let i = 0; i < INCLUDED_KEYS.length; i++) {
        if (includedStr.includes(INCLUDED_KEYS[i])) vec[IDX_INCLUDED_START + i] = 1.0;
    }

    // Security
    for (let item of row.Security_List) {
        for (let i = 0; i < SECURITY_KEYS.length; i++) {
            if (item.includes(SECURITY_KEYS[i])) vec[IDX_SECURITY_START + i] = 1.0;
        }
    }

    // Pet friendly
    let notes = row.Note_List.join(" ");
    if (notes.includes("可養寵物")) vec[IDX_PET] = 1.0;

    return vec;
}

export async function recommend(text, top_k = 5) {
    // Step 1: NER segmentation
    const words = await segmentText(text);
    console.log("ALBERT Segmented Words:", words);

    // Step 2: Feature extraction (reuses existing tagFeatures)
    const features = tagFeatures(words);
    console.log("Extracted Features:", features);

    // Step 3: Build user vector
    const userVec = buildUserVector(features, text);
    console.log("User Vector:", Array.from(userVec));

    // Detect budget limit from raw text (NER may tag 以上/以下 as O)
    let budget_limit = features["預算限制"];
    if (!budget_limit) {
        if (text.includes('以上')) budget_limit = 'above';
        else if (text.includes('以下') || text.includes('以內')) budget_limit = 'below';
    }

    // Get budget value for hard filtering
    let user_budget = userVec[IDX_BUDGET] * 30000; // Denormalize
    if (user_budget < 100) user_budget = null; // No budget specified

    let user_gender = features["性別限制"];
    let pet_pref = features["寵物"];

    console.log("Parsed Budget:", user_budget, "Limit:", budget_limit);

    // Step 4: Score each property
    let results = [];

    for (let row of rentalData) {
        let notes = row.Note_List.join(" ");

        // === Hard exclusion filters (NOT part of cosine) ===
        if (user_budget && budget_limit) {
            if (budget_limit === 'below' && row.Rent_Num > user_budget) continue;
            if (budget_limit === 'above' && row.Rent_Num < user_budget) continue;
        }
        if (user_gender === "限女生" && notes.includes("限男生")) continue;
        if (user_gender === "限男生" && notes.includes("限女生")) continue;
        if (pet_pref === "可養寵物" && notes.includes("禁養寵物")) continue;

        // Build house vector
        const houseVec = buildHouseVector(row);

        // Compute cosine similarity
        let sim = cosineSimilarity(userVec, houseVec);

        // Boost: apply Gaussian bonus for budget proximity (sigma = 2000 NTD)
        if (user_budget && row.Rent_Num < 900000) {
            let budgetSim = gaussianSim(user_budget, row.Rent_Num, 2000);
            sim = sim * 0.7 + budgetSim * 0.3; // Blend: 70% cosine + 30% budget precision
        }

        let percentage = Math.round(sim * 100);
        if (percentage <= 0) continue;

        // Generate match details — list dimensions where both user and house have non-zero values
        let details = [];
        for (let i = 0; i < VECTOR_LENGTH; i++) {
            if (userVec[i] > 0 && houseVec[i] > 0) {
                let name = getDimensionName(i);
                if (!details.includes(name)) details.push(name);
            }
        }

        results.push({
            house: row,
            score: percentage,
            rent_num: row.Rent_Num,
            match_details: details.join(", ")
        });
    }

    // Step 5: Sort by similarity desc, then by rent asc for ties
    results.sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        return a.rent_num - b.rent_num;
    });

    return results.slice(0, top_k).map(item => ({
        id: item.house['網址'],
        title: `${item.house['格局']} | ${item.house['地址']}`,
        price_str: item.house['租金'],
        url: item.house['網址'],
        imgUrl: item.house['圖片網址'] || null,
        score: item.score,
        match_details: item.match_details,
        size: item.house['室內坪數'] || "坪數未提供",
        floor: item.house['樓層'] || "樓層未提供",
        furniture: item.house['家具設施'] || "無特殊設施提供",
        distance: item.house.distance,
        address: item.house['地址']
    }));
}

