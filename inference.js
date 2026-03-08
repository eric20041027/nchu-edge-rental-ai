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
            tokenizer = await AutoTokenizer.from_pretrained('onnx_model_dir');

            // 2. Setup ONNXRuntime Web paths
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

            // 3. Load ONNX model session directly via URI
            if (onProgress) onProgress({ status: 'progress', file: 'model.onnx', loaded: 50, total: 100 });
            session = await ort.InferenceSession.create(window.location.origin + '/onnx_model_dir/model.onnx');
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

        const label = max_idx === 0 ? 'B' : 'I';

        // Decode token to string (Albert format adds ' ', replacing it)
        let char = tokenizer.decode([token_id]).replace(/ /g, '').trim();
        if (!char) continue;

        if (label === 'B') {
            if (current_word) words.push(current_word);
            current_word = char;
        } else if (label === 'I') {
            current_word += char.replace(/^##/, '');
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
        "備註": []
    };

    const furniture_keywords = ["床", "衣櫃", "電話", "網路", "寬頻", "冰箱", "洗衣機", "脫水機", "電視", "第四台", "書桌", "熱水器", "冷氣", "穿衣鏡", "電梯", "車位", "機車", "汽車", "飲水機", "陽台", "曬衣"];
    const included_keywords = ["水費", "電費", "網路費", "管理費", "清潔費", "瓦斯"];
    const security_keywords = ["監視器", "監視系統", "攝影", "感應", "滅火器", "警報", "照明", "逃生", "防盜"];

    for (let i = 0; i < words.length; i++) {
        let word = words[i];

        if (word === "預算" && i + 1 < words.length) {
            if (!isNaN(parseInt(words[i + 1]))) {
                features["預算"] = parseInt(words[i + 1]);
            }
        } else if (!isNaN(parseInt(word)) && !features["預算"]) {
            if (i + 1 < words.length && ["元", "塊", "千", "萬"].includes(words[i + 1])) {
                let multiplier = words[i + 1] === "千" ? 1000 : (words[i + 1] === "萬" ? 10000 : 1);
                features["預算"] = parseInt(word) * multiplier;
            } else if (parseInt(word) > 1000) {
                features["預算"] = parseInt(word);
            }
        }

        if (word.includes("以上")) features["預算限制"] = "above";
        if (word.includes("以下") || word.includes("以內")) features["預算限制"] = "below";

        if (["南區", "西區", "東區", "北區", "中區", "大里", "大里區", "烏日", "市區", "校區", "學校"].includes(word)) {
            features["地址(區域)"] = word;
        }

        if (["套房", "雅房", "整層", "家庭式", "住家"].includes(word)) features["格局(房型)"] = word;
        if (["透天", "透天厝", "公寓", "電梯大樓", "別墅"].includes(word)) features["類型(建築)"] = word;

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
    }

    return features;
}

function formatForCBF(features) {
    let pet_friendly = -1;
    if (features["寵物"] === "可養寵物") pet_friendly = 1;
    else if (features["寵物"] === "禁養寵物") pet_friendly = 0;

    return {
        search_budget: features["預算"],
        budget_limit: features["預算限制"],
        search_region: features["地址(區域)"],
        search_room_type: features["格局(房型)"],
        search_building_type: features["類型(建築)"],
        required_furniture: features["家具設施"],
        required_included_fees: features["租金包含"],
        required_security: features["安全管理與消防"],
        is_pet_friendly: pet_friendly,
        gender_preference: features["性別限制"]
    };
}

export async function recommend(text, top_k = 5) {
    // RUN ALBERT + ORT
    const words = await segmentText(text);
    console.log("ALBERT Segmented Words:", words);

    // Feature Extraction
    const features = tagFeatures(words);
    const cbf_vector = formatForCBF(features);

    let user_budget = cbf_vector.search_budget;
    let target_region = cbf_vector.search_region;
    let target_room = cbf_vector.search_room_type;
    let target_building = cbf_vector.search_building_type;
    let req_furnitures = cbf_vector.required_furniture;
    let req_security = cbf_vector.required_security;
    let user_gender = cbf_vector.gender_preference;
    let pet_friendly = cbf_vector.is_pet_friendly;

    let MAX_THEORETICAL_SCORE = 0.0;
    if (user_budget) MAX_THEORETICAL_SCORE += 20.0;
    if (target_region) MAX_THEORETICAL_SCORE += 20.0;
    if (target_room) MAX_THEORETICAL_SCORE += 20.0;
    if (target_building) MAX_THEORETICAL_SCORE += 15.0;
    MAX_THEORETICAL_SCORE += (req_furnitures.length * 5.0);
    MAX_THEORETICAL_SCORE += (req_security.length * 5.0);

    if (MAX_THEORETICAL_SCORE === 0) MAX_THEORETICAL_SCORE = 1.0;

    let results = [];

    for (let row of rentalData) {
        let score = 0.0;
        let details = [];
        let notes = row.Note_List.join(" ");

        let budget_limit = cbf_vector.budget_limit;
        if (user_budget && budget_limit) {
            if (budget_limit === 'below' && row.Rent_Num > user_budget) {
                score -= 1000.0; details.push("超出預算上限");
            } else if (budget_limit === 'above' && row.Rent_Num < user_budget) {
                score -= 1000.0; details.push("低於預算下限");
            }
        }

        if (user_gender === "限女生" && notes.includes("限男生")) {
            score -= 1000.0; details.push("性別不符(限男)");
        } else if (user_gender === "限男生" && notes.includes("限女生")) {
            score -= 1000.0; details.push("性別不符(限女)");
        }

        if (pet_friendly === 1 && notes.includes("禁養寵物")) {
            score -= 1000.0; details.push("禁養寵物");
        } else if (pet_friendly === 0 && notes.includes("可養寵物")) {
            score -= 50.0; details.push("有其他寵物");
        }

        if (user_budget) {
            let diff = Math.abs(row.Rent_Num - user_budget);
            if (diff <= 500) {
                score += 20.0; details.push("預算完美符合");
            } else {
                let deduction = (diff - 500) / 200.0;
                let awarded = Math.max(0.0, 20.0 - deduction);
                score += awarded;
                if (awarded >= 10) details.push("符合預算");
            }
        }

        let addr = String(row['地址'] || '');
        if (target_region && addr.includes(target_region)) {
            score += 20.0; details.push(`位於${target_region}`);
        }

        let roomStr = String(row['格局'] || '');
        if (target_room && roomStr.includes(target_room)) {
            score += 20.0; details.push(`房型相符(${target_room})`);
        }

        let buildStr = String(row['類型'] || '');
        if (target_building && buildStr.includes(target_building)) {
            score += 15.0; details.push(`建築相符(${target_building})`);
        }

        for (let f of req_furnitures) {
            if (row.Furniture_List.some(item => item.includes(f))) {
                score += 5.0; details.push(`有${f}`);
            }
        }

        for (let s of req_security) {
            if (row.Security_List.some(item => item.includes(s))) {
                score += 5.0; details.push(`有${s}`);
            }
        }

        let percentage_score = (score / MAX_THEORETICAL_SCORE) * 100;
        if (score >= 0) {
            percentage_score = 40 + (percentage_score * 0.6);
        }
        percentage_score = Math.min(Math.max(percentage_score, 0), 100);

        if (score > 0) {
            results.push({
                house: row,
                score: Math.round(percentage_score),
                rent_num: row.Rent_Num,
                match_details: details.join(", ")
            });
        }
    }

    results.sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        return a.rent_num - b.rent_num;
    });

    return results.slice(0, top_k).map(item => {
        // Map back to the API response structure expected by app.js
        return {
            id: item.house['網址'], // unique identifier
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
        };
    });
}
