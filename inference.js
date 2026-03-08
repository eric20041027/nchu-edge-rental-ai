let rentalData = [];

// Initialize CSV data
export async function initData() {
    return new Promise((resolve, reject) => {
        // PapaParse is loaded via CDN in index.html
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

// 模擬原本載入模型的耗時結構，實際上直接完成
export async function initNLP(onProgress) {
    return Promise.resolve();
}

function tagFeatures(text) {
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

    if (!text) return features;

    // 1. 預算解析
    // 尋找例如 "預算 6000", "6000", "6千" 的字眼
    let budgetMatch = text.match(/預算\s*(\d+)/) || text.match(/(\d+)\s*(元|塊|千|萬)/) || text.match(/(\d{3,5})/);
    if (budgetMatch) {
        let num = parseInt(budgetMatch[1]);
        if (budgetMatch[2] === '千') num *= 1000;
        else if (budgetMatch[2] === '萬') num *= 10000;
        else if (num < 100 && text.includes('預算')) num *= 1000; // e.g. "預算 6" -> 6000

        if (num >= 1000 || text.includes("預算")) {
            features["預算"] = num;
        }
    }

    if (text.includes("以上") && !text.includes("以內")) features["預算限制"] = "above";
    if (text.includes("以下") || text.includes("以內")) features["預算限制"] = "below";

    // 2. 地區
    const regions = ["南區", "西區", "東區", "北區", "中區", "大里區", "大里", "烏日", "市區", "校區", "學校"];
    for (let r of regions) {
        if (text.includes(r)) { features["地址(區域)"] = r; break; }
    }

    // 3. 房型
    const rooms = ["套房", "雅房", "整層", "家庭式", "住家"];
    for (let r of rooms) if (text.includes(r)) { features["格局(房型)"] = r; break; }

    // 4. 建築
    const builds = ["透天厝", "透天", "電梯大樓", "公寓", "別墅"];
    for (let b of builds) if (text.includes(b)) { features["類型(建築)"] = b; break; }

    // 5. 家具
    const furniture_keywords = ["床", "衣櫃", "電話", "網路", "寬頻", "冰箱", "洗衣機", "脫水機", "電視", "第四台", "書桌", "熱水器", "冷氣", "穿衣鏡", "電梯", "車位", "機車", "汽車", "飲水機", "陽台", "曬衣"];
    for (let f of furniture_keywords) {
        if (text.includes(f)) features["家具設施"].push(f);
    }

    // 6. 租金包含項目
    const included_keywords = ["水費", "電費", "網路費", "管理費", "清潔費", "瓦斯", "水", "電", "網路"];
    if (text.includes("包") || text.includes("含")) {
        for (let i of included_keywords) {
            let regex = new RegExp(`(包|含).*?${i}`);
            if (regex.test(text)) {
                let key = (i === '水' || i === '電') ? i + '費' : i;
                if (!features["租金包含"].includes(key)) features["租金包含"].push(key);
            }
        }
    }

    // 7. 安全
    const security_keywords = ["監視器", "監視系統", "攝影", "感應", "滅火器", "警報", "照明", "逃生", "防盜"];
    for (let s of security_keywords) {
        if (text.includes(s)) features["安全管理與消防"].push(s);
    }

    // 8. 寵物
    if (text.includes("寵物") || text.includes("貓") || text.includes("狗")) {
        if (text.includes("不") || text.includes("禁") || text.includes("不能") || text.includes("不可")) {
            features["寵物"] = "禁養寵物";
        } else {
            features["寵物"] = "可養寵物";
        }
    }

    // 9. 性別
    if (text.includes("限男") || text.includes("男網") || text.match(/男生/)) features["性別限制"] = "限男生";
    else if (text.includes("限女") || text.includes("女孩") || text.match(/女生/)) features["性別限制"] = "限女生";

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
    const features = tagFeatures(text);
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
