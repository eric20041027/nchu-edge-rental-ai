/**
 * property-features.js — 房源特徵工具(來源判定 / 三態 bool / 可比對文字 / 同義歸一)。
 *
 * 架構精簡批次5 從 inference.js 抽出。rule-based 比對與 explainability 共用。
 * 資料換版時 COLLAPSED_BOOL_FIELDS / BOOL_FIELD_FEATURES 需依新統計更新
 * (見 data_source_misalignment 記憶)。
 */

// 2026-06-14 興大 crawler 補抓 租金包含/安全管理/消防逃生 二級表格 + 衍生特色標籤後，
// has_window 0%→70%、safety_level high 0%→94%、特色 avg 1.6→5.3。
// 資料換版時需依新統計更新此表 (見 data_source_misalignment 記憶)。
// --- Source-aware boolean reliability (bool 空值≠False) ---------------------
// 兩來源爬蟲抓取的欄位子集不同，部分 bool 欄在某來源「整欄崩塌」(≈0% true)，
// 那是「爬蟲沒抓該欄」而非「房子真的沒有」。對崩塌欄的 false 必須視為「未知」
// (回退文字判斷 / 交 AI)，不可當「明確無」硬判，否則系統性誤殺該來源房源。
//
// 崩塌判定來自現有 704 筆 property_data.json 實測 true 比率(<5% 視為崩塌)：
//   has_parking      nchu 61% / dd  0%   → dd 崩塌
//   water_dispenser  nchu 61% / dd  0%   → dd 崩塌
//   has_waste_disposal nchu 0% / dd 99%  → nchu 崩塌(興大頁面真無此欄)
//   has_subsidy      nchu  0% / dd 100%  → nchu 崩塌(興大真無租屋補助欄)
//   has_window       nchu 70% / dd 100%  → 兩來源皆可信(2026-06-14 補抓安全管理表後脫離崩塌)
//   has_elevator / has_balcony 兩來源皆有訊號 → 皆可信
export const COLLAPSED_BOOL_FIELDS = {
    nchu: new Set(['has_waste_disposal', 'has_subsidy']),
    dd:   new Set(['has_parking', 'water_dispenser']),
};

export function propSource(prop) {
    return (prop.url || '').includes('nchu') ? 'nchu' : 'dd';
}

// 三態解讀一個 bool 設施欄：
//   'yes'     — has_xxx===true，明確有
//   'no'      — has_xxx===false 且該欄對此來源可信，明確無
//   'unknown' — has_xxx===false 但該欄對此來源崩塌，未知(交文字/AI 判)
export function boolFieldState(prop, field) {
    if (prop[field] === true) return 'yes';
    if (COLLAPSED_BOOL_FIELDS[propSource(prop)]?.has(field)) return 'unknown';
    return 'no';
}

// 把房源的結構化欄位(furniture/features/notes)+ bool 設施欄一起納入可比對文字,
// 並做同義詞歸一,讓查詢擴展詞(可寵/廚房/獨衛…)能對上房源實際用詞(可養貓/可開伙/獨立衛浴…)。
// 落地率實測:只看 text 17.6% → +結構欄+bool 30.5% → +同義歸一 45.8%。
export const PROP_SYNONYMS = {
    "可寵":["可養貓","可養狗","可養寵物","可養其他寵物"],"寵物友善":["可養貓","可養狗","可養寵物"],
    "廚房":["可開伙","流理台"],"開火":["可開伙","瓦斯","電磁爐"],"自炊":["可開伙"],"可伙":["可開伙"],
    "抽油煙機":["排油煙"],"獨衛":["獨立衛浴","專用衛浴"],"獨立衛浴":["獨衛"],"獨廁":["獨立衛浴","獨衛"],
    "變頻":["冷氣"],"變頻冷氣":["冷氣"],"吹冷氣":["冷氣"],"全新":["新裝潢","新成屋"],
    "管理員":["保全","警衛"],"監視器":["保全","監視"],"門禁":["保全","刷卡"],
    "床架":["床"],"床墊":["床"],"書桌椅":["桌子","書桌","椅子"],
    "天然瓦斯熱水器":["熱水器","瓦斯"],"電熱水器":["熱水器"],
    "全配":["家具","家電"],"全家具":["家具"],"全家電":["家電"],"家具齊全":["家具"],
    "子母車":["垃圾"],"垃圾代收":["垃圾"],"獨立洗衣機":["洗衣機"],"獨洗":["洗衣機"],
    // P2 救援橋 (token 對房源自由描述端 0 命中,但有未架橋的真實同義詞/結構欄):
    // 詳見 data/UNVERIFIABLE_TOKENS_AUDIT.md 與 pipeline/data_prep/audit_expansion_tokens.py。
    "禁菸":["無菸"],
    "採光":["對外窗","窗"],"通風":["對外窗","窗"],
    "安全":["保全"],"刷卡":["保全","門禁"],"女性友善":["限女","女性"],
    "租補":["租金補貼","補貼"],"室友":["雅房","分租"],"合租":["雅房","分租"],
    "隔音":["氣密窗","氣密"],
    // 台水/電費族:靠 buildPropText 納入 electricity_billing 欄(排除「不明」)後可命中。
    "台水":["水費"],"帳單":["台電","台水","電費"],"自繳":["台電","台水"],"標準電費":["台電"],
};
export const BOOL_FIELD_FEATURES = {
    has_elevator:"電梯", has_window:"對外窗", has_balcony:"陽台",
    has_parking:"車位 停車場", has_waste_disposal:"垃圾處理", is_rooftop:"頂樓",
    water_dispenser:"飲水機", private_washer:"獨洗", has_subsidy:"補助", is_taipower:"台電",
};

// 產生房源「完整可比對文字」: text + 結構化欄位 + bool 設施詞。所有房源關鍵字比對統一使用。
export function buildPropText(prop) {
    const parts = [prop.text || ""];
    for (const f of ["furniture", "features", "building_type", "room_type"]) {
        if (prop[f]) parts.push(String(prop[f]).replace(/\//g, " "));
    }
    for (const f of ["notes", "other_fees"]) {
        if (Array.isArray(prop[f])) parts.push(prop[f].join(" "));
    }
    for (const [bk, wd] of Object.entries(BOOL_FIELD_FEATURES)) {
        if (prop[bk] === true) parts.push(wd);
    }
    // 電費/水費計費方式 (台電/台水/獨立電錶) 落在結構欄,非自由文字。納入可比對文字以
    // 支援「台水/標準電費/帳單自繳」等查詢;「不明」是未知值非訊號,排除避免假命中。
    if (prop.electricity_billing && prop.electricity_billing !== "不明") {
        parts.push(String(prop.electricity_billing));
    }
    return parts.join(" ");
}
// 註(2026-06-16 更新):2026-06-14 曾驗證「不重訓、直接餵 enriched 文字」= NO-GO
// (docs/ce_text_layer_decision.md:舊 CE 只認短 prop.text 格式,餵長文字 OOD 崩壞,
// 「要有陽台」+7.9→+0.5)。**根治方案已落地**:C 組重訓 CE on 富化文字
// (notebooks/ce_expansion_augment_experiment.ipynb),訓練+推論都用富化文字 → 消除
// OOD。故 scorePair 現餵 prop.ce_text(預算於 property_data.json,見
// precompute_ce_text.py),非 prop.text。buildPropText 仍只供 rule-based 35% 層使用。

// 房源是否含某特徵詞(含同義歸一): 直接命中, 或任一同義詞命中。
export function propHasFeature(propText, feature) {
    if (propText.includes(feature)) return true;
    const syns = PROP_SYNONYMS[feature];
    if (syns) for (const s of syns) if (propText.includes(s)) return true;
    return false;
}
