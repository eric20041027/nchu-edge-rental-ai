/**
 * explainability.js — 推薦結果的可解釋層(匹配理由 + 衝突偵測)。
 *
 * 架構精簡批次5 從 inference.js 抽出。只被 recommend() 用,依賴 property-features。
 */
import { buildPropText } from './property-features.js';

export function explainMatch(query, prop, constraints) {
    const reasons = [];
    const pText = buildPropText(prop).toLowerCase();
    const q = query.toLowerCase();
    
    // 1. Budget & CP Value
    if (constraints.hasBudgetMention) {
        if (prop.rent <= (constraints.maxBudget || constraints.budget)) {
            if (prop.cp_tag === "high_cp") {
                reasons.push("💎 區域高 CP 值首選");
            } else {
                reasons.push("💰 符合您的預算範圍");
            }
        }
    }

    // 2. Billing & Electricity
    if (q.includes('省錢') || q.includes('台電') || q.includes('怕熱')) {
        if (prop.billing_type === "taipower") {
            reasons.push("⚡ 台電計費，省電好幫手");
        }
    }

    // 3. Service & Convenience (Garbage/Parcels)
    if (q.includes('垃圾') || q.includes('追車') || q.includes('子母車') || q.includes('方便')) {
        if (prop.service_level === "five_star") {
            reasons.push("✨ 免追垃圾車 + 代收包裹");
        } else if (prop.service_level === "basic" || prop.has_waste_disposal) {
            reasons.push("🧹 設有子母車，丟垃圾免煩惱");
        }
    }

    // 4. Distance & Geo Tier
    // 移除:geo_tier 在現有爬蟲資料退化(701/704=core,3=active),無法據以生成可信標籤。
    // 距離訊號改由 OSRM 通勤距離(distance / walk_mins)在排序層處理,不在此產生臆測標籤。

    // 5. Condition & Aesthetics
    if (q.includes('漂亮') || q.includes('質感') || q.includes('新') || q.includes('裝潢')) {
        if (prop.condition === "new") {
            reasons.push("🏠 全新首租，質感第一手");
        } else if (prop.condition === "renovated") {
            reasons.push("🎨 精緻裝潢，充滿設計感");
        }
    }

    // 6. Semantic rules — multi-trigger + explicit property check
    // Each rule: triggers[] (any match in query activates), check(pText, prop) verifies house has it
    const semanticRules = [
        {
            triggers: ['冰箱'],
            check: p => p.includes('冰箱'),
            label: '🧊 附有冰箱'
        },
        {
            triggers: ['洗衣機', '獨洗', '洗衣', '獨立洗'],
            check: p => p.includes('洗衣機') || p.includes('獨立洗'),
            label: '🫧 附獨立洗衣機'
        },
        {
            triggers: ['冷氣', '空調', '怕熱', '夏天熱'],
            check: p => p.includes('冷氣') || p.includes('空調'),
            label: '❄️ 房內附冷氣'
        },
        {
            triggers: ['電視', '看電視', '電視機', 'tv', '液晶', '追劇'],
            check: p => p.includes('電視') || p.includes('液晶'),
            label: '📺 房內附電視'
        },
        {
            triggers: ['廚房', '開伙', '煮飯', '瓦斯', '自炊', '炒菜', '料理', '在家煮', '自己煮', '開火', '煮東西'],
            check: p => p.includes('廚房') || p.includes('開伙') || p.includes('瓦斯') || p.includes('可自炊') || p.includes('電磁爐') || p.includes('排油煙') || p.includes('流理台') || p.includes('爐'),
            label: '🍳 可開伙自炊'
        },
        {
            triggers: ['電梯', '升降梯', '不爬樓', '不用爬', '不想爬', '不要爬', '腿不好', '膝蓋不好'],
            check: p => p.includes('電梯'),
            label: '🛗 有電梯，行動便利'
        },
        {
            triggers: ['貓', '狗', '養貓', '養狗', '帶貓', '帶狗', '毛小孩', '寵物', '貓咪', '狗狗', '可養'],
            check: (p, prop) => !p.includes('禁養') && !p.includes('不可養') && (
                p.includes('可養') || p.includes('寵物') || p.includes('友善') ||
                p.includes('可貓') || p.includes('可狗') || prop.has_pet
            ),
            label: '🐱 友善毛小孩環境'
        },
        {
            triggers: ['補助', '補貼', '租金補貼', '報稅', '入籍', '租屋補助'],
            check: (p, prop) => !p.includes('不可補助') && !p.includes('不可報稅') && !p.includes('不可入籍') && (
                prop.has_subsidy || p.includes('可補助') || p.includes('可報稅') || p.includes('可入籍') || p.includes('補助')
            ),
            label: '📑 可申請政府租金補貼'
        },
        {
            triggers: ['陽台', '晾衣', '晾曬', '晾衫'],
            check: p => p.includes('陽台') || p.includes('晾衣'),
            label: '☀️ 有私人陽台可晾衣'
        },
        {
            triggers: ['窗', '採光', '通風', '光線', '明亮'],
            check: p => p.includes('窗') || p.includes('採光') || p.includes('通風'),
            label: '🪟 採光通風對外窗'
        },
        {
            triggers: ['停車', '車位', '機車位', '腳踏車', '單車', '自行車', '機車'],
            check: p => p.includes('停車') || p.includes('車位') || p.includes('機車'),
            label: '🛵 附有機車停放空間'
        },
        {
            triggers: ['飲水機', '飲水', '開水'],
            check: p => p.includes('飲水機') || p.includes('飲水'),
            label: '💧 配有公共飲水機'
        },
        {
            triggers: ['門禁', '保全', '安全', '管理員', '管理室'],
            check: p => p.includes('門禁') || p.includes('保全') || p.includes('管理員'),
            label: '🔒 有門禁管理，安全有保障'
        },
        {
            triggers: ['網路', 'wifi', 'wi-fi', '無線', '寬頻', '含網路', '附網路'],
            check: p => p.includes('網路') || p.includes('wifi') || p.includes('寬頻'),
            label: '📶 含網路費用'
        },
        {
            triggers: ['熱水', '熱水器', '獨立熱水', '不搶熱水'],
            check: p => p.includes('熱水器') || p.includes('獨立熱水') || p.includes('瓦斯熱水'),
            label: '🚿 獨立熱水，不搶澡'
        },
        {
            triggers: ['全配', '家具', '家電', '附家電', '附家具'],
            check: p => (p.includes('冰箱') || p.includes('冷氣')) && (p.includes('床') || p.includes('桌')),
            label: '🛋️ 家具家電全配'
        },
        // ── 女生安全 ──────────────────────────────────
        {
            triggers: ['女生獨居', '獨居女', '女生住', '女生安全', '怕危險', '治安', '監視器', '女性友善'],
            check: p => p.includes('監視器') || p.includes('女性') || p.includes('門禁') || p.includes('管理員'),
            label: '🛡️ 女性友善 / 安全管理'
        },
        // ── 衛浴獨立 ──────────────────────────────────
        {
            triggers: ['不想共用廁所', '不想共廁', '個人衛浴', '獨立衛浴', '獨衛', '想泡澡', '浴缸'],
            check: p => p.includes('獨衛') || p.includes('獨立衛浴') || p.includes('浴缸') || p.includes('套房'),
            label: '🚿 獨立衛浴不共用'
        },
        // ── 租期彈性 ──────────────────────────────────
        {
            triggers: ['短租', '只租幾個月', '不確定租多久', '剛畢業', '工作不穩定', '彈性租期'],
            check: p => p.includes('短租') || p.includes('彈性') || p.includes('不限租期') || p.includes('月租'),
            label: '📅 租期彈性不限長短'
        },
        // ── 合租 / 室友 ───────────────────────────────
        {
            triggers: ['找室友', '想合租', '不想一個人住', '合租', '分租'],
            check: p => p.includes('室友') || p.includes('合租') || p.includes('分租') || p.includes('雅房'),
            label: '👥 可合租 / 室友同住'
        },
        {
            triggers: ['一個人住', '不想跟人共用', '獨住'],
            check: p => p.includes('套房') || p.includes('獨衛') || p.includes('獨立'),
            label: '🏠 獨立套房不共用'
        },
        // ── 交通通勤 ── 移除:check 的 公車/捷運/交通/生活機能 在爬蟲資料 0 命中,
        //    此規則永遠無法 truthy(dead code)。通勤訊號改由 OSRM distance 處理。
        // ── 在家工作 / WFH ────────────────────────────
        {
            triggers: ['在家工作', 'WFH', '遠距工作', '居家辦公', '書桌', '打報告', '念書', '讀書'],
            check: p => p.includes('書桌') || p.includes('寬頻') || p.includes('網路') || p.includes('安靜'),
            label: '💻 適合居家辦公 / 讀書'
        },
        // ── 預算暗示 ──────────────────────────────────
        {
            triggers: ['學生', '剛出社會', '薪水不多', '不要太貴', '便宜', '省錢', '實惠'],
            check: (p, prop) => prop.cp_tag === 'high_cp' || p.includes('學生') || p.includes('實惠') || p.includes('經濟'),
            label: '💰 經濟實惠 / 學生友善'
        },
        // ── 採光朝向 ──────────────────────────────────
        {
            triggers: ['不要西曬', '採光', '東向', '南向', '對外窗', '明亮'],
            check: p => p.includes('採光') || p.includes('對外窗') || p.includes('東向') || p.includes('南向'),
            label: '🌤️ 採光佳 / 無西曬'
        },
        // ── 安靜 / 隔音 ───────────────────────────────
        {
            triggers: ['怕吵', '安靜', '隔音', '靜巷'],
            check: p => p.includes('隔音') || p.includes('靜巷') || p.includes('氣密') || p.includes('安靜'),
            label: '🔇 安靜隔音佳'
        },
        // ── 夜貓子 / 無門禁 ───────────────────────────
        {
            triggers: ['夜貓子', '作息晚', '晚歸', '無門禁', '24小時'],
            check: p => p.includes('無門禁') || p.includes('24小時') || p.includes('自由進出') || p.includes('不限'),
            label: '🌙 無門禁限制 / 自由進出'
        }
    ];

    semanticRules.forEach(rule => {
        if (reasons.length >= 3) return;
        if (reasons.includes(rule.label)) return;
        const triggered = rule.triggers.some(t => q.includes(t));
        if (!triggered) return;
        if (rule.check(pText, prop)) {
            reasons.push(rule.label);
        }
    });

    // Default highlights if empty
    if (reasons.length === 0) {
        if (prop.cp_tag === "high_cp") reasons.push("💎 區域高 CP 值選");
        if (prop.service_level === "five_star") reasons.push("✨ 高品質社區管理");
        if (prop.billing_type === "taipower") reasons.push("⚡ 電費照台電撥款");
    }

    return [...new Set(reasons)].slice(0, 3);
}

export function checkConflicts(prop, constraints) {
    const { wantsPet, wantsRoomType } = constraints;
    const pText = buildPropText(prop);

    // 1. Room Type Mismatch
    if (wantsRoomType && prop.room_type && prop.room_type !== wantsRoomType) {
        return `此房源為${prop.room_type}，您指定的是${wantsRoomType}`;
    }

    // 2. Pet Conflict
    if (wantsPet && (pText.includes('禁養') || pText.includes('不可養'))) {
        return "此房源禁養寵物";
    }

    // 3. Gender Conflict
    if (constraints.hasGenderMention && constraints.originalText) {
        const orig = constraints.originalText;
        const isMale = orig.includes('男生') || orig.includes('男士') || orig.includes('男性');
        const isFemale = orig.includes('女生') || orig.includes('女士') || orig.includes('女性');
        const isFemaleOnly = pText.includes('限女');
        const isMaleOnly = pText.includes('限男');
        if (isMale && isFemaleOnly) return "此房源僅限女性";
        if (isFemale && isMaleOnly) return "此房源僅限男性";
    }

    // 4. Smoking
    if (constraints.originalText?.includes('抽菸') && (pText.includes('禁菸') || pText.includes('禁止吸菸'))) {
        return "此房源禁止吸菸";
    }

    // 5. 軟性設施不符 — 這些條件在 filterHardExclusions 不硬篩(註:595-597「REMOVED HARD
    //    CONTINUES」,改交給 rule/AI scoring),所以不符的房源仍會出現在結果裡。用 bool 結構欄
    //    『嚴格 === false』(房源明確沒有)才標不符,避免「軟性提及」(parser 對句中含「窗/陽台」
    //    即 requireWindow/Balcony=true)誤標。覆蓋使用者最常問且 CE 富化解鎖的特徵。
    const {
        requireElevator, requireBalcony, requireWindow, requireParking,
        requireWaste, requireCooking,
    } = constraints;

    if (requireElevator && prop.has_elevator === false) return "此房源無電梯";
    if (requireBalcony && prop.has_balcony === false) return "此房源無陽台";
    if (requireWindow && prop.has_window === false) return "此房源無對外窗";
    if (requireParking && prop.has_parking === false) return "此房源無車位";
    if (requireWaste && prop.has_waste_disposal === false) return "此房源無垃圾代收";
    // 可開伙:無 bool 欄,看 notes 是否含「可開伙」;含禁炊字樣或完全沒提 → 不符。
    if (requireCooking) {
        const canCook = (prop.notes || []).some(n => n.includes('可開伙'))
            || pText.includes('可開伙') || pText.includes('可開火');
        const banCook = pText.includes('禁開伙') || pText.includes('不可開伙')
            || pText.includes('不可開火') || pText.includes('禁炊');
        if (banCook || !canCook) return "此房源不可開伙";
    }

    return null;
}

