/**
 * constraint-parser.js — 從 query 文字 / NER 抽出硬約束。
 *
 * 架構精簡批次5 從 inference.js 抽出。兩個純函式(零模組依賴):
 *   parseConstraintsFromText(text) → constraints object
 *   parseBudgetFromNER(budgetSpans) → {budget, limit} | null(NER 補強用)
 * 只被 recommend() 呼叫。
 */

export function parseConstraintsFromText(text) {
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
    const negativeWords = "(謝絕|不要|拒絕|禁|❌|不接受|不想|討厭|避免|不要有|不要找)";
    if (text.match(new RegExp(`${negativeWords}[^。！？\\n]*(頂加|加蓋|頂樓)`))) excludeRooftop = true;
    if (text.match(new RegExp(`${negativeWords}[^。！？\\n]*木板`))) excludeWooden = true;
    if (text.match(new RegExp(`${negativeWords}[^。！？\\n]*凶宅`))) excludeHaunted = true;

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
                if (val >= 1000) {
                    // 4 位數以上幾乎必為租金本身,直接採用。
                    budget = val;
                    hasBudgetMention = true;
                } else if (val < 100) {
                    // 小數字(如「3」「8」)解讀成「3千/8千」風險高:中文數字正規化會把
                    // 「一點(明亮一點)」「三樓」「住一下」的『一/三』也轉成數字,誤判成預算 →
                    // 把全部房源篩掉(此 bug 來源)。僅在有明確預算語境詞時才採用。
                    const hasBudgetCue = /預算|月租|租金|房租|元|塊|[kK]|千|萬|以下|以內|以上/.test(text);
                    if (hasBudgetCue) {
                        budget = val * 1000;
                        hasBudgetMention = true;
                    }
                }
            }
        }
    }

    let wantsRoomType = null;
    if (text.includes('套房')) { hasRoomTypeMention = true; wantsRoomType = '套房'; }
    else if (text.includes('雅房')) { hasRoomTypeMention = true; wantsRoomType = '雅房'; }
    else if (text.includes('工作室')) { hasRoomTypeMention = true; wantsRoomType = '工作室'; }

    // 寵物意圖:區分「要養」vs「不要養」。舊版 wantsPet 只要含『寵物/養貓/養狗』就 true,
    // 把「不要養寵物 / 禁養寵物 / 討厭寵物」誤判成想養 → 反而推可養寵物房源(語意相反)。
    // 偵測否定詞修飾寵物關鍵字 → excludePet(排除可養寵物房源),且此時 wantsPet 不成立。
    const petMention = text.includes('養貓') || text.includes('養狗') || text.includes('寵物') || text.includes('毛小孩') || text.includes('毛孩');
    const petNegated = /(不要|不想|不可|沒有|別|禁|討厭|避免|謝絕|拒絕|怕)[^。！？\n]{0,4}(養貓|養狗|養寵|寵物|毛小孩|毛孩|貓|狗)/.test(text);
    const excludePet = petMention && petNegated;
    const wantsPet = petMention && !petNegated;

    return {
        budget, minBudget, maxBudget, limit, genderUnrestricted, hasGenderMention, hasBudgetMention, hasRoomTypeMention, wantsRoomType,
        wantsUtilityBilling, maxElectricityPrice, requireBalcony, requireWindow, requireParking, requireWaste,
        requireSubsidy, isSocialHousing, excludePet,
        excludeRooftop, excludeWooden, excludeHaunted, maxWalkMins, maxScooterMins,
        wantsPet,
        requireElevator: (text.includes('電梯') || text.includes('升降梯') || text.includes('不爬樓') || text.includes('不用爬') || text.includes('不想爬') || text.includes('不要爬') || text.includes('腿不好') || text.includes('膝蓋不好')),
        requireCooking: (text.includes('開伙') || text.includes('開火') || text.includes('自炊') || text.includes('煮飯') || text.includes('炒菜') || text.includes('在家煮') || text.includes('自己煮')),
        requireWaterDispenser: (text.includes('飲水機')),
        requirePrivateWasher: (text.includes('獨洗') || text.includes('個人洗衣機')),
        requireGuard: (text.includes('代收') || text.includes('包裹') || text.includes('管理員') || text.includes('警衛')),
        originalText: text
    };
}

// --- Explainability: Match Reasons & Conflict Detection ---


export function parseBudgetFromNER(budgetSpans) {
    if (!budgetSpans || budgetSpans.length === 0) return null;
    let budget = null;
    let limit = null;

    for (const span of budgetSpans) {
        // Detect direction from original span
        if (span.includes('以上')) limit = 'above';
        else if (span.includes('以下') || span.includes('以內') || span.includes('內')) limit = limit || 'below';

        let s = span
            .replace(/[一１]/g, '1').replace(/[二２兩]/g, '2').replace(/[三３]/g, '3')
            .replace(/[四４]/g, '4').replace(/[五５]/g, '5').replace(/[六６]/g, '6')
            .replace(/[七７]/g, '7').replace(/[八８]/g, '8').replace(/[九９]/g, '9')
            .replace(/十/g, '10');

        // Handle 萬 notation first (e.g., 1萬2 → 12000)
        const wanMatch = s.match(/(\d+(?:\.\d+)?)萬(\d*)/);
        if (wanMatch) {
            const candidate = parseFloat(wanMatch[1]) * 10000 + (wanMatch[2] ? parseInt(wanMatch[2]) * 1000 : 0);
            if (candidate > 0) { budget = candidate; continue; }
        }
        // Handle 千 / k / K
        s = s.replace(/千/g, '000').replace(/[kK]/g, '000');
        const numMatch = s.match(/(\d{3,})/);
        if (numMatch) {
            const candidate = parseInt(numMatch[1]);
            if (candidate >= 1000) budget = candidate;
        }
    }
    return budget ? { budget, limit: limit || 'below' } : null;
}
