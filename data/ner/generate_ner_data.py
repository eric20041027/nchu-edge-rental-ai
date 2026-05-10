"""Generate synthetic NER training data for LOC / BGT / FEAT token classification."""
import json
import random
from pathlib import Path

random.seed(42)

LOCATIONS = [
    (["南", "區"], "LOC"),
    (["北", "區"], "LOC"),
    (["西", "區"], "LOC"),
    (["東", "區"], "LOC"),
    (["中", "區"], "LOC"),
    (["西", "屯", "區"], "LOC"),
    (["北", "屯", "區"], "LOC"),
    (["南", "屯", "區"], "LOC"),
    (["台", "中", "市", "南", "區"], "LOC"),
    (["台", "中", "市", "北", "區"], "LOC"),
    (["台", "中", "市", "西", "區"], "LOC"),
    (["台", "中", "市", "東", "區"], "LOC"),
    (["中", "興", "大", "學", "附", "近"], "LOC"),
    (["興", "大", "路", "附", "近"], "LOC"),
    (["大", "英", "街", "附", "近"], "LOC"),
    (["台", "中", "火", "車", "站", "附", "近"], "LOC"),
    (["南", "投", "客", "運", "附", "近"], "LOC"),
    (["國", "光", "路", "附", "近"], "LOC"),
    (["學", "府", "路", "附", "近"], "LOC"),
    (["復", "興", "路", "附", "近"], "LOC"),
    (["民", "族", "路", "附", "近"], "LOC"),
    (["建", "成", "路", "附", "近"], "LOC"),
    (["五", "權", "路", "附", "近"], "LOC"),
    (["忠", "明", "南", "路", "附", "近"], "LOC"),
    (["台", "中", "市", "南", "區", "大", "英", "國", "小", "附", "近"], "LOC"),
]

BUDGETS = [
    (["四", "千", "以", "內"], "BGT"),
    (["四", "千", "五"], "BGT"),
    (["四", "千", "五", "百", "元"], "BGT"),
    (["五", "千", "以", "內"], "BGT"),
    (["五", "千", "以", "下"], "BGT"),
    (["五", "千", "元", "以", "內"], "BGT"),
    (["五", "千", "五"], "BGT"),
    (["五", "千", "五", "百", "元"], "BGT"),
    (["六", "千"], "BGT"),
    (["六", "千", "元"], "BGT"),
    (["六", "千", "以", "內"], "BGT"),
    (["六", "千", "以", "下"], "BGT"),
    (["六", "千", "五"], "BGT"),
    (["六", "千", "五", "百", "元"], "BGT"),
    (["七", "千"], "BGT"),
    (["七", "千", "元"], "BGT"),
    (["七", "千", "以", "內"], "BGT"),
    (["七", "千", "以", "下"], "BGT"),
    (["七", "千", "五"], "BGT"),
    (["七", "千", "五", "百", "元"], "BGT"),
    (["八", "千"], "BGT"),
    (["八", "千", "元"], "BGT"),
    (["八", "千", "以", "內"], "BGT"),
    (["八", "千", "五"], "BGT"),
    (["九", "千"], "BGT"),
    (["九", "千", "以", "內"], "BGT"),
    (["一", "萬"], "BGT"),
    (["一", "萬", "以", "內"], "BGT"),
    (["一", "萬", "元", "以", "內"], "BGT"),
    (["五", "千", "到", "七", "千"], "BGT"),
    (["六", "千", "到", "八", "千"], "BGT"),
    (["四", "千", "到", "六", "千"], "BGT"),
    (["七", "千", "到", "一", "萬"], "BGT"),
    (["預", "算", "六", "千"], "BGT"),
    (["預", "算", "七", "千"], "BGT"),
    (["預", "算", "八", "千"], "BGT"),
    (["月", "租", "六", "千"], "BGT"),
    (["月", "租", "七", "千"], "BGT"),
    (["月", "租", "八", "千"], "BGT"),
    (["租", "金", "六", "千"], "BGT"),
    (["租", "金", "七", "千"], "BGT"),
]

FEATURES = [
    (["冷", "氣"], "FEAT"),
    (["變", "頻", "冷", "氣"], "FEAT"),
    (["電", "梯"], "FEAT"),
    (["洗", "衣", "機"], "FEAT"),
    (["網", "路"], "FEAT"),
    (["光", "纖", "網", "路"], "FEAT"),
    (["停", "車", "位"], "FEAT"),
    (["機", "車", "位"], "FEAT"),
    (["獨", "立", "衛", "浴"], "FEAT"),
    (["獨", "衛"], "FEAT"),
    (["廚", "房"], "FEAT"),
    (["陽", "台"], "FEAT"),
    (["子", "母", "車"], "FEAT"),
    (["台", "電", "計", "費"], "FEAT"),
    (["台", "水", "台", "電"], "FEAT"),
    (["含", "電", "費"], "FEAT"),
    (["含", "水", "電"], "FEAT"),
    (["含", "管", "理", "費"], "FEAT"),
    (["家", "具"], "FEAT"),
    (["家", "電"], "FEAT"),
    (["含", "家", "具", "家", "電"], "FEAT"),
    (["冰", "箱"], "FEAT"),
    (["微", "波", "爐"], "FEAT"),
    (["書", "桌"], "FEAT"),
    (["床", "架"], "FEAT"),
    (["衣", "櫃"], "FEAT"),
    (["熱", "水", "器"], "FEAT"),
    (["保", "全"], "FEAT"),
    (["對", "講", "機"], "FEAT"),
    (["女", "生", "專", "用"], "FEAT"),
    (["男", "生", "專", "用"], "FEAT"),
    (["可", "養", "寵", "物"], "FEAT"),
    (["寵", "物", "友", "善"], "FEAT"),
    (["不", "要", "頂", "加"], "FEAT"),
    (["不", "要", "木", "板", "隔", "間"], "FEAT"),
    (["採", "光", "好"], "FEAT"),
    (["通", "風", "良", "好"], "FEAT"),
    (["環", "境", "安", "靜"], "FEAT"),
    (["近", "捷", "運"], "FEAT"),
    (["近", "公", "車", "站"], "FEAT"),
    (["近", "超", "商"], "FEAT"),
    (["近", "公", "園"], "FEAT"),
    (["第", "四", "台"], "FEAT"),
    (["有", "線", "電", "視"], "FEAT"),
    (["獨", "立", "門", "牌"], "FEAT"),
    (["儲", "藏", "室"], "FEAT"),
]

PREFIXES_LOC  = [["我", "想", "找"], ["想", "租"], ["需", "要"], ["尋", "找"], ["我", "要", "找"], []]
PREFIXES_BGT  = [["預", "算"], ["月", "租"], ["租", "金"], []]
PREFIXES_FEAT = [["要", "有"], ["需", "要", "有"], ["希", "望", "有"], ["要"], ["含"], []]

CONNECTORS = [["，"], ["，", "而", "且"], ["，", "並", "且"], ["，", "同", "時"], []]

def bio(tokens, entity_type):
    """Return BIO labels for a token list."""
    if not tokens:
        return []
    return [f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(tokens) - 1)

def make_sample(parts):
    """parts = list of (token_list, label_type | None)"""
    tokens, labels = [], []
    for tok_list, label_type in parts:
        tokens.extend(tok_list)
        if label_type:
            labels.extend(bio(tok_list, label_type))
        else:
            labels.extend(["O"] * len(tok_list))
    assert len(tokens) == len(labels)
    return {"tokens": tokens, "labels": labels}

def rand(lst):
    return random.choice(lst)

def generate_samples(n):
    samples = []

    # Pattern 1: LOC only
    for _ in range(n // 10):
        loc = rand(LOCATIONS)
        pre = rand(PREFIXES_LOC)
        suf = rand([["的", "套", "房"], ["的", "雅", "房"], ["租", "屋"], []])
        samples.append(make_sample([
            (pre, None), (loc[0], loc[1]), (suf, None)
        ]))

    # Pattern 2: BGT only
    for _ in range(n // 10):
        bgt = rand(BUDGETS)
        suf = rand([["的", "套", "房"], ["的", "雅", "房"], ["租", "屋"], []])
        samples.append(make_sample([
            (bgt[0], bgt[1]), (suf, None)
        ]))

    # Pattern 3: FEAT only
    for _ in range(n // 10):
        feat = rand(FEATURES)
        pre = rand(PREFIXES_FEAT)
        suf = rand([["的", "套", "房"], ["的", "房", "間"], []])
        samples.append(make_sample([
            (pre, None), (feat[0], feat[1]), (suf, None)
        ]))

    # Pattern 4: LOC + BGT
    for _ in range(n // 8):
        loc = rand(LOCATIONS)
        bgt = rand(BUDGETS)
        conn = rand(CONNECTORS)
        pre_l = rand(PREFIXES_LOC)
        suf = rand([["的", "套", "房"], ["的", "房"], []])
        if random.random() > 0.5:
            samples.append(make_sample([
                (pre_l, None), (loc[0], loc[1]), (conn, None), (bgt[0], bgt[1]), (suf, None)
            ]))
        else:
            samples.append(make_sample([
                (bgt[0], bgt[1]), (conn, None), (loc[0], loc[1]), (suf, None)
            ]))

    # Pattern 5: LOC + FEAT
    for _ in range(n // 8):
        loc = rand(LOCATIONS)
        feat = rand(FEATURES)
        conn = rand(CONNECTORS)
        pre_l = rand(PREFIXES_LOC)
        pre_f = rand(PREFIXES_FEAT)
        suf = rand([["套", "房"], ["房", "間"], []])
        if random.random() > 0.5:
            samples.append(make_sample([
                (pre_l, None), (loc[0], loc[1]), (conn, None),
                (pre_f, None), (feat[0], feat[1]), (suf, None)
            ]))
        else:
            samples.append(make_sample([
                (pre_f, None), (feat[0], feat[1]), (["的", "套", "房"], None),
                (loc[0], loc[1])
            ]))

    # Pattern 6: BGT + FEAT
    for _ in range(n // 8):
        bgt = rand(BUDGETS)
        feat = rand(FEATURES)
        conn = rand(CONNECTORS)
        pre_f = rand(PREFIXES_FEAT)
        suf = rand([["套", "房"], []])
        samples.append(make_sample([
            (bgt[0], bgt[1]), (conn, None), (pre_f, None), (feat[0], feat[1]), (suf, None)
        ]))

    # Pattern 7: LOC + BGT + FEAT (full)
    for _ in range(n // 4):
        loc = rand(LOCATIONS)
        bgt = rand(BUDGETS)
        feat1 = rand(FEATURES)
        feat2 = rand(FEATURES)
        conn = rand(CONNECTORS)
        pre_l = rand(PREFIXES_LOC)
        pre_f = rand(PREFIXES_FEAT)

        # shuffle order of loc/bgt
        order = random.choice(["loc_bgt", "bgt_loc"])
        if order == "loc_bgt":
            head = [(pre_l, None), (loc[0], loc[1]), (conn, None), (bgt[0], bgt[1])]
        else:
            head = [(bgt[0], bgt[1]), (conn, None), (loc[0], loc[1])]

        tail = [(conn, None), (pre_f, None), (feat1[0], feat1[1])]
        if random.random() > 0.6:
            tail += [(["和"], None), (feat2[0], feat2[1])]

        samples.append(make_sample(head + tail))

    # Pattern 8: two LOC
    for _ in range(n // 12):
        loc1 = rand(LOCATIONS)
        loc2 = rand(LOCATIONS)
        bgt = rand(BUDGETS)
        conn = ["或"]
        samples.append(make_sample([
            (loc1[0], loc1[1]), (conn, None), (loc2[0], loc2[1]),
            (["，"], None), (bgt[0], bgt[1])
        ]))

    # Pattern 9: two FEAT
    for _ in range(n // 12):
        loc = rand(LOCATIONS)
        feat1 = rand(FEATURES)
        feat2 = rand(FEATURES)
        bgt = rand(BUDGETS)
        samples.append(make_sample([
            (loc[0], loc[1]), (bgt[0], bgt[1]),
            (["含"], None), (feat1[0], feat1[1]),
            (["和"], None), (feat2[0], feat2[1])
        ]))

    random.shuffle(samples)
    return samples


if __name__ == "__main__":
    out = Path(__file__).parent
    train = generate_samples(700)
    dev   = generate_samples(150)

    (out / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dev.json").write_text(json.dumps(dev,   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成 train: {len(train)} 筆，dev: {len(dev)} 筆")
