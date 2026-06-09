"""LifestyleMapper — maps colloquial lifestyle intent phrases to property features.

Kept in sync with frontend/js/inference-worker.js semanticExpandQuery()
and frontend/js/inference.js expandQueryIntent().
"""
from __future__ import annotations


# 15+ lifestyle clusters (query phrase → implied property features)
LIFESTYLE_CLUSTERS: dict[str, list[str]] = {
    # Cleanliness / OCD
    "潔癖":     ["全新", "獨洗", "禁菸", "乾淨", "裝潢"],
    "愛乾淨":   ["全新", "獨洗", "禁菸", "乾淨"],
    "稍微潔癖": ["全新", "獨洗", "禁菸"],

    # Cooking / self-catering — extended coverage
    "想在家煮飯": ["可伙", "廚房", "流理台", "瓦斯爐", "電磁爐", "開火"],
    "想自己煮飯": ["可伙", "廚房", "流理台", "瓦斯爐", "開火"],
    "在家開伙":   ["可伙", "廚房", "抽油煙機", "流理台"],
    "想下廚":     ["可伙", "廚房", "抽油煙機", "瓦斯爐"],
    "要下廚":     ["可伙", "廚房", "抽油煙機", "瓦斯爐"],
    "喜歡下廚":   ["可伙", "廚房", "抽油煙機", "瓦斯爐", "流理台"],
    "喜歡自己煮": ["可伙", "廚房", "流理台", "瓦斯爐"],
    "自己煮":     ["廚房", "瓦斯", "開火", "流理台"],
    "自炊":       ["可伙", "廚房", "流理台", "電磁爐", "開火"],
    "省伙食費":   ["廚房", "瓦斯", "開火", "流理台"],
    "省餐費":     ["可伙", "廚房", "流理台"],
    "不想外食":   ["可伙", "廚房", "流理台", "電磁爐"],
    "不吃外食":   ["可伙", "廚房", "流理台", "瓦斯爐"],
    "可以煮東西": ["可伙", "廚房"],
    "要能煮飯":   ["可伙", "廚房", "流理台", "電磁爐"],
    "煮飯":       ["可伙", "廚房", "流理台"],
    "開火":       ["可伙", "廚房", "瓦斯爐", "電磁爐"],
    "要有廚房":   ["廚房", "流理台", "可伙"],
    "有瓦斯":     ["天然瓦斯", "瓦斯爐", "可伙"],
    "天然瓦斯":   ["天然瓦斯", "瓦斯爐", "可伙", "廚房"],

    # Heat / air conditioning
    "怕熱":     ["冷氣", "變頻", "吹冷氣"],
    "夏天":     ["冷氣"],
    "西曬":     ["遮陽", "窗簾", "隔熱"],

    # Ventilation / light
    "怕悶熱":   ["陽台", "採光", "通風", "對外窗"],
    "採光好":   ["落地窗", "採光", "對外窗"],
    "網美":     ["裝潢", "採光", "漂亮", "落地窗"],

    # Laundry
    "獨洗獨曬": ["洗衣機", "陽台", "曬衣", "獨洗"],

    # Parking / vehicle
    "有車":     ["車位", "停車場"],
    "開車":     ["車位", "停車場"],

    # Pets
    "可貓":     ["可寵", "養寵", "寵物友善"],
    "可狗":     ["可寵", "養寵", "寵物友善"],
    "有毛孩":   ["可寵", "寵物"],

    # Utility billing
    "台水電":   ["台電", "台水"],
    "省電費":   ["變頻", "台電"],

    # Convenience / lazy
    "懶人":     ["電梯", "子母車", "垃圾處理", "飲水機"],
    "外送族":   ["管理員", "飲水機", "子母車"],
    "不想出門": ["管理員", "飲水機", "子母車"],
    "不想追垃圾車": ["子母車", "垃圾處理"],

    # Noise / peace / night owls
    "怕吵":     ["隔音", "氣密窗", "禁菸", "靜巷"],
    "安靜":     ["靜巷", "隔音"],
    "夜貓子":   ["無門禁", "24小時", "自由進出"],
    "作息晚":   ["無門禁", "24小時", "自由進出"],
    "晚歸":     ["門禁", "管理員", "安全", "刷卡"],

    # Female safety
    "女生獨居": ["管理員", "門禁", "監視器", "女性友善", "安全"],
    "女生住":   ["管理員", "門禁", "監視器", "安全"],
    "獨居女":   ["管理員", "門禁", "監視器", "女性友善"],
    "女生安全": ["管理員", "門禁", "監視器", "安全"],
    "怕危險":   ["管理員", "門禁", "監視器", "安全"],
    "治安":     ["管理員", "門禁", "監視器", "靜巷", "安全"],

    # Move-in ready / furniture
    "拎包入住":   ["全配", "全家具", "全家電", "冰箱", "洗衣機", "床"],
    "不想買家具": ["全配", "全家具", "家具齊全"],
    "什麼都有":   ["全配", "全家具", "全家電", "冰箱", "洗衣機"],
    "家電齊全":   ["冰箱", "洗衣機", "冷氣", "全家電"],
    "要有冰箱":   ["冰箱", "全配"],
    "要有書桌":   ["書桌", "書桌椅"],
    "要有床":     ["床架", "床墊", "全配"],
    "空屋":       ["空屋", "自備家具"],

    # Private bathroom
    "不想共用廁所": ["獨衛", "獨立衛浴", "套房"],
    "不想共廁":     ["獨衛", "獨立衛浴", "套房"],
    "個人衛浴":     ["獨衛", "獨立衛浴"],
    "獨立衛浴":     ["獨衛", "套房"],
    "想泡澡":       ["浴缸", "獨衛"],
    "要有熱水":     ["熱水器", "天然瓦斯熱水器", "電熱水器"],

    # Flexible lease
    "短租":         ["短期", "彈性租期", "月租", "不限租期"],
    "只租幾個月":   ["短租", "彈性租期", "不限租期"],
    "不確定租多久": ["彈性租期", "短租", "月租"],
    "剛畢業":       ["短租", "彈性", "經濟實惠"],
    "工作不穩定":   ["彈性租期", "短租"],

    # Roommate / sharing
    "找室友":       ["雅房", "分租", "室友", "合租"],
    "想合租":       ["雅房", "分租", "室友", "合租"],
    "不想一個人住": ["雅房", "分租", "室友"],
    "一個人住":     ["獨立套房", "獨衛", "獨廁", "套房"],
    "不想跟人共用": ["獨立套房", "獨衛", "套房"],

    # Transportation
    "騎車上班":   ["機車停車位", "停車"],
    "通勤":       ["近公車", "近捷運", "交通便利"],
    "沒有車":     ["近公車", "生活機能", "便利商店", "交通便利"],
    "不開車":     ["近公車", "近捷運", "生活機能"],
    "上班方便":   ["交通便利", "近公車", "近捷運"],

    # Orientation / balcony
    "不要西曬":   ["非西向", "東向", "北向", "採光"],
    "要有陽台":   ["陽台", "曬衣", "採光", "通風"],
    "不要頂樓":   ["非頂樓", "非頂加"],
    "頂樓加蓋":   ["頂加"],

    # WFH / remote work
    "在家工作":   ["網路", "寬頻", "書桌", "安靜"],
    "WFH":        ["網路", "寬頻", "書桌", "安靜"],
    "遠距工作":   ["網路", "寬頻", "書桌", "安靜"],
    "居家辦公":   ["網路", "寬頻", "書桌", "安靜"],

    # Budget hints
    "學生":       ["學生套房", "經濟實惠", "低價"],
    "剛出社會":   ["經濟實惠", "低價", "套房"],
    "薪水不多":   ["經濟實惠", "低租金", "實惠"],
    "不要太貴":   ["實惠", "低租金", "經濟"],
    "便宜":       ["低租金", "經濟實惠"],

    # Work / study
    "打報告":   ["寬頻", "網路", "書桌"],
    "上網":     ["寬頻", "網路"],
    "念書":     ["書桌", "書桌椅", "安靜", "寬頻"],
    "讀書":     ["書桌", "書桌椅", "安靜", "寬頻"],

    # Elevator / mobility
    "不想爬樓梯": ["電梯", "大樓", "華廈"],
    "搬東西":     ["電梯"],
    "膝蓋不好":   ["電梯", "大樓"],
    "機車":       ["機車停車位"],

    # Life quality
    "高品質":   ["管理員", "電梯", "漂亮", "全新"],
    "首租":     ["全新"],
    "健身":     ["健身房", "交誼廳"],
}


class LifestyleMapper:
    """Maps lifestyle intent phrases in a query to implied property feature keywords."""

    def __init__(self, clusters: dict[str, list[str]] | None = None):
        self.clusters = clusters or LIFESTYLE_CLUSTERS

    def expand_query(self, query: str) -> str:
        """Append inferred feature keywords to the query string."""
        extras: list[str] = []
        for phrase, features in self.clusters.items():
            if phrase in query:
                for f in features:
                    if f not in extras:
                        extras.append(f)
        if extras:
            return query + " " + " ".join(extras)
        return query

    def infer_features(self, query: str) -> list[str]:
        """Return list of implied property features for this query."""
        features: list[str] = []
        for phrase, feats in self.clusters.items():
            if phrase in query:
                for f in feats:
                    if f not in features:
                        features.append(f)
        return features

    def matched_clusters(self, query: str) -> list[str]:
        """Return list of lifestyle cluster names triggered by this query."""
        return [phrase for phrase in self.clusters if phrase in query]
