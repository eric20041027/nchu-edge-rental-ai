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

    # Cooking / self-catering
    "想下廚":   ["可伙", "廚房", "抽油煙機", "瓦斯爐"],
    "要下廚":   ["可伙", "廚房", "抽油煙機", "瓦斯爐"],
    "自己煮":   ["廚房", "瓦斯", "開火"],
    "省伙食費": ["廚房", "瓦斯", "開火"],
    "可以煮東西":["可伙", "廚房"],

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

    # Noise / peace
    "怕吵":     ["隔音", "氣密窗", "禁菸", "靜巷"],
    "安靜":     ["靜巷", "隔音"],

    # Work / study
    "打報告":   ["寬頻", "網路", "書桌"],
    "上網":     ["寬頻", "網路"],

    # Elevator
    "不想爬樓梯": ["電梯", "大樓", "華廈"],
    "搬東西":   ["電梯"],

    # Life quality
    "高品質":   ["管理員", "電梯", "漂亮", "全新"],
    "首租":     ["全新"],
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
