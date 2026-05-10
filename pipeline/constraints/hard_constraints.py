"""Hard constraint filtering — one-strike elimination before AI re-ranking."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedQuery:
    """Structured constraints parsed from natural-language query text."""
    raw: str = ""
    max_budget: int | None = None
    min_budget: int | None = None
    budget_strict: bool = False          # 以下/以內 → strict upper bound
    wants_taipower: bool = False         # 台電計費
    wants_taiwater: bool = False         # 台水計費
    max_elec_price: float | None = None  # 每度上限 (元)
    wants_pet: bool = False
    gender_male_ok: bool = True
    gender_female_ok: bool = True
    exclude_rooftop: bool = False
    exclude_wooden: bool = False
    max_walk_mins: int | None = None
    max_scooter_mins: int | None = None
    required_features: list[str] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "ParsedQuery":
        q = cls(raw=text)
        t = (text
             .replace("一", "1").replace("二", "2").replace("兩", "2")
             .replace("三", "3").replace("四", "4").replace("五", "5")
             .replace("六", "6").replace("七", "7").replace("八", "8")
             .replace("九", "9").replace("十", "10").replace("半", "30"))

        # Budget
        t2 = re.sub(r"(\d+(?:\.\d+)?)萬(\d*)",
                    lambda m: str(int(float(m.group(1)) * 10000) + (int(m.group(2)) * 1000 if m.group(2) else 0)), t)
        t2 = re.sub(r"(\d+)[千kK]", lambda m: str(int(m.group(1)) * 1000), t2)
        rng = re.search(r"(\d{3,})\s*[-~～至到]\s*(\d{3,})", t2)
        if rng:
            q.min_budget, q.max_budget = int(rng.group(1)), int(rng.group(2))
        else:
            m = re.search(r"(\d{3,})", t2)
            if m:
                q.max_budget = int(m.group(1))

        if re.search(r"以下|以內|不超過|上限|最多|不能超", text):
            q.budget_strict = True

        # Utility billing
        q.wants_taipower = bool(re.search(r"台電|獨立電錶|獨立電表", text))
        q.wants_taiwater = bool(re.search(r"台水", text))
        em = re.search(r"度\s*(\d+(?:\.\d+)?)\s*[元塊]", text)
        if em:
            q.max_elec_price = float(em.group(1))

        # Pets
        q.wants_pet = bool(re.search(r"養貓|養狗|寵物|貓|狗|毛孩", text))

        # Gender
        if re.search(r"限女|女性專用", text):
            q.gender_male_ok = False
        if re.search(r"限男|男性專用", text):
            q.gender_female_ok = False

        # Exclusions
        neg = r"(?:謝絕|不要|拒絕|禁|不接受|不想|討厭|避免)"
        q.exclude_rooftop = bool(re.search(rf"{neg}[^。！？\n]*(頂加|加蓋|頂樓)", text))
        q.exclude_wooden  = bool(re.search(rf"{neg}[^。！？\n]*木板", text))

        # Commute
        wm = re.search(r"(?:走路|步行)[^\d]*(\d+)[^\d]*(?:分鐘|分)", t)
        if wm:
            q.max_walk_mins = int(wm.group(1))
        sm = re.search(r"(?:機車|騎車)[^\d]*(\d+)[^\d]*(?:分鐘|分)", t)
        if sm:
            q.max_scooter_mins = int(sm.group(1))

        return q


class HardConstraintFilter:
    """One-strike elimination: any violated hard constraint → property excluded."""

    ELEC_PRICE_SOFT_LIMIT = 5.0   # 台電標準約 4.x 元/度

    def filter(self, properties: list[dict[str, Any]], query: ParsedQuery) -> list[dict[str, Any]]:
        return [p for p in properties if self._passes(p, query)]

    def _passes(self, p: dict[str, Any], q: ParsedQuery) -> bool:
        text = (p.get("text", "") + " " + p.get("furniture", "") +
                " ".join(p.get("notes", []) if isinstance(p.get("notes"), list) else []))

        # 1. Budget (strict upper bound only)
        try:
            rent = int(p.get("rent", 0))
        except (ValueError, TypeError):
            rent = 0
        if q.budget_strict and q.max_budget and rent > q.max_budget:
            return False

        # 2. Pets
        if q.wants_pet and re.search(r"禁養|不可養|禁止養", text):
            return False

        # 3. Electricity price
        if q.max_elec_price:
            billing = p.get("electricity_billing", "")
            m = re.search(r"(\d+(?:\.\d+)?)", billing)
            if m and float(m.group(1)) > q.max_elec_price:
                return False

        # 4. Requires Taipower but property charges premium rate
        if q.wants_taipower and not q.max_elec_price:
            billing = p.get("electricity_billing", "")
            m = re.search(r"(\d+(?:\.\d+)?)", billing)
            if m and float(m.group(1)) >= self.ELEC_PRICE_SOFT_LIMIT:
                return False

        # 5. Structural exclusions
        if q.exclude_rooftop and (str(p.get("is_rooftop", "False")) == "True" or "頂加" in text):
            return False
        if q.exclude_wooden and str(p.get("is_wooden_partition", "False")) == "True":
            return False

        # 6. Gender
        if not q.gender_male_ok and re.search(r"限女|女性專用", text):
            pass   # user is female, female-only is OK
        if not q.gender_female_ok and re.search(r"限男|男性專用", text):
            pass   # user is male, male-only is OK
        if q.gender_male_ok and not q.gender_female_ok:
            if re.search(r"限女|女性專用", text):
                return False
        if q.gender_female_ok and not q.gender_male_ok:
            if re.search(r"限男|男性專用", text):
                return False

        # 7. Commute time (requires distance field in km)
        try:
            dist_km = float(p.get("distance", 0))
        except (ValueError, TypeError):
            dist_km = 0
        if dist_km > 0:
            if q.max_walk_mins is not None:
                walk_mins = round(dist_km / 0.075)
                if walk_mins > q.max_walk_mins + 3:
                    return False
            if q.max_scooter_mins is not None:
                scooter_mins = max(1, round(dist_km / 0.417))
                if scooter_mins > q.max_scooter_mins + 2:
                    return False

        return True
