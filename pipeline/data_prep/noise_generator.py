"""
noise_generator.py — Generate a noisy version of the test set for Group D evaluation.

Usage:
    python -m pipeline.data_prep.noise_generator

Output:
    data/processed/noisy_test.json  (same schema as recommendation_test.json)

Noise types (1-2 applied per sample, seed=42):
    1. Abbreviation substitution  (縮寫替換)
    2. Typo injection             (錯字注入, ~10% chars)
    3. Colloquial/oral style      (口語化 — fillers prepend/append + phrase replacement)
    4. Number format variation    (數字格式 — Chinese numerals or k-notation)
"""
import json
import os
import random
import re
from collections import Counter
from typing import List

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

INPUT_PATH  = os.path.join(_PROJECT_ROOT, "data", "processed", "recommendation_test.json")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "noisy_test.json")


# ── Noise type 1: Abbreviation substitution ───────────────────────────────────

ABBREV_MAP = {
    "中興大學": "興大",
    "台中市":   "台中",
    "台中":     "中",
    "套房":     "套",
    "雅房":     "雅",
    "月租":     "租",
    "有洗衣機": "有洗衣",
    "冷氣":     "冷",
    "台電計費": "台電",
    "保全系統": "保全",
    "以內":     "內",
    "以上":     "上",
}

# Sort by length descending so longer keys are matched first (greedy)
_ABBREV_SORTED = sorted(ABBREV_MAP.items(), key=lambda x: -len(x[0]))


def apply_abbreviation(text: str) -> str:
    """Replace full terms with their abbreviations."""
    for full, short in _ABBREV_SORTED:
        text = text.replace(full, short)
    return text


# ── Noise type 2: Typo injection ──────────────────────────────────────────────

TYPO_MAP = {
    "的": "滴",
    "找": "揾",
    "要": "ㄠˋ",
    "租": "粗",
    "房": "仿",
    "近": "禁",
    "電": "點",
    "費": "廢",
    "設": "射",
    "備": "必",
    "台": "臺",
    "區": "取",
}

TYPO_RATE = 0.10   # ~10% of eligible characters get substituted


def apply_typos(text: str, rng: random.Random) -> str:
    """Replace ~10% of characters that appear in TYPO_MAP with their typo."""
    chars = list(text)
    eligible_indices = [i for i, c in enumerate(chars) if c in TYPO_MAP]
    n_typos = max(1, round(len(eligible_indices) * TYPO_RATE))
    chosen  = rng.sample(eligible_indices, min(n_typos, len(eligible_indices)))
    for i in chosen:
        chars[i] = TYPO_MAP[chars[i]]
    return "".join(chars)


# ── Noise type 3: Colloquial / oral style ─────────────────────────────────────

PREPEND_FILLERS = ["幫我找", "推薦一下", "有沒有", "想問", "請問"]
APPEND_FILLERS  = ["這樣的房子", "的地方", "ㄟ", "啊", "那種的"]

ORAL_REPLACEMENTS = [
    ("我要",  "我想要"),
    ("需要",  "想要"),
    ("找",    "找找看"),
]


def apply_colloquial(text: str, rng: random.Random) -> str:
    """Prepend or append a filler phrase, and apply oral replacements."""
    # Phrase substitutions (apply all that match)
    for src, dst in ORAL_REPLACEMENTS:
        text = text.replace(src, dst)

    # Randomly prepend or append a filler
    action = rng.choice(["prepend", "append"])
    if action == "prepend":
        text = rng.choice(PREPEND_FILLERS) + text
    else:
        text = text + rng.choice(APPEND_FILLERS)
    return text


# ── Noise type 4: Number format variation ────────────────────────────────────

DIGIT_MAP = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}

# Well-known rent ranges for named conversions
_NAMED_AMOUNTS = {
    5000: "五千",
    6000: "六千",
    4500: "四千五",
    4000: "四千",
    7000: "七千",
    8000: "八千",
    3000: "三千",
    3500: "三千五",
    5500: "五千五",
    6500: "六千五",
    7500: "七千五",
    8500: "八千五",
    9000: "九千",
    10000: "一萬",
}


def _to_chinese_numerals(num_str: str) -> str:
    """Convert a digit string to Chinese numerals character by character."""
    return "".join(DIGIT_MAP.get(c, c) for c in num_str)


def _number_replacer(match: re.Match, rng: random.Random) -> str:
    num_str = match.group(0)
    try:
        val = int(num_str)
    except ValueError:
        return num_str

    choices = [num_str]   # always include original as a fallback

    # Named amount
    if val in _NAMED_AMOUNTS:
        choices.append(_NAMED_AMOUNTS[val])

    # k-notation (multiples of 1000)
    if val > 0 and val % 1000 == 0:
        choices.append(f"{val // 1000}k")

    # Chinese numeral digit-by-digit
    choices.append(_to_chinese_numerals(num_str))

    return rng.choice(choices)


def apply_number_variation(text: str, rng: random.Random) -> str:
    """Replace numeric sequences in text with alternative formats."""
    return re.sub(
        r"\d+",
        lambda m: _number_replacer(m, rng),
        text,
    )


# ── Apply noise to a single query ────────────────────────────────────────────

ALL_NOISE_TYPES = [
    "abbreviation",
    "typo",
    "colloquial",
    "number",
]


def apply_noise(query: str, noise_types: List[str], rng: random.Random) -> str:
    """Apply a sequence of noise types to the query string."""
    text = query
    for nt in noise_types:
        if nt == "abbreviation":
            text = apply_abbreviation(text)
        elif nt == "typo":
            text = apply_typos(text, rng)
        elif nt == "colloquial":
            text = apply_colloquial(text, rng)
        elif nt == "number":
            text = apply_number_variation(text, rng)
    return text


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(42)

    print(f"Loading test set from: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  {len(test_data)} samples loaded.")

    noise_counter: Counter = Counter()
    noisy_data = []

    for sample in test_data:
        n_types = rng.choice([1, 2])
        chosen_types = rng.sample(ALL_NOISE_TYPES, n_types)
        noise_counter.update(chosen_types)

        noisy_query = apply_noise(sample["query"], chosen_types, rng)

        noisy_sample = dict(sample)          # shallow copy — only query changes
        noisy_sample["query"]       = noisy_query
        noisy_sample["noise_types"] = chosen_types
        noisy_data.append(noisy_sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(noisy_data, f, ensure_ascii=False, indent=2)

    print(f"\nNoisy test set saved to: {OUTPUT_PATH}")
    print(f"  Total samples: {len(noisy_data)}")
    print(f"\nNoise type distribution:")
    total_applications = sum(noise_counter.values())
    for nt, count in sorted(noise_counter.items()):
        pct = 100 * count / total_applications if total_applications else 0
        print(f"    {nt:<15} {count:>5} ({pct:.1f}%)")

    # Quick sanity check: how many queries were actually changed?
    changed = sum(1 for orig, noisy in zip(test_data, noisy_data)
                  if orig["query"] != noisy["query"])
    print(f"\n  Queries changed: {changed}/{len(test_data)} "
          f"({100*changed/len(test_data):.1f}%)")


if __name__ == "__main__":
    main()
