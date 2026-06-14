"""Single source of truth for semantic expansion rules.

Edit ONLY data/semantic_rules.json, then run:
    python pipeline/data_prep/sync_semantic_rules.py

This regenerates the rule tables in:
  - pipeline/data_prep/lifestyle_mapper.py   (LIFESTYLE_CLUSTERS dict)
  - frontend/js/inference-worker.js          (semanticExpandQuery expansionMap)
  - frontend/js/inference.js                 (expandQueryIntent intentMap)

Each target file has BEGIN/END marker comments delimiting the generated block;
content between the markers is overwritten. Do NOT hand-edit between markers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "data" / "semantic_rules.json"

PY_FILE = ROOT / "pipeline" / "data_prep" / "lifestyle_mapper.py"
WORKER_FILE = ROOT / "frontend" / "js" / "inference-worker.js"
INFER_FILE = ROOT / "frontend" / "js" / "inference.js"


def load_rules() -> dict[str, list[str]]:
    return json.loads(CANON.read_text(encoding="utf-8"))


def replace_block(text: str, begin: str, end: str, body: str) -> str:
    pattern = re.compile(
        re.escape(begin) + r".*?" + re.escape(end), re.DOTALL
    )
    if not pattern.search(text):
        raise ValueError(f"markers not found: {begin!r} .. {end!r}")
    return pattern.sub(begin + "\n" + body + "\n    " + end, text)


def gen_py(rules: dict[str, list[str]]) -> str:
    maxk = max(len(k) for k in rules)
    lines = []
    for k, feats in rules.items():
        feats_str = ", ".join(f'"{f}"' for f in feats)
        pad = " " * (maxk - len(k))
        lines.append(f'    "{k}":{pad} [{feats_str}],')
    return "\n".join(lines)


def gen_js(rules: dict[str, list[str]], indent: str = "        ") -> str:
    maxk = max(len(k) for k in rules)
    lines = []
    for k, feats in rules.items():
        pad = " " * (maxk - len(k))
        lines.append(f'{indent}"{k}":{pad} "{" ".join(feats)}",')
    return "\n".join(lines)


def main() -> None:
    rules = load_rules()

    # Backend python dict
    py = PY_FILE.read_text(encoding="utf-8")
    py = replace_block(
        py,
        "# >>> GENERATED: semantic rules (sync_semantic_rules.py) >>>",
        "# <<< GENERATED <<<",
        gen_py(rules),
    )
    PY_FILE.write_text(py, encoding="utf-8")

    # Frontend worker
    w = WORKER_FILE.read_text(encoding="utf-8")
    w = replace_block(
        w,
        "// >>> GENERATED: semantic rules (sync_semantic_rules.py) >>>",
        "// <<< GENERATED <<<",
        gen_js(rules),
    )
    WORKER_FILE.write_text(w, encoding="utf-8")

    # Frontend inference
    inf = INFER_FILE.read_text(encoding="utf-8")
    inf = replace_block(
        inf,
        "// >>> GENERATED: semantic rules (sync_semantic_rules.py) >>>",
        "// <<< GENERATED <<<",
        gen_js(rules),
    )
    INFER_FILE.write_text(inf, encoding="utf-8")

    print(f"synced {len(rules)} rules -> 3 targets")


if __name__ == "__main__":
    main()
