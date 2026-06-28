"""stage4 第三輪 — 把有區辨力設施特徵線索補進召回用的 `text` 欄位。

根因(本機查證):房源向量從 `text` 編碼(build_property_embeddings.py:17-26,
用 ce_text 會出分佈傷召回)。但飲水機/車位/報稅/曬衣場等特徵線索只在 ce_text,
`text` 裡 0% 有 → 模型召回時看不到線索,補訓練資料也對不上。

解法:把這些特徵的標準線索詞 append 進 `text`(召回用欄位)。平均只加 ~0.7 詞/房源
(46→47 字),遠短於 ce_text 134 字 → 不出 encoder 分佈。

只補【有區辨力(base 3~60%) + 有客觀依據 + text 缺】的特徵:
  飲水機 / 機車位 / 可報稅 / 曬衣場。
排除:台電(ce_text 都 0% 線索,本質結構化)、第四台/沙發(53% 近同質,避新塌縮)、
      含水(text 已 100% 有)。

補完【必須】Colab 重算房源向量 + 本機驗召回不退步(ab_eval ≥0.26)。

用法:python pipeline/data_prep/enrich_text_features.py
產物:frontend/assets/property_data.json(就地富化 text)+ 印報告
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PD_PATH = ROOT / "frontend/assets/property_data.json"


def ce(p: dict) -> str:
    return str(p.get("ce_text", ""))


# 特徵 → (該房源是否命中, 補進 text 的標準線索詞)
FEATURES = [
    ("飲水機", lambda p: p.get("water_dispenser") is True),
    ("機車位", lambda p: p.get("has_parking") is True),
    ("可報稅", lambda p: "可報稅" in ce(p)),
    ("曬衣場", lambda p: "曬衣" in ce(p)),
]


def enrich(props: list[dict]) -> tuple[list[dict], dict]:
    """回 (新 props, 報告)。immutable:回新 list,不改入參。"""
    report = {w: 0 for w, _ in FEATURES}
    out = []
    for p in props:
        text = str(p.get("text", ""))
        add = []
        for word, pred in FEATURES:
            if pred(p) and word not in text:
                add.append(word)
                report[word] += 1
        new_text = text if not add else (text + " " + " ".join(add))
        out.append({**p, "text": new_text})
    return out, report


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="就地寫回 property_data.json(預設只 dry-run 印報告)")
    args = ap.parse_args()

    props = json.loads(PD_PATH.read_text(encoding="utf-8"))
    enriched, rep = enrich(props)

    import statistics as st
    before = st.mean(len(str(p.get("text", ""))) for p in props)
    after = st.mean(len(str(p.get("text", ""))) for p in enriched)

    print("補特徵線索進召回用 text 欄位" + ("" if args.write else "  [DRY-RUN]"))
    print("-" * 56)
    for word, n in rep.items():
        print(f"  {word:8} 補進 {n:4} 間房源 text")
    print(f"  text 平均長度: {before:.0f} → {after:.0f} 字(ce_text 是 134,未出分佈)")

    if args.write:
        PD_PATH.write_text(json.dumps(enriched, ensure_ascii=False), encoding="utf-8")
        print(f"→ {PD_PATH.name} 就地富化")
        print("\n⚠ 下一步必跑:Colab 重算 property_embeddings + 本機驗 ab_eval ≥0.26 不退步")
    else:
        print("→ DRY-RUN 未寫檔。加 --write 才就地富化(且需隨後 Colab 重算向量)")


if __name__ == "__main__":
    main()
