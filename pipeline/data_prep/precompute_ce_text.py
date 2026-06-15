"""把 C 組富化房源文字(property_to_text_enriched)預算進前端 property_data.json。

為什麼要預算而非前端即時組裝:
  C 組 cross-encoder 是用 `property_to_text_enriched` 富化文字訓練的,其基底是
  **generate_dataset.property_to_text**(讀 CSV 的 Python dict)。前端 prop.text 與
  該基底 100% 不一致(缺『步行X分鐘 騎車X分鐘』、路名段號被砍、含水≠含水費),若
  前端用 prop.text 當基底自組富化文字 → 基底就跟訓練分佈不符 → 仍是 OOD
  (重蹈 docs/ce_text_layer_decision.md)。

  唯一保證 byte-exact 對齊訓練的做法:直接在 Python 端用同一個
  property_to_text_enriched 算好,寫進 JSON 的 ce_text 欄,前端 scorePair 直接餵
  prop.ce_text。零手動移植風險。

對齊保證:
  - 房源來源 = generate_dataset.load_properties()(訓練同一份 CSV、同一順序)。
  - 富化邏輯 = augment_with_expansion_map.property_to_text_enriched(C 組訓練同一函式)。
  - 寫回前端 JSON 以 (address, rent) 為鍵比對(非純順序),全 704 筆一一對上。

用法:
    python pipeline/data_prep/precompute_ce_text.py            # 預覽(不寫)
    python pipeline/data_prep/precompute_ce_text.py --write    # 寫進 property_data.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline" / "data_prep"))

import generate_dataset as G  # noqa: E402
# 復用 C 組訓練同一個富化函式,確保 byte-exact 對齊(切勿在此重寫富化邏輯)。
from augment_with_expansion_map import property_to_text_enriched  # noqa: E402

FRONTEND_JSON = ROOT / "frontend" / "assets" / "property_data.json"


def build_key(address: str, rent) -> tuple[str, int]:
    return (str(address or "").strip(), int(rent or 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="寫回 property_data.json")
    args = ap.parse_args()

    props = G.load_properties()
    enriched_by_key: dict[tuple[str, int], str] = {}
    for p in props:
        enriched_by_key[build_key(p["address"], p["rent"])] = property_to_text_enriched(p)

    fe = json.loads(FRONTEND_JSON.read_text(encoding="utf-8"))
    if not isinstance(fe, list):
        print("⚠️ property_data.json 不是 list,結構超出預期", file=sys.stderr)
        return 1

    matched, missing = 0, []
    token_lens = []
    for f in fe:
        key = build_key(f.get("address"), f.get("rent"))
        ce = enriched_by_key.get(key)
        if ce is None:
            missing.append(key)
            continue
        f["ce_text"] = ce
        matched += 1
        token_lens.append(len(ce.split()))

    print(f"=== 預算 ce_text(C 組富化房源文字)===")
    print(f"CSV 房源: {len(props)}  前端房源: {len(fe)}  對上: {matched}  缺: {len(missing)}")
    if missing:
        print(f"⚠️ 未對上(address,rent): {missing[:5]}")
    if token_lens:
        token_lens.sort()
        avg = sum(token_lens) / len(token_lens)
        p50 = token_lens[len(token_lens) // 2]
        p95 = token_lens[int(len(token_lens) * 0.95)]
        print(f"ce_text 空白切分長度: avg={avg:.1f} p50={p50} p95={p95} max={token_lens[-1]}")
        print(f"(注:實際 BERT subword token 數更多;MAX_LENGTH=128 須涵蓋之)")

    print(f"\n=== 範例 ===")
    for f in fe[:3]:
        if "ce_text" in f:
            print(f"  text    : {f['text']}")
            print(f"  ce_text : {f['ce_text']}\n")

    if missing:
        print("❌ 有房源未對上,不寫出(避免部分富化造成不一致)。", file=sys.stderr)
        return 1

    if args.write:
        FRONTEND_JSON.write_text(
            json.dumps(fe, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✅ 已寫回 {FRONTEND_JSON.relative_to(ROOT)}({matched} 筆 ce_text)")
    else:
        print("(預覽模式 — 加 --write 才寫回)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
