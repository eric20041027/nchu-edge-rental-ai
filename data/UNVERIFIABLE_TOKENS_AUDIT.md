# 擴展詞 ↔ 房源資料 可驗證性盤點 (2026-06-14)

針對 704 筆 `frontend/assets/property_data.json` 實測:把每個語意擴展 token 對
房源 blob(`text` + `furniture` + `features` + `notes` + `other_fees` + 電費欄 + `address`)
做 `includes` 比對,統計命中數。**命中 0 = 爬蟲資料無對應實據,屬模型臆測。**

爬蟲使用**受控詞彙**(`可養貓`/`可開伙`/`對外窗`/`保全設施`/`垃圾代收`…),非自由
描述。擴展表多數 token 是對著「想像中的描述用語」寫的,房源端根本不產出該詞。

---

## 本次已移除 (geo,9 詞)

`便利商店` `超商` `近捷運` `近公車` `生活機能` `交通便利` `核心` `核心區` `核心圈`

- rules 表刪 4 條抽空 rule(通勤/沒有車/不開車/上班方便),`生活便利` 留 `興大路`(14/704)。
- `explainMatch` 移除 geo_tier 標籤 + 「🚌交通便利」dead-code 規則。
- 已 commit `c43fbdd`,進 PR#3。

---

## 待審:其餘 0-backing 臆測詞(本次未動,約 60+)

> ⚠️ 刪前需逐一確認**沒有結構欄/bool 欄/PROP_SYNONYMS backing**。
> 下列為「raw blob 直接 0 命中」者,部分可能已被同義橋接救回(例:`廚房`→`可開伙`、
> `可寵`→`可養`、`管理員`→`保全`),**有橋的要留**。真正該刪的是橋不到的。

| 維度 | 0-命中 token |
|---|---|
| 採光朝向 | 採光 通風 落地窗 東向 北向 非西向 遮陽 窗簾 隔熱 |
| 質感 | 全新 漂亮 質感 裝潢 乾淨 |
| 安靜/門禁 | 隔音 靜巷 安靜 無門禁 24小時 自由進出 刷卡 女性友善 安全 監視器 |
| 租期 | 短租 短期 彈性租期 月租 不限租期 彈性 |
| 合租 | 室友 合租 獨立套房 獨廁 |
| 預算暗示 | 經濟實惠 學生套房 低價 低租金 實惠 經濟 便宜 |
| 公設/家具 | 健身房 交誼廳 全配 全家具 全家電 家具齊全 空屋 自備家具 |
| 頂樓 | 非頂樓 非頂加 頂加 |
| 衛浴 | 浴缸 (獨衛/獨立衛浴 → 有 `獨衛` 同義可救,需確認) |
| 電費/補助 | 台水 帳單 自繳 標準電費 租補 (台電→`is_taipower`有 backing;補助→`租金補貼`559 有 backing) |
| 抽菸 | 禁菸 |

## 保留(別誤刪)

- `走路10分` / `騎車10分`:distance sentinel,被 regex 抽成 maxWalkMins,走 OSRM 通勤距離計算。
- 任何有 `PROP_SYNONYMS` / bool 欄 / 結構欄 backing 的詞。

## 復算方式

```bash
node -e '
const p=require("./frontend/assets/property_data.json");
const d=require("./data/semantic_rules.json");
function blob(o){return [o.text,o.furniture,o.features,(o.notes||[]).join(" "),(o.other_fees||[]).join(" "),o.address,o.electricity_billing].join(" ");}
const T=p.map(blob);
const tok=new Set(); for(const v of Object.values(d.rules)) (Array.isArray(v)?v:String(v).split(/[\s　]+/)).forEach(t=>t&&tok.add(t));
for(const t of [...tok].sort()){ const c=T.filter(x=>x.includes(t)).length; if(c<10) console.log(String(c).padStart(4),t); }
'
```
