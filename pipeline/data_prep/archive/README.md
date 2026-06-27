# data_prep/archive — 歸檔的一次性腳本

架構精簡批次4 歸檔。下列腳本是 stage4 泛化階段的**一次性 LLM 資料生成**(Gemini API),
現役管線(build_frontend_data 那條)不再用,零 import。保留供重跑參考,不在 import 路徑內。

- `augment_with_llm.py` — Gemini 生成困難負樣本 + 語意映射(訓練資料合成)
- `silver_labeling.py` — Gemini 生成 query-label 銀標(訓練資料)

> 注意:`silver_labeling.py` module-level 即建 genai.Client(需 GEMINI_API_KEY);
> 歸檔後不會被誤 import。要重跑直接 `python pipeline/data_prep/archive/<檔>.py`。
> (class 版的 SilverLabeler 在 data_prep/labeler.py,與此腳本不同,仍由 __init__ export。)
