from transformers import BertTokenizerFast, pipeline
from optimum.onnxruntime import ORTModelForTokenClassification
import os
from Recommender import RentalRecommender

def load_ws_model():
    
    print("正在載入 CKIP ALBERT Tiny 斷詞模型 (ONNX 版)...")
    
    model_dir = './onnx_model_dir'
    
    if not os.path.exists(model_dir):
        model_dir = './test_input_ONNX/onnx_model_dir'
        
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = ORTModelForTokenClassification.from_pretrained(model_dir)
    
    ws_pipeline = pipeline('token-classification', model=model, tokenizer=tokenizer)
    print("ONNX 模型載入完成！")
    return ws_pipeline

def segment_text(ws_pipeline, text):

    if not text:
        return []

    result = ws_pipeline(text)
    
    words = []
    current_word = ''
    
    for token in result:
        label = token['entity']
        char = token['word']
        
        if char.startswith('['):
            continue
            
        if label == 'B':  
            if current_word:
                words.append(current_word)
            current_word = char
        elif label == 'I': 
            current_word += char
            
    if current_word:
        words.append(current_word)
        
    return words

def tag_features(words):
  
    features = {
        "地址(區域)": None,
        "格局(房型)": None,
        "類型(建築)": None,
        "預算": None,
        "家具設施": [],
        "租金包含": [],
        "安全管理與消防": [],
        "寵物": None,
        "性別限制": None,
        "備註": []
    }
    
    # 定義關鍵字字典
    furniture_keywords = ["床", "衣櫃", "電話", "網路", "寬頻", "冰箱", "洗衣機", "脫水機", "電視", "第四台", "書桌", "熱水器", "冷氣", "穿衣鏡", "電梯", "車位", "機車", "汽車", "飲水機", "陽台", "曬衣"]
    included_keywords = ["水費", "電費", "網路費", "管理費", "清潔費", "瓦斯"]
    security_keywords = ["監視器", "監視系統", "攝影", "感應", "滅火器", "警報", "照明", "逃生", "防盜"]

    for i, word in enumerate(words):
        # 1. 預算判斷
        if word == "預算" and i + 1 < len(words):
            if words[i+1].isdigit():
                features["預算"] = int(words[i+1])
        elif word.isdigit() and not features["預算"]:
            if i + 1 < len(words) and words[i+1] in ["元", "塊", "千", "萬"]:
                # 簡單計算，如果說 6 千就是 6000
                multiplier = 1000 if words[i+1] == "千" else (10000 if words[i+1] == "萬" else 1)
                features["預算"] = int(word) * multiplier
            elif int(word) > 1000: # 假設大於1000的數字很可能是預算
                features["預算"] = int(word)

        # 2. 地區判斷 (例如南區、西區、大里等)
        if word in ["南區", "西區", "東區", "北區", "中區", "大里", "大里區", "烏日", "市區", "校區", "學校"]:
            features["地址(區域)"] = word

        # 3. 房型與建築類型
        if word in ["套房", "雅房", "整層", "家庭式", "住家"]:
            features["格局(房型)"] = word
        if word in ["透天", "透天厝", "公寓", "電梯大樓", "別墅"]:
            features["類型(建築)"] = word

        # 4. 家具與設施
        for f_kw in furniture_keywords:
            if f_kw in word and f_kw not in features["家具設施"]:
                features["家具設施"].append(f_kw)

        # 5. 租金包含項目 (尋找 "包" + 水/電/網路)
        for inc_kw in included_keywords:
            if inc_kw in word:
                # 檢查前面有沒有說「包」
                is_included = False
                for j in range(max(0, i-2), i):
                    if words[j] in ["包", "包含", "含"]:
                        is_included = True
                
                if is_included and inc_kw not in features["租金包含"]:
                    features["租金包含"].append(inc_kw)

        # 6. 安全與消防設施
        for sec_kw in security_keywords:
            if sec_kw in word and sec_kw not in features["安全管理與消防"]:
                features["安全管理與消防"].append(sec_kw)

        # 7. 寵物相關
        if "寵物" in word or "貓" in word or "狗" in word:
            is_allowed = True
            for j in range(max(0, i-2), i):
                if words[j] in ["不", "禁", "不可", "不能"]:
                    is_allowed = False
            features["寵物"] = "可養寵物" if is_allowed else "禁養寵物"
            
        # 8. 性別限制相關
        if "男" in word or "女" in word:
            for j in range(max(0, i-2), i):
                if words[j] in ["限", "只"]:
                    features["性別限制"] = f"限{word}生" 

    return features

def format_for_cbf(features):
    """
    將 NLP 萃取出來的特徵字典，轉換成 Content-Based Filtering (CBF) 推薦模型
    容易處理的格式 (扁平化特徵、數值化特定欄位等)。
    """
    # 判斷寵物是否友善 (-1 代表未提及/不拘, 1 代表可養, 0 代表禁養)
    pet_friendly = -1
    if features.get("寵物") == "可養寵物":
        pet_friendly = 1
    elif features.get("寵物") == "禁養寵物":
        pet_friendly = 0

    cbf_input = {
        "search_budget": features.get("預算"),          # Integer or None
        "search_region": features.get("地址(區域)"),    # String or None
        "search_room_type": features.get("格局(房型)"), # String or None
        "search_building_type": features.get("類型(建築)"), # String or None
        
        # 將串列直接保留，方便後續用 Pandas 的 .apply() 或 Set Intersection 算分
        "required_furniture": features.get("家具設施", []),
        "required_included_fees": features.get("租金包含", []),
        "required_security": features.get("安全管理與消防", []),
        
        "is_pet_friendly": pet_friendly,
        "gender_preference": features.get("性別限制")   # String or None
    }
    
    return cbf_input


if __name__ == "__main__":
    # 0. 載入推薦引擎 (讀取 CSV)
    print("正在初始化房屋匹配引擎...")
    recommender = RentalRecommender("nchu_rental_info.csv")

    # 1. 載入模型
    ws_pipeline = load_ws_model()
    
    # 2. 提供使用者自行輸入斷詞測試
    print("\n--- 測試 CKIP ALBERT 中文斷詞與特徵標記 ---")
    print(" (您可以輸入 'q' 或 'exit' 來退出) ")
    while True:
        user_input = input("\n請輸入您的租屋需求：")
        if user_input.lower() in ['q', 'exit']:
            break
        
        # 進行斷詞
        custom_result = segment_text(ws_pipeline, user_input)
        
        # 進行特徵標記 (給人類或前端看的原始結構)
        extracted_features = tag_features(custom_result)
        
        # 轉換成 CBF 推薦模型用的格式 (機器學習/相似度計算用)
        cbf_features = format_for_cbf(extracted_features)
        
        print("\n--- 分析結果 ---")
        print(f"原始房屋特徵 (Human Readable):")
        for key, value in extracted_features.items():
            if value:
                print(f"  - [{key}]: {value}")
                
        print(f"\n推薦模型輸入向量 (CBF Input Vector):")
        for key, value in cbf_features.items():
            # 為了讓畫面乾淨，過濾掉空的 list 或 None (如果想要全印可以把這個 if 拿掉)
            if value is not None and value != [] and value != -1:
                print(f"  - {key}: {value}")

        print(f"\n=================================")
        print(f"🤖 正在為您搜尋最適合的房屋...")
        recommendations = recommender.recommend(cbf_features, top_k=3)
        
        if recommendations.empty:
            print("找不到符合硬性條件的房屋 (可能預算過嚴格或性別不符)。")
        else:
            for idx, row in recommendations.iterrows():
                print(f"\n⭐ 推薦分數: {row['Score']}")
                print(f"💰 租金: {row['租金']}")
                print(f"📍 地址: {row['地址']} ({row['格局']})")
                print(f"🔗 網址: {row['網址']}")
                print(f"💡 推薦理由: {row['Match_Details']}")
        print(f"=================================\n")
