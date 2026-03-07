from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import traceback
import math

# 載入我們的自家模組
from user_input_process import load_ws_model, segment_text, tag_features, format_for_cbf
from recommender import RentalRecommender

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # 允許跨域請求 (如果是前後端分離開發需要)

print(" 啟動 Flask 伺服器...")
print("載入 ONNX模型與 Recommender 引擎")

# 全域變數預先載入，避免每次 API 請求都重新讀取
ws_pipeline = load_ws_model()
recommender = RentalRecommender("nchu_rental_info.csv")

print("模型載入完畢")

@app.route("/")
def index():
    """ 渲染前端首頁 """
    return send_from_directory('.', 'index.html')

@app.route("/api/recommend", methods=['POST'])
def recommend_api():
    """
    接收前端傳來的純文字
    經過 NLP 斷詞 -> 擷取特徵 -> CVB 格式化 -> 評分推薦
    回傳 Top 3 推薦清單
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
            
        user_input = data['text']
        print(f"\n[API 請求] 收到需求: {user_input}")
        
        # 1. 進行斷詞
        custom_result = segment_text(ws_pipeline, user_input)
        
        # 2. 進行特徵標記
        extracted_features = tag_features(custom_result)
        
        # 3. 轉換成 CBF 推薦模型用的格式
        cbf_features = format_for_cbf(extracted_features)
        
        # 4. 取得推薦結果 (DataFrame)
        recommendations_df = recommender.recommend(cbf_features, top_k=5)
        
        results = []
        if not recommendations_df.empty:
            for _, row in recommendations_df.iterrows():
                # 清洗 NaN 值，因為 JSON 序列化遇到 math.isnan 會報錯，若是 NaN 統一代換成字串或空值
                score = row['Score']
                if pd.isna(score) or math.isnan(score):
                    score = 0
                    
                rent_price_raw = row['租金']
                if pd.isna(rent_price_raw):
                    rent_price_raw = "價格面議"
                
                address_raw = row['地址']
                if pd.isna(address_raw):
                    address_raw = "地址未提供"
                    
                room_type = row['格局']
                if pd.isna(room_type):
                    room_type = ""
                
                title = f"{address_raw} ({room_type})" if room_type else address_raw

                img_url = row.get('圖片網址', "")
                if pd.isna(img_url):
                    img_url = ""
                    
                size = str(row.get('室內坪數', ""))
                if size == "nan": size = "坪數未提供"
                
                floor = str(row.get('樓層', ""))
                if floor == "nan": floor = "樓層未提供"
                
                furniture = str(row.get('家具設施', ""))
                if furniture == "nan": furniture = "無提家具設備"
                
                distance = row.get('距離(km)', -1.0)
                
                # 組裝給前端的物件
                item = {
                    "url": str(row['網址']) if not pd.isna(row['網址']) else "#",
                    "price_str": str(rent_price_raw),
                    "title": title,
                    "score": int(score),
                    "imgUrl": str(img_url),
                    "match_details": str(row['Match_Details']) if not pd.isna(row['Match_Details']) else "",
                    "size": size,
                    "floor": floor,
                    "furniture": furniture,
                    "distance": float(distance) if not pd.isna(distance) else -1.0,
                    "address": str(row['地址']) if not pd.isna(row['地址']) else ""
                }
                results.append(item)
                
        print(f"[API 回應] 成功找到 {len(results)} 筆推薦")
        return jsonify({
            "success": True,
            "data": results,
            "parsed_features": cbf_features # 順便把解析出來的條件傳回去，如果前端想展示
        })

    except Exception as e:
        print(f"[API 錯誤] {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
