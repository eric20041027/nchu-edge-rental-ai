import pandas as pd
import re

class RentalRecommender:
    def __init__(self, csv_path="nchu_rental_info.csv"):
        self.csv_path = csv_path
        self.df = self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        except FileNotFoundError:
            print(f"Warning: {self.csv_path} not found. Please run RentInformCatcher.py first.")
            return pd.DataFrame()

        # 1. 清理租金 (將 "5800 元/月 (1學期繳)" 轉換為數字 5800)
        def clean_rent(rent_str):
            if pd.isna(rent_str):
                return 999999 # 如果沒有租金視為無限大避免比對錯誤
            
            # 使用 Regex 提取第一組連續數字
            match = re.search(r'(\d+)', str(rent_str).replace(',', ''))
            if match:
                return int(match.group(1))
            return 999999
            
        df['Rent_Num'] = df['租金'].apply(clean_rent)

        # 2. 將家具、設施、備註等字串切成 List，方便後續比對交集
        def split_to_list(item_str):
            if pd.isna(item_str):
                return []
            # 檔案中的分隔符號是 '/'
            return [x.strip() for x in str(item_str).split('/') if x.strip()]

        df['Furniture_List'] = df['家具設施'].apply(split_to_list)
        df['Included_List'] = df['租金包含'].apply(split_to_list)
        df['Security_List'] = df['安全管理'].apply(split_to_list)
        df['Note_List'] = df['備註'].apply(split_to_list)

        return df

    def recommend(self, cbf_vector, top_k=5):
        """
        給定從 UserInputProcess 取出的 cbf_vector，對資料庫進行評分並返回 Top K 房屋
        """
        if self.df.empty:
            return pd.DataFrame()

        # 建立副本以放置分數, 避免更動原始資料
        result_df = self.df.copy()
        result_df['Score'] = 0.0
        result_df['Match_Details'] = "" # 紀錄加分/雷區原因

        # 計算最高可能加分 (做為分母)
        # 假設: 地點(20) + 房型(20) + 建築(15) + (每個家俱5分*預期3個) + (每個安全設施5分*預期2個) = 大約 80 分
        # 預算符合(+10)，這是一個大概的估值，你可以依據需求調整演算法上限
        MAX_THEORETICAL_SCORE = 80.0

        for index, row in result_df.iterrows():
            score = 0.0
            details = []

            # --- 1. 硬性約束 (Hard Constraints) ---
            
            # (A) 預算 (Budget): 只要超出預算就算不符合 (或倒扣大量分數)
            user_budget = cbf_vector.get("search_budget")
            if user_budget:
                if row['Rent_Num'] > user_budget:
                    # 預算超過，設定分數極低，標示為不推薦
                    score -= 1000.0
                    details.append("預算超標")
                else:
                    # 預算內，可以給一點基礎分數
                    score += 10.0
                    details.append("符合預算")

            # (B) 性別限制 (Gender Restriction)
            user_gender = cbf_vector.get("gender_preference")
            notes = row['Note_List']
            if user_gender == "限女生" and "限男生" in notes:
                score -= 1000.0
                details.append("性別不符(限男)")
            elif user_gender == "限男生" and "限女生" in notes:
                score -= 1000.0
                details.append("性別不符(限女)")

            # (C) 寵物 (Pets)
            pet_friendly = cbf_vector.get("is_pet_friendly", -1)
            # 1 代表必須可以養寵物
            if pet_friendly == 1 and "禁養寵物" in notes:
                score -= 1000.0
                details.append("禁養寵物")
            # 0 代表使用者不想要有寵物的環境 
            elif pet_friendly == 0 and "可養寵物" in notes:
                score -= 50.0  # 這比較不是硬性雷區，但扣分
                details.append("有其他寵物")


            # --- 2. 軟性物件配對 (Soft Scoring) ---
            
            # (A) 區域 (Region)
            target_region = cbf_vector.get("search_region")
            if target_region and isinstance(row['地址'], str) and target_region in row['地址']:
                score += 20.0
                details.append(f"位於{target_region}")

            # (B) 房型 (Room Type)
            target_room = cbf_vector.get("search_room_type")
            if target_room and isinstance(row['格局'], str) and target_room in row['格局']:
                score += 20.0
                details.append(f"房型相符({target_room})")

            # (C) 建築類型 (Building Type)
            target_building = cbf_vector.get("search_building_type")
            if target_building and isinstance(row['類型'], str) and target_building in row['類型']:
                score += 15.0
                details.append(f"建築相符({target_building})")

            # (D) 家具與設施交集 (Jaccard-like matching)
            req_furnitures = cbf_vector.get("required_furniture", [])
            for f in req_furnitures:
                # 這裡做一個簡單字串包含比對，例如使用者要"網路"，CSV裡有"寬頻網路"也算中
                matched = any(f in item for item in row['Furniture_List'])
                if matched:
                    score += 5.0
                    details.append(f"有{f}")

            # (E) 其他額外需求 (安全、租金包含等)
            req_security = cbf_vector.get("required_security", [])
            for s in req_security:
                matched = any(s in item for item in row['Security_List'])
                if matched:
                    score += 5.0
                    details.append(f"有{s}")

            # 將原始分數轉化為百分比 (Percentage)
            # 確保不會低於 0，也不會超過 100 (如果原分數 < 0 就代表有踩到雷區，等等會被過濾)
            percentage_score = (score / MAX_THEORETICAL_SCORE) * 100
            
            # 基礎分 (至少給個 40% 起跳如果沒踩雷，讓畫面好看一點)
            if score >= 0:
                percentage_score = 40 + (percentage_score * 0.6)
                
            percentage_score = min(max(percentage_score, 0), 100)

            # 儲存分數計算結果
            result_df.at[index, 'Score'] = percentage_score
            result_df.at[index, 'Match_Details'] = ", ".join(details)

        # 過濾掉那些嚴重觸犯硬性條件的 (分數歸零代表被扣分扣爆)
        filtered_df = result_df[result_df['Score'] > 0]

        # 按照分數由高到低排序，如果分數一樣，租金便宜的優先
        sorted_df = filtered_df.sort_values(by=['Score', 'Rent_Num'], ascending=[False, True])

        # 返回前 K 名
        return sorted_df.head(top_k)

# 簡單自我測試
if __name__ == "__main__":
    recommender = RentalRecommender()
    
    # 模擬從 NLP 解析出來的 CBF input vector
    test_vector = {
        "search_budget": 6000,
        "search_region": "南區",
        "search_room_type": "套房",
        "required_furniture": ["冰箱", "洗衣機", "網路"],
        "is_pet_friendly": 1, 
        "gender_preference": "限女生"
    }

    print("測試推薦系統：")
    print(f"輸入條件: {test_vector}")
    
    recommendations = recommender.recommend(test_vector, top_k=3)
    
    if recommendations.empty:
        print("\n找不到符合的房屋。")
    else:
        for idx, row in recommendations.iterrows():
            print(f"\n推薦排名 - 分數: {row['Score']}")
            print(f"網址: {row['網址']}")
            print(f"租金: {row['租金']} (解析後: {row['Rent_Num']})")
            print(f"地址: {row['地址']}")
            print(f"加分細項: {row['Match_Details']}")
