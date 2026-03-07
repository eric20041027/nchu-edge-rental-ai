import pandas as pd
import requests
import time
import math
import os

# 中興大學座標
nchu_coords = (24.1207, 120.6756)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半徑(km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

import re

def get_coords(address):
    headers = {"User-Agent": "RentalBot/1.0 (test)"}
    # 稍微清理地址，把樓層等資訊去掉，幫助 Nominatim 更好辨識
    clean_addr = address.split("F")[0].split("-")[0].replace("~", "").split("樓")[0].split("之")[0]
    
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={clean_addr}&format=json"
        resp = requests.get(url, headers=headers).json()
        if resp and len(resp) > 0:
            return float(resp[0]["lat"]), float(resp[0]["lon"])
            
        # 如果找不到完整門牌，嘗試萃取路段名稱
        match = re.search(r'(.+?[市縣])(.+?[區鄉鎮市])(.+?(?:路|街|大道|段))', clean_addr)
        if match:
            city, dist, street = match.groups()
            # 確保不會組裝出重複的區名 (例如: 台中市南區五權南區路)
            street_only = street.replace(dist, "").replace(city, "")
            q = f"{street_only}, {dist}, {city}"
            url_street = f"https://nominatim.openstreetmap.org/search?q={q}&format=json"
            resp_street = requests.get(url_street, headers=headers).json()
            if resp_street and len(resp_street) > 0:
                return float(resp_street[0]["lat"]), float(resp_street[0]["lon"])
                
        # 最後的退路，嘗試只抓區
        if "區" in address:
            district = address.split("市")[1].split("區")[0] + "區"
            url2 = f"https://nominatim.openstreetmap.org/search?q=台中市{district}&format=json"
            resp2 = requests.get(url2, headers=headers).json()
            if resp2 and len(resp2) > 0:
                return float(resp2[0]["lat"]), float(resp2[0]["lon"])
    except Exception as e:
        print(f"Error fetching {clean_addr}: {e}")
    return None, None

def main():
    target_csv = "nchu_rental_info.csv"
    if not os.path.exists(target_csv):
        print(f"找不到 {target_csv}")
        return

    df = pd.read_csv(target_csv)
    if "距離(km)" not in df.columns:
        df["距離(km)"] = 0.0
    
    # 遍歷每一行加入座標與計算距離
    for idx, row in df.iterrows():
        # 若已經計算過，跳過 (大於0的合理距離)
        if row["距離(km)"] > 0:
            continue
            
        addr = row["地址"]
        print(f"[{idx+1}/{len(df)}] Fetching coordinates for: {addr}")
        lat, lon = get_coords(addr)
        if lat and lon:
            dist = haversine(nchu_coords[0], nchu_coords[1], lat, lon)
            df.at[idx, "距離(km)"] = round(dist, 2)
            print(f" -> Distance: {round(dist, 2)} km")
        else:
            df.at[idx, "距離(km)"] = -1.0 # 標記為無法計算
            print(f" -> Failed to get coordinates.")
        
        # 配合 Nominatim API 的政策限制 (1次/1-1.5秒)
        time.sleep(1.5)
        
    df.to_csv(target_csv, index=False)
    print("距離計算完成！CSV 已更新。")

if __name__ == "__main__":
    main()
