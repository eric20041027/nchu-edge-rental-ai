import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time 
import pandas as pd
import os

def scrape_rent():
    base_index_url = 'https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/index.php'
    base_house_url = "https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/"
    
    all_house_data = [] # 存儲所有物件的詳細資訊
    detail_urls = []
    
    # 1. 取得所有物件的詳細頁面 URL
    PAGE_NUM = 7 # 預計抓取的頁數
    print(f"正在獲取前 {PAGE_NUM} 頁的連結...")
    
    for i in range(PAGE_NUM):
        page_to_use = f"{base_index_url}?start={i * 20}"
        try:
            resp = requests.get(page_to_use, timeout=10)
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'html.parser')  
            
            house_links = soup.find_all('span', class_='list-title')  
            for link in house_links:
                a_tag = link.find('a')
                if a_tag:
                    href = a_tag.get('href')
                    full_url = urljoin(base_house_url, href)
                    detail_urls.append(full_url)
            
            print(f"已完成第 {i+1} 頁連結收集 (目前共 {len(detail_urls)} 筆)")
            time.sleep(0.5) 
        except Exception as e:
            print(f"讀取分頁 {i+1} 失敗: {e}")

    print(f"\n找到 {len(detail_urls)} 筆租屋資訊，準備開始爬取詳細內容...\n")

    # 2. 進入每一個詳細頁面抓取資訊
    for index, url in enumerate(detail_urls):
        try:
            print(f"[{index+1}/{len(detail_urls)}] 正在抓取: {url}")
            resp = requests.get(url, timeout=10)
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            item_info = {"網址": url} # 每一筆資料先存入 URL
            
            # 抓取表格中的所有欄位
            rows = soup.find_all('tr')
            for row in rows:
                cols = row.find_all(['th', 'td'])
                # 使用使用者提供的遍歷邏輯，處理可能的一列多欄情況
                for i in range(0, len(cols) - 1, 2):
                    key = cols[i].get_text(strip=True).replace('：', '').replace(':', '')
                    value = cols[i+1].get_text(strip=True)
                    if key:
                        item_info[key] = value
            
            # 從 fancybox 中尋找第一張房屋圖片 (使用者提供的邏輯)
            img_tag = soup.find('a', {'class': 'fancybox'})
            if img_tag and img_tag.get('href'):
                img_url = urljoin(base_house_url, img_tag.get('href'))
                item_info['圖片網址'] = img_url
            else:
                item_info['圖片網址'] = ""

            # 預設距離為 1.0 (或留空讓後續處理)
            item_info["距離(km)"] = 1.0
            
            all_house_data.append(item_info)
            
            if (index + 1) % 5 == 0:
                time.sleep(1)
                
        except Exception as e:
            print(f"抓取詳細頁面失敗 {url}: {e}")

    # 3. 轉換為 Pandas DataFrame 並寫入 CSV
    if all_house_data:
        df = pd.DataFrame(all_house_data)
        
        # 確保與原 CSV 欄位對齊
        expected_cols = [
            "網址", "地址", "格局", "類型", "室內坪數", "租金", "空房間數", "押金", 
            "安全標章", "樓層", "聯絡人", "電話", "家具設施", "租金包含", "另計費用", 
            "安全管理", "消防逃生", "備註", "圖片網址", "距離(km)"
        ]
        
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""
        
        df = df[expected_cols]
        
        output_filename = "nchu_rental_info.csv"
        df.to_csv(output_filename, index=False, encoding='utf-8-sig') 
        print(f"\n--- 抓取完成 ---")
        print(f"總共抓取 {len(all_house_data)} 筆資料")
        print(f"檔案已儲存至: {output_filename}")
    else:
        print("未抓取到任何資料。")

if __name__ == "__main__":
    scrape_rent()