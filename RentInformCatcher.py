import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time 
import pandas as pd
 


def scrape_rent():
        base_index_url = 'https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/index.php'
        base_house_url = "https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/"
        resp = requests.get(base_index_url)
        soup = BeautifulSoup(resp.text, 'html5lib')
        all_data = []
        detailed_data = []
        detail_urls = []
        page_url = []
        house_links = soup.find_all('span',class_ = 'list-title')
        PAGE_NUM = 7
        for i in range(PAGE_NUM):
            page_to_use = base_index_url + '?start=' + str((i+1)*20)
            page_url.append(page_to_use)
                    
        for page in page_url:
            resp = requests.get(page)
            soup = BeautifulSoup(resp.text, 'html5lib')  
            house_links = soup.find_all('span',class_ = 'list-title')  
            for link in house_links:
                a_tag = link.find('a')
                if a_tag:
                    href = a_tag.get('href')
                    full_url = urljoin(base_house_url, href)
                    #print(full_url)
                    detail_urls.append(full_url)
        print(f"找到 {len(detail_urls)} 筆租屋資訊，準備開始爬取詳細內容...")
        
scrape_rent()
    

    
    
    
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time 
import pandas as pd

def scrape_rent():
    base_index_url = 'https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/index.php'
    base_house_url = "https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/"
    
    all_house_data = [] #存儲所有物件的詳細資訊
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
            
            print(f"已完成第 {i+1} 頁連結收集")
            time.sleep(0.5) 
        except Exception as e:
            print(f"讀取分頁 {i+1} 失敗: {e}")

    print(f"找到 {len(detail_urls)} 筆租屋資訊，準備開始爬取詳細內容...\n")

    # 2. 進入每一個詳細頁面抓取資訊
    for index, url in enumerate(detail_urls):
        try:
            print(f"[{index+1}/{len(detail_urls)}] 正在抓取: {url}")
            resp = requests.get(url, timeout=10)
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            item_info = {"網址": url} # 每一筆資料先存入 URL
            
            rows = soup.find_all('tr')
            for row in rows:
                cols = row.find_all(['th', 'td'])
                for i in range(0, len(cols) - 1, 2):
                    key = cols[i].get_text(strip=True).replace('：', '').replace(':', '')
                    value = cols[i+1].get_text(strip=True)
                    if key:
                        item_info[key] = value
            
            all_house_data.append(item_info)
            if (index + 1) % 5 == 0:
                time.sleep(1)
                
        except Exception as e:
            print(f"抓取詳細頁面失敗 {url}: {e}")

    # 3. 轉換為 Pandas DataFrame 並寫入 Excel
    if all_house_data:
        df = pd.DataFrame(all_house_data)
        
        # 整理欄位順序（可選）：將網址放到最後或最前
        cols = ['網址'] + [c for c in df.columns if c != '網址']
        df = df[cols]
        
        filename = "nchu_rental_info.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 避免 Excel 打開亂碼
        print(f"\n--- 抓取完成 ---")
        print(f"總共抓取 {len(all_house_data)} 筆資料")
        print(f"檔案已儲存至: {filename}")
    else:
        print("未抓取到任何資料。")

if __name__ == "__main__":
    scrape_rent()