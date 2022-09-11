import requests
import os
import time

start_time = time.time()
keyword = 'princess'  # 下關鍵字查詢
if not os.path.exists(keyword): 
    os.mkdir(keyword)  # 建立資料夾

headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
}

# 查詢網址
template1 = f'https://unsplash.com/napi/search/photos?query={keyword}'
template2 = '&per_page=20&page={}&xp=unsplash-plus-2%3AControl'
template = template1 + template2

# 算要爬的總頁數
url = template1 + '&per_page=20&page=1&xp=unsplash-plus-2%3AControl'  
res = requests.get(url=url, headers=headers)
res_json = res.json()
print('總頁數為:\t', res_json['total_pages'])

# 開爬
for page in range(res_json['total_pages']):  # 一次載全部
    url = template.format(page)  # 依頁數變換網址
    res = requests.get(url=url, headers=headers)
    res_json = res.json()
    for index, res_json_result in enumerate(res_json['results']):  # 每頁20張
        url_img = res_json_result['urls']['small']
        img = requests.get(url_img)  # 下載圖片
        location = f'./{keyword}/page_{page}_index_{index}.jpg'
        
        with open(location, "wb") as file:  # 開啟資料夾及命名圖片檔
            file.write(img.content)  # 寫入圖片的二進位碼
            
end_time = time.time()
print('總運行時間:\t', end_time - start_time)