import requests
import os


if not os.path.exists("images"): 
    os.mkdir("images")  # 建立資料夾
    
template = 'https://unsplash.com/napi/search/photos?query=suit&per_page=20&page={}&xp=unsplash-plus-2%3AControl'
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
}

for page in range(3):
    url = template.format(page)
    res = requests.get(url=url, headers=headers)
    res_json = res.json()
    
    for index, res_json_result in enumerate(res_json['results']):  # 每頁20張
        url_img = res_json_result['urls']['small']
        img = requests.get(url_img)  # 下載圖片
        
        location = './images/page_{}_index_{}.jpg'.format(page, index)
        with open(location, "wb") as file:  # 開啟資料夾及命名圖片檔
            file.write(img.content)  # 寫入圖片的二進位碼
            
            
            
