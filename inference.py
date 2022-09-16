import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from PIL import Image
import shutil
import time
import os
from yolov5_6_2.detect import parse_opt, main
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import cv2


# 創建檢測器Class, 讀取讀片及預測圖片
class Detector(object):
    def __init__(self, net_kind, num_classes=6):
        super(Detector, self).__init__()
        kind = net_kind.lower()
        if kind == 'large':
            self.net = mobilenet_v3_large(num_classes=num_classes)
        elif kind == 'small':
            self.net = mobilenet_v3_small(num_classes=num_classes)

        self.net.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 設置GPU device
        self.net = self.net.to(self.device)

    def load_weights(self, weight_path):
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(weight_path))
        else:
            self.net.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

    # 檢測器
    def detect(self, weight_path, pic_path, img=False):
        self.load_weights(weight_path=weight_path)
        if not img:
            img = Image.open(pic_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        net_output = self.net(img_tensor)
        _, predicted = torch.max(net_output, dim=1)
        result = predicted[0].item()
        return result


def delete_dir(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")


def cut_pic(path_img, path_txt):
    img = cv2.imread(path_img)
    with open(path_txt, mode='r') as file:
        content = file.read()
        
    location = list()
    for label_location in content.split('\n'):
        if label_location:
            location.append(label_location.split(' ')[1:])

    img_list = list()
    # background_record = np.full_like(img, 255)  # 全白當記錄

    for content_img in location:
        Xmin, Ymin, Xmax, Ymax = content_img
        img_clip = img[int(Ymin):int(Ymax), int(Xmin):int(Xmax), :]  # 切出來的人
        img_clip = cv2.resize(img_clip, (224, 224))
        # background_record[int(Ymin):int(Ymax), int(Xmin):int(Xmax), :] = [0, 0, 0]  # 人的位置反黑
        img_list.append(img_clip)

    # background_record_resize = cv2.resize(background_record, (224, 224))  # for speed

    # background = img.copy()
    # background = cv2.resize(background, (224, 224))
    # background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    # background_hsv_list = list()
    # a, b, _ = background_hsv.shape

    # for h in range(a):
    #     for w in range(b):
    #         x, y, z = background_record_resize[h, w, :]
    #         if x != 0 or y != 0 or z != 0:
    #             background_hsv_list.append(background_hsv[h, w, :])

    return img_list


def close_center(h, s, v):
    # print('執行算距離程式')
    color = {
        '黑': (90, 122.5, 23),
        '灰': (90, 21.5, 133),
        '白': (90, 15, 238),
        '紅1': (10, 149, 150.5),
        '紅2': (168, 149, 150.5),
        '澄': (18, 149, 150.5),
        '黃': (30, 149, 150.5),
        '綠': (56, 149, 150.5),
        '青': (88.5, 149, 150.5),
        '藍': (112, 149, 150.5),
        '紫': (140, 149, 150.5)
    }
    global_min = 257*257*257
    min_color = 'None'
    for key in color.keys():
        distance = ((h - color[key][0])**2 + (s - color[key][1])**2 + (v - color[key][2])**2)**0.5
        if distance <= global_min:
            min_color = key
            global_min = distance  # 最小距離

    if min_color == '紅1' or '紅2':
        return '紅'
    else:
        return min_color


def color_check(H, S, V):
    color = 'None'
    if (0 <= H <= 180) & (0 <= S <= 255) & (0 <= V < 46):
        color = '黑'
    elif (0 <= H <= 180) & (0 <= S <= 43) & (46 <= V <= 220):
        color = '灰'
    elif (0 <= H <= 180) & (0 <= S <= 30) & (221 <= V <= 255):
        color = '白'
    elif (0 <= H <= 10 or 156 <= H <= 180) & (43 <= S <= 255) & (46 <= V <= 255):
        color = '紅'
    elif (11 <= H <= 25) & (43 <= S <= 255) & (46 <= V <= 255):
        color = '澄'
    elif (26 <= H <= 34) & (43 <= S <= 255) & (46 <= V <= 255):
        color = '黃'
    elif (35 <= H <= 77) & (43 <= S <= 255) & (46 <= V <= 255):
        color = '綠'
    elif (78 <= H <= 99) & (43 <= S <= 255) & (46 <= V <= 255):
        color = '青'
    elif (100 <= H <= 124) & (43 <= S <= 255) & (46 <= V <= 255):
        color = '藍'
    elif (125 <= H <= 155) & (43 <= S <= 255) & (46 <= V <= 255):
        color = '紫'

    if color == 'None':
        color = close_center(H, S, V)

    return color


def hsv_label(hsv):
    information = "#H:{},S:{},V:{}".format(int(hsv[0]), int(hsv[1]), int(hsv[2]))
    return information


def plot_image(path, color_labels, plt_values, ordered_colors):
    # load image
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # for plot

    # plots
    plt.figure(figsize=(14, 8))
    plt.subplot(221)
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(222)
    plt.pie(plt_values, labels=color_labels, colors=ordered_colors, startangle=90)
    plt.axis('equal')
    plt.show()


def kmeans(path, img_list, k=6, plot=True):

    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(img_list)
    label_counts = Counter(labels)  # Counter({0: 1781, 2: 1379, 3: 401, 5: 356, 4: 93, 1: 86})
    index = label_counts.most_common(1)[0][0]  # 第幾個群集的點座標數量最多
    HSV_values = clt.cluster_centers_[index]

    if plot:
        label_counts = Counter(labels)  # Counter({0: 1781, 2: 1379, 3: 401, 5: 356, 4: 93, 1: 86})
        ordered_colors = clt.cluster_centers_ / 255  # 作圖的顏色
        color_labels = [hsv_label(i) for i in clt.cluster_centers_]  # 畫圖的label(HSV數值)
        plt_values = [label_counts[i] for i in range(k)]  # 每個群集對應到的點座標的數量
        plot_image(path, color_labels, plt_values, ordered_colors)  # 圖示

    return HSV_values


def model(img_name, path='yolov5_6_2/runs/detect/exp', 
    path_weights='best_test.pt', num_classes=4, net='large'):

    if os.path.isdir(path):
        delete_dir(path)

    # Yolov5 物件偵測
    opt = parse_opt()
    main(opt)
    print('Yolo執行完畢')

    # 解析分割後的圖片
    path_img = f'./yolov5_6_2/data/images/{img_name}.jpg'  # 原始圖片放置路徑
    path_txt = f'./yolov5_6_2/runs/detect/exp/labels/{img_name}.txt'  # 物件偵測產出的座標txt
    img_person_list = cut_pic(path_img, path_txt)

    # 人物風格
    result = list()
    detector = Detector(net, num_classes=num_classes)
    for each_person in img_person_list:
        pil_image = Image.fromarray(each_person)
        predict_result = detector.detect(path_weights, '', pil_image)  # 丟圖片路徑即可
        result.append(predict_result)
        # print('預測結果為:\t', predict_result)

    return result


if __name__ == '__main__':
    '''
    path_img = r'./yolov5_6_2/data/images'  # 照片要存到這個資料夾裡面
    path= 'yolov5_6_2/runs/detect/exp'  # 模型跑完之後的圖片會存到這邊
    result = model('4_1121')  # 回傳照片每個人的服裝風格(list形式), 傳入照片名稱(不用+.jpg)
    建議在最後整個運行完之後把資料夾刪掉, 可以自行斟酌要不要+延遲時間
    '''
    start_time = time.time()
    result = model('4_1121')  
    if not result:
        print('未偵測到照片有人')
    else:
        for i in result:
            print('預測結果為:\t', i)
    end_time = time.time()
    print('總運行時間為:\t', end_time - start_time)
    
    time.sleep(3)
    # 若存在路徑則刪除
    path= 'yolov5_6_2/runs/detect/exp'
    if os.path.isdir(path):
        delete_dir(path)

    # # Yolov5 物件偵測
    # opt = parse_opt()
    # main(opt)
    # print('Yolo執行完畢')

    # # 解析分割後的圖片
    # img_name = '4_1121'
    # path_img = f'./yolov5_6_2/data/images/{img_name}.jpg'  # 原始圖片放置路徑
    # path_txt = f'./yolov5_6_2/runs/detect/exp/labels/{img_name}.txt'  # 物件偵測產出的座標txt
    # img_person_list = cut_pic(path_img, path_txt)

    # # 場景色系
    # HSV_values = kmeans(path_img, background_hsv_list, k=6, plot=False)
    # print('主導色的HSV值為:\n', HSV_values)
    # H, S, V = HSV_values
    # Dominant_Color = color_check(H, S, V)
    # print('主導色為:\n', Dominant_Color)

    # # 人物風格
    # detector = Detector('large', num_classes=4)
    # path_weights = r'best_test.pt'
    # for each_person in img_person_list:
    #     pil_image = Image.fromarray(each_person)
    #     predict_result = detector.detect(path_weights, '', pil_image)  # 丟圖片路徑即可
    #     print('預測結果為:\t', predict_result)


    # # 程式運行完刪除路徑
    # time.sleep(5)
    # path = 'yolov5_6_2/runs/detect/exp'
    # if os.path.isdir(path):
    #     delete_dir(path)




