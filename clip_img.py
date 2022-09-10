import time

import cv2
import numpy as np


def cut_pic(path_img, path_txt):
    img = cv2.imread(path_img)
    with open(path_txt, mode='r') as file:
        content = file.read()
    location = list()
    for label_location in content.split('\n'):
        if label_location:
            location.append(label_location.split(' ')[1:])

    img_list = list()
    background_record = np.full_like(img, 255)  # 全白當記錄

    for content_img in location:
        Xmin, Ymin, Xmax, Ymax = content_img
        img_clip = img[int(Ymin):int(Ymax), int(Xmin):int(Xmax), :]  # 切出來的人
        img_clip = cv2.resize(img_clip, (224, 224))
        background_record[int(Ymin):int(Ymax), int(Xmin):int(Xmax), :] = [0, 0, 0]  # 人的位置反黑
        img_list.append(img_clip)

    background_record_resize = cv2.resize(background_record, (224, 224))  # for speed

    background = img.copy()
    background = cv2.resize(background, (224, 224))
    background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    background_hsv_list = list()
    a, b, _ = background_hsv.shape

    for h in range(a):
        for w in range(b):
            x, y, z = background_record_resize[h, w, :]
            if x != 0 or y != 0 or z != 0:
                background_hsv_list.append(background_hsv[h, w, :])

    return background_hsv_list, img_list


if __name__ == '__main__':
    start_time = time.time()
    img_name = '111'
    path_img = r'./data/images/{}.jpg'.format(img_name)  # 原始圖片放置路徑
    path_txt = r'./runs/detect/exp/labels/{}.txt'.format(img_name)  # 物件偵測產出的座標txt
    _, _ = cut_pic(path_img, path_txt)
    end_time = time.time()
    print('執行時間:\t', end_time - start_time)



