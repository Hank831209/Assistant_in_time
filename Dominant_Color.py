from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import cv2
# import time


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


if __name__ == '__main__':
    a = 1








