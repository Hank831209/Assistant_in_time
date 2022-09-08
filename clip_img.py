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
    background_rgb = img.copy()

    for content_img in location:
        Xmin, Ymin, Xmax, Ymax = content_img
        img_clip = img[int(Ymin):int(Ymax), int(Xmin):int(Xmax), :]  # 切出來的人
        img_clip = cv2.resize(img_clip, (224, 224))
        background_rgb[int(Ymin):int(Ymax), int(Xmin):int(Xmax), :] = [0, 0, 0]  # 人的位置反黑
        img_list.append(img_clip)

    background = cv2.resize(background_rgb, (224, 224))
    background_hsv = background.copy()
    background_hsv = cv2.cvtColor(background_hsv, cv2.COLOR_BGR2HSV)
    background_hsv_list = list()
    a, b, _ = background.shape

    for h in range(a):
        for w in range(b):
            x, y, z = background[h, w, :]
            if x != 0 or y != 0 or z != 0:
                background_hsv_list.append(background_hsv[h, w, :])

    return background_hsv_list, img_list


# if __name__ == '__main__':
    # path_img = r'../yolov5-6.2/data/images/1_7.jpg'  # 原始圖片
    # path_txt = r'../yolov5-6.2/runs/detect/exp6/labels/1_7.txt'  # 物件偵測產出的txt
    # img = cv2.imread(path_img)
    # print(img.shape)
    # H, W, _ = img.shape
    # with open(path_txt, mode='r') as file:
    #     content = file.read()
    # location = list()
    # for label_location in content.split('\n'):
    #     if label_location:
    #         location.append(label_location.split(' ')[1:])
    # print(location)
    #
    # img_list = list()
    # background_rgb = img.copy()
    #
    # for content_img in location:
    #     Xmin, Ymin, Xmax, Ymax = content_img
    #     img_clip = img[int(Ymin):int(Ymax), int(Xmin):int(Xmax), :]  # 切出來的人
    #     img_clip = cv2.resize(img_clip, (224, 224))
    #     background_rgb[int(Ymin):int(Ymax), int(Xmin):int(Xmax), :] = [0, 0, 0]  # 人的位置反黑
    #     img_list.append(img_clip)
    #
    # background = cv2.resize(background_rgb, (224, 224))
    # background_hsv = background.copy()
    # background_hsv = cv2.cvtColor(background_hsv, cv2.COLOR_BGR2HSV)
    # # cv2.imshow('background', background)
    # background_hsv_list = list()
    # a, b, _ = background.shape
    # print(a, type(a))
    # for h in range(a):
    #     for w in range(b):
    #         x, y, z = background[h, w, :]
    #         if x != 0 or y != 0 or z != 0:
    #             background_hsv_list.append(background_hsv[h, w, :])
    # #
    # print(len(background_hsv_list))
    # print(background_hsv_list)
    #
    # for index, i in enumerate(img_list):
    #     cv2.imshow('img_{}'.format(index), i)  # 切割出來的人
    # img = cv2.resize(img, (224, 224))
    # # cv2.imshow('img_original', img)
    # # cv2.waitKey(0)









