import numpy as np
import cv2
import os


# path = r'../Data/Pic/0'
for i in range(1, 5+1):
    path = r'../Data/Pic/{}'.format(i)
    for old_name in os.listdir(path):
        old_path = os.path.join(path, old_name)
        new_path = r'../Data/Pic/Resize/{}/{}'.format(i, old_name)
        img1 = cv2.imread(old_path)
        img = cv2.resize(img1, (224, 224))
        cv2.imwrite(new_path, img)
        print('{}已儲存'.format(old_name))


# # 圖片前處理
# def img_transform(img):
#     img = cv2.resize(img, (28, 28))
#     img = trans(img)
#     img = torch.reshape(img, (-1, 3, 28, 28))
#     return img

# path_train = r'./Data/hymenoptera_data/train'
# path_test = r'./Data/hymenoptera_data/test'


# net = Classification()
# print(net)
# A = torch.ones((64, 3, 256, 256))
# output = net(A)
# print(output)




