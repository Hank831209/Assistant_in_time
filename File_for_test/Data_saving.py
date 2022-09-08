import numpy as np
import cv2
import os
from torchvision import transforms
import torch
from Project_model import MyData, Classification

trans = transforms.ToTensor()


# # 圖片前處理
def img_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = trans(img)
    # img = torch.reshape(img, (-1, 1, 256, 256))
    return img


# net = Classification()
# print(net)
# A = torch.ones((64, 3, 256, 256))
# output = net(A)
# print(output)


path_train = r'./Data/Pic'
# path_test = r'./Data/hymenoptera_data/test'
# save data
i = 0
train_data = 0
train_label = 0
for label in os.listdir(path_train):  # label
    for name_image in os.listdir(os.path.join(path_train, label)):  # img_filename
        path_image = os.path.join(path_train, label, name_image)
        img = cv2.imread(path_image)
        img = img_transform(img)  # reshape 過後的 tensor
        a = torch.tensor(int(label))  # label 沒reshape
        a = torch.reshape(a, (-1, 1))  # reshape過的label
        if i == 0:
            train_data = img
            train_label = a
        else:
            train_data = torch.cat((train_data, img), 0)
            train_label = torch.cat((train_label, a), 0)
        i = i + 1
        print(i)


torch.save(train_data, './Data/train_data_gray.pt')
torch.save(train_label, './Data/train_targets_gray.pt')
# torch.save(train_data, './Data/train_data_gray_0.pt')
# torch.save(train_label, './Data/train_targets_gray_0.pt')
print('資料已保存')
print(train_data.shape)
print(train_label.shape)

# i = 0
# test_data = 0
# test_label = 0
# for label in os.listdir(path_test):  # label
#     for name_image in os.listdir(os.path.join(path_test, label)):  # img_filename
#         path_image = os.path.join(path_test, label, name_image)
#         img = cv2.imread(path_image)
#         img = img_transform(img)  # reshape 過後的 tensor
#         a = torch.tensor(int(label))  # label 沒reshape
#         a = torch.reshape(a, (-1, 1))  # reshape過的label
#         if i == 0:
#             test_data = img
#             test_label = a
#         else:
#             test_data = torch.cat((test_data, img), 0)
#             test_label = torch.cat((test_label, a), 0)
#         i = i + 1

# # save data
# torch.save(train_data, './Data/train_data.pt')
# torch.save(train_label, './Data/train_targets.pt')
# torch.save(test_data, './Data/test_data.pt')
# torch.save(test_label, './Data/test_targets.pt')
# print('Program End')


# train_data = torch.load('./Data/train_data.pt')
# train_targets = torch.load('./Data/train_targets.pt')
# test_data = torch.load('./Data/test_data.pt')
# test_targets = torch.load('./Data/test_targets.pt')
# # ---------------------------------

# for check
# path_data = r'./Data'
# train_dataset = MyData(root=path_data, train=True)
# test_dataset = MyData(root=path_data, train=False)
# train_data = train_data
# train_targets = train_targets
# print(train_data.shape)










