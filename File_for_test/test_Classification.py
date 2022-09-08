import torch
from Project_model import Classification
import cv2
from torchvision import transforms

'''
    輸入圖片位置及可預測類別
'''


def img_transform(img):
    img = cv2.resize(img, (32, 32))
    trans = transforms.ToTensor()
    img = trans(img)
    img = torch.reshape(img, (1, 3, 32, 32))
    return img


def predict_img(path):
    img = cv2.imread(path)
    img = img_transform(img)
    net = Classification()
    net.load_state_dict(torch.load(r'./Data/net_params.pkl'))
    net.eval()
    with torch.no_grad():
        dic = {0: 'ants', 1: 'bees'}
        prediction = net(img)
        return dic[prediction.argmax(dim=1).item()]


# ----------------------------------------------------------------------------------
path_img = r'./Data/hymenoptera_data/train/0/0013035.jpg'
prediction_img = predict_img(path_img)
print(prediction_img)

