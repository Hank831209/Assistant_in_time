import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from PIL import Image
import shutil
import time
import os
from yolov5_6_2.detect import parse_opt, main
import numpy as np
import cv2
from pathlib import Path


def delete_dir(path_img, result_dir):
    if os.path.isfile(path_img):  # 刪除原始圖片
        os.remove(path_img)
    if os.path.isdir(result_dir):  # 刪除偵測完的圖片資料夾
        shutil.rmtree(result_dir)


def cut_pic(path_img, path_txt):
    img = cv2.imread(path_img)
    with open(path_txt, mode='r') as file:
        content = file.read()
        
    location = list()
    for label_location in content.split('\n'):
        if label_location:
            location.append(label_location.split(' ')[1:])

    img_list = list()

    for content_img in location:
        Xmin, Ymin, Xmax, Ymax = content_img
        img_clip = img[int(Ymin):int(Ymax), int(Xmin):int(Xmax), :]  # 切出來的人
        img_clip = cv2.resize(img_clip, (224, 224))
        img_list.append(img_clip)

    return img_list
# -------------------------------------------------------------------------------------------------


# 創建檢測器Class, 讀取讀片及預測圖片
class Detector(object):
    def __init__(self, net_kind='large', num_classes=4):
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
    def detect(self, weight_path, img):
        # img 可為PIL image 或 img path
        self.load_weights(weight_path=weight_path)
        if type(img) == str:
            img = Image.open(img).convert('RGB')
        else:
            img = img
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        net_output = self.net(img_tensor)
        _, predicted = torch.max(net_output, dim=1)
        result = predicted[0].item()
        return result  # int


def model(path_img, yolo=True, result_dir='./detect/exp', 
    path_weights='best.pt', num_classes=4, net='large'):

    # 刪除之前的測試結果
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)

    if yolo:
        # Yolov5 物件偵測
        opt = parse_opt()
        main(opt)
        print('Yolo Done')

        # 解析分割後的圖片
        result = list()
        img_name = Path(path_img).stem  # 圖片名稱(不含.jpg)
        path_txt = os.path.join(result_dir, 'labels', f'{img_name}.txt')

        if not os.path.isfile(path_txt):  # 未偵測到人
            return result  
        else:  # 有偵測到人
            img_person_list = cut_pic(path_img, path_txt)
            detector = Detector(net, num_classes=num_classes)
            for each_person in img_person_list:
                pil_image = Image.fromarray(cv2.cvtColor(each_person, cv2.COLOR_BGR2RGB))
                predict_result = detector.detect(path_weights, pil_image)  
                result.append(predict_result)  
            return result  # type: int
    else:
        detector = Detector(net, num_classes=num_classes)
        predict_result = detector.detect(path_weights, path_img)  # 丟圖片路徑即可
        return predict_result


if __name__ == '__main__':
    start_time = time.time()
    pic_path = 'static/1_1.jpg'
    result = model(pic_path, yolo=True)
    print('預測結果為:\t', result)
    print('type(result)', type(result))
    end_time = time.time()
    print('總時間:\t', end_time - start_time)


    '''
    path_img = r'./yolov5_6_2/data/images'  # 照片要存到這個資料夾裡面
    path= 'yolov5_6_2/runs/detect/exp'  # 模型跑完之後的圖片會存到這邊
    result = model('4_1121')  # 回傳照片每個人的服裝風格(list形式), 傳入照片名稱(不用+.jpg)
    建議在最後整個運行完之後把資料夾刪掉, 可以自行斟酌要不要+延遲時間
    '''
    # start_time = time.time()
    # result = model('16794691849152')  
    # print('result:\t', result)
    # if not result:
    #     print('未偵測到照片有人')
    # else:
    #     for i in result:
    #         print('預測結果為:\t', i)
    # end_time = time.time()
    # print('總運行時間為:\t', end_time - start_time)
    
    # time.sleep(3)
    # # 若存在路徑則刪除
    # path= 'yolov5_6_2/runs/detect/exp'
    # if os.path.isdir(path):
    #     delete_dir(path)




