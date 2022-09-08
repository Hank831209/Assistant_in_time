import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from PIL import Image
import shutil
from Dominant_Color import kmeans, color_check
from clip_img import cut_pic
from detect import parse_opt, main
import time


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
        self.net.load_state_dict(torch.load(weight_path))

    # 檢測器
    def detect(self, weight_path, pic_path, img):
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


if __name__ == '__main__':
    start_time = time.time()
    opt = parse_opt()
    main(opt)

    print('Yolo執行完畢')

    img_name = '111'
    path_img = r'./data/images/{}.jpg'.format(img_name)  # 原始圖片
    path_txt = r'./runs/detect/exp/labels/{}.txt'.format(img_name)  # 物件偵測產出的txt
    background_hsv_list, img_person_list = cut_pic(path_img, path_txt)

    # 算場景色系
    HSV_values = kmeans(path_img, background_hsv_list, k=6, plot=False)
    print('主導色的HSV值為:\n', HSV_values)
    H, S, V = HSV_values
    Dominant_Color = color_check(H, S, V)
    print('主導色為:\n', Dominant_Color)

    # 辨識人物風格
    detector = Detector('large', num_classes=5)
    path_weights = r'best_test.pt'
    for each_person in img_person_list:
        pil_image = Image.fromarray(each_person)
        predict_result = detector.detect(path_weights, '', pil_image)  # 丟圖片路徑即可
        print('預測結果為:\t', predict_result)

    end_time = time.time()
    print('總運行時間為:\t', end_time - start_time)

    # time.sleep(5)
    # # # 程式運行完刪除路徑
    # path = 'runs/detect/exp'
    # if os.path.isdir(path):
    #     try:
    #         shutil.rmtree(path)
    #     except OSError as e:
    #         print(e)
    #     else:
    #         print("The directory is deleted successfully")





