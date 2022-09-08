import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
from PIL import Image


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
    detector = Detector('small', num_classes=6)
    path_weights = r'./Data/weights/best.pkl'
    path_img = r'./Data/Pic/Resize/Raw_Data/3/3_3.jpg'
    predict_result = detector.detect(path_weights, path_img, img=False)  # 丟圖片路徑即可
    print('預測結果為:\t', predict_result)






