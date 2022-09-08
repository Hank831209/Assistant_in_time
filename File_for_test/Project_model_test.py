from torch import nn
from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2
import torch


class MyData(Dataset):

    def __init__(self, root, train):
        self.train = train
        self.root = root  # r'C:/workspace/Tibame_Project/Data'
        train_data = torch.load(os.path.join(self.root, 'train_data.pt'))
        train_targets = torch.load(os.path.join(self.root, 'train_targets.pt'))
        test_data = torch.load(os.path.join(self.root, 'test_data.pt'))
        test_targets = torch.load(os.path.join(self.root, 'test_targets.pt'))

        if self.train:
            self.data = train_data
            self.targets = train_targets
        else:
            self.data = test_data
            self.targets = test_targets

    def __getitem__(self, idx):
        data = (self.data[idx], self.targets[idx])
        return data

    def __len__(self):
        return int(self.data.shape[0])


# ----------------------------------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),  # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# ---------------------------------------------------------------------------------------------------------


class MyData_old(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # './Data/hymenoptera_data/train', training data的資料夾
        self.label_dir = label_dir  # '0' ---> label(設為資料夾名稱)
        self.path = os.path.join(self.root_dir, self.label_dir)  # '.Data/hymenoptera_data/train/0'
        self.img_path = os.listdir(self.path)  # 所有圖片檔案名稱的list, ex: [00013035.jpg, ...]

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 第 idx 張圖片的檔案名稱, ex: 00013035.jpg
        img_item_path = os.path.join(self.path, img_name)  # ex: '.Data/hymenoptera_data/train/0/00013035.jpg'
        img = cv2.imread(img_item_path)
        img = self.transform(img)  # 圖片前處理(轉tensor等等)
        label = int(self.label_dir)  # class0 ---> ants, ...
        data = (img, label)
        return data

    def __len__(self):
        return len(self.img_path)

    @staticmethod
    def transform(img):
        # img ---> PIL image
        img = cv2.resize(img, (28, 28))
        trans = transforms.ToTensor()
        img = trans(img)
        return img


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 幾分類問題
            nn.Softmax(dim=1)
        )

    def forward(self, x):  # x = x.view(x.size(0), -1)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x




