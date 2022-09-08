from torch import nn
from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2
import torch


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(128 * 128, 100),
            nn.Tanh(),
            nn.Linear(100, 20),
            nn.Tanh(),
            nn.Linear(20, 5),
            nn.Tanh(),
            nn.Linear(5, 3),  # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.Tanh(),
            nn.Linear(5, 20),
            nn.Tanh(),
            nn.Linear(20, 100),
            nn.Tanh(),
            nn.Linear(100, 128*128),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class MyData(Dataset):

    def __init__(self, root, train):
        self.train = train
        self.root = root
        if self.train:
            self.data = torch.load(os.path.join(self.root, 'train_data_gray.pt'))
            self.targets = torch.load(os.path.join(self.root, 'train_targets_gray.pt'))
        else:
            self.data = torch.load(os.path.join(self.root, 'test_data.pt'))
            self.targets = torch.load(os.path.join(self.root, 'test_targets.pt'))

    def __getitem__(self, idx):
        data = (self.data[idx], int(self.targets[idx]))
        return data

    def __len__(self):
        return int(self.data.shape[0])


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 2, 0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 2, 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),  # 幾分類問題
            nn.Softmax(dim=1)
        )

    def forward(self, x):  # x = x.view(x.size(0), -1)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x
