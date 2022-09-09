import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
from dataset import Clothes
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pretty_confusion_matrix import pp_matrix_from_data


kind = 'large'
num_classes = 5
weight_path = 'best_test.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 設置GPU device
if kind == 'large':
    net = mobilenet_v3_large(num_classes=num_classes)
else:
    net = mobilenet_v3_small(num_classes=num_classes)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dir = r'data/Pic/Resize_0908/Raw_Data'
test_data = Clothes(data_dir=test_dir, transform=test_transform)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, drop_last=False)

net.load_state_dict(torch.load(weight_path))
net = net.to(device)

loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(device)

net.eval()
correct = 0.
total = 0.
total_loss = 0.
with torch.no_grad():
    for (img, label) in test_loader:
        img, label = img.to(device), label.to(device)
        out = net(img)
        loss = loss_func(out, label)
        total_loss += loss
        _, predicted = torch.max(F.softmax(out.data, dim=1), dim=1)
        print('valid label:\n', label)
        print('valid predicted:\n', predicted)

        total += label.size(0)
        correct += (predicted == label).sum()

    accuracy = correct / total
print(r'Test accuracy： {}%'.format(100 * accuracy))





