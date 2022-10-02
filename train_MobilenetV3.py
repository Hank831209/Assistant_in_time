import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from dataset import Clothes
import matplotlib.pyplot as plt
import numpy as np
import time
from pretty_confusion_matrix import pp_matrix_from_data


def plot_loss(epoch, train, test):  # Loss Curve
    x = np.arange(epoch)
    plt.plot(x, train, 'r--', x, test, 'b^')
    plt.xlabel('Each Epoch')
    plt.ylabel('Total Loss')
    plt.title('Loss Curve')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.show()


def plot_accuracy(epoch, train, test):  # Accuracy Curve
    x = np.arange(epoch)
    plt.plot(x, train, 'r--', x, test, 'b^')
    plt.xlabel('Each Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.show()

    
def plot_confusion_matrix(test_loader, net, device):
    cpu = torch.device('cpu')
    y_true = list()
    y_predict = list()
    net.eval()
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            output = net(img)
            _, predicted = torch.max(output, 1)
            y_predict.extend(predicted.view(-1).detach().to(cpu).numpy())
            y_true.extend(label.view(-1).detach().to(cpu).numpy())

    pp_matrix_from_data(y_true, y_predict, columns=['Baby', 'Princess', 'Casual Wear', 'Gentleman'], 
                        cmap='rainbow', pred_val_axis='x')


# 超參數
start_time = time.time()
MAX_EPOCH = 250
BATCH_SIZE = 30  # 太大的話有些電腦會出錯(GPU負荷不了會出錯)
LR = 0.0001
num_class = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 設置GPU device
cpu = torch.device('cpu')
model_name = 'large'  # 模型種類
    
# 資料前處理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 讀取資料
root = r'Data/Pic/0913A/Split_Data'
train_dir = os.path.join(root, 'train')
test_dir = os.path.join(root, 'test')
train_data = Clothes(data_dir=train_dir, transform=train_transform)
test_data = Clothes(data_dir=test_dir, transform=test_transform)

# 切分BATCH_SIZE
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
print('已讀取完資料')

# 建立模型
if model_name == 'small':
    net = mobilenet_v3_small(num_classes=num_class)
else:
    net = mobilenet_v3_large(num_classes=num_class)
net = net.to(device)

# 損失函數
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(device)

# 優化器
optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))

# 紀錄數據
accuracy_train_global = torch.tensor(0)
accuracy_test_global = torch.tensor(0)
train_curve = list()
test_curve = list()
train_accuracy_curve = list()
test_accuracy_curve = list()

# 訓練
print('開始訓練')
for epoch in range(MAX_EPOCH):
    correct = 0.     # 每個epoch預測正確的總資料數
    total = 0.       # 每個epoch跑過的總資料數
    total_loss = 0.  # 每個epoch的total loss
    net.train()
    for step, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)
        out = net(img)
        optimizer.zero_grad()
        loss = loss_func(out, label)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print('epoch:{}, Train loss:{:.4f}'.format(epoch, loss.data.item()))
        _, predicted = torch.max(F.softmax(out.data, dim=1), dim=1)
        total += label.size(0)  # 總資料數
        correct += (predicted == label).sum()
    print("============================================")
    accuracy = correct / total
    if accuracy > accuracy_train_global:
        torch.save(net.state_dict(), './Data/weights/best_train.pt')
        print('訓練集準確率由：', accuracy_train_global.item(), '上升至：', accuracy.item(),
              '已更新並保存數值為Data/weights/best_train.pt')
        accuracy_train_global = accuracy
    print(r'第{}個epoch的 Train accuracy： {}%'.format(epoch, 100 * accuracy))
    print(r'第{}個epoch的 Train Total Loss： {}'.format(epoch, total_loss))
    train_curve.append(total_loss.detach().to(cpu).numpy())  # plot loss
    train_accuracy_curve.append(accuracy.to(cpu).numpy())  # plot accuracy

    # 驗證
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
            total += label.size(0)
            correct += (predicted == label).sum()

        accuracy = correct / total
        if accuracy > accuracy_test_global:
            torch.save(net.state_dict(), './Data/weights/best_test.pt')
            print('準確率由：', accuracy_test_global.item(), '上升至：', accuracy.item(),
                  '已更新並保存數值為Data/weights/best_test.pt')
            accuracy_test_global = accuracy
            best_test_net = net  # for plot
        print(r'第{}個epoch的 Test accuracy： {}%'.format(epoch, 100 * accuracy))
        print(r'第{}個epoch的 Test Total Loss： {}'.format(epoch, total_loss))
        test_curve.append(total_loss.detach().to(cpu).numpy())  # plot loss
        test_accuracy_curve.append(accuracy.to(cpu).numpy())  # plot accuracy

print('最佳訓練集準確率為:\n', accuracy_train_global.item())
print('最佳測試集準確率為:\n', accuracy_test_global.item())
torch.save(net.state_dict(), './Data/weights/last.pt')
print('訓練完畢，權重已保存為：Data/weights/last.pt')
end_time = time.time()
print('{}個Epoch總運行時間為:\t'.format(MAX_EPOCH), (end_time - start_time) / 60, '分鐘')

print('開始畫Loss圖')
plot_loss(MAX_EPOCH, train_curve, test_curve)  # 畫Debug圖

print('開始畫Accuracy圖')
plot_accuracy(MAX_EPOCH, train_accuracy_curve, test_accuracy_curve)

print('開始畫Confusion Matrix')
plot_confusion_matrix(test_loader, best_test_net, device)  # 畫Confusion Matrix(最準的測試集)
print('程式結束')





