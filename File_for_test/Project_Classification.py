import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from Project_model import MyData, Classification
from torch.utils.tensorboard import SummaryWriter
# from PIL import Image


if __name__ == '__main__':
    # 提取數據集
    path_Data = r'./Data'
    train_dataset = MyData(root=path_Data, train=True)
    test_dataset = MyData(root=path_Data, train=False)

    # 資料集的資料總數
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)

    # 批訓練
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True, drop_last=False, num_workers=2)

    # 建立模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 設置GPU device
    net = Classification()
    net = net.to(device)  # 調用GPU

    # 損失函數
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(device)  # 調用GPU
    # 優化器
    learning_rate = 0.005
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # training and testing
    # writer = SummaryWriter('logs')
    total_train_step = 0  # 記錄畫圖用的x軸座標
    EPOCH = 15
    for epoch in range(EPOCH):
        print("-------第 {} 輪訓練開始-------".format(epoch + 1))
        # 訓練
        net.train()  # self.train = True
        for (img, label) in train_loader:
            img = img.to(device)  # 調用GPU
            label = label.to(device)  # 調用GPU
            output = net(img)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 畫圖記錄誤差
            total_train_step = total_train_step + 1
            if total_train_step % 10 == 0:
                print("訓練次數：{}, Loss: {}".format(total_train_step, loss.item()))
                # writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 驗證
        net.eval()  # self.train = False
        total_correct = 0
        total_test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for (imgs, labels) in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = net(imgs)
                loss = loss_func(outputs, labels)
                total_test_loss = total_test_loss + loss.item()
                predict_correct_num = (outputs.argmax(dim=1) == labels).sum().item()
                total_correct = total_correct + predict_correct_num

        accuracy = total_correct / test_data_size
        print("epoch: {}, 驗證集上的總Loss: {}".format(epoch, total_test_loss))
        print("epoch: {}, 驗證集上的正確率: {}".format(epoch, accuracy))
        # writer.add_scalar("test_loss", total_test_loss, epoch)
        # writer.add_scalar("test_accuracy", accuracy, epoch + 1)

    # torch.save(net, "./Data/net.pkl")  # 存整個網路
    # torch.save(net.state_dict(), './Data/net_params.pkl')  # 只存網路參數(官方推薦)
    print("模型已保存")
    # writer.close()


