import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from torch.utils.data import DataLoader
from Project_model import AutoEncoder, MyData
import os
import time


if __name__ == '__main__':
    # torch.manual_seed(1)
    # 設置GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')

    # 提取數據集
    Path_Data = r'./Data'
    train_dataset = MyData(root=Path_Data, train=True)
    # test_dataset = MyData(root=Path_Data, train=False)

    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, drop_last=False, num_workers=2)

    # # # plot one example
    print(train_dataset.data.size())     # (60000, 28, 28), data
    print(train_dataset.targets.size())     # (60000), label
    # Pic_1 = torch.reshape(train_dataset.data[1], (256, 256))
    plt.imshow(train_dataset.data[1].numpy(), cmap='gray')
    plt.title('%i' % train_dataset.targets[1])
    plt.show()

# -----------------------------------------------------------------------------------------------

    # # 建立模型
    autoencoder = AutoEncoder()
    autoencoder = autoencoder.to(device)

    # 優化器
    learning_rate = 0.005  # learning rate
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss(reduction='sum')
    loss_func = loss_func.to(device)

    # # initialize figure
    N_TEST_IMG = 5
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(N_TEST_IMG, 2))
    plt.ion()  # continuously plot

    # original data (first row) for viewing
    view_data = train_dataset.data[:N_TEST_IMG].view(-1, 128 * 128).type(torch.FloatTensor) / 255.
    for i in range(N_TEST_IMG):
        # 圖片的第一列(), 原始圖片
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (128, 128)), cmap='gray')
        a[0][i].set_xticks(())  # 刪除x, y座標軸刻度
        a[0][i].set_yticks(())  # 刪除x, y座標軸刻度

    start_time = time.time()  # 計時
    EPOCH = 20
    for epoch in range(EPOCH):
        # 訓練模式
        autoencoder.train()
        for step, (x, label) in enumerate(train_loader):
            b_x = x.view(-1, 128 * 128)  # batch x, shape (batch, 28*28)
            b_y = x.view(-1, 128 * 128)  # batch y, shape (batch, 28*28)
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            _, decoded = autoencoder(b_x)
            loss = loss_func(decoded, b_y)      # mean square error
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: {}'.format(loss.item()))

                # plotting decoded image (second row)
                view_data = view_data.to(device)
                _, decoded_data = autoencoder(view_data)
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    # 顯示decoder轉譯出來的圖案
                    a[1][i].imshow(np.reshape(decoded_data.to(device_cpu).data.numpy()[i], (128, 128)), cmap='gray')
                    a[1][i].set_xticks(())  # 刪除x, y座標軸刻度
                    a[1][i].set_yticks(())  # 刪除x, y座標軸刻度
                plt.draw()
                plt.pause(0.05)

    plt.ioff()
    plt.show()
    end_time = time.time()  # 計時
    print('總訓練時間:\t', end_time - start_time)
    # visualize in 3D plot
    N_SHOW_IMG = 200
    view_data = train_dataset.data[:N_SHOW_IMG].view(-1, 128*128).type(torch.FloatTensor)/255.
    view_data = view_data.to(device)
    encoded_data, _ = autoencoder(view_data)
    fig = plt.figure(num=2)  # figure number
    # ax = Axes3D(fig) 寫法新版本已移除, 警告提示建議改為下面兩行的寫法
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # 設定X, Y, Z軸資料
    encoded_data = encoded_data.to(device_cpu)
    X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
    # -----------------------------------------check------------------------------------------------------------
    values = torch.reshape(train_dataset.targets[:N_SHOW_IMG], (-1, ))
    values = values.numpy()
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255*s/9))
        ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())  # 刻度顯示
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()
    torch.save(autoencoder.state_dict(), './Data/autoencoder_params.pkl')  # 存檔
    print('Program End')

