import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from Project_model_test import AutoEncoder


if __name__ == '__main__':
    # torch.manual_seed(1)    # reproducible, 固定隨機, for debug
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 設置GPU device
    N_TEST_IMG = 5  # 對照圖數量

    # Mnist digits dataset
    train_data = torchvision.datasets.MNIST(
        root='../Data/mnist/',
        train=True,                                     # this is training data
        transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=True,                        # download it if you don't have it
    )

    # plot one example
    print(train_data.data.size())     # (60000, 28, 28), data
    print(train_data.targets.size())     # (60000), label
    plt.imshow(train_data.data[1].numpy(), cmap='gray')
    plt.title('%i' % train_data.targets[1])
    plt.show()

    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    BATCH_SIZE = 64
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# -----------------------------------------------------------------------------------------------
    # 建立模型
    autoencoder = AutoEncoder()
    # autoencoder.to(device)

    # 優化器
    learning_rate = 0.005  # learning rate
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    # loss_func.to(device)

    # initialize figure
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()   # continuously plot

    # original data (first row) for viewing
    view_data = train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
    for i in range(N_TEST_IMG):
        # 圖片的第一列(), 原始圖片
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(())  # 刪除x, y座標軸刻度
        a[0][i].set_yticks(())  # 刪除x, y座標軸刻度

    EPOCH = 1
    for epoch in range(EPOCH):
        # 訓練模式
        autoencoder.train()
        for step, (x, label) in enumerate(train_loader):
            b_x = x.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
            b_y = x.view(-1, 28 * 28)  # batch y, shape (batch, 28*28)
            encoded, decoded = autoencoder(b_x)
            loss = loss_func(decoded, b_y)      # mean square error
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

                # plotting decoded image (second row)
                _, decoded_data = autoencoder(view_data)
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    # 顯示decoder轉譯出來的圖案
                    a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(())  # 刪除x, y座標軸刻度
                    a[1][i].set_yticks(())  # 刪除x, y座標軸刻度
                plt.draw()
                plt.pause(0.05)

    plt.ioff()
    plt.show()

    # visualize in 3D plot
    view_data = train_data.data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
    # encoded_data: AutoEncoder轉譯出來的圖片向量
    encoded_data, _ = autoencoder(view_data)  # encoded_data:
    fig = plt.figure(num=2)  # figure number
    # ax = Axes3D(fig) 寫法新版本已移除, 警告提示建議改為下面兩行的寫法
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # 設定X, Y, Z軸資料
    X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
    values = train_data.targets[:200].numpy()
    print('type(values):\t', type(values))
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255*s/9))
        ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())  # 刻度顯示
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()
    # torch.save(autoencoder.state_dict(), './Data/autoencoder_params.pkl')  # 存檔
    print('Program End')

