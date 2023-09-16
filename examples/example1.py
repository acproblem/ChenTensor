import numpy as np
import gzip
import ChenTensor as ct
from ChenTensor import network, loss, optim, data


# Load data. 定义加载数据的函数
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as file:
        magic = int.from_bytes(file.read(4), byteorder='big')
        num_images = int.from_bytes(file.read(4), byteorder='big')
        num_rows = int.from_bytes(file.read(4), byteorder='big')
        num_cols = int.from_bytes(file.read(4), byteorder='big')

        image_data = file.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, num_rows, num_cols)

        return images


# Load labels. 定义加载标签的函数
def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as file:
        magic = int.from_bytes(file.read(4), byteorder='big')
        num_labels = int.from_bytes(file.read(4), byteorder='big')

        label_data = file.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)

        return labels


# Dataset 数据集
images = load_mnist_images('../dataset/mnist/train-images-idx3-ubyte.gz').reshape((60000, 1, 28, 28)) / 255
labels = load_mnist_labels('../dataset/mnist/train-labels-idx1-ubyte.gz')
trainset = data.TensorDataset(ct.tensor(images[:50000]), ct.tensor(labels[:50000], dtype=ct.int32))
testset = data.TensorDataset(ct.tensor(images[50000:]), ct.tensor(labels[50000:], dtype=ct.int32))


# DataLoader 数据集加载器
train_batch_size, test_batch_size = 256, 256
trainloader = data.DataLoader(trainset, train_batch_size)
testloader = data.DataLoader(testset, test_batch_size)


# Network 网络
class MyNet(network.Network):
    def __init__(self):
        self.net = network.Sequential([
            network.Conv2D(1, 6, kernel_size=(5, 5), padding=2),
            network.MaxPool2D(kernel_size=(2, 2), stride=2),
            network.Conv2D(6, 16, kernel_size=5),
            network.BatchNorm2D(16),
            network.MaxPool2D(2, 2),
            network.Flatten(),
            network.Linear(16 * 5 * 5, 120),
            network.LeakyReLU(0.1),
            network.Linear(120, 84),
            network.ReLU(),
            network.Linear(84, 10)
        ])

    def forward(self, inputs):
        return self.net(inputs)


net = MyNet()

# Optimizer 优化器
# opt = optim.GD(net.parameters(), lr=0.1)
# opt = optim.Momentum(net.parameters(), lr=0.1)
# opt = optim.RMSprop(net.parameters(), 0.001)
opt = optim.Adam(net.parameters())

# Loss function 损失函数
lossfunc = loss.CrossEntropyLoss()

# training 训练
epochs = 1
for epoch in range(epochs):
    for i, (X, y) in enumerate(trainloader):
        y_pred = net(X)

        l = lossfunc(y_pred, y)

        l.backward()

        opt.step()

        opt.zero_grad()

        print(f"Epoch: {epoch + 1}/{epochs}, iter: {i + 1}, loss: {l}")

# Predicting 预测
net.eval()
y_pred = np.zeros(len(testset), dtype=np.int32)
y_true = np.zeros(len(testset), dtype=np.int32)
for i, (X, y) in enumerate(testloader):
    y_pred[i * test_batch_size: (i + 1) * test_batch_size] = net(X).data.argmax(axis=1)
    y_true[i * test_batch_size: (i + 1) * test_batch_size] = y.data

print("accuracy: ", np.mean(y_pred == y_true))

# -----     optim.GD(net.parameters(), lr=0.1)     -----
# Epoch: 1/1, iter: 194, loss:  0.358778
# Epoch: 1/1, iter: 195, loss:  0.253631
# Epoch: 1/1, iter: 196, loss:  0.513843
# accuracy:  0.9036

# -----     optim.Momentum(net.parameters())     -----
# Epoch: 1/1, iter: 194, loss:  0.390568
# Epoch: 1/1, iter: 195, loss:  0.379375
# Epoch: 1/1, iter: 196, loss:  0.26708
# accuracy:  0.9003

# -----     optim.RMSProp(net.parameters(), 0.001)     -----
# Epoch: 1/1, iter: 194, loss:  0.161929
# Epoch: 1/1, iter: 195, loss:  0.206539
# Epoch: 1/1, iter: 196, loss:  0.164884
# accuracy:  0.9436

# -----     optim.Adam(net.parameters(), eps=1e-1)     -----
# Epoch: 1/1, iter: 194, loss:  0.101545
# Epoch: 1/1, iter: 195, loss:  0.162649
# Epoch: 1/1, iter: 196, loss:  0.11746
# accuracy:  0.9632
