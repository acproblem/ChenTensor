# <center>ChenTensor
## Introduction
This is a deep learning framework. It uses dynamic graph technology and runs on the CPU. It uses C++ as the underlying implementation and provides Python APIs. It consists of six main modules: tensor module, data module, general function module, loss function module, neural network module, and optimizer module.

这是一个深度学习框架。它采用动态图技术，在CPU上运行。它使用C++作为底层实现并且提供Python的API。它由6大部分组成：张量模块、数据模块、通用函数模块、损失函数模块、神经网络模块和优化器模块。

For a more detailed understanding of the library, please refer to the documentation.

若想更加深入了解该库，请查看说明文档。

## Setup

Please download the "ChenTensor" folder and all its contents. Place the folder in any location. Create a Python file in the same location, and write code:

请下载“ChenTensor”文件夹及其该文件夹下的所有内容，将该文件夹放在任意位置，在相同的位置创建python文件并编写代码：

```python
import ChenTensor as ct
```

If there are no error messages, the installation is successful.

如果没有报错信息，则安装成功。

## Quick start
Taking the MNIST dataset as an example, establish a convolutional neural network, train and test it.

这里以MNIST数据集为例，建立一个卷积神经网络，训练并测试。

```python
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
```

## Range

1. Data types:
    - 32-bit float
    - 64-bit float
    - 32-bit integer
    - 64-bit integer
2. Tensor operations:
    - Four arithmetic operations
    - Common functions: exp, log, sqrt, etc.
    - Common trigonometric functions
    - Common hyperbolic function
    - Matrix multiplication
    - Others: minus, mean, variance, concatenate, etc.
3. Network Models:
    - Linear layers
    - Convolution layers
    - Pooling layers
    - Recurrent layers
    - Batch normalization layers
    - Dropout layers
    - Activation layers
    - Others: flatten layers, sequential layers
4. Loss functions:
    - Regression: MSELoss
    - classification: CrossEntropyLoss
5. Optimizers:
    - Common gradient descent
    - Momentum
    - RMSprop
    - Adam
6. Data IO:
    - Dataset
    - DataLoader



1. 数据类型：
    - 32位浮点数
    - 64位浮点数
    - 32位整数
    - 64位整数
2. 张量操作：
    - 四则运算
    - 常见函数：指数、对数、根号等
    - 常见三角函数
    - 常见双曲函数
    - 矩阵乘法
    - 其他：取负号运算、均值、方差、连接等
3. 网络模型：
    - 线性层（全连接层）
    - 卷积层
    - 池化层
    - 循环网络层
    - 批归一化层
    - 丢弃层
    - 激活函数层
    - 其他：平铺，序列
4. 损失函数：
    - 回归：MSELoss
    - 分类：CrossEntropyLoss
5. 优化器：
    - 普通梯度下降
    - Momentum
    - RMSprop
    - Adam
6. 数据IO：
    - 数据集类
    - 数据加载器类

## Framework implementation

The underlying framework is implemented in C++ and uses the `xtensor` multidimensional array library to construct tensors. By using template technology, dynamic graphs and rich operators (such as four arithmetic operations, exp, log, trigonometric functions, matrix multiplication, convolution, pooling, etc.) have been implemented. Automatic broadcasting is achieved during tensor operation.

Bind classes and functions written in C++ to Python through the `Pybind11` library, and write Python APIs.

该框架底层由C++实现，使用了`xtensor`多维数组库构建张量。采用模板技术，实现了动态图和丰富的算子（如：四则运算、指数、对数、三角函数、矩阵乘法、卷积、池化等）。张量运算时实现了自动广播。

通过`Pybind11`库将C++编写的类和函数绑定到Python，并编写Python的API。

## Contact

E-mail: 1935663328@qq.com

