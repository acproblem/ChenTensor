[TOC]



# ChenTensor Documentation

## Introduction
This is a deep learning framework. It uses dynamic graph technology and runs on the CPU. It uses C++ as the underlying implementation and provides Python APIs. It consists of six main modules: tensor module, data module, general function module, loss function module, neural network module, and optimizer module.

这是一个深度学习框架。它采用动态图技术，在CPU上运行。它使用C++作为底层实现并且提供Python的API。它由6大部分组成：张量模块、数据模块、通用函数模块、损失函数模块、神经网络模块和优化器模块。

1. 张量：提供了四种数据类型的张量：`TensorFloat32`, `TensorFloat64`, `TensorInt32`和`TensorInt64`。通过函数`ChenTensor.tensor`创建张量，更多细节请看`Tensor`一节。
2. 数据：提供了数据集类和数据集加载器类，可以更加灵活地控制数据的加载。这尤其对大量数据很有效，不必一次性将所有数据都加载到内存。
3. 通用函数：提供了丰富的函数库，包括但不限于：矩阵乘法、三角函数、双曲函数、指数函数、对数函数等。
4. 损失函数：提供了两类损失函数分别用于回归问题和分类问题。
5. 神经网络：提供了常用的神经网络模块，包括但不限于：全连接层、卷积层、池化层、循环网络层、激活函数层等。
6. 优化器：提供了常用的优化器，包括：普通梯度下降、Momentum、RMSprop、Adam。

## Setup
Please download the "ChenTensor" folder and all its contents. Place the folder in any location. Create a Python file in the same location, and write code:

请下载“ChenTensor”文件夹及其该文件夹下的所有内容，将该文件夹放在任意位置，在相同的位置创建python文件并编写代码：

```python
import ChenTensor as ct
```

If there are no error messages, the installation is successful.

如果没有报错信息，则安装成功。

## Tensor

`ChenTensor `provides four data types of tensors: `TensorFloat32 `,` TensorFloat64 `,` TensorInt32 `, and` TensorInt64 `, collectively referred to as` Tensor`. Tensors can be created using the factory method `ChenTensor. sensor`. The specific function details are as follows:

`ChenTensor`提供了四种数据类型的张量：`TensorFloat32`, `TensorFloat64`, `TensorInt32`和`TensorInt64`，统称为`Tensor`，可以通过工厂方法`ChenTensor.tensor`来创建张量，具体函数细节如下：

```python
tensor(arr=0.0, dtype=<Dtype.float32: 2>, requires_grad=False)
    Create a tensor.
    创建一个张量。

    Parameters:
        arr : numpy.ndarray, list, int, float
            
        dtype : ChenTensor.Dtype, default = ChenTensor.float32
           	You can also pass ChenTensor.float32, ChenTensor.float64,
        	ChenTensor.int32 and ChenTensor.int64.
            你也可以传递ChenTensor.float32, ChenTensor.float64,
            ChenTensor.int32 和 ChenTensor.int64。
            
        requires_grad : bool, default = False
            Create a tensor with gradient if requires_grad is True.
            如果requires_grad是True，则创建一个带梯度的张量。

    Returns:
        TensorFloat32 if dtype is ChenTensor.float32.
        TensorFloat64 if dtype is ChenTensor.float64.
        TensorInt32 if dtype is ChenTensor.int32.
        TensorInt64 if dtype is ChenTensor.int64.

    Examples:
        >>> import ChenTensor as ct
        >>> import numpy as np
        >>> ct.tensor([0, 1, 2])
        { 0.,  1.,  2.}
        >>> ct.tensor(1)
         1.
        >>> ct.tensor(np.array([1, 2, 3]), dtype=ct.int32, requires_grad=True)
        {1, 2, 3}
```

Tensor classes provide multiple methods, and `Tensor` is used instead of any of the four types of tensor classes. First, take a look at the four operation operators and positive and negative operators of tensor class overloading:

张量类提供了多种方法，下面用`Tensor`代替四种类型张量类的任意一种。首先看一看张量类重载的四则运算操作符和正负操作符：

```python
__add__(self: Tensor, tensor: Tensor) -> Tensor
	Add two tensors.
    加法。
    
__sub__(self: Tensor, tensor: Tensor) -> Tensor
	Subtract two tensors.
    减法。

__mul__(self: Tensor, tensor: Tensor) -> Tensor
	Multiply two tensors.
    乘法。
   
__truediv__(self: Tensor, tensor: Tensor) -> Tensor
	Divide two tensors.
    除法。
    
__pos__(self: Tensor) -> Tensor
	Positive operation.
    取正号运算。

__neg__(self: Tensor) -> Tensor
	Negative operation.
    取负号运算。

Examples:
    >>> import ChenTensor as ct
    >>> a = ct.tensor([1, 2, 3])
    >>> b = ct.tensor([2, 3, 4])
    >>> a + b
    { 3.,  5.,  7.}
    >>> a - b
    {-1., -1., -1.}
    >>> a * b
    {  2.,   6.,  12.}
    >>> a / b
    { 0.5     ,  0.666667,  0.75    }
    >>> +a
    { 1.,  2.,  3.}
    >>> -a
    {-1., -2., -3.}
```

Tensors also provide two read-only attributes and two read-write attributes:

张量还提供了两个只读属性和两个读写属性：

```python
Readonly properties defined here:
dtype
	Return data type.
    返回数据类型。
    
requires_grad
	Returns True if the tensor contains gradients, otherwise returns False.
    如果张量带有梯度，则返回True，否则返回False。
    
----------------------------------------------------------------------
Readwrite properties defined here:
data
	numpy.ndarray. Get/Set data of tensor. 访问或设置张量。
grad
	numpy.ndarray. Get/Set gradient of tensor. 访问或设置张量的梯度。

Examples:
    >>> import ChenTensor as ct
    >>> a = ct.tensor([1, 2, 3])
    >>> a.data
    array([1., 2., 3.], dtype=float32)
    >>> a.data = 0
    >>> a
     0.
    >>> b = ct.tensor([1, 2, 3], dtype=ct.int32, requires_grad=True)
    >>> b.dtype
    <Dtype.int32: 0>
    >>> b.requires_grad
    True
    >>> b.grad
    array(0)
```

In addition, tensors also contain three commonly used methods:

除此之外，张量还含有三个常用方法：

```python
backward(self : Tensor) -> None
	Solving the Gradient of Dynamic Graphs.
    求解动态图中数据节点的梯度。

reshape(self: Tensor, shape: List[int]) -> None
	Change shape of tensor.
    改变张量形状。

shape(self: Tensor) -> List[int]
	Get shape of tensor.
    获取张量形状。
   

Examples:
    >>> import ChenTensor as ct
    >>> a = ct.tensor([1, 2, 3], requires_grad=True)
    >>> b = ct.tensor([4, 5, 6], requires_grad=True)
    >>> d = ct.functional.sum(a * b)
    >>> d
     32.
    >>> d.backward()
    >>> a.grad
    array([4., 5., 6.], dtype=float32)
    >>> b.grad
    array([1., 2., 3.], dtype=float32)
    
    >>> a = ct.tensor([1, 2, 3, 4])
    >>> a.shape()
    [4]
    >>> a.reshape([2, 2])
    >>> a.shape()
    [2, 2]
    >>> a
    {{ 1.,  2.},
     { 3.,  4.}}
```



## Data

The `ChenTensor.data` module has two main classes: `Dataset` and `DataLoader`.

`ChenTensor.data`模块有两个主要的类：`Dataset`和`DataLoader`。

### Dataset

Dataset abstract class, all other dataset classes need to inherit this class and implement `__ getitem__` And`__ len__` Method. More details are as follows:

数据集抽象类，其他所有的数据集类都需继承该类，并且实现`__getitem__`和`__len__`方法。更多细节如下：

```python
class Dataset(builtins.object)
 |  This is a abstract class. All custom dataset classes must inherit this class.
 |  Subclasses need to implement the `__getitem__` method and the `__len__` method.
 |  这是一个抽象类。所有自定义数据集类必须继承该类。
    子类需实现`__getitem__`和`__len__`方法。
 |
 |  Attributes:
 |
 |  Methods:
 |      __getitem__(self, item) : Get tensor. 获取张量。
 |      __len__(self) : Get the number of data. 获取数据条数。
 |
 |  Methods defined here:
 |
 |  __getitem__(self, item)
 |      Get tensor.
 |  	获取张量。
 |
 |      Parameters:
 |          item : int
 |
 |      Returns:
 |          Tensor
 |
 |  __init__(self)
 |      Constructor.
 |		构造方法。
 |
 |  __len__(self)
 |      Get the number of data.
 |		获取数据条数。
 |
 |      Parameters:
 |
 |      Returns:
 |          int : The number of data. 数据条数。
```

The example program is as follows:

示例程序如下：

```python
import ChenTensor as ct
from ChenTensor import data
import numpy as np

class MyDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.y = np.array([0, 1], dtype=np.int32)
    
    def __getitem__(self, item):
        return ct.tensor(self.X[item]), ct.tensor(self.y[item], dtype=ct.int32)
    
    def __len__(self):
        return self.X.shape[0]

dataset = MyDataset()
print("dataset[0]: ", dataset[0])
print("len(dataset): ", len(dataset))

# ------     Result     ------
# dataset[0]:  ({ 1.,  2.}, 0)
# len(dataset):  2
```

For ease of use, the framework provides the `TensorDataset` class, which can be constructed using several tensors with equal 0-axis lengths. This class inherits from `Dataset` and implements`__ getitem__` And`__ len__` Method. More details are as follows:

为了方便使用，框架提供了`TensorDataset`类，可以使用若干0轴长度相等的张量构造该类对象。该类继承自`Dataset`并实现了`__getitem__`和`__len__`方法。更多细节如下：

```python
class TensorDataset(ChenTensor.data.Dataset.Dataset)
 |  TensorDataset(*tensors)
 |
 |  This is a concrete class that inherits from Dataset. 
 |  Encapsulate several tensors into a dataset.
 |	这是一个继承自Dataset的具体类。将若干张量封装为数据集。
 |
 |  Attributes:
 |      tensors : tuple of tensors
 |
 |  Methods:
 |      __init__(self, *tensors) : Transfer several tensors to construct an object.
 |			传递若干张量用于构造对象。
 |
 |  Methods defined here:
 |
 |  __init__(self, *tensors)
 |      Transfer several tensors to construct an object.
 |		传递若干张量用于构造对象。
 |
 |      Parameters:
 |          *tensors : tuple of tensors.
```

示例程序：

```python
import ChenTensor as ct
from ChenTensor import data
import numpy as np

X = ct.tensor([[1, 2], [3, 4]])
y = ct.tensor([0, 1], dtype=ct.int32)

dataset = TensorDataset(X, y)
print("dataset[0]: ", dataset[0])
print("len(dataset): ", len(dataset))

# ------     Result     ------
# dataset[0]:  ({ 1.,  2.}, 0)
# len(dataset):  2
```

### DataLoader

This class is used for data loading, passing the `Dataset` object and batch size `batch_ Size` to construct this type of object. This class implements`__iter__` And`__next__` method for iteratively obtaining small batches of data. More details are as follows:

该类用于数据加载，传递`Dataset`对象和批量大小`batch_size`构造该类对象。该类实现了`__iter__`和`__next__`方法，用于迭代地获取小批量数据。更多细节如下：

```python
class DataLoader(builtins.object)
 |  This is a class for loading data in small batches.
 |	这是一个加载小批量数据的类。
 |
 |  Attributes:
 |      dataset : Dataset
 |      batch_size : int
 |
 |  Methods:
 |      __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True) : Constructor.
 |      __iter__(self) : Return a iterable object.
 |      __next__(self) : Get the next data.
```

Example program:

示例程序：

```python
import ChenTensor as ct
from ChenTensor import data
import numpy as np

X = ct.tensor([[1, 2], [3, 4], [4, 5], [6, 7], [8, 9]])
y = ct.tensor([0, 1, 2, 3, 4], dtype=ct.int32)

dataset = TensorDataset(X, y)

loader = data.DataLoader(dataset, 2)

for i, (X_batch, y_batch) in enumerate(loader):
    print(X_batch)
    print(y_batch)
    print()

# -----     Result     -----
# {{ 3.,  4.},
#  { 1.,  2.}}
# {1, 0}
# 
# {{ 6.,  7.},
#  { 4.,  5.}}
# {3, 2}
# 
# {{ 8.,  9.}}
# {4}
```



## Function

The module `ChenTensor.functional` provides a wealth of functions.

`ChenTensor.functional`模块提供了丰富的函数。

### Matrix multiplication

1. **mm**

    ```python
    mm(mat1: Tensor, mat2: Tensor) -> Tensor
    	The function of matrix multiplication matrix.
        矩阵乘矩阵。
    
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([[1, 2], [3, 4]])
        >>> b = ct.tensor([[5, 6], [7, 8]])
        >>> f.mm(a, b)
        {{ 19.,  22.},
         { 43.,  50.}}
    ```

    

2. **mv**

    ```python
    mv(mat: Tensor, vec: Tensor) -> Tensor
    	The function of matrix multiplication vector.
        矩阵乘向量。
    
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([[1, 2], [3, 4]])
        >>> b = ct.tensor([5, 6])
        >>> f.mv(a, b)
        { 17.,  39.}
    ```

    

3. **linear**

    
   
   ```python
   linear(input: Tensor, weight: Tensor, bias: Tensor) -> Tensor
       Linear layer function.
       线性层函数。
      
   Examples:
       >>> import ChenTensor as ct
       >>> from ChenTensor import functional as f
       >>> a = ct.tensor([[1, 2], [3, 4]])
       >>> w = ct.tensor([[5, 6], [7, 8]])
       >>> b = ct.tensor([1, 1])
       >>> f.linear(a, w, b)
       {{ 20.,  23.},
        { 44.,  51.}}
   ```
   
   

### Common function

1. **exp**

    ```python
    exp(tensor: Tensor) -> Tensor
        Exponential function.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([-1, 0, 1, 2])
        >>> f.exp(a)
        { 0.367879,  1.      ,  2.718282,  7.389056}
    ```

    

2. **log**

    ```python
    log(tensor: Tensor) -> Tensor
        Logarithmic function based on `e`.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([1, 2, 3])
        >>> f.log(a)
        { 0.      ,  0.693147,  1.098612}
    ```

    

3. **sqrt**

    ```python
    sqrt(tensor: Tensor) -> Tensor
        Sqrt function.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([1, 2, 3])
        >>> f.sqrt(a)
        { 1.      ,  1.414214,  1.732051}
    ```

    

### Mean and sum

1. **mean**

    ```python
    1. mean(tensor: Tensor) -> Tensor
    2. mean(tensor: Tensor, axis: List[int]) -> Tensor
    3. mean(tensor: Tensor, axis: int) -> Tensor
        Calculate the mean of a tensor.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([[1, 2], [3, 4]])
        >>> f.mean(a)
         2.5
        >>> f.mean(a, 0)
        { 2.,  3.}
        >>> f.mean(a, [0, 1])
         2.5
    ```

    

2. **sum**

    ```python
    1. sum(tensor: Tensor) -> Tensor
    2. sum(tensor: Tensor, axis: List[int]) -> Tensor
    3. sum(tensor: Tensor, axis: int) -> Tensor
      	Calculate the sum of a tensor.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([[1, 2], [3, 4]])
        >>> f.sum(a)
         10.
        >>> f.sum(a, 0)
        { 4.,  6.}
        >>> f.sum(a, [1])
        { 3.,  7.}
    ```

    

### Activation function

1. **sigmoid**

    ```python
    sigmoid(tensor: Tensor) -> Tensor
        Sigmoid function.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([[1, 2], [3, 4]])
        >>> f.sigmoid(a)
        {{ 0.731059,  0.880797},
         { 0.952574,  0.982014}}
    ```

    

2. **relu**

    ```python
    relu(tensor: Tensor) -> Tensor
        ReLU function.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([-1, 0, 1])
        >>> f.relu(a)
        { 0.,  0.,  1.}
    ```

    

3. **leaky_relu**

    ```python
    leaky_relu(tensor: Tensor, alpha: float) -> Tensor
        Leaky ReLU function.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([-1, 0, 1])
        >>> f.leaky_relu(a, 0.2)
        {-0.2     ,  0.      ,  1.      }
    ```

    

4. **tanh**

    Please refer to the `Hyperbolic function` section for more details.

    详见双曲函数（`Hyperbolic function`）一节。



### Trigonometric function

1. **sin**

    ```python
    sin(tensor: Tensor) -> Tensor
        Sin function.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> import math
        >>> a = ct.tensor([-math.pi, 0, math.pi, 1])
        >>> f.sin(a)
        { 8.742278e-08,  0.000000e+00, -8.742278e-08,  8.414710e-01}
    ```

    

2. **cos**

    ```python
    cos(tensor: Tensor) -> Tensor
        Cos function.
    ```

    

3. **tan**

    ```python
    tan(tensor: Tensor) -> Tensor
        Tan function.
    ```

    

4. **asin**

    ```python
    asin(tensor: Tensor) -> Tensor
        Asin function.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([-1, 0, 1])
        >>> f.asin(a)
        {-1.570796,  0.      ,  1.570796}
    ```

    

5. **acos**

    ```python
    acos(tensor: Tensor) -> Tensor
        Acos function.
    ```

    

6. **atan**

    ```python
    atan(tensor: Tensor) -> Tensor
        Atan function.
    ```

    

### Hyperbolic function

1. **sinh**

    ```python
    sinh(tensor: Tensor) -> Tensor
        Sinh function.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([-1, 0, 1])
        >>> f.sinh(a)
        {-1.175201,  0.      ,  1.175201}
    ```

    

2. **cosh**

    ```python
    cosh(tensor: Tensor) -> Tensor
        Cosh function.
    ```

    

3. **tanh**

    ```python
    tanh(tensor: Tensor) -> Tensor
        Tanh function.
    ```

    

### Others

1. **transpose**

    ```python
    transpose(tensor: Tensor, permutation: List[int] = []) -> Tensor
        Transpose a tensor.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([[1, 2], [3, 4]])
        >>> f.transpose(a)
        {{ 1.,  3.},
         { 2.,  4.}}
        >>> b = ct.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> f.transpose(b, [0, 2, 1])
        {{{ 1.,  3.},
          { 2.,  4.}},
         {{ 5.,  7.},
          { 6.,  8.}}}
    ```

    

2. **squeeze**

    ```python
    squeeze(tensor: Tensor) -> Tensor
        Remove axis with dimension size 1.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([[1, 2]])
        >>> f.squeeze(a)
        { 1.,  2.}
    ```

    

3. **dropout**

    ```python
    dropout(tensor: Tensor, p: float = 0.5) -> Tensor
        Dropout function.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([1, 2, 3, 4, 5, 6])
        >>> f.dropout(a)
        {  0.,   4.,   6.,   0.,   0.,  12.}
        >>> f.dropout(a, 0.9)
        { 0.,  0.,  0.,  0.,  0.,  0.}
    ```

    

4. **flatten**

    ```python
    flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor
    	Flatten the array within the interval of axis [start_dim, end_dim].
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([[1, 2], [3, 4]])
        >>> f.flatten(a)
        { 1.,  2.,  3.,  4.}
    ```

    

5. **union_tensor**

    ```python
    union_tensor(tensors: List[Tensor]) -> Tensor
        Merge a series of tensors with the same shape on the newly created 0-axis.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([1, 2])
        >>> b = ct.tensor([3, 4])
        >>> f.union_tensor([a, b])
        {{ 1.,  2.},
         { 3.,  4.}}
    ```

    

6. **batch_norm1d**

    ```python
    batch_norm1d(input: Tensor, gamma: Tensor, beta: Tensor) -> Tensor
        Batch normalization 1-D function.
       
    Examples:
        >>> import ChenTensor as ct
        >>> from ChenTensor import functional as f
        >>> a = ct.tensor([[1, 2], [3, 4]])
        >>> gamma = ct.tensor([1, 1])
        >>> beta = ct.tensor([0, 0])
        >>> f.batch_norm1d(a, gamma, beta)
        {{-0.999995, -0.999995},
         { 0.999995,  0.999995}}
    ```

    



## Loss

The module `ChenTensor.loss` provides two types of loss function classes: `MSELoss` and `CrossEntropyLoss`. Loss function class implements `__call__` method: Transfer predicted labels and true labels to calculate and return loss value.

`ChenTensor.loss`模块提供了两种损失函数类：`MSELoss`和`CrossEntropyLoss`。损失函数类实现了`__call__`方法，传递预测标签和真实标签，计算损失并返回。

```python
__call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor
	Calculate loss value of `y_pred` and `y_true`.
```



1. **MSELoss**

    ```python
    >>> import ChenTensor as ct
    >>> from ChenTensor import loss
    >>> y_true = ct.tensor([0, 1, 2])
    >>> y_pred = ct.tensor([0.1, 1.1, 2.1])
    >>> loss.MSELoss()(y_pred, y_true)
     0.01
    ```

    

2. **CrossEntropyLoss**

    ```python
    >>> import ChenTensor as ct
    >>> from ChenTensor import loss
    >>> y_pred = ct.tensor([[1, 9], [8, 2]])
    >>> y_true = ct.tensor([1, 0], dtype=ct.int32)
    >>> loss.CrossEntropyLoss()(y_pred, y_true)
     0.001406
    ```

    

## Network

The module `ChenTensor.network` provides commonly used neural network models. This module has a abstract base class named `Network`, which provides  methods such as `forward`. All other neural network classes directly or indirectly inherit this abstract base class and override methods such as `forward`.

ChenTensor.network模块提供了常用的神经网络模型，该模块有一个抽象基类`Network`，提供了`forward`等方法，其他所有的神经网络类都直接或间接地继承该抽象基类，并重写`forward`等方法。

```python
class Network(builtins.object)
 |  This is a network abstract class. All custom network classes must inherit this class.
 |  Subclasses need to implement the `forward` method.
 |  这是一个网络抽象类，所有自定义网络必须继承该类，子类需重写`forward`方法。
 |
 |  Attributes:
 |      mode : str
 |          "train" or "eval".
 |
 |  Methods:
 |      __init__(self) : Constructor.
    
 |      forward(self, inputs) : Forward propagation. Return calculation result.
    
 |      __call__(self, inputs) : Equivalent to the `forward` method.
                 
 |      parameters(self) : Get learnable parameters. Subclasses can override it.
                 
 |      type(self) : Get type of network.
                 
 |      __str__(self) : Return a description for object.
                 
 |      __repr__(self) : Equivalent to the `__str__` method.
                 
 |      train(self) : Set the network to training mode.
                 
 |      eval(self) : Set the network to evaluation mode.
 |
```

The following will introduce each method one by one:

下面将一一介绍每种方法：

```python
__init__(self)
	Constructor.
	构造函数。
```



```python
forward(self, inputs: Tensor) -> Tensor
	Forward propagation. Return calculation result.
    Subclasses need to implement the method.
    前向传播，返回计算结果，子类必须重写该方法。
```



```python
__call__(self, inputs: Tensor) -> Tensor
	Equivalent to the `forward` method.
	等价于`forward`方法。
```



```python
parameters(self) -> List[Tensor]
	Get learnable parameters. Subclasses can override it.
	If a subclass does not override it, the default implementation is
	to return learnable parameters for all member properties
	belonging to the network type.
	获取可学习的参数，子类可以重写它。
	子类如果不重写它，则采取默认实现：返回属于网络（Network）类型的所有成员属性的可学习参数。
```



```python
type(self) -> NetType
	Get type of network.
   
Examples:
    >>> from ChenTensor import network
    >>> net = network.Network()
    >>> net.type()
    <NetType.Undefined: 0>
    >>> net = network.Linear(2, 3)
    >>> net.type()
    <NetType.Linear: 1>
```



```python
__str__(self) -> str
	Return a description for object.
   
Examples:
    >>> from ChenTensor import network
    >>> str(network.Network())
    'Network()'
    >>> str(network.Linear(1, 2))
    'Linear(in_features=1, out_features=2, bias=True)'
```



```python
__repr__(self) -> str
	Equivalent to the `__str__` method.

Examples:
	>>> from ChenTensor import network
	>>> network.Network()
    Network()
    >>> network.Linear(2, 3)
    Linear(in_features=2, out_features=3, bias=True)
```



```python
train(self) -> None
	Set the network to training mode.
eval(self) -> None
	Set the network to evaluation mode.
```



The following will introduce all neural network classes provided by the module `ChenTensor.network`.

下面介绍`ChenTensor.network`提供的神经网络类：

### Linear layers

1. **Linear**

    ```python
    class Linear(ChenTensor.network.network.Network)
     |  Linear(in_features, out_features, bias=True, dtype=<Dtype.float32: 2>)
     |
     |  This is Linear Layer class that inherit Network.
     |
     |  Attributes:
     |      in_features : int
     |          The number of input's features.
     |      out_features : int
     |          The number of output's features.
     |      requires_bias : bool
     |          Whether offset item is required.
     |      weight : Tensor
     |          Weight term.
     |      bias : Tensor
     |          Bias term.
     |
     |  Methods defined here:
     |
     |  __init__(self, in_features, out_features, bias=True, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          in_features : int
     |              The number of input's features.
     |          out_features : int
     |              The number of output's features.
     |          bias : bool
     |              Whether offset item is required.
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |              shape : [batch_size, num_features]
     |
     |      Returns:
     |          Tensor
     |              shape : [batch_size, num_features]
     |
    ```

    



### Convolution layers

1. **Conv2D**

    ```python
    class Conv2D(ChenTensor.network.network.Network)
     |  Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, padding_value=0, bias=True, dtype=<Dtype.float32: 2>)
     |
     |  This is Convolution 2-D Layer class that inherit Network.
     |
     |  Attributes:
     |      in_channels : int
     |          The number of input's channels.
     |      out_channels : int
     |          The number of output's channels.
     |      kernel_size : tuple or list of int (length = 2)
     |          Convolutional kernel size.
     |      stride : tuple or list of int (length = 2)
     |          Convolutional kernel step size.
     |      padding : tuple or list of int (length = 2)
     |          Padding size.
     |      dilation : tuple or list of int (length = 2)
     |          Dilation size.
     |      padding_value : int or float
     |          Padding value.
     |      requires_bias : bool
     |          Whether offset item is required.
     |      weight : Tensor
     |          Convolutional kernel. shape : [out_channels, in_channels, kernel_size[0], kernel_size[1]]
     |      bias : Tensor
     |          Bias term. shape : [out_channels]
     |
     |  Methods defined here:
     |
     |  __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, padding_value=0, bias=True, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          in_channels : int
     |              The number of input's channels.
     |          out_channels : int
     |              The number of output's channels.
     |          kernel_size : tuple or list of int (length = 2)
     |              Convolutional kernel size.
     |          stride : tuple or list of int (length = 2)
     |              Convolutional kernel step size.
     |          padding : tuple or list of int (length = 2)
     |              Padding size.
     |          dilation : tuple or list of int (length = 2)
     |              Dilation size.
     |          padding_value : int or float
     |              Padding value.
     |          bias : bool
     |              Whether offset item is required.
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |              shape : [batch_size, in_channels, input_height, input_width]
     |
     |      Returns:
     |          Tensor
     |              shape : [batch_size, out_channels, output_height, output_width]
     |
    ```

    

### Pooling layers

1. **MaxPool2D**

    ```python
    class MaxPool2D(ChenTensor.network.network.Network)
     |  MaxPool2D(kernel_size, stride=1, padding=0, dilation=1, dtype=<Dtype.float32: 2>)
     |
     |  This is Maximum Pooling 2-D Layer class that inherit Network.
     |
     |  Attributes:
     |      kernel_size : tuple or list of int (length = 2)
     |          Pooling kernel size.
     |      stride : tuple or list of int (length = 2)
     |          Pooling kernel step size.
     |      padding : tuple or list of int (length = 2)
     |          Padding size.
     |      dilation : tuple or list of int (length = 2)
     |          Dilation size.
     |
     |  Methods defined here:
     |
     |  __init__(self, kernel_size, stride=1, padding=0, dilation=1, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          kernel_size : tuple or list of int (length = 2)
     |              Pooling kernel size.
     |          stride : tuple or list of int (length = 2)
     |              Pooling kernel step size.
     |          padding : tuple or list of int (length = 2)
     |              Padding size.
     |          dilation : tuple or list of int (length = 2)
     |              Dilation size.
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |              shape : [batch_size, in_channels, input_height, input_width]
     |
     |      Returns:
     |          Tensor
     |              shape : [batch_size, out_channels, output_height, output_width]
     |
    ```

    

### Recurrent layers

1. **RNNBase**

    ```python
    class RNNBase(ChenTensor.network.network.Network)
     |  RNNBase(input_size, hidden_size, bias=True, dtype=<Dtype.float32: 2>)
     |
     |  This is a RNN network abstract class. All custom RNN network classes must inherit this class.
     |  Subclasses need to implement the `forward` method.
     |
     |  Attributes:
     |      input_size : int
     |          The number of input's features.
     |      hidden_size : int
     |          The number of hidden data's features.
     |      requires_bias : bool
     |          Whether offset item is required.
     |
     |  Methods defined here:
     |
     |  __init__(self, input_size, hidden_size, bias=True, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          input_size : int
     |              The number of input's features.
     |          hidden_size : int
     |              The number of hidden data's features.
     |          bias : bool
     |              Whether offset item is required.
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs, hidden)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |              Input of time t. shape : [batch_size, input_size]
     |          hidden : Tensor
     |              Hidden input of time t. shape : [batch_size, hidden_size]
     |
     |      Returns:
     |          Tensor
     |              Hidden output of time t. shape : [batch_size, hidden_size]
     |
    ```

    

2. **RNN**

    ```python
    class RNN(ChenTensor.network.RNNBase.RNNBase)
     |  RNN(input_size, hidden_size, activation='tanh', bias=True, dtype=<Dtype.float32: 2>)
     |
     |  This is RNN cell class that inherit RNNBase.
     |
     |  Attributes:
     |      activation : str
     |          "tanh" or "sigmoid" or "relu".
     |
     |  Methods defined here:
     |
     |  __init__(self, input_size, hidden_size, activation='tanh', bias=True, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          input_size : int
     |              The number of input's features.
     |          hidden_size : int
     |              The number of hidden data's features.
     |          activation : str
     |              "tanh" or "sigmoid" or "relu".
     |          bias : bool
     |              Whether offset item is required.
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs, hidden)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |              Input of time t. shape : [batch_size, input_size]
     |          hidden : Tensor
     |              Hidden input of time t. shape : [batch_size, hidden_size]
     |
     |      Returns:
     |          Tensor
     |              Hidden output of time t. shape : [batch_size, hidden_size]
     |
    ```

    

3. **GRU**

    ```python
    class GRU(ChenTensor.network.RNNBase.RNNBase)
     |  GRU(input_size, hidden_size, bias=True, dtype=<Dtype.float32: 2>)
     |
     |  This is GRU cell class that inherit RNNBase.
     |
     |  Attributes:
     |
     |  Methods defined here:
     |
     |  __init__(self, input_size, hidden_size, bias=True, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          input_size : int
     |              The number of input's features.
     |          hidden_size : int
     |              The number of hidden data's features.
     |          bias : bool
     |              Whether offset item is required.
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs, hidden)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |              Input of time t. shape : [batch_size, input_size]
     |          hidden : Tensor
     |              Hidden input of time t. shape : [batch_size, hidden_size]
     |
     |      Returns:
     |          Tensor
     |              Hidden output of time t. shape : [batch_size, hidden_size]
     |
    ```

    

4. **LSTM**

    ```python
    class LSTM(ChenTensor.network.RNNBase.RNNBase)
     |  LSTM(input_size, hidden_size, bias=True, dtype=<Dtype.float32: 2>)
     |
     |  This is GRU cell class that inherit RNNBase.
     |
     |  Attributes:
     |
     |  Methods defined here:
     |
     |  __init__(self, input_size, hidden_size, bias=True, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          input_size : int
     |              The number of input's features.
     |          hidden_size : int
     |              The number of hidden data's features.
     |          bias : bool
     |              Whether offset item is required.
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs, hidden, c)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |              Input of time t. shape : [batch_size, input_size]
     |          hidden : Tensor
     |              Hidden input of time t. shape : [batch_size, hidden_size]
     |
     |      Returns:
     |          Tensor
     |              Hidden output of time t. shape : [batch_size, hidden_size]
     |
    ```

    

### Batch normalization layers

1. **BatchNorm1D**

    ```python
    class BatchNorm1D(ChenTensor.network.network.Network)
     |  BatchNorm1D(num_features, eps=1e-05, momentum=0.9, dtype=<Dtype.float32: 2>)
     |
     |  This is Batch Normalization 1-D Layer class that inherit Network.
     |
     |  Attributes:
     |      num_features : int
     |          The number of features.
     |      momentum : float
     |          It is used to compute mean and variance of running time.
     |      eps : float
     |          A value for making the denominator not zero.
     |
     |  Methods defined here:
     |
     |  __init__(self, num_features, eps=1e-05, momentum=0.9, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          num_features : int
     |              The number of features.
     |          momentum : float
     |              It is used to compute mean and variance of running time.
     |          eps : float
     |              A value for making the denominator not zero.
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |              shape : [batch_size, num_features]
     |
     |      Returns:
     |          Tensor
     |              shape : [batch_size, num_features]
     |
    ```

    

2. **BatchNorm2D**

    ```python
    class BatchNorm2D(ChenTensor.network.network.Network)
     |  BatchNorm2D(num_channels, eps=1e-05, momentum=0.9, dtype=<Dtype.float32: 2>)
     |
     |  This is Batch Normalization 2-D Layer class that inherit Network.
     |
     |  Attributes:
     |      num_features : int
     |          The number of features.
     |      momentum : float
     |          It is used to compute mean and variance of running time.
     |      eps : float
     |          A value for making the denominator not zero.
     |
     |  Methods defined here:
     |
     |  __init__(self, num_channels, eps=1e-05, momentum=0.9, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          num_features : int
     |              The number of features.
     |          momentum : float
     |              It is used to compute mean and variance of running time.
     |          eps : float
     |              A value for making the denominator not zero.
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |              shape : [batch_size, in_channels, input_height, input_width]
     |
     |      Returns:
     |          Tensor
     |              shape : [batch_size, out_channels, output_height, output_width]
     |
    ```

    



### Dropout layers

1. **Dropout**

    ```python
    class Dropout(ChenTensor.network.network.Network)
     |  Dropout(p=0.5, dtype=<Dtype.float32: 2>)
     |
     |  This is Dropout Layer class that inherit Network.
     |
     |  Attributes:
     |      probability : float
     |          The dropout probability.
     |
     |  Methods defined here:
     |
     |  __init__(self, p=0.5, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          p : int or float
     |              The dropout probability. It must in [0, 1].
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |
     |      Returns:
     |          Tensor
     |
    ```

    

### Activation layers

1. **Sigmoid**

    ```python
    class Sigmoid(ChenTensor.network.network.Network)
     |  Sigmoid(dtype=<Dtype.float32: 2>)
     |
     |  This is Sigmoid Layer class that inherit Network.
     |      f(x) = 1 / (1 + exp(-x))
     |
     |  Attributes:
     |
     |  Methods defined here:
     |
     |  __init__(self, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |
     |      Returns:
     |          Tensor
     |
    ```

    

2. **ReLU**

    ```python
    class ReLU(ChenTensor.network.network.Network)
     |  ReLU(dtype=<Dtype.float32: 2>)
     |
     |  This is ReLU Layer class that inherit Network.
     |      f(x) = x if x >= 0 else 0
     |
     |  Attributes:
     |
     |  Methods defined here:
     |
     |  __init__(self, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |
     |      Returns:
     |          Tensor
     |
    ```

    

3. **LeakyReLU**

    ```python
    class LeakyReLU(ChenTensor.network.network.Network)
     |  LeakyReLU(alpha=0.01, dtype=<Dtype.float32: 2>)
     |
     |  This is LeakyReLU Layer class that inherit Network.
     |      f(x) = x if x >= 0 else alpha * x
     |
     |  Attributes:
     |      alpha : int or float
     |
     |  Methods defined here:
     |
     |  __init__(self, alpha=0.01, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          alpha : int or float
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |
     |      Returns:
     |          Tensor
     |
    ```

    

4. **Tanh**

    ```python
    class Tanh(ChenTensor.network.network.Network)
     |  Tanh(dtype=<Dtype.float32: 2>)
     |
     |  This is Tanh Layer class that inherit Network.
     |      f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
     |
     |  Attributes:
     |
     |  Methods defined here:
     |
     |  __init__(self, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |
     |      Returns:
     |          Tensor
     |
    ```

    

### Others

1. **Flatten**

    ```python
    class Flatten(ChenTensor.network.network.Network)
     |  Flatten(start_dim=1, end_dim=-1, dtype=<Dtype.float32: 2>)
     |
     |  This is Flatten Layer class that inherit Network.
     |  It flattens the data with dimensions within [start_dim, end_dim].
     |
     |  Attributes:
     |      start_dim : int
     |          The start dimension.
     |      end_dim : int
     |          The end dimension.
     |
     |  Methods defined here:
     |
     |  __init__(self, start_dim=1, end_dim=-1, dtype=<Dtype.float32: 2>)
     |      Constructor.
     |
     |      Parameters:
     |          start_dim : int
     |              The start dimension.
     |          end_dim : int
     |              The end dimension.
     |          dtype : ChenTensor.Dtype
     |              Data type.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |
     |      Returns:
     |          Tensor
     |              shape : [batch_size, out_features]
     |
    ```

    

2. **Sequential**

    ```python
    class Sequential(ChenTensor.network.network.Network)
     |  Sequential(net_lst)
     |
     |  This is Sequential Layer class that inherit Network.
     |
     |  Attributes:
     |
     |  Methods defined here:
     |
     |  __init__(self, net_lst)
     |      Constructor.
     |
     |      Parameters:
     |          net_lst : tuple or list of Networks.
     |
     |  forward(self, inputs)
     |      Forward propagation. Return calculation result.
     |
     |      Parameters:
     |          inputs : Tensor
     |
     |      Returns:
     |          Tensor
     |
     |  get(self, idx)
     |      Get the idx-th Network.
     |
     |      Parameters:
     |          idx : int
     |
     |      Returns:
     |          Network
     |
     |  size(self)
     |      Get the number of networks.
     |
     |      Parameters:
     |
     |      Returns:
     |          int : The number of networks.
     |
    ```

    

## Optimizer

The module `ChenTensor.optim` provides four optimizers. All optimizers implement `zero_grad` and `step` methods. The method `zero_grad` is used to clear gradient to 0.  The method `step` be used to update learnable parameters.

`ChenTensor.optim`模块提供了四种优化器，每种优化器都实现了`zero_grad`和`step`方法。`zero_grad`用于将梯度清零，`step`方法用于更新可学习参数。

Example:

示例：

```python
from ChenTensor import network, optim

net = network.Linear(2, 3)
opt = optim.GD(net.parameters())

opt.zero_grad()  # Clear gradient.
opt.step()  # Update learnable parameters.
```



### GD

```python
class GD(builtins.object)
 |  GD(paras, lr=0.01)
 |
 |  Ordinary gradient descent optimizer.
 |
 |      Parameters:
 |          paras : list of tensor
 |              List of learnable parameters.
 |          lr : float or int
 |              learning rate.
```



### Momentum

```python
class Momentum(builtins.object)
 |  Momentum(paras, lr=0.01, momentum=0.9)
 |
 |  Momentum gradient descent optimizer.
 |
 |      Parameters:
 |          paras : list of tensor
 |              List of learnable parameters.
 |          lr : float or int
 |              learning rate.
 |          momentum : float or int.
```



### RMSprop

```python
class RMSprop(builtins.object)
 |  RMSprop(paras, lr=0.01, alpha=0.99, eps=1e-08)
 |
 |  Momentum gradient descent optimizer
 |
 |      Parameters:
 |          paras : list of tensor
 |              List of learnable parameters.
 |          lr : float or int
 |              learning rate.
 |          alpha : float or int
 |          eps : float
 |              A value for making the denominator not zero.
```



### Adam

```python
class Adam(builtins.object)
 |  Adam(paras, lr=0.01, beta=(0.9, 0.999), eps=1e-08)
 |
 |  Adam gradient descent optimizer
 |
 |      Parameters:
 |          paras : list of tensor
 |              List of learnable parameters.
 |          lr : float or int
 |              learning rate.
 |          beta : tuple of float or int (length = 2)
 |          eps : float
 |              A value for making the denominator not zero.
```

