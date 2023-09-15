//
// Created by Lenovo on 2023/8/29.
// 网络类型。
//

#ifndef CHENTENSOR_NETTYPE_H
#define CHENTENSOR_NETTYPE_H

enum class NetType {
    Undefined,
    Linear,
    Sequential,
    Dropout,
    ReLU, Sigmoid, LeakyReLU, Tanh,
    Conv1D, Conv2D, Conv3D,
    MaxPool1D, MaxPool2D, MaxPool3D,
    AvgPool1D, AvgPool2D, AvgPool3D,
    Flatten,
    BatchNorm1D, BatchNorm2D, BatchNorm3D,
    RNN, GRU, LSTM
};


#endif //CHENTENSOR_NETTYPE_H
