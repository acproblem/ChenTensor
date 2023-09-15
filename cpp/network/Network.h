//
// Created by Lenovo on 2023/8/20.
// 神经网络抽象类。所有的网络继承该网络。
//

#ifndef CHENTENSOR_NETWORK_H
#define CHENTENSOR_NETWORK_H

#include "tensor/Tensor.h"
#include "NetType.h"
#include <vector>


/// 神经网络基类
template<typename T>
class Network {
public:
    /// 前向传播接口
    virtual Tensor<T> forward(Tensor<T> input) = 0;

    virtual /// 重载 () 运算符
    inline Tensor<T> operator()(Tensor<T> input) {
        return this->forward(input);
    }

    /// 获取网络参数
    virtual std::vector<Tensor<T>> parameters() = 0;

    /// 获取网络类型
    virtual NetType type() {
        return NetType::Undefined;
    }


    virtual ~Network() {}

};


#endif //CHENTENSOR_NETWORK_H
