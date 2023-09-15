//
// Created by Lenovo on 2023/8/30.
// 绑定 ReLU 层到 Python。
//

#ifndef CHENTENSOR_PYRELU_H
#define CHENTENSOR_PYRELU_H


#include "network/ReLU.h"
#include <memory>


/// 绑定到 Python 中的 ReLU 层
template<typename T>
class PyReLU : public Network<T> {
private:
    std::shared_ptr<ReLU<T>> net;

public:
    PyReLU() : net(ReLU<float>::create()) {}

    virtual NetType type() override {
        return net->type();
    }

    virtual Tensor<T> forward(Tensor<T> input) override {
        return net->forward(input);
    }

    virtual std::vector<Tensor<T>> parameters() override {
        return net->parameters();
    }
};


#endif //CHENTENSOR_PYRELU_H
