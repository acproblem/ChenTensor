//
// Created by Lenovo on 2023/8/30.
// 绑定到 Python 中的 Sigmoid 层。
//

#ifndef CHENTENSOR_PYSIGMOID_H
#define CHENTENSOR_PYSIGMOID_H


#include "network/Sigmoid.h"
#include <memory>


/// 绑定到 Python 中的 Sigmoid 层
template<typename T>
class PySigmoid : public Network<T> {
private:
    std::shared_ptr<Sigmoid<T>> net;

public:
    PySigmoid() : net(Sigmoid<float>::create()) {}

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


#endif //CHENTENSOR_PYSIGMOID_H
