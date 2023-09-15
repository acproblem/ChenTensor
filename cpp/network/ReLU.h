//
// Created by Lenovo on 2023/8/23.
// ReLU 激活函数层类。
//

#ifndef CHENTENSOR_RELU_H
#define CHENTENSOR_RELU_H

#include "Network.h"
#include "functional/__LayerFunc__/__ActivationLayerFunc__.h"
#include <memory>

/// ReLU 激活函数层
template<typename T>
class ReLU : public Network<T> {
private:
    ReLU() {}

    ReLU(const ReLU<T> &);

    ReLU<T> &operator=(const ReLU<T> &);

public:
    static std::shared_ptr<ReLU<T>> create() {
        return std::shared_ptr<ReLU<T>>(new ReLU<T>());
    }

    Tensor<T> forward(Tensor<T> input) override {
        return relu(input);
    }

    std::vector<Tensor<T>> parameters() override {
        return {};
    }

    NetType type() override {
        return NetType::ReLU;
    }
};

#endif //CHENTENSOR_RELU_H
