//
// Created by Lenovo on 2023/8/24.
// Sigmoid 激活函数层类。
//

#ifndef CHENTENSOR_SIGMOID_H
#define CHENTENSOR_SIGMOID_H

#include "Network.h"
#include "functional/__LayerFunc__/__ActivationLayerFunc__.h"
#include <memory>

/// Sigmoid 激活函数层
template<typename T>
class Sigmoid : public Network<T> {
private:
    Sigmoid() {}

    Sigmoid(const Sigmoid<T> &);

    Sigmoid<T> &operator=(const Sigmoid<T> &);

public:
    static std::shared_ptr<Sigmoid<T>> create() {
        return std::shared_ptr<Sigmoid<T>>(new Sigmoid<T>());
    }

    Tensor<T> forward(Tensor<T> input) override {
        return sigmoid(input);
    }

    std::vector<Tensor<T>> parameters() override {
        return {};
    }

    NetType type() override {
        return NetType::Sigmoid;
    }
};

#endif //CHENTENSOR_SIGMOID_H
