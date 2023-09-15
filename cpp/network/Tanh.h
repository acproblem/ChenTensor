//
// Created by Lenovo on 2023/9/9.
// Tanh 层。
//

#ifndef CHENTENSOR_TANH_H
#define CHENTENSOR_TANH_H


#include "Network.h"
#include "functional/__LayerFunc__/__ActivationLayerFunc__.h"
#include <memory>

/// Tanh 激活函数层
template<typename T>
class Tanh : public Network<T> {
private:
    Tanh() {}

    Tanh(const Tanh<T> &);

    Tanh<T> &operator=(const Tanh<T> &);

public:
    static std::shared_ptr<Tanh<T>> create() {
        return std::shared_ptr<Tanh<T>>(new Tanh<T>());
    }

    Tensor<T> forward(Tensor<T> input) override {
        return tanh(input);
    }

    std::vector<Tensor<T>> parameters() override {
        return {};
    }

    NetType type() override {
        return NetType::Tanh;
    }
};


#endif //CHENTENSOR_TANH_H
