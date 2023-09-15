//
// Created by Lenovo on 2023/9/7.
// 带动量（Momentum）的梯度下降优化器。
//

#ifndef CHENTENSOR_MOMENTUM_H
#define CHENTENSOR_MOMENTUM_H


#include "optim/Optim.h"
#include "tensor/Tensor.h"

/// 普通梯度下降优化器
template<typename T>
class Momentum : public Optim {
private:
    std::vector<Tensor<T>> paras;  // 需要优化的参数
    const double lr;  // 学习率
    const double momentum;  // 动量项系数
    std::vector<xt::xarray<T>> avg_grads;  // 指数加权平均梯度

public:
    Momentum(const std::vector<Tensor<T>> &parameters, double lr = 0.01, double momentum = 0.9)
            : paras(parameters), lr(lr), momentum(momentum) {
        if (momentum < 0 || momentum > 1)
            throw std::runtime_error("The parameter 'momentum' must be between 0 and 1.");

        for (auto &para: paras)
            avg_grads.push_back(xt::zeros<T>(para.shape()));
    }

    virtual void zero_grad() override {
        for (auto &it: paras)
            it.grad().fill(0);
    }

    virtual void step() override {
        for (std::size_t i = 0; i < paras.size(); ++i) {
            avg_grads[i] =
                    momentum * avg_grads[i] + (1 - momentum) * paras[i].grad();
            paras[i].data() -= lr * avg_grads[i];
        }
    }
};


#endif //CHENTENSOR_MOMENTUM_H
