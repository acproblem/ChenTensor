//
// Created by Lenovo on 2023/8/21.
// 普通梯度下降优化器。
//

#ifndef CHENTENSOR_GD_H
#define CHENTENSOR_GD_H

#include "optim/Optim.h"
#include "tensor/Tensor.h"

/// 普通梯度下降优化器
template<typename T>
class GD : public Optim {
private:
    std::vector<Tensor<T>> paras;  // 需要优化的参数
    const double lr;  // 学习率

public:
    GD(const std::vector<Tensor<T>> &parameters, double lr = 0.01) : paras(parameters), lr(lr) {}

    virtual void zero_grad() override {
        for (auto &it: paras)
            it.grad().fill(0);
    }

    virtual void step() override {
        for (auto &it: paras)
            it.data() -= lr * it.grad();
    }

};

#endif //CHENTENSOR_GD_H
