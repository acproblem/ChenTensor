//
// Created by Lenovo on 2023/9/7.
// 实现 RMSprop 算法优化器。
//

#ifndef CHENTENSOR_RMSPROP_H
#define CHENTENSOR_RMSPROP_H


#include "optim/Optim.h"
#include "tensor/Tensor.h"

/// RMSprop 算法优化器。
template<typename T>
class RMSprop : public Optim {
private:
    std::vector<Tensor<T>> paras;  // 需要优化的参数
    const double lr;  // 学习率
    const double alpha;  // 平滑系数
    const double eps;
    std::vector<xt::xarray<T>> rms_grads;

public:
    RMSprop(const std::vector<Tensor<T>> &parameters, double lr = 0.01, double alpha = 0.99, double eps = 1e-8)
            : paras(parameters), lr(lr), alpha(alpha), eps(eps) {
        if (alpha < 0 || alpha > 1)
            throw std::runtime_error("The parameter 'alpha' must be between 0 and 1.");

        for (auto &para: paras)
            rms_grads.push_back(xt::zeros<T>(para.shape()));
    }

    virtual void zero_grad() override {
        for (auto &it: paras)
            it.grad().fill(0);
    }

    virtual void step() override {
        for (std::size_t i = 0; i < paras.size(); ++i) {
            rms_grads[i] = alpha * rms_grads[i] + (1 - alpha) * (paras[i].grad() * paras[i].grad());
            paras[i].data() -= lr * paras[i].grad() / (xt::sqrt(rms_grads[i]) + eps);
        }
    }
};


#endif //CHENTENSOR_RMSPROP_H
