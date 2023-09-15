//
// Created by Lenovo on 2023/9/7.
// 实现 Adam 算法优化器。
//

#ifndef CHENTENSOR_ADAM_H
#define CHENTENSOR_ADAM_H


#include "optim/Optim.h"
#include "tensor/Tensor.h"

/// Adam 算法优化器。
template<typename T>
class Adam : public Optim {
private:
    std::vector<Tensor<T>> paras;  // 需要优化的参数
    const double lr;  // 学习率
    const std::array<double, 2> beta;  // 指数加权平均系数
    const double eps;
    std::vector<xt::xarray<T>> avg_grads;
    std::vector<xt::xarray<T>> rms_grads;

    std::array<double, 2> beta_t;  // beta[0]^t, beta[1]^t

public:
    Adam(const std::vector<Tensor<T>> &parameters, double lr = 0.01, const std::array<double, 2> &beta = {0.9, 0.999},
         double eps = 1e-8)
            : paras(parameters), lr(lr), beta(beta), eps(eps), beta_t({1.0, 1.0}) {
        if (beta[0] <= 0 || beta[0] >= 1)
            throw std::runtime_error("The parameter 'beta[0]' must be between intervals (0, 1)");
        if (beta[1] <= 0 || beta[1] >= 1)
            throw std::runtime_error("The parameter 'beta[1]' must be between intervals (0, 1)");

        for (auto &para: paras) {
            avg_grads.push_back(xt::zeros<T>(para.shape()));
            rms_grads.push_back(xt::zeros<T>(para.shape()));
        }
    }

    virtual void zero_grad() override {
        for (auto &it: paras)
            it.grad().fill(0);
    }

    virtual void step() override {
        xt::xarray<T> avg_grad_correct, rms_grad_correct;
        for (std::size_t i = 0; i < paras.size(); ++i) {
            avg_grads[i] = beta[0] * avg_grads[i] + (1 - beta[0]) * paras[i].grad();
            rms_grads[i] = beta[1] * rms_grads[i] + (1 - beta[1]) * (paras[i].grad() * paras[i].grad());

            beta_t[0] *= beta[0];
            beta_t[1] *= beta[1];
            avg_grad_correct = avg_grads[i] / (1 - beta_t[0]);
            rms_grad_correct = rms_grads[i] / (1 - beta_t[1]);

            paras[i].data() -= lr * avg_grad_correct / (xt::sqrt(rms_grad_correct) + eps);
        }
    }
};


#endif //CHENTENSOR_ADAM_H
