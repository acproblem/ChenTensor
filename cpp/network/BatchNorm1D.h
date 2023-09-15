//
// Created by Lenovo on 2023/9/11.
// BatchNorm1D 层。
//

#ifndef CHENTENSOR_BATCHNORM1D_H
#define CHENTENSOR_BATCHNORM1D_H

#include "Network.h"
#include "functional/functional.h"
#include <memory>


/// BatchNorm1D 层
template<typename T>
class BatchNorm1D : public Network<T> {
private:
    /// 可学习参数
    Tensor<T> m_gamma, m_beta;

    /// 滑动平均和滑动方差
    Tensor<T> moving_mean, moving_var;

    Tensor<T> m_eps;

    Tensor<T> m_momentum;

    BatchNorm1D(std::size_t num_features, T eps = 1e-5, T momentum = 0.9)
            : m_eps(eps, false), m_momentum(momentum, false) {
        m_gamma = Tensor<T>(xt::ones<T>({num_features}), true);
        m_beta = Tensor<T>(xt::zeros<T>({num_features}), true);
        moving_mean = Tensor<T>(xt::zeros<T>({num_features}), false);
        moving_var = Tensor<T>(xt::zeros<T>({num_features}), false);
    }

    BatchNorm1D(const BatchNorm1D<T> &);

    BatchNorm1D<T> &operator=(const BatchNorm1D<T> &);

public:
    static std::shared_ptr<BatchNorm1D<T>> create(std::size_t num_features, T eps = 1e-5, T momentum = 0.9) {
        if (num_features == 0)
            throw std::runtime_error("The parameter `num_features` must be greater than zero.");
        if (momentum < 0.0 || momentum > 1.0)
            throw std::runtime_error("The parameter `momentum' must be in [0, 1].");
        return std::shared_ptr<BatchNorm1D<T>>(new BatchNorm1D<T>(num_features, eps, momentum));
    }

    Tensor<T> forward(Tensor<T> input) override {
        Tensor<T> mu = mean(input, 0);
        Tensor<T> x_sub_mu = input - mu;
        Tensor<T> var = mean(x_sub_mu * x_sub_mu, 0);

        moving_mean = m_momentum * moving_mean + (Tensor<T>(1, false) - m_momentum) * Tensor<T>(mu.data());
        moving_var = m_momentum * moving_var + (Tensor<T>(1, false) - m_momentum) * Tensor<T>(var.data());

        Tensor<T> norm_input = (input - mu) / (sqrt(var + m_eps));

        return norm_input * m_gamma + m_beta;
    }

    Tensor<T> eval_forward(Tensor<T> input) {
        Tensor<T> norm_input = (input - moving_mean) / (sqrt(moving_var + m_eps));
        return norm_input * m_gamma + m_beta;
    }

    std::vector<Tensor<T>> parameters() override {
        return {m_gamma, m_beta};
    }

    NetType type() override {
        return NetType::BatchNorm1D;
    }

    std::size_t num_features() const {
        return m_gamma.data().size();
    }

    T momentum() const {
        return m_momentum.data()();
    }

    T eps() const {
        return m_eps.data()();
    }
};


#endif //CHENTENSOR_BATCHNORM1D_H
