//
// Created by Lenovo on 2023/9/9.
// LeakyReLU 层。
//

#ifndef CHENTENSOR_LEAKYRELU_H
#define CHENTENSOR_LEAKYRELU_H


#include "Network.h"
#include "functional/__LayerFunc__/__ActivationLayerFunc__.h"
#include <memory>

/// LeakyReLU 激活函数层
template<typename T>
class LeakyReLU : public Network<T> {
private:
    double m_alpha;

    LeakyReLU(double alpha = 0.01) : m_alpha(alpha) {}

    LeakyReLU(const LeakyReLU<T> &);

    LeakyReLU<T> &operator=(const LeakyReLU<T> &);

public:
    static std::shared_ptr<LeakyReLU<T>> create(double alpha) {
        return std::shared_ptr<LeakyReLU<T>>(new LeakyReLU<T>(alpha));
    }

    Tensor<T> forward(Tensor<T> input) override {
        return leaky_relu(input, m_alpha);
    }

    std::vector<Tensor<T>> parameters() override {
        return {};
    }

    NetType type() override {
        return NetType::LeakyReLU;
    }

    double alpha() const {
        return m_alpha;
    }
};


#endif //CHENTENSOR_LEAKYRELU_H
