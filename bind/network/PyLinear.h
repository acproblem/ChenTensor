//
// Created by Lenovo on 2023/8/29.
// PyLinear 类。该类将绑定到 Python 中。
//

#ifndef CHENTENSOR_PYLINEAR_H
#define CHENTENSOR_PYLINEAR_H

#include "network/Linear.h"
#include <memory>


/// PyLinear 类。该类将绑定到 Python 中。
template<typename T>
class PyLinear : public Network<T> {
private:
    std::shared_ptr<Linear<T>> net;

public:
    PyLinear(unsigned int in_features, unsigned int out_features, bool bias = true)
            : net(Linear<T>::create(in_features, out_features, bias)) {}

    Tensor<T> weight() {
        return net->weight();
    }

    Tensor<T> bias() {
        return net->bias();
    }

    NetType type() override {
        return NetType::Linear;
    }

    Tensor<T> forward(Tensor<T> input) override {
        return net->forward(input);
    }

    std::vector<Tensor<T>> parameters() override {
        return net->parameters();
    }
};

#endif //CHENTENSOR_PYLINEAR_H
