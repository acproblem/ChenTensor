//
// Created by Lenovo on 2023/8/30.
// 绑定到 Python 的 Dropout 类。
//

#ifndef CHENTENSOR_PYDROPOUT_H
#define CHENTENSOR_PYDROPOUT_H


#include "network/Dropout.h"
#include <memory>


/// 绑定到 Python 中的 Dropout 层
template<typename T>
class PyDropout : public Network<T> {
private:
    std::shared_ptr<Dropout<T>> net;

public:
    PyDropout(double p = 0.5)
            : net(Dropout<float>::create(p)) {}

    /// 获取丢弃概率
    double probability() const {
        return net->probability();
    }

    virtual NetType type() override {
        return net->type();
    }

    virtual Tensor<T> forward(Tensor<T> input) override {
        return net->forward(input);
    }

    virtual std::vector<Tensor<T>> parameters() override {
        return net->parameters();
    }
};


#endif //CHENTENSOR_PYDROPOUT_H
