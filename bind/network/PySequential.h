//
// Created by Lenovo on 2023/8/30.
// 绑定到 Python 的 Sequential 层。
//

#ifndef CHENTENSOR_PYSEQUENTIAL_H
#define CHENTENSOR_PYSEQUENTIAL_H


#include "network/Sequential.h"
#include <memory>


/// 绑定到 Python 中的 Sigmoid 层
template<typename T>
class PySequential : public Network<T> {
private:
    std::shared_ptr<Sequential<T>> net;

public:
    PySequential(const std::vector<std::shared_ptr<Network<T>>> &nets) : net(Sequential<float>::create(nets)) {}

    /// 获取子网络数量
    std::size_t size() const {
        return net->size();
    }

    /// 获取第 i 个子网络
    const std::shared_ptr<Network<T>> &get(std::size_t idx) {
        return net->get(idx);
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


#endif //CHENTENSOR_PYSEQUENTIAL_H
