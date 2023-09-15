//
// Created by Lenovo on 2023/9/4.
// Flatten 层。
//

#ifndef CHENTENSOR_FLATTEN_H
#define CHENTENSOR_FLATTEN_H

#include "Network.h"
#include "functional/__OtherFunc__.h"
#include <memory>


/// Flatten 层。
template<typename T>
class Flatten : public Network<T> {
private:
    int m_start_dim, m_end_dim;

    Flatten(int start_dim = 1, int end_dim = -1) : m_start_dim(start_dim), m_end_dim(end_dim) {}

    Flatten(const Flatten<T> &);

    Flatten<T> &operator=(const Flatten<T> &);

public:
    static std::shared_ptr<Flatten<T>> create(int start_dim = 1, int end_dim = -1) {
        return std::shared_ptr<Flatten<T>>(new Flatten<T>(start_dim, end_dim));
    }

    /// 获取 start_dim
    int start_dim() const {
        return m_start_dim;
    }

    /// 获取 end_dim
    int end_dim() const {
        return m_end_dim;
    }

    Tensor<T> forward(Tensor<T> input) override {
        return flatten(input, m_start_dim, m_end_dim);
    }

    std::vector<Tensor<T>> parameters() override {
        return {};
    }

    NetType type() override {
        return NetType::Flatten;
    }
};


#endif //CHENTENSOR_FLATTEN_H
