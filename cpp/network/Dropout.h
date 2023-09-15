//
// Created by Lenovo on 2023/8/23.
// Dropout 层类。
//

#ifndef CHENTENSOR_DROPOUT_H
#define CHENTENSOR_DROPOUT_H

#include "Network.h"
#include "functional/__LayerFunc__/__DropoutLayerFunc__.h"
#include <memory>

/// Dropout 层
template<typename T>
class Dropout : public Network<T> {
private:
    double p;  // 丢弃概率

    Dropout(double p = 0.5) : p(p) {}

    Dropout(const Dropout<T> &);

    Dropout<T> &operator=(const Dropout<T> &);

public:
    static std::shared_ptr<Dropout<T>> create(double p = 0.5) {
        if (p < 0.0 || p > 1.0)
            throw std::runtime_error("The probability of dropout must be in [0.0, 1.0].");
        return std::shared_ptr<Dropout<T>>(new Dropout<T>(p));
    }

    /// 获取丢弃概率
    double probability() const {
        return p;
    }

    Tensor<T> forward(Tensor<T> input) override {
        return dropout(input, p);
    }

    std::vector<Tensor<T>> parameters() override {
        return {};
    }

    NetType type() override {
        return NetType::Dropout;
    }
};

#endif //CHENTENSOR_DROPOUT_H
