//
// Created by Lenovo on 2023/8/30.
// Optim helper class。为了绑定纯虚函数而设立。
//

#ifndef CHENTENSOR_PYOPTIM_H
#define CHENTENSOR_PYOPTIM_H

#include "optim/Optim.h"


/// Optim helper class。为了绑定纯虚函数而设立。
class PyOptim : public Optim {
public:
    using Optim::Optim;

    /// 梯度清零
    void zero_grad() override {
        PYBIND11_OVERRIDE_PURE(void, Optim, zero_grad);
    }

    /// 更新参数
    void step() override {
        PYBIND11_OVERRIDE_PURE(void, Optim, step);
    }
};

#endif //CHENTENSOR_PYOPTIM_H
