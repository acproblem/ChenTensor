//
// Created by Lenovo on 2023/8/28.
// 激活函数算子节点。包含：Sigmoid、ReLU 等。
//

#ifndef CHENTENSOR___ACTIVATIONLAYERNODE___H
#define CHENTENSOR___ACTIVATIONLAYERNODE___H

#include "autograd/__UnaryOpNode__.h"

/// sigmoid 函数算子节点类
template<typename T>
class __SigmoidNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::where(input->data >= 0,
                              1 / (1 + xt::exp(-input->data)),
                              xt::exp(input->data) / (xt::exp(input->data) + 1));
    }

    /// 实现返现传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() * res->data * (1 - res->data);
            input->backward();
        }
    }
};


/// ReLU 函数算子节点类
template<typename T>
class __ReluNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::maximum(input->data, 0.0);
    }

    /// 实现返现传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            xt::xarray<T> _zero(0), _one(1);
            input->grad() += res->grad() * ((input->data > _zero) * _one + (input->data <= _zero) * _zero);
            input->backward();
        }
    }
};


/// LeakyReLU 函数算子节点类
template<typename T>
class __LeakyReluNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    double alpha;

    __LeakyReluNode__(const std::shared_ptr<__DataNode__<T>> &input, const std::shared_ptr<__DataNode__<T>> &res,
                      double alpha = 0.01) : __UnaryOpNode__<T>(input, res), alpha(alpha) {}

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::where(input->data >= 0, input->data, alpha * input->data);
    }

    /// 实现返现传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            xt::xarray<T> _zero(0), _one(1);
            input->grad() += res->grad() * ((input->data > _zero) * _one + (input->data <= _zero) * alpha);
            input->backward();
        }
    }
};


/// Tanh 函数算子节点类
//template<typename T>
//class __TanhNode__ : public __UnaryOpNode__<T> {
//public:
//    using __UnaryOpNode__<T>::input;
//    using __UnaryOpNode__<T>::res;
//    using __UnaryOpNode__<T>::__UnaryOpNode__;
//
//    /// 实现前向传播
//    virtual void forward() {
//        auto res = this->res.lock();
//        res->data = xt::where(input->data >= 0,
//                              (1 - xt::exp(-2 * input->data)) / (1 + xt::exp(-2 * input->data)),
//                              (xt::exp(2 * input->data) - 1) / (xt::exp(2 * input->data) + 1));
//    }
//
//    /// 实现返现传播
//    virtual void backward() {
//        auto res = this->res.lock();
//
//        // 链式法则求导
//        if (input->requires_grad()) {
//            input->grad() += res->grad() * (1 - res->data * res->data);
//            input->backward();
//        }
//    }
//};


#endif //CHENTENSOR___ACTIVATIONLAYERNODE___H
