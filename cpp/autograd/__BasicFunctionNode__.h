//
// Created by Lenovo on 2023/8/28.
// 基本函数类。包括：指数函数、对数函数等。
//

#ifndef CHENTENSOR___BASICFUNCTIONNODE___H
#define CHENTENSOR___BASICFUNCTIONNODE___H

#include "__UnaryOpNode__.h"


/// 指数函数 e^x 算子节点类
template<typename T>
class __ExpNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::exp(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() * res->data;
            input->backward();
        }
    }
};


/// 自然对数函数 log x 算子节点类，以 e 为底
template<typename T>
class __LogNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::log(input->data);
    }

    /// 实现返现传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() / input->data;
            input->backward();
        }
    }
};


/// 根号算子节点
template<typename T>
class __SqrtNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::sqrt(input->data);
    }

    /// 实现返现传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() / (2 * res->data);
            input->backward();
        }
    }
};


#endif //CHENTENSOR___BASICFUNCTIONNODE___H
