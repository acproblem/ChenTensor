//
// Created by Lenovo on 2023/9/9.
// 三角函数算子节点。
//

#ifndef CHENTENSOR___TRIGONOMETRICOPNODE___H
#define CHENTENSOR___TRIGONOMETRICOPNODE___H

#include "__UnaryOpNode__.h"


/// sin 算子节点类
template<typename T>
class __SinNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::sin(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() * xt::cos(input->data);
            input->backward();
        }
    }
};


/// cos 算子节点类
template<typename T>
class __CosNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::cos(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += -res->grad() * xt::sin(input->data);
            input->backward();
        }
    }
};


/// tan 算子节点类
template<typename T>
class __TanNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::tan(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() * (1 + res->data * res->data);
            input->backward();
        }
    }
};


/// asin 算子节点类
template<typename T>
class __AsinNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::asin(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() / (xt::sqrt(1 - input->data * input->data));
            input->backward();
        }
    }
};


/// acos 算子节点类
template<typename T>
class __AcosNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::acos(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += -res->grad() / (xt::sqrt(1 - input->data * input->data));
            input->backward();
        }
    }
};


/// atan 算子节点类
template<typename T>
class __AtanNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::atan(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() / (1 + input->data * input->data);
            input->backward();
        }
    }
};


/// sinh 算子节点类
template<typename T>
class __SinhNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::sinh(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() * xt::cosh(input->data);
            input->backward();
        }
    }
};


/// cosh 算子节点类
template<typename T>
class __CoshNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::cosh(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() * xt::sinh(input->data);
            input->backward();
        }
    }
};


/// tanh 算子节点类
template<typename T>
class __TanhNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::tanh(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad() * (1 - res->data * res->data);
            input->backward();
        }
    }
};


#endif //CHENTENSOR___TRIGONOMETRICOPNODE___H
