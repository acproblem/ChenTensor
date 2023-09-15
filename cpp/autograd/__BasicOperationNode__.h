//
// Created by Lenovo on 2023/8/28.
// 基本运算算子，提供：加、减、乘、除、正、负等基本算子
//

#ifndef CHENTENSOR___BASICOPERATIONNODE___H
#define CHENTENSOR___BASICOPERATIONNODE___H

#include "__UnaryOpNode__.h"
#include "__BinOpNode__.h"


/// 加法算子
template<typename TL, typename TR>
class __AddNode__ : public __BinOpNode__<TL, TR> {
public:
    using typename __BinOpNode__<TL, TR>::res_type;
    using __BinOpNode__<TL, TR>::__BinOpNode__;
    using __BinOpNode__<TL, TR>::left;
    using __BinOpNode__<TL, TR>::right;
    using __BinOpNode__<TL, TR>::res;

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();
        res->data = left->data + right->data;
    }

    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (left->requires_grad()) {
            __add_grad__<TL, res_type>(left, res->grad());
            left->backward();
        }
        if (right->requires_grad()) {
            __add_grad__<TR, res_type>(right, res->grad());
            right->backward();
        }
    }
};


/// 减法算子
template<typename TL, typename TR>
class __SubNode__ : public __BinOpNode__<TL, TR> {
public:
    using typename __BinOpNode__<TL, TR>::res_type;
    using __BinOpNode__<TL, TR>::__BinOpNode__;
    using __BinOpNode__<TL, TR>::left;
    using __BinOpNode__<TL, TR>::right;
    using __BinOpNode__<TL, TR>::res;

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();
        res->data = left->data - right->data;
    }

    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (left->requires_grad()) {
            __add_grad__<TL, res_type>(left, res->grad());
            left->backward();
        }
        if (right->requires_grad()) {
            __add_grad__<TR, res_type>(right, -res->grad());
            right->backward();
        }
    }
};


/// 乘法算子
template<typename TL, typename TR>
class __MulNode__ : public __BinOpNode__<TL, TR> {
public:
    using typename __BinOpNode__<TL, TR>::res_type;
    using __BinOpNode__<TL, TR>::__BinOpNode__;
    using __BinOpNode__<TL, TR>::left;
    using __BinOpNode__<TL, TR>::right;
    using __BinOpNode__<TL, TR>::res;

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();
        res->data = left->data * right->data;
    }

    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (left->requires_grad()) {
            __add_grad__<TL, res_type>(left, res->grad() * right->data);
            left->backward();
        }
        if (right->requires_grad()) {
            __add_grad__<TR, res_type>(right, res->grad() * left->data);
            right->backward();
        }
    }
};


/// 除法算子
template<typename TL, typename TR>
class __DivNode__ : public __BinOpNode__<TL, TR> {
public:
    using typename __BinOpNode__<TL, TR>::res_type;
    using __BinOpNode__<TL, TR>::__BinOpNode__;
    using __BinOpNode__<TL, TR>::left;
    using __BinOpNode__<TL, TR>::right;
    using __BinOpNode__<TL, TR>::res;

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();
        res->data = left->data / right->data;
    }

    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (left->requires_grad()) {
            __add_grad__<TL, res_type>(left, res->grad() / right->data);
            left->backward();
        }
        if (right->requires_grad()) {
            __add_grad__<TR, res_type>(right, res->grad() * (-left->data) / (right->data * right->data));
            right->backward();
        }
    }
};


/// 正号操作符
template<typename T>
class __PlusNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播接口
    virtual void forward() {
        auto res = this->res.lock();
        res->data = input->data;  // 数据拷贝
    }

    /// 实现反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += res->grad();
            input->backward();
        }
    }
};


/// 负号操作符
template<typename T>
class __MinusNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播接口
    virtual void forward() {
        auto res = this->res.lock();
        res->data = -input->data;
    }

    /// 实现反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += -res->grad();
            input->backward();
        }
    }
};


#endif //CHENTENSOR___BASICOPERATIONNODE___H
