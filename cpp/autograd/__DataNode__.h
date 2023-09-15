//
// Created by Lenovo on 2023/8/16.
// 该文件包括 "数据节点抽象类" 以及派生的两个具体类："带有梯度的数据节点类" 和 "不带梯度的数据节点类"
// 为了成为计算图的一部分，数据节点有两类指针：指向前方算子节点的 shared_ptr 和 指向后方算子节点的 weak_ptr
// 数据节点的由来有两种：
//      1. 通过计算得到。此时指向前方算子节点的 shared_ptr 不为空；
//      2. 用户自定义张量时得到。此时，该数据节点不是通过计算得到，因此没有前方算子节点，则指向前方算子节点的 shared_ptr 为空。
// 该数据节点可能会参与一种或多种运算、也可能不参与运算，因此后方算子节点有 0 至 多个，使用 std::list<weak_ptr<__OpNode__>> 存储。
//

#ifndef CHENTENSOR___DATANODE___H
#define CHENTENSOR___DATANODE___H

#include <xtensor/xarray.hpp>
#include <memory>
#include <list>
#include <stdexcept>
#include "__OpNode__.h"


/// 在这里声明算子节点
class __OpNode__;


/// 数据节点类（抽象类），派生出两个类：带有梯度的数据节点 和 不带梯度的数据节点，计算图中的数据节点，可指向算子节点
template<typename T>
class __DataNode__ {
public:
    /// 类型
    typedef T value_type;

    /// 数据张量
    xt::xarray<T> data;

    /// 指向前方算子节点
    std::shared_ptr<__OpNode__> pre_op;

    /// 指向后方算子节点
    std::list<std::weak_ptr<__OpNode__>> next_ops;

public:
    __DataNode__(const xt::xarray<T> &data = xt::xarray<T>()) : data(data) {}

    /// 返回是否含有梯度
    virtual bool requires_grad() = 0;

    /// 返回梯度引用
    virtual xt::xarray<T> &grad() = 0;

    /// 反向传播
    virtual void backward() = 0;

    virtual ~__DataNode__() {}
};


/// 带有梯度的数据节点
template<typename T>
class __GradDataNode__ : public __DataNode__<T> {
private:
    /// 梯度
    xt::xarray<T> m_grad;

public:
    __GradDataNode__(const xt::xarray<T> &data = xt::xarray<T>()) : __DataNode__<T>(data),
                                                                    m_grad(0) {}

    /// 返回是否含有梯度
    virtual bool requires_grad() {
        return true;
    }

    /// 返回梯度引用
    virtual xt::xarray<T> &grad() {
        return m_grad;
    }

    /// 反向传播
    virtual void backward() {
        // 链式法则，调用其余节点的反向传播
        if (this->pre_op) {
            this->pre_op->backward();
            this->pre_op.reset();
        }
    }
};


/// 不带梯度的数据节点，返回梯度方法抛出异常
template<typename T>
class __NonGradDataNode__ : public __DataNode__<T> {
public:
    __NonGradDataNode__(const xt::xarray<T> &data = xt::xarray<T>()) : __DataNode__<T>(data) {}

    /// 返回是否含有梯度
    virtual bool requires_grad() {
        return false;
    }

    /// 返回梯度引用，抛出异常
    virtual xt::xarray<T> &grad() {
        throw std::runtime_error("Type \"__NonGradDataNode__\" has not gradient.");
    }

    /// 反向传播
    virtual void backward() {
        throw std::runtime_error("Type \"__NonGradDataNode__\" can not backward.");
    }
};


#endif //CHENTENSOR___DATANODE___H
