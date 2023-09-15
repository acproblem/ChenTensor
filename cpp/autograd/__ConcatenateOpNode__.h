//
// Created by Lenovo on 2023/9/1.
// 实现连接算子。
//

#ifndef CHENTENSOR___CONCATENATEOPNODE___H
#define CHENTENSOR___CONCATENATEOPNODE___H

#include "__OpNode__.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xtensor.hpp>
#include <vector>
#include <memory>


/// 合并算子，将一系列同类型同形状的数据节点合并起来。
/// 具体为将 {a1, a2, ... , an} 在新建的 0 轴上排列起来。
template<typename T>
class __UnionNode__ : public __OpNode__ {
public:
    /// 输入数据节点列表
    std::vector<std::shared_ptr<__DataNode__<T>>> inputs;

    /// 结果节点
    std::weak_ptr<__DataNode__<T>> res;

public:
    __UnionNode__(const std::vector<std::shared_ptr<__DataNode__<T>>> &inputs,
                  const std::shared_ptr<__DataNode__<T>> &res)
            : inputs(inputs), res(res) {}

    virtual void forward() override {
        auto res = this->res.lock();
        if (inputs.empty())
            res->data = xt::xarray<T>(0);
        else {
            auto shape = inputs[0]->data.shape();
            shape.insert(shape.begin(), inputs.size());
            res->data = xt::empty<T>(shape);
            for (int i = 0; i < inputs.size(); i++)
                xt::view(res->data, i, xt::all()) = inputs[i]->data;
        }
    }

    virtual void backward() override {
        auto res = this->res.lock();

        // 链式法则求导
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs[i]->requires_grad()) {
                inputs[i]->grad() += xt::view(res->grad(), i, xt::all());
                inputs[i]->backward();
            }
        }
    }

    virtual ~__UnionNode__() {
        for (int i = 0; i < inputs.size(); i++) {
            __reset__(this->inputs[i], this);
        }
    }
};


/// 连接算子，将一系列同类型数据节点连接起来
template<typename T>
class __ConcatNode__ : public __OpNode__ {
public:
    /// 输入数据节点列表
    std::vector<std::shared_ptr<__DataNode__<T>>> inputs;

    /// 结果节点
    std::weak_ptr<__DataNode__<T>> res;

    /// 连接的轴
    int axis;

public:
    __ConcatNode__(const std::vector<std::shared_ptr<__DataNode__<T>>> inputs,
                   const std::shared_ptr<__DataNode__<T>> res, int axis = 0)
            : inputs(inputs), res(res), axis(axis) {}

    /// 实现前向传播
    virtual void forward() override {
        if (inputs.empty())
            return;
        auto res = this->res.lock();
        res->data = *inputs[0];
        for (int i = 1; i < inputs.size(); i++)
            res->data = xt::concatenate(xt::xtuple(res->data, *inputs[i]), axis);
    }

    /// 实现反向传播


};

#endif //CHENTENSOR___CONCATENATEOPNODE___H
