//
// Created by Lenovo on 2023/8/28.
// 提供线性层算子节点（带有偏置 bias）。
//

#ifndef CHENTENSOR___LINEARLAYERNODE___H
#define CHENTENSOR___LINEARLAYERNODE___H

#include "autograd/__OpNode__.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xtensor.hpp>


/// 线性层算子节点
template<typename TI, typename TP>  // TI : Type of input, TP : Type of parameters.
class __LinearNode__ : public __OpNode__ {
public:
    /// 输入操作数，数据节点指针
    std::shared_ptr<__DataNode__<TI>> input;

    /// 参数操作数，数据节点指针
    std::shared_ptr<__DataNode__<TP>> weight, bias;

    /// 结果类型
    using res_type = typename decltype(input->data * weight->data)::value_type;

    /// 计算结果，数据节点指针
    std::weak_ptr<__DataNode__<res_type>> res;

public:
    __LinearNode__(const std::shared_ptr<__DataNode__<TI>> &input, const std::shared_ptr<__DataNode__<TP>> &weight,
                   const std::shared_ptr<__DataNode__<TP>> &bias, const std::shared_ptr<__DataNode__<res_type>> &res)
            : input(input), weight(weight), bias(bias), res(res) {}

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::linalg::dot(input->data, weight->data) + bias->data;
    }

    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += xt::linalg::dot(res->grad(), xt::transpose(weight->data));
            input->backward();
        }
        if (weight->requires_grad()) {
            weight->grad() += xt::linalg::dot(xt::transpose(input->data), res->grad());
            weight->backward();
        }
        if (bias->requires_grad()) {
            bias->grad() += xt::sum(res->grad(), 0);
            bias->backward();
        }
    }

    /// 析构时，取消前面数据节点对该算子节点的 weak_ptr，释放内存。
    virtual ~__LinearNode__() {
        __reset__(this->input, this);
        __reset__(this->weight, this);
        __reset__(this->bias, this);
    }
};


#endif //CHENTENSOR___LINEARLAYERNODE___H
