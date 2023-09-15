//
// Created by Lenovo on 2023/8/28.
// 提供 Dropout 算子节点。
//

#ifndef CHENTENSOR___DROPOUTLAYERNODE___H
#define CHENTENSOR___DROPOUTLAYERNODE___H

#include "autograd/__OpNode__.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xtensor.hpp>


/// Dropout 算子节点
template<typename T>
class __DropoutNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 丢弃概率
    double p;

    /// 丢弃矩阵
    xt::xarray<bool> dropout_mat;

public:
    __DropoutNode__(const std::shared_ptr<__DataNode__<T>> &input, const std::shared_ptr<__DataNode__<T>> &res,
                    double p) : __UnaryOpNode__<T>(input, res), p(p) {}

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();
        dropout_mat = xt::random::rand<double>(input->data.shape()) >= p;
        if (p == 1.0)
            res->data = xt::zeros<T>(input->data.shape());
        else
            res->data = (dropout_mat * input->data) / (1 - p);
//        res->data = p == 1.0 ? xt::zeros<T>(input->data.shape()) : (dropout_mat * input->data) / (1 - p);
    }

    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += dropout_mat * res->grad();
            input->backward();
        }
    }
};


#endif //CHENTENSOR___DROPOUTLAYERNODE___H
