//
// Created by Lenovo on 2023/8/28.
// 提供激活函数。包括：Sigmoid、ReLU 等。
//

#ifndef CHENTENSOR___ACTIVATIONLAYERFUNC___H
#define CHENTENSOR___ACTIVATIONLAYERFUNC___H

#include "autograd/__LayerNode__/__ActivationLayerNode__.h"
#include "tensor/Tensor.h"


/// Sigmoid 函数
template<typename TT>
Tensor<TT>
sigmoid(const Tensor<TT> &input) {
    xt::xarray<TT> _one(1);
    Tensor<TT> res(_one / (_one + xt::exp(-input.data())), input.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __SigmoidNode__<TT>(input.ptr, res.ptr));
        __reference_unary_op__(input, res, op);
    }
    return res;
}


/// ReLU 函数
template<typename TT>
Tensor<TT>
relu(const Tensor<TT> &input) {
    Tensor<TT> res(xt::maximum(input.data(), 0.0), input.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __ReluNode__<TT>(input.ptr, res.ptr));
        __reference_unary_op__(input, res, op);
    }
    return res;
}


/// LeakyReLU 函数
template<typename TT>
Tensor<TT>
leaky_relu(const Tensor<TT> &input, double alpha = 0.01) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __LeakyReluNode__<TT>(input.ptr, res.ptr, alpha));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


#endif //CHENTENSOR___ACTIVATIONLAYERFUNC___H
