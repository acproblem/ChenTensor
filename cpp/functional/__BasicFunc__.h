//
// Created by Lenovo on 2023/8/28.
// 基础函数。包括：指数、对数 等。
//

#ifndef CHENTENSOR___BASICFUNC___H
#define CHENTENSOR___BASICFUNC___H

#include "autograd/__BasicFunctionNode__.h"
#include "tensor/Tensor.h"


/// 指数函数
template<typename TT>
Tensor<TT>
exp(const Tensor<TT> &input) {
    Tensor<TT> res(xt::exp(input.data()), input.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __ExpNode__<TT>(input.ptr, res.ptr));
        __reference_unary_op__(input, res, op);
    }
    return res;
}


/// 对数函数，以 e 为底
template<typename TT>
Tensor<TT>
log(const Tensor<TT> &input) {
    Tensor<TT> res(xt::log(input.data()), input.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __LogNode__<TT>(input.ptr, res.ptr));
        __reference_unary_op__(input, res, op);
    }
    return res;
}


/// 根号函数
template<typename TT>
Tensor<TT>
sqrt(const Tensor<TT> &input) {
    Tensor<TT> res(xt::sqrt(input.data()), input.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __SqrtNode__<TT>(input.ptr, res.ptr));
        __reference_unary_op__(input, res, op);
    }
    return res;
}

#endif //CHENTENSOR___BASICFUNC___H
