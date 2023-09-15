//
// Created by Lenovo on 2023/8/28.
// 其他的函数。
//

#ifndef CHENTENSOR___OTHERFUNC___H
#define CHENTENSOR___OTHERFUNC___H

#include "autograd/__OtherOpNode__.h"
#include "tensor/Tensor.h"


/// squeeze 函数
template<typename TT>
Tensor<TT>
squeeze(const Tensor<TT> &input) {
    Tensor<TT> res(xt::squeeze(input.data()), input.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __SqueezeNode__<TT>(input.ptr, res.ptr));
        __reference_unary_op__(input, res, op);
    }
    return res;
}


/// flatten 函数
template<typename TT>
Tensor<TT>
flatten(const Tensor<TT> &input, int start_dim = 0, int end_dim = -1) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __FlattenOpNode__<TT>(input.ptr, res.ptr, start_dim, end_dim));
    op->forward();

    if (res.requires_grad())
        __reference_unary_op__(input, res, op);

    return res;
}


#endif //CHENTENSOR___OTHERFUNC___H
