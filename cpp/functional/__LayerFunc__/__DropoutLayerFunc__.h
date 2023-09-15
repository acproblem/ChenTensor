//
// Created by Lenovo on 2023/8/28.
// 提供 Dropout 层函数。
//

#ifndef CHENTENSOR___DROPOUTLAYERFUNC___H
#define CHENTENSOR___DROPOUTLAYERFUNC___H

#include "autograd/__LayerNode__/__DropoutLayerNode__.h"
#include "tensor/Tensor.h"


/// dropout 函数
template<typename TT>
Tensor<TT>
dropout(const Tensor<TT> &input, double p = 0.5) {
    if (p < 0.0 || p > 1.0)
        throw std::runtime_error("The probability of dropout must be in [0.0, 1.0].");

    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __DropoutNode__<TT>(input.ptr, res.ptr, p));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


#endif //CHENTENSOR___DROPOUTLAYERFUNC___H
