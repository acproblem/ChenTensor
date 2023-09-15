//
// Created by Lenovo on 2023/9/5.
// 交叉熵损失函数。
//

#ifndef CHENTENSOR___CROSSENTROPYLOSSFUNC___H
#define CHENTENSOR___CROSSENTROPYLOSSFUNC___H

#include "tensor/Tensor.h"
#include "autograd/__LossNode__/__CrossEntropyLossNode__.h"


/// 交叉熵损失函数
template<typename TL, typename TR>
Tensor<TL>
cross_entropy_loss(const Tensor<TL> &y_pred, const Tensor<TR> &y_true) {
    Tensor<TL> res(y_pred.requires_grad());
    std::shared_ptr<__OpNode__> op(new __CrossEntropyLossNode__<TL, TR>(y_pred.ptr, y_true.ptr, res.ptr));
    op->forward();
    if (res.requires_grad())
        __reference_bin_op__(y_pred, y_true, res, op);
    return res;
}


#endif //CHENTENSOR___CROSSENTROPYLOSSFUNC___H
