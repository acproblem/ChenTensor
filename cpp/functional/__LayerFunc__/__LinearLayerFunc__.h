//
// Created by Lenovo on 2023/8/28.
// 提供线性层函数。
//

#ifndef CHENTENSOR___LINEARLAYERFUNC___H
#define CHENTENSOR___LINEARLAYERFUNC___H

#include "autograd/__LayerNode__/__LinearLayerNode__.h"
#include "tensor/Tensor.h"


/// linear 函数
template<typename TI, typename TP>
auto
linear(const Tensor<TI> &input, const Tensor<TP> &weight, const Tensor<TP> bias) -> Tensor<typename decltype(
input.data() * weight.data())::value_type> {
    if (input.data().dimension() != 2 || weight.data().dimension() != 2 || bias.data().dimension() != 1)
        throw std::runtime_error("Shape of \"input\", \"weight\" and \"bias\" mismatch.");

    Tensor<typename decltype(input.data() + weight.data())::value_type> res(
            xt::linalg::dot(input.data(), weight.data()) + bias.data(),
            input.requires_grad() || weight.requires_grad() || bias.requires_grad());

    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __LinearNode__<TI, TP>(input.ptr, weight.ptr, bias.ptr, res.ptr));
        input.ptr->next_ops.push_back(op);
        weight.ptr->next_ops.push_back(op);
        bias.ptr->next_ops.push_back(op);
        res.ptr->pre_op = op;
    }

    return res;
}


#endif //CHENTENSOR___LINEARLAYERFUNC___H
