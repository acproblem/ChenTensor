//
// Created by Lenovo on 2023/9/3.
// 卷积层函数。
//

#ifndef CHENTENSOR___CONVLAYERFUNC___H
#define CHENTENSOR___CONVLAYERFUNC___H

#include "autograd/__LayerNode__/__Conv2DLayerNode__.h"
#include "tensor/Tensor.h"


/// conv2d 函数
template<typename TI, typename TP>
auto
conv2d(const Tensor<TI> &input, const Tensor<TP> &weight, const Tensor<TP> bias,
       const std::array<std::size_t, 2> &stride = {1, 1},
       const std::array<std::size_t, 2> &padding = {0, 0},
       const std::array<std::size_t, 2> &dilation = {1, 1},
       double padding_value = 0) -> Tensor<typename decltype(input.data() * weight.data())::value_type> {

    if (input.data().dimension() != 4 || weight.data().dimension() != 4 || bias.data().dimension() != 1)
        throw std::runtime_error("Shape of `input`, `weight` and `bias` mismatch.");
    if (input.data().shape()[1] != weight.data().shape()[1] || weight.data().shape()[0] != bias.data().shape()[0])
        throw std::runtime_error("Shape of `input`, `weight` and `bias` mismatch.");
    if (weight.data().size() < 1)
        throw std::runtime_error("Kernel size must be greater than zero.");
    if (stride[0] < 1 || stride[1] < 1)
        throw std::runtime_error("The parameter `stride` must be greater than zero.");
    if (dilation[0] < 1 || stride[1] < 1)
        throw std::runtime_error("The parameter `dilation` must be greater than zero.");
    if (input.data().shape()[2] + 2 * padding[0] < weight.data().shape()[2] ||
        input.data().shape()[3] + 2 * padding[1] < weight.data().shape()[3])
        throw std::runtime_error("Kernel shape can't be greater than input shape.");

    Tensor<typename decltype(input.data() + weight.data())::value_type> res(
            input.requires_grad() || weight.requires_grad() || bias.requires_grad());

    std::shared_ptr<__OpNode__> op(new __Conv2DLayerNode__<TI, TP>(input.ptr, weight.ptr, bias.ptr, res.ptr,
                                                                   stride, padding, dilation, padding_value));
    op->forward();

    if (res.requires_grad()) {
        input.ptr->next_ops.push_back(op);
        weight.ptr->next_ops.push_back(op);
        bias.ptr->next_ops.push_back(op);
        res.ptr->pre_op = op;
    }

    return res;
}


/// 重载版本的 conv2d 函数，不带 bias
template<typename TI, typename TP>
auto
conv2d(const Tensor<TI> &input, const Tensor<TP> &weight,
       const std::array<std::size_t, 2> &stride = {1, 1},
       const std::array<std::size_t, 2> &padding = {0, 0},
       const std::array<std::size_t, 2> &dilation = {1, 1},
       double padding_value = 0) -> Tensor<typename decltype(input.data() * weight.data())::value_type> {

    if (input.data().dimension() != 4 || weight.data().dimension() != 4)
        throw std::runtime_error("Shape of `input` and `weight` mismatch.");
    if (input.data().shape()[1] != weight.data().shape()[1])
        throw std::runtime_error("Shape of `input` and `weight` mismatch.");
    if (input.data().shape()[2] + 2 * padding[0] < weight.data().shape()[2] ||
        input.data().shape()[3] + 2 * padding[1] < weight.data().shape()[3])
        throw std::runtime_error("Kernel shape can't be greater than input shape.");

    Tensor<typename decltype(input.data() + weight.data())::value_type> res(
            input.requires_grad() || weight.requires_grad());

    std::shared_ptr<__OpNode__> op(new __Conv2DLayerNode__<TI, TP>(input.ptr, weight.ptr, res.ptr,
                                                                   stride, padding, dilation, padding_value));
    op->forward();

    if (res.requires_grad()) {
        input.ptr->next_ops.push_back(op);
        weight.ptr->next_ops.push_back(op);
        res.ptr->pre_op = op;
    }

    return res;
}


#endif //CHENTENSOR___CONVLAYERFUNC___H
