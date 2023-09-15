//
// Created by Lenovo on 2023/9/4.
// 池化层函数。
//

#ifndef CHENTENSOR___MAXPOOLLAYERFUNC___H
#define CHENTENSOR___MAXPOOLLAYERFUNC___H

#include "autograd/__LayerNode__/__MaxPool2DLayerNode__.h"
#include "tensor/Tensor.h"


/// 2-D 最大池化函数
template<typename T>
Tensor<T>
maxpool2d(const Tensor<T> &input, const std::array<unsigned int, 2> &kernel_size,
          const std::array<unsigned int, 2> &stride = {1, 1},
          const std::array<unsigned int, 2> &padding = {0, 0},
          const std::array<unsigned int, 2> &dilation = {1, 1}) {

    if (input.data().dimension() != 4)
        throw std::runtime_error("Dim of `input` must be 4.");
    if (kernel_size[0] < 1 || kernel_size[1] < 1)
        throw std::runtime_error("Kernel size must be greater than zero.");
    if (stride[0] < 1 || stride[1] < 1)
        throw std::runtime_error("The parameter `stride` must be greater than zero.");
    if (dilation[0] < 1 || stride[1] < 1)
        throw std::runtime_error("The parameter `dilation` must be greater than zero.");
    if (input.data().shape()[2] + 2 * padding[0] < kernel_size[0] ||
        input.data().shape()[3] + 2 * padding[1] < kernel_size[1])
        throw std::runtime_error("Kernel shape can't be greater than input shape.");

    Tensor<T> res(input.requires_grad());

    std::shared_ptr<__OpNode__> op(new __MaxPool2DLayerNode__<T>(input.ptr, res.ptr, kernel_size,
                                                                 stride, padding, dilation));
    op->forward();

    if (res.requires_grad())
        __reference_unary_op__(input, res, op);

    return res;
}


#endif //CHENTENSOR___MAXPOOLLAYERFUNC___H
