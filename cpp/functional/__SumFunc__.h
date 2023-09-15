//
// Created by Lenovo on 2023/8/28.
// 求和函数。
//

#ifndef CHENTENSOR___SUMFUNC___H
#define CHENTENSOR___SUMFUNC___H

#include "autograd/__SumNode__.h"
#include "tensor/Tensor.h"


/// 全局求和函数，输出为标量，形状为：{}
template<typename TT>
Tensor<TT>
sum(const Tensor<TT> &input) {
    Tensor<TT> res(xt::sum(input.data()), input.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __GlobSumNode__<TT>(input.ptr, res.ptr));
        __reference_unary_op__(input, res, op);
    }
    return res;
}


// 按轴求和函数
template<typename TT>
Tensor<TT>
sum(const Tensor<TT> &input, const std::vector<std::size_t> &axis) {
    std::vector<std::size_t> axis2(axis);
    if (axis2.empty()) {
        for (std::size_t i = 0; i < input.data().dimension(); ++i)
            axis2.push_back(i);
    }

    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __SumNode__<TT>(input.ptr, res.ptr, axis2));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);

    return res;
}


// 按轴求和函数
template<typename TT>
Tensor<TT>
sum(const Tensor<TT> &input, std::size_t axi) {
    std::vector<std::size_t> axis2{axi};

    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __SumNode__<TT>(input.ptr, res.ptr, axis2));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);

    return res;
}


#endif //CHENTENSOR___SUMFUNC___H
