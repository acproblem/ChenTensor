//
// Created by Lenovo on 2023/8/28.
// 矩阵操作函数。包括：矩阵乘向量、矩阵乘矩阵、转置 等。
//

#ifndef CHENTENSOR___MATRIXOPERATIONFUNC___H
#define CHENTENSOR___MATRIXOPERATIONFUNC___H

#include "autograd/__MatrixOperationNode__.h"
#include "tensor/Tensor.h"


/// 矩阵乘向量函数
template<typename TL, typename TR>
auto
mv(const Tensor<TL> &left, const Tensor<TR> &right) -> res_tensor_type {
    if (left.ptr->data.shape().size() != 2 || right.ptr->data.shape().size() != 1)
        throw std::runtime_error("Ensure the data dimensions of the matrix and vector are correct.");
    res_tensor_type res(xt::linalg::dot(left.data(), right.data()), left.requires_grad() || right.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __MVNode__<TL, TR>(left.ptr, right.ptr, res.ptr));
        __reference_bin_op__(left, right, res, op);
    }
    return res;
}


/// 矩阵乘矩阵函数
template<typename TL, typename TR>
auto
mm(const Tensor<TL> &left, const Tensor<TR> &right) -> res_tensor_type {
    if (left.ptr->data.shape().size() != 2 || right.ptr->data.shape().size() != 2)
        throw std::runtime_error("Ensure the data dimension of the matrix is correct.");
    res_tensor_type res(xt::linalg::dot(left.data(), right.data()), left.requires_grad() || right.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __MMNode__<TL, TR>(left.ptr, right.ptr, res.ptr));
        __reference_bin_op__(left, right, res, op);
    }
    return res;
}


/// 转置函数
template<typename TT>
Tensor<TT>
transpose(const Tensor<TT> &input, const std::vector<size_t> &permutation = {}) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __TransposeNode__<TT>(input.ptr, res.ptr, permutation));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


#endif //CHENTENSOR___MATRIXOPERATIONFUNC___H
