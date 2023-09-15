//
// Created by Lenovo on 2023/9/9.
// 三角函数。
//

#ifndef CHENTENSOR___TRIGONOMETRICFUNC___H
#define CHENTENSOR___TRIGONOMETRICFUNC___H

#include "tensor/Tensor.h"
#include "autograd/__TrigonometricOpNode__.h"


/// sin 函数
template<typename TT>
Tensor<TT>
sin(const Tensor<TT> &input) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __SinNode__<TT>(input.ptr, res.ptr));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


/// cos 函数
template<typename TT>
Tensor<TT>
cos(const Tensor<TT> &input) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __CosNode__<TT>(input.ptr, res.ptr));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


/// tan 函数
template<typename TT>
Tensor<TT>
tan(const Tensor<TT> &input) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __TanNode__<TT>(input.ptr, res.ptr));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


/// asin 函数
template<typename TT>
Tensor<TT>
asin(const Tensor<TT> &input) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __AsinNode__<TT>(input.ptr, res.ptr));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


/// acos 函数
template<typename TT>
Tensor<TT>
acos(const Tensor<TT> &input) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __AcosNode__<TT>(input.ptr, res.ptr));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


/// atan 函数
template<typename TT>
Tensor<TT>
atan(const Tensor<TT> &input) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __AtanNode__<TT>(input.ptr, res.ptr));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


/// sinh 函数
template<typename TT>
Tensor<TT>
sinh(const Tensor<TT> &input) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __SinhNode__<TT>(input.ptr, res.ptr));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


/// cosh 函数
template<typename TT>
Tensor<TT>
cosh(const Tensor<TT> &input) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __CoshNode__<TT>(input.ptr, res.ptr));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


/// tanh 函数
template<typename TT>
Tensor<TT>
tanh(const Tensor<TT> &input) {
    Tensor<TT> res(input.requires_grad());
    std::shared_ptr<__OpNode__> op(new __TanhNode__<TT>(input.ptr, res.ptr));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(input, res, op);
    return res;
}


#endif //CHENTENSOR___TRIGONOMETRICFUNC___H
