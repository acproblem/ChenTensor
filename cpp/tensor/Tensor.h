//
// Created by Lenovo on 2023/8/16.
// 张量类。重载算数运算符。
//

#ifndef CHENTENSOR_TENSOR_H
#define CHENTENSOR_TENSOR_H

#include <iostream>
#include <memory>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include "autograd/__BasicOperationNode__.h"
#include "autograd/__OtherOpNode__.h"


/// 张量类型，对 __DataNode__ 类进行包装，以对象管理资源
template<typename T>
class Tensor {
public:
    std::shared_ptr<__DataNode__<T>> ptr;

public:
    Tensor(const xt::xarray<T> &data = xt::xarray<T>()) : ptr(new __NonGradDataNode__<T>(data)) {}

    Tensor(const xt::xarray<T> &data, bool requires_grad) {
        if (requires_grad)
            ptr.reset(new __GradDataNode__<T>(data));
        else
            ptr.reset(new __NonGradDataNode__<T>(data));
    }

    explicit Tensor(bool requires_grad) : Tensor(xt::xarray<T>(), requires_grad) {}

    xt::xarray<T> &data() {
        return ptr->data;
    }

    const xt::xarray<T> &data() const {
        return ptr->data;
    }

    bool requires_grad() const {
        return ptr->requires_grad();
    }

    xt::xarray<T> &grad() {
        return ptr->grad();
    }

    const xt::xarray<T> &grad() const {
        return ptr->grad();
    }

    void backward() {
        if (!requires_grad())
            throw std::runtime_error("Non-gradient _tensor can not backward.");
        ptr->grad() = xt::ones<T>(ptr->data.shape());
        ptr->backward();  // 链式法则迭代调用
    }

    std::vector<std::size_t> shape() {
        auto s1 = ptr->data.shape();
        return std::vector<std::size_t>(s1.begin(), s1.end());
    }

    void reshape(const std::vector<std::size_t> &shape) {
        ptr->data.reshape(shape);
        if (ptr->requires_grad() && ptr->grad().size() == ptr->data.size())
            ptr->grad().reshape(shape);
    }

    Tensor<T> get(unsigned int idx);

};


/// 输出重载
template<typename T>
inline std::ostream &operator<<(std::ostream &out, const Tensor<T> &obj) {
    out << obj.data();
    return out;
}


// **********      重载加、减、乘、除、正、负 运算符      **********


#define res_tensor_type Tensor<typename decltype(left.data() + right.data())::value_type>


/// 让数据节点引用算子节点（二元算子）
template<typename TL, typename TR, typename TRES>
void
__reference_bin_op__(const Tensor<TL> &left, const Tensor<TR> &right, const Tensor<TRES> &res,
                     const std::shared_ptr<__OpNode__> &op) {
    left.ptr->next_ops.push_back(op);
    right.ptr->next_ops.push_back(op);
    res.ptr->pre_op = op;
}


/// 让数据节点引用算子节点（一元算子）
template<typename TT, typename TRES>
void
__reference_unary_op__(const Tensor<TT> &input, const Tensor<TRES> &res, const std::shared_ptr<__OpNode__> &op) {
    input.ptr->next_ops.push_back(op);
    res.ptr->pre_op = op;
}


/// 重载加法运算符，对张量中元素相加，可以广播
template<typename TL, typename TR>
auto
operator+(const Tensor<TL> &left, const Tensor<TR> &right) -> res_tensor_type {
    res_tensor_type res(left.data() + right.data(), left.requires_grad() || right.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __AddNode__<TL, TR>(left.ptr, right.ptr, res.ptr));
        __reference_bin_op__(left, right, res, op);
    }
    return res;
}


/// 重载减法运算符，对张量中元素相减，可以广播
template<typename TL, typename TR>
auto
operator-(const Tensor<TL> &left, const Tensor<TR> &right) -> res_tensor_type {
    res_tensor_type res(left.data() - right.data(), left.requires_grad() || right.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __SubNode__<TL, TR>(left.ptr, right.ptr, res.ptr));
        __reference_bin_op__(left, right, res, op);
    }
    return res;
}


/// 重载乘法运算符，对张量中元素相乘，可以广播
template<typename TL, typename TR>
auto
operator*(const Tensor<TL> &left, const Tensor<TR> &right) -> res_tensor_type {
    res_tensor_type res(left.data() * right.data(), left.requires_grad() || right.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __MulNode__<TL, TR>(left.ptr, right.ptr, res.ptr));
        __reference_bin_op__(left, right, res, op);
    }
    return res;
}


/// 重载除法运算符，对张量中元素相除，可以广播
template<typename TL, typename TR>
auto
operator/(const Tensor<TL> &left, const Tensor<TR> &right) -> res_tensor_type {
    res_tensor_type res(left.data() / right.data(), left.requires_grad() || right.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __DivNode__<TL, TR>(left.ptr, right.ptr, res.ptr));
        __reference_bin_op__(left, right, res, op);
    }
    return res;
}


/// 重载正号运算符
template<typename TT>
Tensor<TT> operator+(const Tensor<TT> &input) {
    Tensor<TT> res(input.data(), input.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __PlusNode__<TT>(input.ptr, res.ptr));
        __reference_unary_op__(input, res, op);
    }
    return res;
}


/// 重载负号运算符
template<typename TT>
Tensor<TT> operator-(const Tensor<TT> &input) {
    Tensor<TT> res(-input.data(), input.requires_grad());
    if (res.requires_grad()) {
        std::shared_ptr<__OpNode__> op(new __MinusNode__<TT>(input.ptr, res.ptr));
        __reference_unary_op__(input, res, op);
    }
    return res;
}

template<typename T>
Tensor<T> Tensor<T>::get(unsigned int idx) {
    if (ptr->data.dimension() == 0)
        throw std::runtime_error("Dim of the tensor must be greater than zero.");
    if (idx >= ptr->data.shape()[0])
        throw std::runtime_error("Idx out of range.");

    Tensor<T> res(requires_grad());
    std::shared_ptr<__OpNode__> op(new __GetNode__<T>(ptr, res.ptr, idx));
    op->forward();
    if (res.requires_grad())
        __reference_unary_op__(*this, res, op);

    return res;
}


#endif //CHENTENSOR_TENSOR_H
