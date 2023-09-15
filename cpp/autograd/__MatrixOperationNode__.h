//
// Created by Lenovo on 2023/8/28.
// 矩阵运算算子节点。包括：矩阵乘向量、矩阵乘矩阵、转置 等。
//

#ifndef CHENTENSOR___MATRIXOPERATIONNODE___H
#define CHENTENSOR___MATRIXOPERATIONNODE___H

#include "__BinOpNode__.h"
#include "__UnaryOpNode__.h"

/// 矩阵乘向量算子
template<typename TL, typename TR>
class __MVNode__ : public __BinOpNode__<TL, TR> {
public:
    using typename __BinOpNode__<TL, TR>::res_type;
    using __BinOpNode__<TL, TR>::__BinOpNode__;
    using __BinOpNode__<TL, TR>::left;
    using __BinOpNode__<TL, TR>::right;
    using __BinOpNode__<TL, TR>::res;

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::linalg::dot(left->data, right->data);
    }

    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (left->requires_grad()) {
            left->grad() += xt::linalg::dot(xt::view(res->grad(), xt::all(), xt::newaxis()),
                                            xt::view(right->data, xt::newaxis(), xt::all()));
            left->backward();
        }
        if (right->requires_grad()) {
            right->grad() += xt::linalg::dot(xt::transpose(left->data), res->grad());
            right->backward();
        }
    }
};


/// 矩阵乘矩阵算子
template<typename TL, typename TR>
class __MMNode__ : public __BinOpNode__<TL, TR> {
public:
    using typename __BinOpNode__<TL, TR>::res_type;
    using __BinOpNode__<TL, TR>::__BinOpNode__;
    using __BinOpNode__<TL, TR>::left;
    using __BinOpNode__<TL, TR>::right;
    using __BinOpNode__<TL, TR>::res;

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::linalg::dot(left->data, right->data);
    }

    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (left->requires_grad()) {
            left->grad() += xt::linalg::dot(res->grad(), xt::transpose(right->data));
            left->backward();
        }
        if (right->requires_grad()) {
            right->grad() += xt::linalg::dot(xt::transpose(left->data), res->grad());
            right->backward();
        }
    }
};


/// 转置算子
template<typename T>
class __TransposeNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    std::vector<size_t> permutation;

    __TransposeNode__(const std::shared_ptr<__DataNode__<T>> input, const std::shared_ptr<__DataNode__<T>> res,
                      const std::vector<size_t> &permutation = {}) : __UnaryOpNode__<T>::__UnaryOpNode__(input, res),
                                                                     permutation(permutation) {
        if (permutation.empty()) {
            for (int i = input->data.shape().size() - 1; i >= 0; i--)
                this->permutation.push_back(i);
        }
    }

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::transpose(input->data, permutation);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            std::vector<size_t> inv_permutation(permutation.size());
            for (int i = 0; i < permutation.size(); i++)
                inv_permutation[permutation[i]] = i;
            input->grad() += xt::transpose(res->grad(), inv_permutation);
            input->backward();
        }
    }
};


#endif //CHENTENSOR___MATRIXOPERATIONNODE___H
