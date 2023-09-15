//
// Created by Lenovo on 2023/8/28.
// 其他的算子节点。包括：squeeze 等。
//

#ifndef CHENTENSOR___OTHEROPNODE___H
#define CHENTENSOR___OTHEROPNODE___H

#include "__UnaryOpNode__.h"


/// squeeze 算子
template<typename T>
class __SqueezeNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::squeeze(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            auto tmp = res->grad();
            tmp.reshape(input->data.shape());
            input->grad() += tmp;
            input->backward();
        }
    }
};


/// flatten 算子节点
template<typename T>
class __FlattenOpNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    int start_dim, end_dim;

    __FlattenOpNode__(const std::shared_ptr<__DataNode__<T>> &input, const std::shared_ptr<__DataNode__<T>> &res,
                      int start_dim, int end_dim) : __UnaryOpNode__<T>(input, res) {
        // 将 start_dim 和 end_dim 正确转化（start_dim 和 end_dim 可能是负值，代表反向的索引）
        if (start_dim < 0)
            start_dim = input->data.dimension() + start_dim;
        if (start_dim < 0 || start_dim >= input->data.dimension())
            throw std::runtime_error("The parameter `start_dim` out of range.");

        if (end_dim < 0)
            end_dim = input->data.dimension() + end_dim;
        if (end_dim < 0 || end_dim >= input->data.dimension())
            throw std::runtime_error("The parameter `end_dim` out of range.");

        this->start_dim = start_dim;
        this->end_dim = end_dim;
    }

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();

        // 计算结果数组的形状 res_shape
        auto input_shape = input->data.shape();
        std::vector<int> res_shape;
        for (int i = 0; i < start_dim; i++) res_shape.push_back(input_shape[i]);
        res_shape.push_back(-1);
        for (int i = end_dim + 1; i < input_shape.size(); i++) res_shape.push_back(input_shape[i]);

        // 计算结果数组
        res->data = input->data;
        res->data.reshape(res_shape);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            auto tmp = res->grad();
            tmp.reshape(input->data.shape());
            input->grad() += tmp;
            input->backward();
        }
    }
};


/// 实现 get 算子，获取第 i 条数据
template<typename T>
class __GetNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    unsigned int idx;

    __GetNode__(const std::shared_ptr<__DataNode__<T>> &input, const std::shared_ptr<__DataNode__<T>> &res,
                unsigned int idx)
            : __UnaryOpNode__<T>(input, res), idx(idx) {}

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::view(input->data, idx);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            if (input->data.shape() != input->grad().shape())
                input->grad() = xt::zeros<T>(input->data.shape());
            xt::view(input->grad(), idx) += res->grad();
            input->backward();
        }
    }
};


#endif //CHENTENSOR___OTHEROPNODE___H
