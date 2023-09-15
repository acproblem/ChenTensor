//
// Created by Lenovo on 2023/8/28.
// 求和算子节点。包括：全局求和 等。
//

#ifndef CHENTENSOR___SUMNODE___H
#define CHENTENSOR___SUMNODE___H

#include "__UnaryOpNode__.h"
#include "__BinOpNode__.h"


/// 全局求和算子，输出是标量，形状为：{}
template<typename T>
class __GlobSumNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::sum(input->data);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            input->grad() += xt::broadcast(res->grad(), input->data.shape());
            input->backward();
        }
    }
};


/// 求和算子，沿着轴进行求和
template<typename T>
class __SumNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

    std::vector<std::size_t> axis;  // 求和的轴

    __SumNode__(const std::shared_ptr<__DataNode__<T>> &input, const std::shared_ptr<__DataNode__<T>> &res,
                const std::vector<std::size_t> &axis) : __UnaryOpNode__<T>(input, res), axis(axis) {
        std::sort(this->axis.begin(), this->axis.end());
    }

    /// 实现前向传播
    virtual void forward() {
        auto res = this->res.lock();
        res->data = xt::sum(input->data, axis);
    }

    /// 实现反向传播
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            xt::xarray<T> grad = res->grad();
            auto shape = grad.shape();
            for (auto axi: axis)
                shape.insert(shape.begin() + axi, 1);
            grad.reshape(shape);
            input->grad() += xt::broadcast(grad, input->data.shape());
            input->backward();
        }
    }
};

#endif //CHENTENSOR___SUMNODE___H
