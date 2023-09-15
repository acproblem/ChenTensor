//
// Created by Lenovo on 2023/9/4.
// 交叉熵损失算子节点。
//

#ifndef CHENTENSOR___CROSSENTROPYLOSSNODE___H
#define CHENTENSOR___CROSSENTROPYLOSSNODE___H

#include "autograd/__BinOpNode__.h"


/// 交叉熵损失算子节点。
template<typename TL, typename TR>
class __CrossEntropyLossNode__ : public __OpNode__ {
public:
    /// 左操作数（预测概率），数据节点指针
    std::shared_ptr<__DataNode__<TL>> left;

    /// 右操作数（真实类别），数据节点指针
    std::shared_ptr<__DataNode__<TR>> right;

    /// 计算损失的结果，数据节点指针
    std::weak_ptr<__DataNode__<TL>> res;

    /// 记录中间变量（logsoftmax 的输出）
    xt::xarray<TL> logsoft;

    __CrossEntropyLossNode__(const std::shared_ptr<__DataNode__<TL>> &left,
                             const std::shared_ptr<__DataNode__<TR>> &right,
                             const std::shared_ptr<__DataNode__<TL>> &res)
            : left(left), right(right), res(res) {
        if (right->requires_grad())
            throw std::runtime_error("The second parameter don't hava gradient.");
        if (left->data.dimension() != 2 || xt::squeeze(right->data).dimension() != 1)
            throw std::runtime_error("Input data shape mismatch.");
        if (left->data.shape()[0] != right->data.size())
            throw std::runtime_error("Input data shape mismatch.");
        if (xt::amin(right->data)() < 0)
            throw std::runtime_error("The second parameter must be a tensor not less than zero.");
        if (xt::amax(right->data)() >= left->data.shape()[1])
            throw std::runtime_error(
                    "The maximum value of the second parameter does not exceed the `shape[1] - 1` of the first parameter.");
    }

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();

        // 首先 logsoftmax
        xt::xarray<TL> left_max = xt::amax(left->data, 1);
        left_max.reshape({left_max.size(), 1});
        logsoft = left->data - left_max;
        xt::xarray<TL> log_sum = xt::log(xt::sum(xt::exp(logsoft), 1));
        log_sum.reshape({log_sum.size(), 1});
        logsoft -= log_sum;

        // 接着计算损失
        res->data = xt::xarray<TL>(0);
        int n = log_sum.shape()[0];
        for (int i = 0; i < n; i++)
            res->data -= xt::xarray<TL>(logsoft(i, right->data(i)));
        res->data /= xt::xarray<TL>(n);
    }

    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (left->requires_grad()) {
            xt::xarray<TL> grad = xt::zeros<TL>(left->data.shape());
            int n = left->data.shape()[0], m = left->data.shape()[1];
            for (int i = 0; i < n; i++) {
                TL logsoft_grad = res->grad()() / static_cast<TL>(n);
                for (int j = 0; j < m; j++) {
                    if (j == right->data(i))
                        grad(i, j) = (logsoft_grad * (xt::exp(xt::xarray<TL>(logsoft(i, j))) - 1))();
                    else
                        grad(i, j) = (logsoft_grad * xt::exp(xt::xarray<TL>(logsoft(i, j))))();
                }
            }
            left->grad() += grad;
            left->backward();
        }
    }


    /// 析构函数
    virtual ~__CrossEntropyLossNode__() {
        __reset__(this->left, this);
        __reset__(this->right, this);
    }
};


///// 交叉熵损失算子节点。
//template<typename TL, typename TR>
//class __CrossEntropyLossNode__ : public __OpNode__ {
//public:
//    /// 左操作数（预测概率），数据节点指针
//    std::shared_ptr<__DataNode__<TL>> left;
//
//    /// 右操作数（真实类别），数据节点指针
//    std::shared_ptr<__DataNode__<TR>> right;
//
//    /// 计算损失的结果，数据节点指针
//    std::weak_ptr<__DataNode__<TL>> res;
//
//    /// 记录中间变量（softmax 的输出）
//    xt::xarray<TL> soft;
//
//    __CrossEntropyLossNode__(const std::shared_ptr<__DataNode__<TL>> &left,
//                             const std::shared_ptr<__DataNode__<TR>> &right,
//                             const std::shared_ptr<__DataNode__<TL>> &res)
//            : left(left), right(right), res(res) {
//        if (right->requires_grad())
//            throw std::runtime_error("The second parameter don't hava gradient.");
//        if (left->data.dimension() != 2 || xt::squeeze(right->data).dimension() != 1)
//            throw std::runtime_error("Input data shape mismatch.");
//        if (left->data.shape()[0] != right->data.size())
//            throw std::runtime_error("Input data shape mismatch.");
//        if (xt::amin(right->data)() < 0)
//            throw std::runtime_error("The second parameter must be a tensor not less than zero.");
//        if (xt::amax(right->data)() >= left->data.shape()[1])
//            throw std::runtime_error(
//                    "The maximum value of the second parameter does not exceed the `shape[1] - 1` of the first parameter.");
//    }
//
//    /// 前向传播接口
//    virtual void forward() {
//        auto res = this->res.lock();
//
//        // 首先 softmax
//        soft = xt::exp(left->data);
//        xt::xarray<float> sum_soft = xt::sum(soft, 1);
//        sum_soft.reshape({sum_soft.size(), 1});
//        soft /= sum_soft;
//        std::cout << "soft : " << soft << std::endl;
//
//        // 接着计算损失
//        res->data = xt::xarray<TL>(0);
//        int n = soft.shape()[0];
//        for (int i = 0; i < n; i++)
//            res->data -= xt::log(xt::xarray<TL>(soft(i, right->data(i))));
//        res->data /= n;
//    }
//
//    /// 反向传播接口
//    virtual void backward() {
//        auto res = this->res.lock();
//
//        // 链式法则求导
//        if (left->requires_grad()) {
//            xt::xarray<TL> grad = xt::zeros<TL>(left->data.shape());
//            int n = left->data.shape()[0], m = left->data.shape()[1];
//            for (int i = 0; i < n; i++) {
//                TL soft_grad = -res->grad()() / (n * soft(i, right->data(i)));
//                for (int j = 0; j < m; j++) {
//                    if (j == right->data(i))
//                        grad(i, j) = soft_grad * (soft(i, j) - soft(i, j) * soft(i, j));
//                    else
//                        grad(i, j) = soft_grad * (-soft(i, right->data(i)) * soft(i, j));
//                }
//            }
//            std::cout << "grad : " << grad << std::endl;
//            left->grad() += grad;
//            left->backward();
//        }
//    }
//
//
//    /// 析构函数
//    virtual ~__CrossEntropyLossNode__() {
//        __reset__(this->left, this);
//        __reset__(this->right, this);
//    }
//};


#endif //CHENTENSOR___CROSSENTROPYLOSSNODE___H
