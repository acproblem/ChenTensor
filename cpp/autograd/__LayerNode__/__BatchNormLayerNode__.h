//
// Created by Lenovo on 2023/9/10.
// Batch Normalization 算子节点。
//

#ifndef CHENTENSOR___BATCHNORMLAYERNODE___H
#define CHENTENSOR___BATCHNORMLAYERNODE___H


#include "autograd/__OpNode__.h"


template<typename T>
class __BatchNorm1DLayerNode__ : public __OpNode__ {
public:
    /// 输入操作数，数据节点指针
    std::shared_ptr<__DataNode__<T>> input;

    /// 参数操作数，数据节点指针
    std::shared_ptr<__DataNode__<T>> gamma, beta;

    /// 计算结果，数据节点指针
    std::weak_ptr<__DataNode__<T>> res;

    /// 缓存标准化的输入数据
    xt::xarray<T> norm_data;

    double eps;

public:
    __BatchNorm1DLayerNode__(const std::shared_ptr<__DataNode__<T>> &input,
                             const std::shared_ptr<__DataNode__<T>> &gamma,
                             const std::shared_ptr<__DataNode__<T>> &beta,
                             const std::shared_ptr<__DataNode__<T>> &res,
                             double eps = 1e-5)
            : input(input), gamma(gamma), beta(beta), eps(eps) {}

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();

        xt::xarray<T> mean = xt::mean(input->data, 0);
        xt::xarray<T> var = xt::variance(input->data, 0);
        norm_data = ((input->data - mean) / xt::sqrt(var + eps));

        res->data = norm_data * gamma + beta;
    }


    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            cal_grad_of_input(res);
            input->backward();
        }
        if (gamma->requires_grad()) {
            gamma->grad() += xt::sum(res->grad() * norm_data, 0);
            gamma->backward();
        }
        if (beta->requires_grad()) {
            beta->grad() += xt::sum(res->grad(), 0);
            beta->backward();
        }
    }

    /// 析构时，取消前面数据节点对该算子节点的 weak_ptr，释放内存。
    virtual ~__BatchNorm1DLayerNode__() {
        __reset__(this->input, this);
        __reset__(this->gamma, this);
        __reset__(this->beta, this);
    }
};


#endif //CHENTENSOR___BATCHNORMLAYERNODE___H
