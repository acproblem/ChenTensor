//
// Created by Lenovo on 2023/8/21.
// 线性层类。包括两类：带有偏置的线性层 和 不带偏置的线性层。
//

#ifndef CHENTENSOR_LINEAR_H
#define CHENTENSOR_LINEAR_H

#include "Network.h"
#include "functional/__LayerFunc__/__LinearLayerFunc__.h"
#include "functional/__MatrixOperationFunc__.h"
#include <cmath>
#include <memory>


/// 带有偏置的线性层
template<typename T>
class LinearWithBias;


/// 不带偏置的线性层
template<typename T>
class LinearWithoutBias;


/// 线性层抽象类
template<typename T>
class Linear : public Network<T> {
public:
    /// 获取权重接口
    virtual Tensor<T> weight() = 0;

    /// 获取偏置接口
    virtual Tensor<T> bias() = 0;

    /// 获取输入特征数
    virtual unsigned int in_features() const = 0;

    /// 获取输出特征数
    virtual unsigned int out_features() const = 0;

    /// 是否含有偏置
    virtual bool requires_bias() const = 0;

    static std::shared_ptr<Linear<T>> create(unsigned int in_features, unsigned int out_features, bool bias = true) {
        if (bias)
            return LinearWithBias<T>::create(in_features, out_features);
        else
            return LinearWithoutBias<T>::create(in_features, out_features);
    }

    NetType type() override {
        return NetType::Linear;
    }

    virtual ~Linear() {}
};


/// 带有偏置的线性层
template<typename T>
class LinearWithBias : public Linear<T> {
private:
    Tensor<T> m_weight;
    Tensor<T> m_bias;
    unsigned int m_in_features, m_out_features;

    LinearWithBias(unsigned int in_features, unsigned int out_features)
            : m_in_features(in_features), m_out_features(out_features) {
        double k = 1.0 / in_features;
        m_weight = Tensor<T>(xt::random::rand<double>({in_features, out_features}, -std::sqrt(k), std::sqrt(k)), true);
        m_bias = Tensor<T>(xt::random::rand<double>({out_features}, -std::sqrt(k), std::sqrt(k)), true);
    }

    LinearWithBias(const LinearWithoutBias<T> &);

    LinearWithBias &operator=(const LinearWithoutBias<T> &);

public:
    static std::shared_ptr<LinearWithBias<T>> create(unsigned int in_features, unsigned int out_features) {
        return std::shared_ptr<LinearWithBias<T>>(new LinearWithBias<T>(in_features, out_features));
    }

    /// 获取权重
    virtual Tensor<T> weight() override {
        return m_weight;
    }

    /// 获取偏置
    virtual Tensor<T> bias() override {
        return m_bias;
    }

    Tensor<T> forward(Tensor<T> input) override {
        if (input.data().shape().size() != 2)
            throw std::runtime_error("Dim of input must be 2.");
        return linear(input, m_weight, m_bias);
    }

    std::vector<Tensor<T>> parameters() override {
        return {m_weight, m_bias};
    }

    /// 获取输入特征数
    virtual unsigned int in_features() const {
        return m_in_features;
    }

    /// 获取输出特征数
    virtual unsigned int out_features() const {
        return m_out_features;
    }

    /// 是否含有偏置
    virtual bool requires_bias() const {
        return true;
    }
};


/// 不带偏置的线性层
template<typename T>
class LinearWithoutBias : public Linear<T> {
private:
    Tensor<T> m_weight;
    unsigned int m_in_features, m_out_features;

    LinearWithoutBias(unsigned int in_features, unsigned int out_features)
            : m_in_features(in_features), m_out_features(out_features) {
        double k = 1.0 / in_features;
        m_weight = Tensor<T>(xt::random::rand<double>({in_features, out_features}, -std::sqrt(k), std::sqrt(k)), true);
    }

    LinearWithoutBias(const LinearWithoutBias<T> &);

    LinearWithoutBias &operator=(const LinearWithoutBias<T> &);

public:
    static std::shared_ptr<LinearWithoutBias<T>> create(unsigned int in_features, unsigned int out_features) {
        return std::shared_ptr<LinearWithoutBias<T>>(new LinearWithoutBias<T>(in_features, out_features));
    }

    /// 获取权重
    virtual Tensor<T> weight() override {
        return m_weight;
    }

    /// 获取偏置
    virtual Tensor<T> bias() override {
        throw std::runtime_error("The object of \"LinearWithoutBias\" doesn't have the attribute of bias.");
    }

    Tensor<T> forward(Tensor<T> input) override {
        if (input.data().shape().size() != 2)
            throw std::runtime_error("Dim of input must be 2.");
        return mm(input, m_weight);
    }

    std::vector<Tensor<T>> parameters() override {
        return {m_weight};
    }

    /// 获取输入特征数
    virtual unsigned int in_features() const {
        return m_in_features;
    }

    /// 获取输出特征数
    virtual unsigned int out_features() const {
        return m_out_features;
    }

    /// 是否含有偏置
    virtual bool requires_bias() const {
        return false;
    }
};


#endif //CHENTENSOR_LINEAR_H
