//
// Created by Lenovo on 2023/9/3.
// 2-D 卷积层。
//

#ifndef CHENTENSOR_CONV2D_H
#define CHENTENSOR_CONV2D_H

#include "Network.h"
#include "functional/__LayerFunc__/__ConvLayerFunc__.h"
#include <memory>


/// 带有偏置的 2-D 卷积层
template<typename T>
class Conv2DWithBias;


/// 不带偏置的 2-D 卷积层
template<typename T>
class Conv2DWithoutBias;


/// 2-D 卷积层抽象类
template<typename T>
class Conv2D : public Network<T> {
public:
    /// 获取权重（卷积核）接口
    virtual Tensor<T> weight() = 0;

    /// 获取偏置接口
    virtual Tensor<T> bias() = 0;

    /// 获取输入通道值
    virtual std::size_t in_channels() const = 0;

    /// 获取输出通道值
    virtual std::size_t out_channels() const = 0;

    /// 获取 kernel_size
    virtual const std::array<std::size_t, 2> &kernel_size() const = 0;

    /// 获取 stride
    virtual const std::array<std::size_t, 2> &stride() const = 0;

    /// 获取 padding
    virtual const std::array<std::size_t, 2> &padding() const = 0;

    /// 获取 dilation
    virtual const std::array<std::size_t, 2> &dilation() const = 0;

    /// 获取 padding_value
    virtual T padding_value() const = 0;

    /// 是否含有偏置
    virtual bool requires_bias() const = 0;

    static std::shared_ptr<Conv2D<T>> create(std::size_t in_channels, std::size_t out_channels,
                                             const std::array<std::size_t, 2> &kernel_size,
                                             const std::array<std::size_t, 2> &stride = {1, 1},
                                             const std::array<std::size_t, 2> &padding = {0, 0},
                                             const std::array<std::size_t, 2> &dilation = {1, 1},
                                             typename T padding_value = 0, bool bias = true) {
        if (bias)
            return Conv2DWithBias<T>::create(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                             padding_value);
        else
            return Conv2DWithoutBias<T>::create(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                padding_value);
    }

    NetType type() override {
        return NetType::Conv2D;
    }

    virtual ~Conv2D() {}
};


/// 带有偏置的 2-D 卷积层
template<typename T>
class Conv2DWithBias : public Conv2D<T> {
private:
    Tensor<T> m_weight;
    Tensor<T> m_bias;
    std::size_t m_in_channels, m_out_channels;
    std::array<std::size_t, 2> m_kernel_size, m_stride, m_padding, m_dilation;
    typename T m_padding_value;

    Conv2DWithBias(std::size_t in_channels, std::size_t out_channels,
                   const std::array<std::size_t, 2> &kernel_size,
                   const std::array<std::size_t, 2> &stride = {1, 1},
                   const std::array<std::size_t, 2> &padding = {0, 0},
                   const std::array<std::size_t, 2> &dilation = {1, 1},
                   typename T padding_value = 0) : m_in_channels(in_channels), m_out_channels(out_channels),
                                                   m_kernel_size(kernel_size), m_stride(stride), m_padding(padding),
                                                   m_dilation(dilation), m_padding_value(padding_value) {
        double k = 1.0 / (in_channels * kernel_size[0] * kernel_size[1]);
        m_weight = Tensor<T>(
                xt::random::rand<double>({out_channels, in_channels, kernel_size[0], kernel_size[1]}, -std::sqrt(k),
                                         std::sqrt(k)), true);
        m_bias = Tensor<T>(xt::random::rand<double>({out_channels}, -std::sqrt(k), std::sqrt(k)), true);
    }

    Conv2DWithBias(const Conv2DWithoutBias<T> &);

    Conv2DWithBias &operator=(const Conv2DWithoutBias<T> &);

public:
    static std::shared_ptr<Conv2DWithBias<T>> create(std::size_t in_channels, std::size_t out_channels,
                                                     const std::array<std::size_t, 2> &kernel_size,
                                                     const std::array<std::size_t, 2> &stride = {1, 1},
                                                     const std::array<std::size_t, 2> &padding = {0, 0},
                                                     const std::array<std::size_t, 2> &dilation = {1, 1},
                                                     typename T padding_value = 0) {
        return std::shared_ptr<Conv2DWithBias<T>>(
                new Conv2DWithBias<T>(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                      padding_value));
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
        return conv2d(input, m_weight, m_bias, m_stride, m_padding, m_dilation, m_padding_value);
    }

    std::vector<Tensor<T>> parameters() override {
        return {m_weight, m_bias};
    }

    /// 获取输入通道值
    virtual std::size_t in_channels() const {
        return m_in_channels;
    }

    /// 获取输出通道值
    virtual std::size_t out_channels() const {
        return m_out_channels;
    }

    /// 获取 kernel_size
    virtual const std::array<std::size_t, 2> &kernel_size() const {
        return m_kernel_size;
    }

    /// 获取 stride
    virtual const std::array<std::size_t, 2> &stride() const {
        return m_stride;
    }

    /// 获取 padding
    virtual const std::array<std::size_t, 2> &padding() const {
        return m_padding;
    }

    /// 获取 dilation
    virtual const std::array<std::size_t, 2> &dilation() const {
        return m_dilation;
    }

    /// 获取 padding_value
    virtual T padding_value() const {
        return m_padding_value;
    }

    /// 是否含有偏置
    virtual bool requires_bias() const {
        return true;
    }
};


/// 不带偏置的 2-D 卷积层
template<typename T>
class Conv2DWithoutBias : public Conv2D<T> {
private:
    Tensor<T> m_weight;
    std::size_t m_in_channels, m_out_channels;
    std::array<std::size_t, 2> m_kernel_size, m_stride, m_padding, m_dilation;
    typename T m_padding_value;

    Conv2DWithoutBias(std::size_t in_channels, std::size_t out_channels,
                      const std::array<std::size_t, 2> &kernel_size,
                      const std::array<std::size_t, 2> &stride = {1, 1},
                      const std::array<std::size_t, 2> &padding = {0, 0},
                      const std::array<std::size_t, 2> &dilation = {1, 1},
                      typename T padding_value = 0) : m_in_channels(in_channels), m_out_channels(out_channels),
                                                      m_kernel_size(kernel_size),
                                                      m_stride(stride), m_padding(padding), m_dilation(dilation),
                                                      m_padding_value(padding_value) {
        double k = 1.0 / (in_channels * kernel_size[0] * kernel_size[1]);
        m_weight = Tensor<T>(xt::random::rand<double>({in_channels, out_channels, kernel_size[0], kernel_size[1]},
                                                      -std::sqrt(k), std::sqrt(k)), true);
    }

    Conv2DWithoutBias(const Conv2DWithoutBias<T> &);

    Conv2DWithoutBias &operator=(const Conv2DWithoutBias<T> &);

public:
    static std::shared_ptr<Conv2DWithoutBias<T>> create(std::size_t in_channels, std::size_t out_channels,
                                                        const std::array<std::size_t, 2> &kernel_size,
                                                        const std::array<std::size_t, 2> &stride = {1, 1},
                                                        const std::array<std::size_t, 2> &padding = {0, 0},
                                                        const std::array<std::size_t, 2> &dilation = {1, 1},
                                                        typename T padding_value = 0) {
        return std::shared_ptr<Conv2DWithoutBias<T>>(new Conv2DWithoutBias<T>(in_channels, out_channels, kernel_size,
                                                                              stride, padding, dilation,
                                                                              padding_value));
    }

    /// 获取权重
    virtual Tensor<T> weight() override {
        return m_weight;
    }

    /// 获取偏置
    virtual Tensor<T> bias() override {
        throw std::runtime_error("The object of `Conv2DWithoutBias` doesn't have the attribute of bias.");
    }

    Tensor<T> forward(Tensor<T> input) override {
        return conv2d(input, m_weight, m_stride, m_padding, m_dilation, m_padding_value);
    }

    std::vector<Tensor<T>> parameters() override {
        return {m_weight};
    }

    /// 获取输入通道值
    virtual std::size_t in_channels() const {
        return m_in_channels;
    }

    /// 获取输出通道值
    virtual std::size_t out_channels() const {
        return m_out_channels;
    }

    /// 获取 kernel_size
    virtual const std::array<std::size_t, 2> &kernel_size() const {
        return m_kernel_size;
    }

    /// 获取 stride
    virtual const std::array<std::size_t, 2> &stride() const {
        return m_stride;
    }

    /// 获取 padding
    virtual const std::array<std::size_t, 2> &padding() const {
        return m_padding;
    }

    /// 获取 dilation
    virtual const std::array<std::size_t, 2> &dilation() const {
        return m_dilation;
    }

    /// 获取 padding_value
    virtual T padding_value() const {
        return m_padding_value;
    }

    /// 是否含有偏置
    virtual bool requires_bias() const {
        return false;
    }
};

#endif //CHENTENSOR_CONV2D_H
