//
// Created by Lenovo on 2023/9/4.
// 池化层。
//

#ifndef CHENTENSOR_MAXPOOL2D_H
#define CHENTENSOR_MAXPOOL2D_H

#include "Network.h"
#include "functional/__LayerFunc__/__MaxPoolLayerFunc__.h"
#include <memory>


/// 池化层
template<typename T>
class MaxPool2D : public Network<T> {
private:
    std::array<unsigned int, 2> m_kernel_size, m_stride, m_padding, m_dilation;

    MaxPool2D(const std::array<unsigned int, 2> &kernel_size,
              const std::array<unsigned int, 2> &stride = {1, 1},
              const std::array<unsigned int, 2> &padding = {0, 0},
              const std::array<unsigned int, 2> &dilation = {1, 1})
            : m_kernel_size(kernel_size), m_stride(stride), m_padding(padding), m_dilation(dilation) {}

    MaxPool2D(const MaxPool2D<T> &);

    MaxPool2D<T> &operator=(const MaxPool2D<T> &);

public:
    /// 工厂函数，创建网络
    static std::shared_ptr<MaxPool2D<T>> create(const std::array<unsigned int, 2> &kernel_size,
                                                const std::array<unsigned int, 2> &stride = {1, 1},
                                                const std::array<unsigned int, 2> &padding = {0, 0},
                                                const std::array<unsigned int, 2> &dilation = {1, 1}) {
        return std::shared_ptr<MaxPool2D<T>>(new MaxPool2D<T>(kernel_size, stride, padding, dilation));
    }

    /// 前向传播
    virtual Tensor<T> forward(Tensor<T> input) override {
        return maxpool2d(input, m_kernel_size, m_stride, m_padding, m_dilation);
    }

    /// 获取参数
    virtual std::vector<Tensor<T>> parameters() override {
        return {};
    }

    /// 返回网络类型
    virtual NetType type() override {
        return NetType::MaxPool2D;
    }

    /// 获取核大小
    const std::array<unsigned int, 2> &kernel_size() const {
        return m_kernel_size;
    }

    /// 获取步长
    const std::array<unsigned int, 2> &stride() const {
        return m_stride;
    }

    /// 获取填充大小
    const std::array<unsigned int, 2> &padding() const {
        return m_padding;
    }

    /// 获取 dilation
    const std::array<unsigned int, 2> &dilation() const {
        return m_dilation;
    }
};

#endif //CHENTENSOR_MAXPOOL2D_H
