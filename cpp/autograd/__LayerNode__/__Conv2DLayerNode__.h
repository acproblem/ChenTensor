//
// Created by Lenovo on 2023/9/2.
// 提供 2-D 卷积算子节点。
//

#ifndef CHENTENSOR___CONV2DLAYERNODE___H
#define CHENTENSOR___CONV2DLAYERNODE___H

#include "autograd/__OpNode__.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xstrided_view.hpp>

using namespace xt::placeholders;


/// 2-D 卷积算子节点
template<typename TI, typename TP>  // TI : Type of input, TP : Type of parameters.
class __Conv2DLayerNode__ : public __OpNode__ {
public:
    /// 输入操作数，数据节点指针
    std::shared_ptr<__DataNode__<TI>> input;

    /// 参数操作数，数据节点指针
    std::shared_ptr<__DataNode__<TP>> weight, bias;

    /// 结果类型
    using res_type = typename decltype(input->data * weight->data)::value_type;

    /// 计算结果，数据节点指针
    std::weak_ptr<__DataNode__<res_type>> res;

    std::size_t batch_size, in_height, in_width;
    std::size_t in_channels, out_channels, kernel_height, kernel_width;
    std::size_t stride_height, stride_width, padding_height, padding_width, dilation_height, dilation_width;
    std::size_t pad_in_height, pad_in_width;
    std::size_t out_height, out_width;

    typename TI padding_value;

    std::size_t K_H, K_W;  // 感受野的高 (K_H) 和宽 (K_W)

    xt::xarray<TI> pad_input;

    xt::xarray<TI> col;  // im2col 的输出矩阵，
    // shape : {batch_size * out_height * out_width, in_channels * kernel_height * kernel_width}

    xt::xarray<TP> kernel_mat;  // 卷积核变形为矩阵
    // shape : {out_channels, in_channels * kernel_height * kernel_width}

public:
    __Conv2DLayerNode__(const std::shared_ptr<__DataNode__<TI>> &input,
                        const std::shared_ptr<__DataNode__<TP>> &weight,
                        const std::shared_ptr<__DataNode__<TP>> &bias,
                        const std::shared_ptr<__DataNode__<res_type>> &res,
                        const std::array<std::size_t, 2> &stride = {1, 1},
                        const std::array<std::size_t, 2> &padding = {0, 0},
                        const std::array<std::size_t, 2> &dilation = {1, 1},
                        typename TI padding_value = 0)
            : input(input), weight(weight), bias(bias), res(res),
              in_channels(weight->data.shape()[1]), out_channels(weight->data.shape()[0]),
              kernel_height(weight->data.shape()[2]), kernel_width(weight->data.shape()[3]),
              batch_size(input->data.shape()[0]),
              in_height(input->data.shape()[2]), in_width(input->data.shape()[3]),
              stride_height(stride[0]), stride_width(stride[1]),
              padding_height(padding[0]), padding_width(padding[1]),
              dilation_height(dilation[0]), dilation_width(dilation[1]),
              padding_value(padding_value) {

        pad_in_height = in_height + 2 * padding_height;
        pad_in_width = in_width + 2 * padding_width;
        K_H = 1 + (kernel_height - 1) * dilation_height;  // 感受野高
        out_height = 1 + (pad_in_height - K_H) / stride_height;
        K_W = 1 + (kernel_width - 1) * dilation_width;  // 感受野宽
        out_width = 1 + (pad_in_width - K_W) / stride_width;
    }

    __Conv2DLayerNode__(const std::shared_ptr<__DataNode__<TI>> &input,
                        const std::shared_ptr<__DataNode__<TP>> &weight,
                        const std::shared_ptr<__DataNode__<res_type>> &res,
                        const std::array<std::size_t, 2> &stride = {1, 1},
                        const std::array<std::size_t, 2> &padding = {0, 0},
                        const std::array<std::size_t, 2> &dilation = {1, 1},
                        typename TI padding_value = 0)
            : __Conv2DLayerNode__(input, weight, nullptr, res, stride, padding, dilation) {}

    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();

        pad_input = xt::ones<TI>({batch_size, in_channels, pad_in_height, pad_in_width}) * padding_value;
        xt::view(pad_input, xt::all(), xt::all(), xt::range(padding_height, pad_in_height - padding_height),
                 xt::range(padding_width, pad_in_width - padding_width)) = input->data;

        kernel_mat = weight->data;
        kernel_mat.reshape({(int) out_channels, -1});

        col = im2col();

        xt::xarray<TI> z = xt::linalg::dot(col, xt::transpose(kernel_mat));
        z.reshape({batch_size, out_height, out_width, out_channels});
        if (bias)
            z += bias->data;

        res->data = xt::transpose(z, {0, 3, 1, 2});
    }


    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            cal_grad_of_input(res);
            input->backward();
        }
        if (weight->requires_grad()) {
            cal_grad_of_weight(res);
            weight->backward();
        }
        if (bias && bias->requires_grad()) {
            cal_grad_of_bias(res);
            bias->backward();
        }
    }

    /// 析构时，取消前面数据节点对该算子节点的 weak_ptr，释放内存。
    virtual ~__Conv2DLayerNode__() {
        __reset__(this->input, this);
        __reset__(this->weight, this);
        if (bias)
            __reset__(this->bias, this);
    }

private:
    /// 对 Input 求导
    void cal_grad_of_input(const std::shared_ptr<__DataNode__<res_type>> &res) {
        xt::xarray<res_type> res_grad = xt::transpose(res->grad(), {0, 2, 3, 1});
        res_grad.reshape({-1, (int) out_channels});

        xt::xarray<TI> col_grad = xt::linalg::dot(res_grad, kernel_mat);

        // 定义梯度
        xt::xarray<TI> pad_grad = xt::zeros<TI>(pad_input.shape());

        // 输入的梯度 col2im
        std::size_t outsize = out_height * out_width;
        for (std::size_t h = 0; h < out_height; ++h) {
            std::size_t h_min = h * stride_height;
            std::size_t h_max = h_min + K_H;
            std::size_t h_start = h * out_width;
            for (std::size_t w = 0; w < out_width; ++w) {
                std::size_t w_min = w * stride_width;
                std::size_t w_max = w_min + K_W;
                auto pad_grad_window = xt::view(pad_grad, xt::all(), xt::all(),
                                                xt::range(h_min, h_max, dilation_height),
                                                xt::range(w_min, w_max, dilation_width));
//                xt::xarray<res_type> col_grad_window = xt::view(col_grad, xt::range(h_start + w, _, outsize));
//                col_grad_window.reshape(pad_grad_window.shape());
//                pad_grad_window += col_grad_window;
                pad_grad_window += xt::reshape_view(xt::view(col_grad, xt::range(h_start + w, _, outsize)),
                                                   pad_grad_window.shape());

            }
        }

        // 增加梯度
        input->grad() += xt::view(pad_grad, xt::all(), xt::all(),
                                  xt::range(padding_height, pad_in_height - padding_height),
                                  xt::range(padding_width, pad_in_width - padding_width));
    }

    /// 对 weight 求导
    void cal_grad_of_weight(const std::shared_ptr<__DataNode__<res_type>> &res) {
        xt::xarray<res_type> res_grad = xt::transpose(res->grad(), {0, 2, 3, 1});
        res_grad.reshape({-1, (int) out_channels});

        xt::xarray<TP> weight_grad = xt::linalg::dot(xt::transpose(res_grad), col);
        weight_grad.reshape({out_channels, in_channels, kernel_height, kernel_width});

        weight->grad() += weight_grad;
    }

    /// 对 bias 求导
    void cal_grad_of_bias(const std::shared_ptr<__DataNode__<res_type>> &res) {
        bias->grad() += xt::sum(res->grad(), {0, 2, 3});
    }

    xt::xarray<TI> im2col() {
        xt::xarray<TI> col = xt::empty<TI>(
                {batch_size * out_height * out_width, in_channels * kernel_height * kernel_width});

        std::size_t axis1_len = in_channels * kernel_height * kernel_width;
        std::size_t outsize = out_height * out_width;
        for (std::size_t h = 0; h < out_height; ++h) {
            std::size_t h_min = h * stride_height;
            std::size_t h_max = h_min + K_H;
            std::size_t h_start = h * out_width;
            for (std::size_t w = 0; w < out_width; ++w) {
                std::size_t w_min = w * stride_width;
                std::size_t w_max = w_min + K_W;
//                xt::xarray<TI> tmp = xt::view(pad_input, xt::all(), xt::all(), xt::range(h_min, h_max, dilation_height),
//                                              xt::range(w_min, w_max, dilation_width));
//                tmp.reshape({(int) batch_size, -1});
//                xt::view(col, xt::range(h_start + w, _, outsize), xt::all()) = tmp;
                auto tmp = xt::view(pad_input, xt::all(), xt::all(), xt::range(h_min, h_max, dilation_height),
                                    xt::range(w_min, w_max, dilation_width));
                xt::view(col, xt::range(h_start + w, _, outsize), xt::all()) = xt::reshape_view(tmp,
                                                                                                {batch_size,
                                                                                                 axis1_len});
            }
        }

        return col;
    }
};


#endif //CHENTENSOR___CONV2DLAYERNODE___H
