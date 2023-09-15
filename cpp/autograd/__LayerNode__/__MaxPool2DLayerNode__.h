//
// Created by Lenovo on 2023/9/3.
// 提供 2-D 最大池化算子节点。
//

#ifndef CHENTENSOR___MAXPOOL2DLAYERNODE___H
#define CHENTENSOR___MAXPOOL2DLAYERNODE___H

#include "autograd/__UnaryOpNode__.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xtensor.hpp>


/// 2-D 最大池化算子节点
template<typename T>
class __MaxPool2DLayerNode__ : public __UnaryOpNode__<T> {
public:
    using __UnaryOpNode__<T>::input;
    using __UnaryOpNode__<T>::res;
    using __UnaryOpNode__<T>::__UnaryOpNode__;

//    std::array<unsigned int, 2> kernel_size, stride, padding, dilation;

//    T padding_value;
//
//    int X_H, X_W, K_H, K_W;  // 填充张量的高 (X_H) 和宽 (X_W) ，感受野的高 (K_H) 和宽 (K_W)
//
//    xt::xarray<T> pad_input;

    std::size_t batch_size, in_height, in_width;
    std::size_t channels, kernel_height, kernel_width;
    std::size_t stride_height, stride_width, padding_height, padding_width, dilation_height, dilation_width;
    std::size_t pad_in_height, pad_in_width;
    std::size_t out_height, out_width;

    std::size_t K_H, K_W;  // 感受野的高 (K_H) 和宽 (K_W)

    xt::xarray<T> pad_input;

    xt::xarray<T> padding_value;

    xt::xarray<T> col;  // im2col 的输出矩阵，
    // shape : {batch_size * out_height * out_width, in_channels, kernel_height * kernel_width}

//    xt::xarray<int> idxs;  // col 矩阵第 3 个轴中最大元素的索引，存储下来，便于反向传播


public:
    __MaxPool2DLayerNode__(const std::shared_ptr<__DataNode__<T>> &input,
                           const std::shared_ptr<__DataNode__<T>> &res,
                           const std::array<unsigned int, 2> &kernel_size,
                           const std::array<unsigned int, 2> &stride = {1, 1},
                           const std::array<unsigned int, 2> &padding = {0, 0},
                           const std::array<unsigned int, 2> &dilation = {1, 1})
            : __UnaryOpNode__<T>(input, res), batch_size(input->data.shape()[0]),
              in_height(input->data.shape()[2]), in_width(input->data.shape()[3]),
              channels(input->data.shape()[1]),
              kernel_height(kernel_size[0]), kernel_width(kernel_size[1]),
              stride_height(stride[0]), stride_width(stride[1]),
              padding_height(padding[0]), padding_width(padding[1]),
              dilation_height(dilation[0]), dilation_width(dilation[1]) {

        pad_in_height = in_height + 2 * padding_height;
        pad_in_width = in_width + 2 * padding_width;
        K_H = 1 + (kernel_height - 1) * dilation_height;  // 感受野高
        out_height = 1 + (pad_in_height - K_H) / stride_height;
        K_W = 1 + (kernel_width - 1) * dilation_width;  // 感受野宽
        out_width = 1 + (pad_in_width - K_W) / stride_width;
    }


    /// 前向传播接口
    virtual void forward() {
        auto res = this->res.lock();

        padding_value = xt::amin(input->data)(0);
        pad_input = xt::ones<T>({batch_size, channels, pad_in_height, pad_in_width}) * padding_value;
        xt::view(pad_input, xt::all(), xt::all(), xt::range(padding_height, pad_in_height - padding_height),
                 xt::range(padding_width, pad_in_width - padding_width)) = input->data;

        col = im2col();

        xt::xarray<T> res_mat = xt::amax(col, 2);
        res_mat.reshape({batch_size, out_height, out_width, channels});

        res->data = xt::transpose(res_mat, {0, 3, 1, 2});

        // 定义结果张量
//        res->data = xt::empty<T>({batch_size, channels, out_height, out_width});

//        // 计算
//        for (int i = 0; i < out_height; i++) {
//            for (int j = 0; j < out_width; j++) {
//                auto tmp = xt::view(pad_input, xt::all(), xt::all(),
//                                    xt::range(i * stride_height, i * stride_height + K_H, dilation_height),
//                                    xt::range(j * stride_width, j * stride_width + K_W, dilation_width));
//                xt::view(res->data, xt::all(), xt::all(), i, j) = xt::amax(tmp, {2, 3});
//            }
//        }
    }


    /// 反向传播接口
    virtual void backward() {
        auto res = this->res.lock();

        // 链式法则求导
        if (input->requires_grad()) {
            cal_grad_of_input(res);
            input->backward();
        }
    }

private:
    /// 对 Input 求导
    void cal_grad_of_input(const std::shared_ptr<__DataNode__<T>> &res) {
        auto idxs = xt::argmax(col, 2);

        xt::xarray<T> res_grad = xt::transpose(res->grad(), {0, 2, 3, 1});
        res_grad.reshape({-1, (int) channels});

        xt::xarray<T> col_grad = xt::zeros<T>(col.shape());
        auto N = col.shape()[0], M = col.shape()[1];
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                col_grad(i, j, idxs(i, j)) = res_grad(i, j);
            }
        }

        // 定义梯度
        xt::xarray<T> pad_grad = xt::zeros<T>(pad_input.shape());

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
//                xt::xarray<T> col_grad_window = xt::view(col_grad, xt::range(h_start + w, _, outsize));
//                col_grad_window.reshape(pad_grad_window.shape());
//                pad_grad_window += col_grad_window;
                pad_grad_window = xt::reshape_view(xt::view(col_grad, xt::range(h_start + w, _, outsize)),
                                                   pad_grad_window.shape());

            }
        }

        // 增加梯度
        input->grad() += xt::view(pad_grad, xt::all(), xt::all(),
                                  xt::range(padding_height, pad_in_height - padding_height),
                                  xt::range(padding_width, pad_in_width - padding_width));
//        // 定义梯度
//        xt::xarray<T> grad = xt::zeros<T>(pad_input.shape());
//
//        // 计算梯度
//        for (int n = 0; n < batch_size; n++) {  // 第 n 条数据反向传播
//            for (int c = 0; c < channels; c++) {  // 输出通道 c 反向传播
//                for (int i = 0; i < out_height; i++) {
//                    for (int j = 0; j < out_width; j++) {
//                        auto grad_tmp = xt::view(grad, n, c,
//                                                 xt::range(i * stride_height, i * stride_height + K_H, dilation_height),
//                                                 xt::range(j * stride_width, j * stride_width + K_W, dilation_width));
//                        auto data_tmp = xt::view(pad_input, n, c,
//                                                 xt::range(i * stride_height, i * stride_height + K_H, dilation_height),
//                                                 xt::range(j * stride_width, j * stride_width + K_W, dilation_width));
//                        auto idx = xt::argmax(data_tmp)();
//                        auto idx0 = idx / kernel_height;
//                        auto idx1 = idx % kernel_height;
//                        grad_tmp(idx0, idx1) += res->grad()(n, c, i, j);
//                    }
//                }
//            }
//        }
//
//        // 增加梯度
//        input->grad() += xt::view(grad, xt::all(), xt::all(), xt::range(padding_height, pad_in_height - padding_height),
//                                  xt::range(padding_width, pad_in_height - padding_width));
    }

    xt::xarray<T> im2col() {
        xt::xarray<T> col = xt::empty<T>(
                {batch_size * out_height * out_width, channels, kernel_height * kernel_width});

        std::size_t axis1_len = kernel_height * kernel_width;
        std::size_t outsize = out_height * out_width;
        for (std::size_t h = 0; h < out_height; ++h) {
            std::size_t h_min = h * stride_height;
            std::size_t h_max = h_min + K_H;
            std::size_t h_start = h * out_width;
            for (std::size_t w = 0; w < out_width; ++w) {
                std::size_t w_min = w * stride_width;
                std::size_t w_max = w_min + K_W;

//                xt::xarray<T> tmp = xt::view(pad_input, xt::all(), xt::all(), xt::range(h_min, h_max, dilation_height),
//                                             xt::range(w_min, w_max, dilation_width));
//                tmp.reshape({(int) batch_size, (int) channels, -1});
//                xt::view(col, xt::range(h_start + w, _, outsize), xt::all()) = tmp;
                auto tmp = xt::view(pad_input, xt::all(), xt::all(), xt::range(h_min, h_max, dilation_height),
                                    xt::range(w_min, w_max, dilation_width));
                xt::view(col, xt::range(h_start + w, _, outsize), xt::all()) = xt::reshape_view(tmp,
                                                                                                {batch_size,
                                                                                                 channels,
                                                                                                 axis1_len});
            }
        }

        return col;
    }
};


#endif //CHENTENSOR___MAXPOOL2DLAYERNODE___H
