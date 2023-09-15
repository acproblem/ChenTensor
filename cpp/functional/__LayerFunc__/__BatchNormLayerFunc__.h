//
// Created by Lenovo on 2023/9/10.
// Batch normalization 层函数。
//

#ifndef CHENTENSOR___BATCHNORMLAYERFUNC___H
#define CHENTENSOR___BATCHNORMLAYERFUNC___H

#include "autograd/__LayerNode__/__BatchNormLayerNode__.h"
#include "functional/__MeanFunc__.h"
#include "functional/__BasicFunc__.h"
#include "tensor/Tensor.h"


/// BatchNorm1D
template <typename T>
Tensor<T> batch_norm1d(const Tensor<T> &input, const Tensor<T> &gamma, const Tensor<T> &beta) {
    Tensor<T> eps(1e-5, false);
    Tensor<T> mu = mean(input, 0);
    Tensor<T> x_sub_mu = input - mu;
    Tensor<T> var = mean(x_sub_mu * x_sub_mu, 0);
    Tensor<T> norm_input = (input - mu) / (sqrt(var + eps));
    return norm_input * gamma + beta;
}


#endif //CHENTENSOR___BATCHNORMLAYERFUNC___H
