//
// Created by Lenovo on 2023/8/28.
// 求平均函数。
//

#ifndef CHENTENSOR___MEANFUNC___H
#define CHENTENSOR___MEANFUNC___H

#include "__SumFunc__.h"


/// 全局平均函数，输出为标量，形状为：{}
template<typename TT>
Tensor<TT>
mean(const Tensor<TT> &input) {
    return sum(input) / Tensor<TT>(xt::xarray<TT>(input.ptr->data.size()));
}


/// 按轴平均函数
template<typename TT>
Tensor<TT>
mean(const Tensor<TT> &input, const std::vector<std::size_t> &axis) {
    int n = input.data().size();  // 数量
    if (!axis.empty()) {
        n = 1;
        auto shape = input.data().shape();
        for (auto axi: axis)
            n *= shape[axi];
    }

    return sum(input, axis) / Tensor<TT>(xt::xarray<TT>(n));
}


/// 按轴平均函数
template<typename TT>
Tensor<TT>
mean(const Tensor<TT> &input, std::size_t axi) {
    return sum(input, axi) / Tensor<TT>(xt::xarray<TT>(input.data().shape()[axi]));
}


#endif //CHENTENSOR___MEANFUNC___H
