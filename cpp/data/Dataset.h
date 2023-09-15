//
// Created by Lenovo on 2023/9/1.
// 数据集抽象类。
//

#ifndef CHENTENSOR_DATASET_H
#define CHENTENSOR_DATASET_H

#include "tensor/Tensor.h"
#include <tuple>


/// 数据集类。
/// TX : Type of X, TY : Type of Y.
template <typename TX, typename TY>
class Dataset {
public:
    /// 获取第 idx 个数据
    virtual std::tuple<Tensor<TX>, Tensor<TY>> operator[](std::size_t idx) = 0;

    /// 返回总数据条数
    virtual unsigned int size() const = 0;
};

#endif //CHENTENSOR_DATASET_H
