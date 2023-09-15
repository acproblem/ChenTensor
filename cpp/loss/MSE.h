//
// Created by Lenovo on 2023/8/22.
// 均方误差损失函数
//

#ifndef CHENTENSOR_MSE_H
#define CHENTENSOR_MSE_H

#include "tensor/Tensor.h"
#include "functional/__OtherFunc__.h"
#include "functional/__MeanFunc__.h"

/// 判断两个数组，去掉值为 1 的维度后，是否形状相等
template <typename TL, typename TR>
bool areShapesEqualAfterSqueeze(const xt::xarray<TL> &arr1, const xt::xarray<TR> &arr2) {
    auto shape1 = arr1.shape(), shape2 = arr2.shape();

    // 移除大小为 1 的维度
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] == 1) {
            shape1.erase(shape1.begin() + i);
            --i;
        }
    }
    for (size_t i = 0; i < shape2.size(); ++i) {
        if (shape2[i] == 1) {
            shape2.erase(shape2.begin() + i);
            --i;
        }
    }

    // 比较形状
    return shape1 == shape2;
}


/// 均方误差损失函数
class MSELoss {
public:

    /// 重载调用运算符 ()，计算损失
    template <typename TL, typename TR>
    auto operator() (const Tensor<TL> &y_pred, const Tensor<TR> &y_true) {
        if (!areShapesEqualAfterSqueeze(y_pred.data(), y_true.data()))
            throw std::runtime_error("Shapes of both \"y_pred\" and \"y_true\" mismatch.");

        auto tmp = squeeze(y_pred) - squeeze(y_true);
        return mean(tmp * tmp);
    }
};


#endif //CHENTENSOR_MSE_H
