//
// Created by Lenovo on 2023/9/5.
// 交叉熵损失函数。
//

#ifndef CHENTENSOR_CROSSENTROPY_H
#define CHENTENSOR_CROSSENTROPY_H

#include "tensor/Tensor.h"
#include "functional/functional.h"


/// 交叉熵损失函数
class CrossEntropyLoss {
public:

    /// 重载调用运算符 ()，计算损失
    template <typename TL, typename TR>
    auto operator() (const Tensor<TL> &y_pred, const Tensor<TR> &y_true) {
        return cross_entropy_loss(y_pred, y_true);
    }
};


#endif //CHENTENSOR_CROSSENTROPY_H
