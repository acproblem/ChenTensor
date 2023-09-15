//
// Created by Lenovo on 2023/9/2.
// 提供连接张量的函数。
//

#ifndef CHENTENSOR___CONCATENATEFUNC___H
#define CHENTENSOR___CONCATENATEFUNC___H

#include "autograd/__ConcatenateOpNode__.h"
#include "tensor/Tensor.h"


/// 合并数据函数
/// 具体为将 {a1, a2, ... , an} 在新建的 0 轴上排列起来。
template<typename TT>
Tensor<TT>
union_tensor(const std::vector<Tensor<TT>> &inputs) {
    if (inputs.empty())
        throw std::runtime_error("The number of `inputs` can not be zero.");

    // 判断各个张量形状是否相等
    auto shape = inputs[0].ptr->data.shape();
    for (int i = 1; i < inputs.size(); i++) {
        if (shape != inputs[i].ptr->data.shape())
            throw std::runtime_error("The shape of the input tensor is not entirely the same.");
    }

    // 结果张量，判断其是否含有梯度
    bool requires_grad = false;
    for (auto it: inputs) {
        if (it.requires_grad()) {
            requires_grad = true;
            break;
        }
    }
    Tensor<TT> res(requires_grad);

    // 前向传播
    std::vector<std::shared_ptr<__DataNode__<TT>>> inputs_ptr;
    for (auto it: inputs)
        inputs_ptr.push_back(it.ptr);
    std::shared_ptr<__OpNode__> op(new __UnionNode__<TT>(inputs_ptr, res.ptr));
    op->forward();

    // 构建计算图（如果含有梯度的话）
    if (res.requires_grad()) {
        for (auto it: inputs)
            it.ptr->next_ops.push_back(op);
        res.ptr->pre_op = op;
    }

    return res;
}

#endif //CHENTENSOR___CONCATENATEFUNC___H
