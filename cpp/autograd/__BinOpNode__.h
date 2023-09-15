//
// Created by Lenovo on 2023/8/17.
// 二元操作符算子节点抽象类。
//

#ifndef CHENTENSOR___BINOPNODE___H
#define CHENTENSOR___BINOPNODE___H

#include "__OpNode__.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xtensor.hpp>


/// 二元操作符节点抽象类，定义基本的结构
template<typename TL, typename TR>
class __BinOpNode__ : public __OpNode__ {
public:
    /// 左操作数，数据节点指针
    std::shared_ptr<__DataNode__<TL>> left;

    /// 右操作数，数据节点指针
    std::shared_ptr<__DataNode__<TR>> right;

    /// 结果类型
    using res_type = typename decltype(left->data + right->data)::value_type;

    /// 加法结果，数据节点指针
    std::weak_ptr<__DataNode__<res_type>> res;

public:
    __BinOpNode__(const std::shared_ptr<__DataNode__<TL>> &left, const std::shared_ptr<__DataNode__<TR>> &right,
                  const std::shared_ptr<__DataNode__<res_type>> &res)
            : left(left), right(right), res(res) {
    }

    /// 操作符的析构函数需要做一件事：将前向数据节点对自身的 weak_ptr 去掉
    virtual ~__BinOpNode__() {
        __reset__(this->left, this);
        __reset__(this->right, this);
    }
};


/// 销毁二元操作符，传入 左操作数、右操作数 和 结果操作数 的智能指针，传入数据节点指针
template<typename TL, typename TR>
inline void
__reset_bin_op__(std::shared_ptr<__DataNode__<TL>> &left, std::shared_ptr<__DataNode__<TR>> &right,
                 std::shared_ptr<__DataNode__<typename __BinOpNode__<TL, TR>::res_type>> &res, __OpNode__ *op) {
    __reset__(left, op);
    __reset__(right, op);
    res->pre_op.reset();
}


/// 逆广播。a 被广播为 b，求解出被广播的轴
template<typename TL, typename TR>
std::vector<int> __get_axis_of_broadcast__(const xt::xarray<TL> &a, const xt::xarray<TR> &b) {
    std::vector<int> axis;
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    int diff_dim = b_shape.size() - a_shape.size();
    for (int i = 0; i < b_shape.size(); i++) {
        if (i < diff_dim)
            axis.push_back(i);
        else if (b_shape[i] != a_shape[i - diff_dim] && a_shape[i - diff_dim] == 1)
            axis.push_back(i);
    }
    return axis;
}


/// 传递一个数据节点和计算出的梯度，将该梯度加到数据节点上，并且考虑广播问题
/// TD : Type of data. TG : Type of gradient.
template<typename TD, typename TG>
void __add_grad__(const std::shared_ptr<__DataNode__<TD>> &data, const xt::xarray<TG> &grad) {
    std::vector<int> axis = __get_axis_of_broadcast__(data->data, grad);
    if (!axis.empty()) {
        data->grad() += xt::sum(grad, axis);
        data->grad().reshape(data->data.shape());
    } else
        data->grad() += grad;
}


#endif //CHENTENSOR___BINOPNODE___H
