//
// Created by Lenovo on 2023/8/17.
// 一元操作符算子节点抽象类。
//

#ifndef CHENTENSOR___UNARYOPNODE___H
#define CHENTENSOR___UNARYOPNODE___H

#include "__OpNode__.h"


/// 一元操作符节点抽象类，定义基本的结构
template<typename T>
class __UnaryOpNode__ : public __OpNode__ {
public:
    /// 输入的操作数，数据节点指针
    std::shared_ptr<__DataNode__<T>> input;

    /// 运算结果，数据节点指针
    std::weak_ptr<__DataNode__<T>> res;

public:
    __UnaryOpNode__(const std::shared_ptr<__DataNode__<T>> &input, const std::shared_ptr<__DataNode__<T>> &res)
            : input(input), res(res) {
    }

    /// 析构时，取消前面数据节点对该算子节点的 weak_ptr，释放内存。
    virtual ~__UnaryOpNode__() {
        __reset__(this->input, this);
    }
};


/// 销毁一元操作符，传入 左操作数、右操作数 和 结果操作数 的智能指针，传入数据节点指针
template<typename T>
inline void
__reset_unary_op__(std::shared_ptr<__DataNode__<T>> &input, std::shared_ptr<__DataNode__<T>> &res, __OpNode__ *op) {
    __reset__(input, op);
    res->pre_op.reset();
}


#endif //CHENTENSOR___UNARYOPNODE___H
