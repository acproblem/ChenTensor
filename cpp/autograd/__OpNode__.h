//
// Created by Lenovo on 2023/8/16.
// 算子节点抽象类。
//

#ifndef CHENTENSOR___OPNODE___H
#define CHENTENSOR___OPNODE___H

#include "__DataNode__.h"


/// 在这里声明数据节点
template<typename T>
class __DataNode__;


/// 算子节点抽象类，提供前向传播和反向传播接口
class __OpNode__ {
public:
    /// 前向传播接口
    virtual void forward() = 0;

    /// 反向传播接口
    virtual void backward() = 0;

    virtual ~__OpNode__() {}
};


/// 自定义断言函数，判断数据节点是否存在，如果数据指针为空，则抛出 std::runtime_error 异常
template<typename T>
inline void __assert_datanode_notnull__(const std::shared_ptr<__DataNode__<T>> &data_node) {
    if (!data_node)
        throw std::runtime_error("The data Node does not exists.");
}


/// 删除 前方数据节点 对 后方操作节点 的 weak_ptr，用于销毁动态图
template<typename T>
void __reset__(std::shared_ptr<__DataNode__<T>> &data_node, __OpNode__ *op_node) {
    for (auto it = data_node->next_ops.begin(); it != data_node->next_ops.end(); it++) {
        auto ptr = it->lock();
        if (!ptr || ptr.get() == op_node) {
            data_node->next_ops.erase(it);
            break;
        }
    }
}


#endif //CHENTENSOR___OPNODE___H
