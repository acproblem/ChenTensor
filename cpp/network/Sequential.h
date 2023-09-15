//
// Created by Lenovo on 2023/8/21.
// 序列层类，多个网络层按顺序合并为一个序列层。
//

#ifndef CHENTENSOR_SEQUENTIAL_H
#define CHENTENSOR_SEQUENTIAL_H

#include "Network.h"
#include <memory>
#include <vector>

/// 神经网络容器，容纳一个网络列表
template<typename T>
class Sequential : public Network<T> {
private:
    std::vector<std::shared_ptr<Network<T>>> nets;

    Sequential(const std::vector<std::shared_ptr<Network<T>>> &nets) : nets(nets) {}

    Sequential(const Sequential<T> &);

    Sequential<T> &operator=(const Sequential<T> &);

public:
    static std::shared_ptr<Sequential<T>> create(const std::vector<std::shared_ptr<Network<T>>> &nets) {
        return std::shared_ptr<Sequential<T>>(new Sequential<T>(nets));
    }

    /// 前向传播接口
    virtual Tensor<T> forward(Tensor<T> input) {
        for (auto &it: nets)
            input = it->forward(input);
        return input;
    }

    /// 获取网络参数
    virtual std::vector<Tensor<T>> parameters() {
        std::vector<Tensor<T>> paras;
        for (auto &it: nets) {
            auto new_paras = it->parameters();
            paras.insert(paras.end(), new_paras.begin(), new_paras.end());
        }
        return paras;
    }

    /// 获取网络类型
    NetType type() {
        return NetType::Sequential;
    }

    /// 获取子网络数量
    std::size_t size() const {
        return nets.size();
    }

    /// 获取第 i 个子网络
    const std::shared_ptr<Network<T>> &get(std::size_t idx) {
        return nets[idx];
    }
};


#endif //CHENTENSOR_SEQUENTIAL_H
