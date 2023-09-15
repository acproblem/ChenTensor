//
// Created by Lenovo on 2023/8/21.
//

#ifndef CHENTENSOR_OPTIM_H
#define CHENTENSOR_OPTIM_H

/// 优化器接口
class Optim {
public:
    /// 梯度清零
    virtual void zero_grad() = 0;

    /// 更新参数
    virtual void step() = 0;

    virtual ~Optim() {}
};


#endif //CHENTENSOR_OPTIM_H
