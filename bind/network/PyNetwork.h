//
// Created by Lenovo on 2023/8/30.
// Network helper class。为了绑定纯虚函数。
//

#ifndef CHENTENSOR_PYNETWORK_H
#define CHENTENSOR_PYNETWORK_H


/// Network helper class. 为了能够绑定纯虚函数
template<typename T>
class PyNetwork : public Network<T> {
public:
    using Network<T>::Network;

    Tensor<T> forward(Tensor<T> input) override {
        PYBIND11_OVERRIDE_PURE(Tensor<T>, Network<T>, forward, input);
    }

    std::vector<Tensor<T>> parameters() override {
        PYBIND11_OVERRIDE_PURE(std::vector<Tensor<T>>, Network<T>, parameters);
    }
};


#endif //CHENTENSOR_PYNETWORK_H
