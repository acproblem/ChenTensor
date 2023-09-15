//
// Created by Lenovo on 2023/8/29.
// 绑定各种网络模型到 Python。
//

#include "network/model.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY                // numpy C api loading

#include "xtensor-python/pyarray.hpp"     // Numpy bindings

#include "PyNetwork.h"
#include "network/model.h"
//#include "PyLinear.h"

namespace py = pybind11;


/// 绑定数据类型为 T 的网络
template<typename T>
void bind1(py::module_ &m, const std::string &typeName) {
    // 绑定网络抽象基类
    py::class_<Network<T>, std::shared_ptr<Network<T>>, PyNetwork<T>>(m, ("Network" + typeName).c_str())
            .def(py::init<>())
            .def("forward", &Network<T>::forward)
            .def("__call__", &Network<T>::operator())
            .def("parameters", &Network<T>::parameters)
            .def("type", &Network<T>::type);

    // 绑定线性层
    py::class_<Linear<T>, std::shared_ptr<Linear<T>>, Network<T>>(m, ("Linear" + typeName).c_str())
            .def(py::init(&Linear<T>::create), "Create a linear layer.", py::arg("in_features"),
                 py::arg("out_features"), py::arg("bias") = true)
            .def_property_readonly("weight", &Linear<T>::weight, "Get weight of linear layer.")
            .def_property_readonly("bias", &Linear<T>::bias, "Get bias of linear layer.")
            .def_property_readonly("requires_bias", &Linear<T>::requires_bias, "Does the network require bias?")
            .def_property_readonly("in_features", &Linear<T>::in_features, "Get number of input's features.")
            .def_property_readonly("out_features", &Linear<T>::out_features, "Get number of output's features.");

    // 绑定 Dropout 层
    py::class_<Dropout<T>, std::shared_ptr<Dropout<T>>, Network<T>>(m, ("Dropout" + typeName).c_str())
            .def(py::init(&Dropout<T>::create), "Create a dropout layer.", py::arg("probability") = 0.5)
            .def_property_readonly("probability", &Dropout<T>::probability, "Get dropout probability.");

    // 绑定 ReLU 层
    py::class_<ReLU<T>, std::shared_ptr<ReLU<T>>, Network<T>>(m, ("ReLU" + typeName).c_str())
            .def(py::init(&ReLU<T>::create), "Create a relu layer.");

    // 绑定 LeakyReLU 层
    py::class_<LeakyReLU<T>, std::shared_ptr<LeakyReLU<T>>, Network<T>>(m, ("LeakyReLU" + typeName).c_str())
            .def(py::init(&LeakyReLU<T>::create), "Create a leaky relu layer.")
            .def_property_readonly("alpha", &LeakyReLU<T>::alpha, "Get negative slope.");

    // 绑定 Sigmoid 层
    py::class_<Sigmoid<T>, std::shared_ptr<Sigmoid<T>>, Network<T>>(m, ("Sigmoid" + typeName).c_str())
            .def(py::init(&Sigmoid<T>::create), "Create a sigmoid layer.");

    // 绑定 Tanh 层
    py::class_<Tanh<T>, std::shared_ptr<Tanh<T>>, Network<T>>(m, ("Tanh" + typeName).c_str())
            .def(py::init(&Tanh<T>::create), "Create a tanh layer.");

    // 绑定 Sequential 层
    py::class_<Sequential<T>, std::shared_ptr<Sequential<T>>, Network<T>>(m, ("Sequential" + typeName).c_str())
            .def(py::init(&Sequential<T>::create), "Create a sequential layer.", py::arg("nets"))
            .def("size", &Sequential<T>::size, "Return a number of nets.")
            .def("get", &Sequential<T>::get, "Get the i-th net.", py::arg("i"));

    // 绑定 2-D 卷积层
    py::class_<Conv2D<T>, std::shared_ptr<Conv2D<T>>, Network<T>>(m, ("Conv2D" + typeName).c_str())
            .def(py::init(&Conv2D<T>::create), "Create a 2-D convolution layer.", py::arg("in_channels"),
                 py::arg("out_channels"), py::arg("kernel_size"), py::arg("stride") = std::array<unsigned int, 2>{1, 1},
                 py::arg("padding") = std::array<unsigned int, 2>{0, 0},
                 py::arg("dilation") = std::array<unsigned int, 2>{1, 1}, py::arg("padding_value") = 0,
                 py::arg("bias") = true)
            .def_property_readonly("weight", &Conv2D<T>::weight, "Get weight(kernel) of convolution layer.")
            .def_property_readonly("bias", &Conv2D<T>::bias, "Get bias of convolution layer.")
            .def_property_readonly("requires_bias", &Conv2D<T>::requires_bias, "Does the network require bias?")
            .def_property_readonly("in_channels", &Conv2D<T>::in_channels, "Get number of input's channels.")
            .def_property_readonly("out_channels", &Conv2D<T>::out_channels, "Get number of output's channels.")
            .def_property_readonly("kernel_size", &Conv2D<T>::kernel_size, "Get kernel size of convolution layer.")
            .def_property_readonly("stride", &Conv2D<T>::stride, "Get stride of convolution layer.")
            .def_property_readonly("padding", &Conv2D<T>::padding, "Get padding of convolution layer.")
            .def_property_readonly("dilation", &Conv2D<T>::dilation, "Get dilation of convolution layer.")
            .def_property_readonly("padding_value", &Conv2D<T>::padding_value,
                                   "Get padding value of convolution layer.");

    // 绑定 2-D 池化层
    py::class_<MaxPool2D<T>, std::shared_ptr<MaxPool2D<T>>, Network<T>>(m, ("MaxPool2D" + typeName).c_str())
            .def(py::init(&MaxPool2D<T>::create), "Create a 2-D maximum pooling layer.", py::arg("kernel_size"),
                 py::arg("stride") = std::array<unsigned int, 2>{1, 1},
                 py::arg("padding") = std::array<unsigned int, 2>{0, 0},
                 py::arg("dilation") = std::array<unsigned int, 2>{1, 1})
            .def_property_readonly("kernel_size", &MaxPool2D<T>::kernel_size,
                                   "Get kernel size of maximum pooling layer.")
            .def_property_readonly("stride", &MaxPool2D<T>::stride, "Get stride of maximum pooling layer.")
            .def_property_readonly("padding", &MaxPool2D<T>::padding, "Get padding of maximum pooling layer.")
            .def_property_readonly("dilation", &MaxPool2D<T>::dilation, "Get dilation of maximum pooling layer.");

    // 绑定 Flatten 层
    py::class_<Flatten<T>, std::shared_ptr<Flatten<T>>, Network<T>>(m, ("Flatten" + typeName).c_str())
            .def(py::init(&Flatten<T>::create), "Create a flatten layer.", py::arg("start_dim"), py::arg("end_dim"))
            .def_property_readonly("start_dim", &Flatten<T>::start_dim, "Get start dim.")
            .def_property_readonly("end_dim", &Flatten<T>::end_dim, "Get end dim.");

    // 绑定 BatchNorm1D 层
    py::class_<BatchNorm1D<T>, std::shared_ptr<BatchNorm1D<T>>, Network<T>>(m, ("BatchNorm1D" + typeName).c_str())
            .def(py::init(&BatchNorm1D<T>::create), "Create a batch normalization 1-D layer.",
                 py::arg("num_features"), py::arg("eps") = 1e-5, py::arg("momentum") = 0.9)
            .def("eval_forward", &BatchNorm1D<T>::eval_forward,
                 "Forward propagation during evaluation.", py::arg("input"))
            .def_property_readonly("num_features", &BatchNorm1D<T>::num_features, "Get number of features.")
            .def_property_readonly("momentum", &BatchNorm1D<T>::momentum, "Get momentum.")
            .def_property_readonly("eps", &BatchNorm1D<T>::eps, "Get eps.");

    // 绑定 BatchNorm2D 层
    py::class_<BatchNorm2D<T>, std::shared_ptr<BatchNorm2D<T>>, Network<T>>(m, ("BatchNorm2D" + typeName).c_str())
            .def(py::init(&BatchNorm2D<T>::create), "Create a batch normalization 2-D layer.",
                 py::arg("num_channels"), py::arg("eps") = 1e-5, py::arg("momentum") = 0.9)
            .def("eval_forward", &BatchNorm2D<T>::eval_forward,
                 "Forward propagation during evaluation.", py::arg("input"))
            .def_property_readonly("num_channels", &BatchNorm2D<T>::num_channels, "Get number of channels.")
            .def_property_readonly("momentum", &BatchNorm2D<T>::momentum, "Get momentum.")
            .def_property_readonly("eps", &BatchNorm2D<T>::eps, "Get eps.");
}


/// 绑定到 Python
PYBIND11_MODULE(_network, m) {
    xt::import_numpy();

    m.doc() = "All kinds of networks.";

    // 绑定网络类型（枚举类）
    py::enum_<NetType>(m, "NetType")
            .value("Undefined", NetType::Undefined)
            .value("Linear", NetType::Linear)
            .value("Sequential", NetType::Sequential)
            .value("Dropout", NetType::Dropout)

            .value("ReLU", NetType::ReLU)
            .value("Sigmoid", NetType::Sigmoid)
            .value("LeakyReLU", NetType::LeakyReLU)
            .value("Tanh", NetType::Tanh)

            .value("Conv1D", NetType::Conv1D)
            .value("Conv2D", NetType::Conv2D)
            .value("Conv3D", NetType::Conv3D)

            .value("MaxPool1D", NetType::MaxPool1D)
            .value("MaxPool2D", NetType::MaxPool2D)
            .value("MaxPool3D", NetType::MaxPool3D)

            .value("AvgPool1D", NetType::AvgPool1D)
            .value("AvgPool2D", NetType::AvgPool2D)
            .value("AvgPool3D", NetType::AvgPool3D)

            .value("Flatten", NetType::Flatten)

            .value("BatchNorm1D", NetType::BatchNorm1D)
            .value("BatchNorm2D", NetType::BatchNorm2D)
            .value("BatchNorm3D", NetType::BatchNorm3D)

            .value("RNN", NetType::RNN)
            .value("GRU", NetType::GRU)
            .value("LSTM", NetType::LSTM);
//            .export_values();

//    // 绑定网络抽象基类
//    py::class_<Network<float_t>, std::shared_ptr<Network<float_t>>, PyNetwork<float_t>>(m, "NetworkFloat32")
//            .def(py::init<>())
//            .def("forward", &Network<float_t>::forward)
//            .def("__call__", &Network<float_t>::operator())
//            .def("parameters", &Network<float_t>::parameters)
//            .def("type", &Network<float_t>::type);
//
//    // 绑定线性层
//    py::class_<Linear<float_t>, std::shared_ptr<Linear<float_t>>, Network<float_t>>(m, "LinearFloat32")
//            .def(py::init(&Linear<float_t>::create), "Create a linear layer.", py::arg("in_features"),
//                 py::arg("out_features"), py::arg("bias") = true)
//            .def("weight", &Linear<float_t>::weight, "Get weight of linear layer.")
//            .def("bias", &Linear<float_t>::bias, "Get bias of linear layer.");
//
//    // 绑定 Dropout 层
//    py::class_<Dropout<float_t>, std::shared_ptr<Dropout<float_t>>, Network<float_t>>(m, "DropoutFloat32")
//            .def(py::init(&Dropout<float_t>::create), "Create a dropout layer.", py::arg("probability") = 0.5)
//            .def("probability", &Dropout<float_t>::probability, "Get dropout probability.");
//
//    // 绑定 ReLU 层
//    py::class_<ReLU<float_t>, std::shared_ptr<ReLU<float_t>>, Network<float_t>>(m, "ReLUFloat32")
//            .def(py::init(&ReLU<float_t>::create), "Create a relu layer.");
//
//    // 绑定 Sigmoid 层
//    py::class_<Sigmoid<float_t>, std::shared_ptr<Sigmoid<float_t>>, Network<float_t>>(m, "SigmoidFloat32")
//            .def(py::init(&Sigmoid<float_t>::create), "Create a sigmoid layer.");
//
//    // 绑定 Sequential 层
//    py::class_<Sequential<float_t>, std::shared_ptr<Sequential<float_t>>, Network<float_t>>(m, "SequentialFloat32")
//            .def(py::init(&Sequential<float_t>::create), "Create a sequential layer.", py::arg("nets"))
//            .def("size", &Sequential<float_t>::size, "Return a number of nets.")
//            .def("get", &Sequential<float_t>::get, "Get the i-th net.", py::arg("i"));

    bind1<int32_t>(m, "Int32");
    bind1<int64_t>(m, "Int64");
    bind1<float_t>(m, "Float32");
    bind1<double_t>(m, "Float64");

}