//
// Created by Lenovo on 2023/8/30.
// 绑定各种优化器到 Python。
//

#include "network/model.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY                // numpy C api loading

#include "xtensor-python/pyarray.hpp"     // Numpy bindings

#include "optim/optimizer.h"
#include "PyOptim.h"

namespace py = pybind11;


/// 绑定 T 数据类型的各种优化器
template<typename T>
void bind1(py::module_ &m, const std::string &typeName) {
    // 绑定一般的梯度下降优化器
    py::class_<GD<T>, Optim>(m, ("GD" + typeName).c_str())
            .def(py::init<const std::vector<Tensor<T>> &, double>(), "Create a GD optimizer.",
                 py::arg("parameters"), py::arg("lr") = 0.01);

    // 绑定带动量（momentum）的梯度下降优化器
    py::class_<Momentum<T>, Optim>(m, ("Momentum" + typeName).c_str())
            .def(py::init<const std::vector<Tensor<T>> &, double, double>(), "Create a Momentum optimizer.",
                 py::arg("parameters"), py::arg("lr") = 0.01, py::arg("momentum") = 0.9);

    // 绑定 RMSprop 算法的梯度下降优化器
    py::class_<RMSprop<T>, Optim>(m, ("RMSprop" + typeName).c_str())
            .def(py::init<const std::vector<Tensor<T>> &, double, double, double>(), "Create a RMSprop optimizer.",
                 py::arg("parameters"), py::arg("lr") = 0.01, py::arg("alpha") = 0.99, py::arg("eps") = 1e-8);

    // 绑定 Adam 算法的梯度下降优化器
    py::class_<Adam<T>, Optim>(m, ("Adam" + typeName).c_str())
            .def(py::init<const std::vector<Tensor<T>> &, double, const std::array<double, 2> &, double>(),
                 "Create a Adam optimizer.", py::arg("parameters"), py::arg("lr") = 0.01,
                 py::arg("beta") = std::array<double, 2>{0.9, 0.999}, py::arg("eps") = 1e-8);
}

/// 绑定到 Python
PYBIND11_MODULE(_optim, m) {
    xt::import_numpy();

    m.doc() = "All kinds of optimizers.\nNote : When constructing the optimizer, parameters must be passed. "
              "The parameter here is a list containing several tensors of the `same data type`.";


    // 绑定优化器抽象基类
    py::class_<Optim, PyOptim>(m, "Optim")
            .def(py::init<>())
            .def("zero_grad", &Optim::zero_grad, "Set the gradient of parameters to zero.")
            .def("step", &Optim::step, "Update all parameters.");


//    // 绑定一般的梯度下降优化器
//    py::class_<GD<float>, Optim>(m, "GDFloat32")
//            .def(py::init<const std::vector<Tensor<float>> &, double>(), "Create a GD optimizer.",
//                 py::arg("parameters"), py::arg("lr") = 0.01);
    bind1<float_t>(m, "Float32");
    bind1<double_t>(m, "Float64");
    bind1<int32_t>(m, "Int32");
    bind1<int64_t>(m, "Int64");

}