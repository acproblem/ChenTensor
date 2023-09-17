//
// Created by Lenovo on 2023/8/29.
// 绑定 tensor 到 python。
//

#include "tensor/Tensor.h"
#include <sstream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY                // numpy C api loading

#include "xtensor-python/pyarray.hpp"     // Numpy bindings

namespace py = pybind11;


/// 数据类型 Dtype
enum class Dtype {
    _INT32, _INT64, _FLOAT32, _FLOAT64
};


/// 绑定 operator op<TL, TR> 到 Python。
/// 这里包括 + - * / 运算符。
template<typename TL, typename TR>
void bind1_part(py::class_<Tensor<TL>> &c) {
    c.def("__add__", operator+ < TL, TR > , "Add two tensors.", py::arg("tensor"))
            .def("__sub__", operator- < TL, TR > , "Subtract two tensors.", py::arg("tensor"))
            .def("__mul__", operator* < TL, TR > , "Multiply two tensors.", py::arg("tensor"))
            .def("__truediv__", operator/ < TL, TR > , "Divide two tensors.", py::arg("tensor"));
}

/// 绑定 Tensor<T> 到 Python。
template<typename T>
void bind1(py::class_<Tensor<T>> &c, const char *className) {


    c.def(py::init([]() { return Tensor<T>(); }), "Create a default tensor.");
    c.def(py::init([](const xt::pyarray<T> &data) { return Tensor<T>(data); }),
          "Create a tensor for specified data.", py::arg("data"));
    c.def(py::init([](const xt::pyarray<T> &data, bool requires_grad) {
              return Tensor<T>(data, requires_grad);
          }), "Create a tensor for specified data and specified differentiability.", py::arg("data"),
          py::arg("requires_grad"));

    c.def_property("data", [](const Tensor<T> &t) { return xt::pyarray<T>(t.data()); },
                   [](Tensor<T> &t, const xt::pyarray<T> &data) { t.data() = data; });
    c.def_property("grad", [](const Tensor<T> &t) { return xt::pyarray<T>(t.grad()); },
                   [](Tensor<T> &t, const xt::pyarray<T> &grad) { t.grad() = grad; });
    c.def_property_readonly("requires_grad", &Tensor<T>::requires_grad)
            .def("__repr__", [](const Tensor<T> &t) {
                std::ostringstream out;
                out << t;
                return out.str();
            });

    c.def("backward", &Tensor<T>::backward, "Solving the Gradient of Dynamic Graphs.")
            .def("shape", &Tensor<T>::shape, "Get shape of tensor.")
            .def("reshape", &Tensor<T>::reshape, "Change shape of tensor.", py::arg("shape"));

    c.def("__getitem__", &Tensor<T>::get, "Return one of sample.", py::arg("idx"));

    c.def("__pos__", &operator+ < T > , "Positive operation.")
            .def("__neg__", &operator- < T > , "Negative operation.");

    bind1_part<T, int32_t>(c);
    bind1_part<T, int64_t>(c);
    bind1_part<T, float_t>(c);
    bind1_part<T, double_t>(c);
}

/// 绑定到 Python
PYBIND11_MODULE(_tensor, m) {
    xt::import_numpy();

    m.doc() = "All kinds of tensor classes.";

    /// 绑定数据类型 Dtype
    py::enum_<Dtype>(m, "Dtype")
            .value("int32", Dtype::_INT32)
            .value("int64", Dtype::_INT64)
            .value("float32", Dtype::_FLOAT32)
            .value("float64", Dtype::_FLOAT64)
            .export_values();

//    py::class_<Tensor<float>>(m, "TensorFloat32")
//            .def(py::init([]() { return Tensor<float>(); }), "Create a default tensor.")
//            .def(py::init([](const xt::pyarray<float> &data) { return Tensor<float>(data); }),
//                 "Create a tensor for specified data.", py::arg("data"))
//            .def(py::init([](const xt::pyarray<float> &data, bool requires_grad) {
//                     return Tensor<float>(data, requires_grad);
//                 }), "Create a tensor for specified data and specified differentiability.", py::arg("data"),
//                 py::arg("requires_grad"))
//            .def_property("data", [](const Tensor<float> &t) { return xt::pyarray<float>(t.data()); },
//                          [](Tensor<float> &t, const xt::pyarray<float> &data) { t.data() = data; })
//            .def_property("grad", [](const Tensor<float> &t) { return xt::pyarray<float>(t.grad()); },
//                          [](Tensor<float> &t, const xt::pyarray<float> &grad) { t.grad() = grad; })
//            .def_property_readonly("requires_grad", &Tensor<float>::requires_grad)
//            .def("__repr__", [](const Tensor<float> &t) {
//                std::ostringstream out;
//                out << t;
//                return out.str();
//            })
//            .def("backward", &Tensor<float>::backward, "Solving the Gradient of Dynamic Graphs.")
//            .def("shape", &Tensor<float>::shape, "Get shape of tensor.")
//            .def("reshape", &Tensor<float>::reshape, "Change shape of tensor.", py::arg("shape"))
//            .def("__pos__", &operator+ < float > , "Positive operation.")
//            .def("__neg__", &operator- < float > , "Negative operation.")
//            .def("__add__", operator+ < float, float > , "Add two tensors.", py::arg("tensor"))
//            .def("__sub__", operator- < float, float > , "Subtract two tensors.", py::arg("tensor"))
//            .def("__mul__", operator* < float, float > , "Multiply two tensors.", py::arg("tensor"))
//            .def("__truediv__", operator/ < float, float > , "Divide two tensors.", py::arg("tensor"));

    py::class_<Tensor<int32_t>> c1(m, "TensorInt32");
    c1.def_property_readonly("dtype", [](const Tensor<int32_t> &t) { return Dtype::_INT32; });
    bind1<int32_t>(c1, "TensorInt32");

    py::class_<Tensor<int64_t>> c2(m, "TensorInt64");
    c2.def_property_readonly("dtype", [](const Tensor<int64_t> &t) { return Dtype::_INT64; });
    bind1<int64_t>(c2, "TensorInt64");

    py::class_<Tensor<float_t>> c3(m, "TensorFloat32");
    c3.def_property_readonly("dtype", [](const Tensor<float_t> &t) { return Dtype::_FLOAT32; });
    bind1<float_t>(c3, "TensorFloat32");

    py::class_<Tensor<double_t>> c4(m, "TensorFloat64");
    c4.def_property_readonly("dtype", [](const Tensor<double_t> &t) { return Dtype::_FLOAT64; });
    bind1<double_t>(c4, "TensorFloat64");

}