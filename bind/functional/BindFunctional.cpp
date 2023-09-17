//
// Created by Lenovo on 2023/8/29.
// 绑定各种函数到 Python。
//

#include "functional/functional.h"
#include <sstream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY                // numpy C api loading

#include "xtensor-python/pyarray.hpp"     // Numpy bindings

namespace py = pybind11;


/// 绑定模板参数个数为 1 的 function 到 Python。
template<typename T>
void bind1(py::module_ &m) {
    m.def("exp", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&exp<T>), "Exponential function.", py::arg("tensor"));
    m.def("log", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&log<T>), "Logarithmic function based on `e`.",
          py::arg("tensor"));
    m.def("sqrt", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&sqrt<T>), "Sqrt function.", py::arg("tensor"));

    m.def("transpose", &transpose<T>, "Transpose a tensor.", py::arg("tensor"),
          py::arg("permutation") = std::vector<size_t>());

    m.def("mean", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&mean<T>), "Calculate the mean of a tensor.",
          py::arg("tensor"));
    m.def("mean", static_cast<Tensor<T> (*)(const Tensor<T> &, const std::vector<std::size_t> &)>(&mean<T>),
          "Calculate the mean of a tensor.", py::arg("tensor"), py::arg("axis"));
    m.def("mean", static_cast<Tensor<T> (*)(const Tensor<T> &, std::size_t)>(&mean<T>),
          "Calculate the mean of a tensor.", py::arg("tensor"), py::arg("axis"));

    m.def("sum", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&sum<T>), "Calculate the sum of a tensor.",
          py::arg("tensor"));
    m.def("sum", static_cast<Tensor<T> (*)(const Tensor<T> &, const std::vector<std::size_t> &)>(&sum<T>),
          "Calculate the sum of a tensor.", py::arg("tensor"), py::arg("axis"));
    m.def("sum", static_cast<Tensor<T> (*)(const Tensor<T> &, std::size_t)>(&sum<T>),
          "Calculate the sum of a tensor.", py::arg("tensor"), py::arg("axis"));

    m.def("squeeze", &squeeze<T>, "Remove axis with dimension size 1.", py::arg("tensor"));
    m.def("sigmoid", &sigmoid<T>, "Sigmoid function.", py::arg("tensor"));
    m.def("relu", &relu<T>, "ReLU function.", py::arg("tensor"));
    m.def("leaky_relu", &leaky_relu<T>, "Leaky ReLU function.", py::arg("tensor"), py::arg("alpha"));

    m.def("dropout", &dropout<T>, "Dropout function.", py::arg("tensor"), py::arg("p") = 0.5);
    m.def("flatten", &flatten<T>, "Flatten the array within the interval of axis [start_dim, end_dim].",
          py::arg("input"), py::arg("start_dim") = 0, py::arg("end_dim") = -1);
    m.def("union_tensor", &union_tensor<T>, "Merge a series of tensors with the same shape on the newly created 0-axis",
          py::arg("tensors"));

    m.def("sin", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&sin<T>), "Sin function.", py::arg("tensor"));
    m.def("cos", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&cos<T>), "Cos function.", py::arg("tensor"));
    m.def("tan", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&tan<T>), "Tan function.", py::arg("tensor"));
    m.def("asin", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&asin<T>), "Asin function.", py::arg("tensor"));
    m.def("acos", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&acos<T>), "Acos function.", py::arg("tensor"));
    m.def("atan", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&atan<T>), "Atan function.", py::arg("tensor"));
    m.def("sinh", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&sinh<T>), "Sinh function.", py::arg("tensor"));
    m.def("cosh", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&cosh<T>), "Cosh function.", py::arg("tensor"));
    m.def("tanh", static_cast<Tensor<T> (*)(const Tensor<T> &)>(&tanh<T>), "Tanh function.", py::arg("tensor"));

    m.def("batch_norm1d", &batch_norm1d<T>, "Batch normalization 1-D function.", py::arg("input"),
          py::arg("gamma"), py::arg("beta"));
}


/// 绑定模板参数个数为 2 的 function 到 Python。
template<typename TL, typename TR>
void bind2(py::module_ &m) {
    m.def("mv", &mv<TL, TR>, "The function of matrix multiplication vector.", py::arg("mat"), py::arg("vec"));
    m.def("mm", &mm<TL, TR>, "The function of matrix multiplication matrix.", py::arg("mat1"), py::arg("mat2"));
    m.def("linear", &linear<TL, TR>, "Linear layer function.", py::arg("input"), py::arg("weight"),
          py::arg("bias"));
}


/// 绑定到 Python
PYBIND11_MODULE(_functional, m) {
    xt::import_numpy();

    m.doc() = "All kinds of functions.";

    // 绑定各种函数

//    m.def("exp", &exp<float>, "Exponential function.", py::arg("_tensor"));
//    m.def("log", &log<float>, "Logarithmic function based on `e`.", py::arg("_tensor"));
//    m.def("mv", &mv<float, float>, "The function of matrix multiplication vector.", py::arg("mat"), py::arg("vec"));
//    m.def("mm", &mm<float, float>, "The function of matrix multiplication matrix.", py::arg("mat1"), py::arg("mat2"));
//    m.def("transpose", &transpose<float>, "Transpose a _tensor.", py::arg("_tensor"),
//          py::arg("permutation") = std::vector<size_t>());
//    m.def("mean", &mean<float>, "Calculate the mean of a _tensor.", py::arg("_tensor"));
//    m.def("sum", &sum<float>, "Calculate the sum of a _tensor.", py::arg("_tensor"));
//    m.def("squeeze", &squeeze<float>, "Remove axis with dimension size 1.", py::arg("_tensor"));
//    m.def("sigmoid", &sigmoid<float>, "Sigmoid function.", py::arg("_tensor"));
//    m.def("relu", &relu<float>, "ReLU function.", py::arg("_tensor"));
//    m.def("dropout", &dropout<float>, "Dropout function.", py::arg("_tensor"), py::arg("p") = 0.5);
//    m.def("linear", &linear<float, float>, "Linear layer function.", py::arg("input"), py::arg("weight"),
//          py::arg("bias"));

//    m.def("exp", static_cast<Tensor<int32_t> (*)(const Tensor<int32_t> &)>(&exp<int32_t>), "Exponential function.",
//          py::arg("_tensor"));
//    m.def("log", &log<int32_t>, "Logarithmic function based on `e`.", py::arg("_tensor"));

    bind1<int32_t>(m);
    bind1<int64_t>(m);
    bind1<float_t>(m);
    bind1<double_t>(m);

    bind2<int32_t, int32_t>(m);
    bind2<int32_t, int64_t>(m);
    bind2<int32_t, float_t>(m);
    bind2<int32_t, double_t>(m);

    bind2<int64_t, int32_t>(m);
    bind2<int64_t, int64_t>(m);
    bind2<int64_t, float_t>(m);
    bind2<int64_t, double_t>(m);

    bind2<float_t, int32_t>(m);
    bind2<float_t, int64_t>(m);
    bind2<float_t, float_t>(m);
    bind2<float_t, double_t>(m);

    bind2<double_t, int32_t>(m);
    bind2<double_t, int64_t>(m);
    bind2<double_t, float_t>(m);
    bind2<double_t, double_t>(m);

}