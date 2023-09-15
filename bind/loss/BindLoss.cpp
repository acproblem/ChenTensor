//
// Created by Lenovo on 2023/8/30.
// 绑定各种损失函数到 Python。
//
#include "loss/loss.h"
#include <sstream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY                // numpy C api loading

#include "xtensor-python/pyarray.hpp"     // Numpy bindings

namespace py = pybind11;


/// 绑定 损失函数 TLoss 类的函数 operator()<TP, TT> 。
/// TP : Type of y_pred, TT : Type of y_true.
template<class TLoss, typename TP, typename TT>
void bind1_part(py::class_<TLoss> &c) {
    c.def("__call__", &TLoss::operator() < TP, TT > , "Calculate loss value of `y_pred` and `y_true`.",
          py::arg("y_pred"), py::arg("y_true"));
}

/// 绑定 损失函数 TLoss 类。（真实值和预测值可以是任意类型）
template<class TLoss>
void bind1(py::class_<TLoss> &c) {
    c.def(py::init<>());

    bind1_part<TLoss, int32_t, int32_t>(c);
    bind1_part<TLoss, int32_t, int64_t>(c);
    bind1_part<TLoss, int32_t, float_t>(c);
    bind1_part<TLoss, int32_t, double_t>(c);

    bind1_part<TLoss, int64_t, int32_t>(c);
    bind1_part<TLoss, int64_t, int64_t>(c);
    bind1_part<TLoss, int64_t, float_t>(c);
    bind1_part<TLoss, int64_t, double_t>(c);

    bind1_part<TLoss, float_t, int32_t>(c);
    bind1_part<TLoss, float_t, int64_t>(c);
    bind1_part<TLoss, float_t, float_t>(c);
    bind1_part<TLoss, float_t, double_t>(c);

    bind1_part<TLoss, double_t, int32_t>(c);
    bind1_part<TLoss, double_t, int64_t>(c);
    bind1_part<TLoss, double_t, float_t>(c);
    bind1_part<TLoss, double_t, double_t>(c);
}


/// 绑定 损失函数 TLoss 类。（真实值只能为整型，预测值可以是任意类型）
template<class TLoss>
void bind2(py::class_<TLoss> &c) {
    c.def(py::init<>());

    bind1_part<TLoss, int32_t, int32_t>(c);
    bind1_part<TLoss, int32_t, int64_t>(c);

    bind1_part<TLoss, int64_t, int32_t>(c);
    bind1_part<TLoss, int64_t, int64_t>(c);

    bind1_part<TLoss, float_t, int32_t>(c);
    bind1_part<TLoss, float_t, int64_t>(c);

    bind1_part<TLoss, double_t, int32_t>(c);
    bind1_part<TLoss, double_t, int64_t>(c);
}

/// 绑定到 Python
PYBIND11_MODULE(_loss, m) {
    xt::import_numpy();

    m.doc() = "All kinds of loss functions.";

//    // 绑定 MSE 损失
//    py::class_<MSELoss>(m, "MSELoss")
//            .def(py::init<>())
//            .def("__call__", &MSELoss::operator() < float, float > , "Calculate loss value of `y_pred` and `y_true`.",
//                 py::arg("y_pred"), py::arg("y_true"));


    bind1<MSELoss>(py::class_<MSELoss>(m, "MSELoss"));
    bind2<CrossEntropyLoss>(py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss"));
}