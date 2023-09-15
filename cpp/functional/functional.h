//
// Created by Lenovo on 2023/8/28.
// 将所有对张量的函数都包含进该头文件。
//

#ifndef CHENTENSOR_FUNCTIONAL_H
#define CHENTENSOR_FUNCTIONAL_H

#include "__BasicFunc__.h"
#include "__MatrixOperationFunc__.h"
#include "__MeanFunc__.h"
#include "__OtherFunc__.h"
#include "__SumFunc__.h"
#include "__ConcatenateFunc__.h"
#include "__TrigonometricFunc__.h"

#include "functional/__LayerFunc__/__ActivationLayerFunc__.h"
#include "functional/__LayerFunc__/__DropoutLayerFunc__.h"
#include "functional/__LayerFunc__/__LinearLayerFunc__.h"
#include "functional/__LayerFunc__/__ConvLayerFunc__.h"
#include "functional/__LayerFunc__/__MaxPoolLayerFunc__.h"
#include "functional/__LayerFunc__/__BatchNormLayerFunc__.h"

#include "functional/__LossFunc__/__CrossEntropyLossFunc__.h"


#endif //CHENTENSOR_FUNCTIONAL_H
