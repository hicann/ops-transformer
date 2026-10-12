/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ATTR_LAYOUT_INDEX = 0;
constexpr uint32_t SOFTMAX_MAX_INDEX = 0;
constexpr uint32_t SOFTMAX_SUM_INDEX = 1;
constexpr uint32_t SUB_INDEX_NUM = 1;
constexpr ge::DataType OUTPUT_TYPE = ge::DT_FLOAT;

static ge::graphStatus InferShapeDenseLightningIndexerSoftmaxLse(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DenseLightningIndexerSoftmaxLse", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter DenseLightningIndexerSoftmaxLse InferDataType impl.");
    const gert::Shape* queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape* keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* inputLayoutPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutPtr);
    std::string inputLayoutPtrStr = std::string(inputLayoutPtr);
    OP_CHECK_IF(inputLayoutPtrStr != "TND" && inputLayoutPtrStr != "BSND",
                OP_LOGE(context, "The attr layout_query should be TND or BSND, but got %s.", inputLayoutPtrStr.c_str()),
                return ge::GRAPH_FAILED);

    gert::Shape* softmaxMaxShape = context->GetOutputShape(SOFTMAX_MAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);
    gert::Shape* softmaxSumShape = context->GetOutputShape(SOFTMAX_SUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);

    softmaxMaxShape->SetDimNum(queryShape->GetDimNum() - SUB_INDEX_NUM);
    softmaxSumShape->SetDimNum(queryShape->GetDimNum() - SUB_INDEX_NUM);
    if (inputLayoutPtrStr == "BSND") {
        OP_CHECK_IF(queryShape->GetDimNum() != 4,
                    OP_LOGE(context, "Layout BSND, queryDims (%zu) must be 4!", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        softmaxMaxShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        softmaxMaxShape->SetDim(1, keyShape->GetDim(2));   // 1:Dim N2
        softmaxMaxShape->SetDim(2, queryShape->GetDim(1)); // 2:Dim S
        softmaxSumShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        softmaxSumShape->SetDim(1, keyShape->GetDim(2));   // 1:Dim N2
        softmaxSumShape->SetDim(2, queryShape->GetDim(1)); // 2:Dim S
    } else {
        OP_CHECK_IF(queryShape->GetDimNum() != 3,
                    OP_LOGE(context, "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        softmaxMaxShape->SetDim(0, keyShape->GetDim(1));   // 0:Dim N2
        softmaxMaxShape->SetDim(1, queryShape->GetDim(0)); // 0:Dim T
        softmaxSumShape->SetDim(0, keyShape->GetDim(1));   // 0:Dim N2
        softmaxSumShape->SetDim(1, queryShape->GetDim(0)); // 0:Dim T
    }

    OP_LOGD(context->GetNodeName(), "DenseLightningIndexerSoftmaxLse InferShape end.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDenseLightningIndexerSoftmaxLse(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DenseLightningIndexerSoftmaxLse", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter DenseLightningIndexerSoftmaxLse InferDataType impl.");
    context->SetOutputDataType(SOFTMAX_MAX_INDEX, OUTPUT_TYPE);
    context->SetOutputDataType(SOFTMAX_SUM_INDEX, OUTPUT_TYPE);
    OP_LOGD(context->GetNodeName(), "DenseLightningIndexerSoftmaxLse InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DenseLightningIndexerSoftmaxLse)
    .InferShape(InferShapeDenseLightningIndexerSoftmaxLse)
    .InferDataType(InferDataTypeDenseLightningIndexerSoftmaxLse);
} // namespace ops
