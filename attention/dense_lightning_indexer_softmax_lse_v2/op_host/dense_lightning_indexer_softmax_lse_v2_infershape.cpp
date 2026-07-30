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
 * \file dense_lightning_indexer_softmax_lse_v2_infershape.cpp
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
constexpr uint32_t SOFTMAX_LSE_INDEX = 0;

static ge::graphStatus InferShapeDenseLightningIndexerSoftmaxLseV2(gert::InferShapeContext *context)
{
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutPtr);
    std::string inputLayoutStr = std::string(inputLayoutPtr);

    gert::Shape *softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);

    if (inputLayoutStr == "BSND") {
        softmaxLseShape->SetDimNum(3);
        softmaxLseShape->SetDim(0, queryShape->GetDim(0));
        softmaxLseShape->SetDim(1, keyShape->GetDim(2));
        softmaxLseShape->SetDim(2, queryShape->GetDim(1));
    } else {
        softmaxLseShape->SetDimNum(2);
        softmaxLseShape->SetDim(0, keyShape->GetDim(1));
        softmaxLseShape->SetDim(1, queryShape->GetDim(0));
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDenseLightningIndexerSoftmaxLseV2(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DenseLightningIndexerSoftmaxLseV2)
    .InferShape(InferShapeDenseLightningIndexerSoftmaxLseV2)
    .InferDataType(InferDataTypeDenseLightningIndexerSoftmaxLseV2);
} // namespace ops
