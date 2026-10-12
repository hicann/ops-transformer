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
 * \file lightning_indexer_grad_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t DY_INDEX = 2;
constexpr uint32_t SPARSE_INDICES_INDEX = 3;
constexpr uint32_t WEIGHTS_INDEX = 4;

constexpr uint32_t ATTR_HEADNUM_INDEX = 0;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 2;
constexpr uint32_t ATTR_PRETOKENS_INDEX = 3;
constexpr uint32_t ATTR_NEXTTOKENS_INDEX = 4;
constexpr uint32_t ATTR_DETERMINSTIC_INDEX = 5;

static ge::graphStatus InferShapeLightningIndexerGrad(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Enter LightningIndexerGrad runtime infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape* keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape* weightShape = context->GetInputShape(WEIGHTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);

    gert::Shape* dqShape = context->GetOutputShape(0);
    gert::Shape* dkShape = context->GetOutputShape(1);
    gert::Shape* dweightShape = context->GetOutputShape(2);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* inputLayouPtr = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayouPtr);
    const int64_t* sparse_mode = attrs->GetInt(ATTR_SPARSE_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparse_mode);
    std::string inputLayoutPtrStr = std::string(inputLayouPtr);

    if ((std::string(inputLayoutPtrStr) != "TND") && (std::string(inputLayoutPtrStr) != "BSND")) {
        OP_LOGE(context, "The attr layout_query should be TND or BSND, but got %s.", inputLayoutPtrStr.c_str());
        return ge::GRAPH_FAILED;
    }

    if (inputLayoutPtrStr == "BSND") {
        if (queryShape->GetDimNum() != 4) {
            OP_LOGE(context, "Layout BSND, queryDims (%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (queryShape->GetDimNum() != 3) {
            OP_LOGE(context, "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    }

    *dqShape = *queryShape;
    *dkShape = *keyShape;
    *dweightShape = *weightShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexerGrad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType defaultType = context->GetInputDataType(0);

    // dq, outidx:0
    context->SetOutputDataType(0, defaultType);
    // dk, outidx:1
    context->SetOutputDataType(1, defaultType);
    // dweights, outidx:2
    context->SetOutputDataType(2, defaultType);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexerGrad)
    .InferShape(InferShapeLightningIndexerGrad)
    .InferDataType(InferDataTypeLightningIndexerGrad);
} // namespace ops
