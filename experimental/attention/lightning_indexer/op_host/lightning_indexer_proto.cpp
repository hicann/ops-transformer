/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 4;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 2;

static ge::graphStatus InferShapeLightningIndexer(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("LightningIndexer", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    gert::Shape *sparseIndicesShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndicesShape);
    gert::Shape *sparseValuesShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseValuesShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KEY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKeyPtr);
    const int64_t *seleced_count = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seleced_count);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    OP_CHECK_IF(inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND",
                OP_LOGE(context->GetNodeName(), "The attr layout_query should be TND or BSND, but got %s.",
                        inputLayoutQueryPtrStr.c_str()),
                return ge::GRAPH_FAILED);

    sparseIndicesShape->SetDimNum(queryShape->GetDimNum());
    sparseValuesShape->SetDimNum(queryShape->GetDimNum());

    if (inputLayoutQueryPtrStr == "BSND") {
        OP_CHECK_IF(queryShape->GetDimNum() != 4,
                    OP_LOGE(context->GetNodeName(), "Layout BSND, queryDims (%zu) must be 4!", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        sparseIndicesShape->SetDim(1, queryShape->GetDim(1)); // 1:Dim S
        sparseIndicesShape->SetDim(2, keyShape->GetDim(2));   // 2:Dim N
        sparseIndicesShape->SetDim(3, *seleced_count);        // 3:Dim K

        sparseValuesShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        sparseValuesShape->SetDim(1, queryShape->GetDim(1)); // 1:Dim S
        sparseValuesShape->SetDim(2, keyShape->GetDim(2));   // 2:Dim N
        sparseValuesShape->SetDim(3, *seleced_count);        // 3:Dim K

    } else {
        OP_CHECK_IF(queryShape->GetDimNum() != 3,
                    OP_LOGE(context->GetNodeName(), "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));            // 0:Dim T
        int32_t nDimIndex = (inputLayoutKeyPtrStr == "PA_BSND") ? 2 : 1; // 2:Key Dim N
        sparseIndicesShape->SetDim(1, keyShape->GetDim(nDimIndex));      // 1:Dim N
        sparseIndicesShape->SetDim(2, *seleced_count);                   // 2:Dim K

        sparseValuesShape->SetDim(0, queryShape->GetDim(0));       // 0:Dim T
        sparseValuesShape->SetDim(1, keyShape->GetDim(nDimIndex)); // 1:Dim N
        sparseValuesShape->SetDim(2, *seleced_count);              // 2:Dim K
    }
    OP_LOGD(context->GetNodeName(), "LightningIndexer InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexer(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("LightningIndexer", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter LightningIndexer InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType sparseIndicesType = ge::DT_INT32;
    ge::DataType queryType = context->GetInputDataType(QUERY_INDEX);
    // attention_out, outidx:0
    context->SetOutputDataType(0, sparseIndicesType);
    context->SetOutputDataType(1, queryType);
    OP_LOGD(context->GetNodeName(), "LightningIndexer InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexer)
    .InferShape(InferShapeLightningIndexer)
    .InferDataType(InferDataTypeLightningIndexer);
} // namespace ops
