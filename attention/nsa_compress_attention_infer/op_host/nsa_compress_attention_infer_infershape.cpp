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
 * \file nsa_compress_attention_infer_proto.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;

namespace ops {

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t ATTEN_OUTPUT_INDEX = 0;
constexpr uint32_t TOPK_OUTPUT_INDEX = 1;
constexpr uint32_t KV_NUM_HEADS_ATTR_INDEX = 1;
constexpr uint32_t SELECT_BLOCK_ATTR_INDEX = 3;
constexpr uint32_t LAYOUT_ATTR_INDEX = 7;
static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;

static ge::graphStatus InferShapeNsaCompressAttentionInfer(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    gert::Shape* outShape = context->GetOutputShape(ATTEN_OUTPUT_INDEX);
    const gert::Shape* valueShape = context->GetInputShape(VALUE_INPUT_INDEX);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* kvHeadNumsPtr = attrs->GetAttrPointer<int64_t>(KV_NUM_HEADS_ATTR_INDEX);
    const int64_t* selectBlockCountPtr = attrs->GetAttrPointer<int64_t>(SELECT_BLOCK_ATTR_INDEX);
    const char* layOutPtr = attrs->GetAttrPointer<char>(LAYOUT_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, layOutPtr);

    auto kvHeadNums = *kvHeadNumsPtr;
    auto selectBlockCount = *selectBlockCountPtr;
    std::string layOutStr = std::string(layOutPtr);
    if (layOutStr != "TND" && layOutStr != "BSND") {
        OP_LOGE(context, "The layout should be TND and BSND, but got %s.", layOutStr.c_str());
        return GRAPH_FAILED;
    }

    auto shapeD2 = valueShape->GetDim(DIM_NUM_2);
    *outShape = *queryShape;

    gert::Shape* topkIndicesOutShape = context->GetOutputShape(TOPK_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIndicesOutShape);
    *topkIndicesOutShape = *queryShape;

    if (layOutStr == "TND") {
        outShape->SetDim(DIM_NUM_2, shapeD2 / kvHeadNums);
        topkIndicesOutShape->SetDim(DIM_NUM_1, kvHeadNums);
        topkIndicesOutShape->SetDim(DIM_NUM_2, selectBlockCount);
    } else if (layOutStr == "BSND") {
        outShape->SetDim(DIM_NUM_3, shapeD2 / kvHeadNums);
        topkIndicesOutShape->SetDim(DIM_NUM_2, kvHeadNums);
        topkIndicesOutShape->SetDim(DIM_NUM_3, selectBlockCount);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeNsaCompressAttentionInfer(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DataType::DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NsaCompressAttentionInfer)
    .InferShape(InferShapeNsaCompressAttentionInfer)
    .InferDataType(InferDataTypeNsaCompressAttentionInfer);
} // namespace ops
