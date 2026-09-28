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
 * \file attention_worker_combine_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

constexpr size_t INPUT_IDX_SCHEDULE_CONTEXT = 0;
constexpr size_t INPUT_IDX_EXPERT_SCALES = 1;
constexpr size_t INPUT_IDX_LAYER_ID = 2;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_NEXT_LAYER_ID = 1;
constexpr size_t IDX_ZERO = 0;
constexpr size_t IDX_ONE = 1;
constexpr int64_t DIM_NUM = 2;
constexpr int64_t TOKEN_DTYPE_FP16 = 0;
constexpr int64_t TOKEN_DTYPE_BF16 = 1;

graphStatus InferShape4AttentionWorkerCombine(gert::InferShapeContext *context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShape4AttentionWorkerCombine");

    // input
    const gert::Shape *scheduleContextInputShape = context->GetInputShape(INPUT_IDX_SCHEDULE_CONTEXT);
    const gert::Shape *expertScalesInputShape = context->GetInputShape(INPUT_IDX_EXPERT_SCALES); // [BS, K]
    const gert::Shape *layerIdInputShape = context->GetInputShape(INPUT_IDX_LAYER_ID);           // [1, ]
    OP_CHECK_NULL_WITH_CONTEXT(context, scheduleContextInputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertScalesInputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, layerIdInputShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto hiddenSize = attrs->GetAttrPointer<int64_t>(IDX_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context, hiddenSize);

    // output
    gert::Shape *yShape = context->GetOutputShape(OUTPUT_IDX_Y); // [BS, H]
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(DIM_NUM);
    yShape->SetDim(IDX_ZERO, expertScalesInputShape->GetDim(IDX_ZERO));
    yShape->SetDim(IDX_ONE, *hiddenSize);

    gert::Shape *nextLayerIdShape = context->GetOutputShape(OUTPUT_IDX_NEXT_LAYER_ID); // [1, ]
    OP_CHECK_NULL_WITH_CONTEXT(context, nextLayerIdShape);

    *nextLayerIdShape = *layerIdInputShape; // [1, ]
    OP_LOGD(context->GetNodeName(), "End to do InferShape4AttentionWorkerCombine");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4AttentionWorkerCombine(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferDtype4AttentionWorkerCombine enter");

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto tokenDtype = attrs->GetAttrPointer<int64_t>(IDX_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context, tokenDtype);
    OP_CHECK_IF(*tokenDtype < 0 || *tokenDtype > 4,
                OP_LOGE(context->GetNodeName(), "token_dtype must be 0, 1, 2, 3 or 4."), return GRAPH_FAILED);
    if (*tokenDtype != TOKEN_DTYPE_FP16) {
        context->SetOutputDataType(OUTPUT_IDX_Y, ge::DT_BF16);
    } else {
        context->SetOutputDataType(OUTPUT_IDX_Y, ge::DT_FLOAT16);
    }

    auto layer_id_input_dtype = context->GetInputDataType(INPUT_IDX_LAYER_ID);
    context->SetOutputDataType(OUTPUT_IDX_NEXT_LAYER_ID, layer_id_input_dtype);

    OP_LOGD(context->GetNodeName(), "InferDtype4AttentionWorkerCombine end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AttentionWorkerCombine)
    .InferShape(InferShape4AttentionWorkerCombine)
    .InferDataType(InferDtype4AttentionWorkerCombine);
} // namespace ops
