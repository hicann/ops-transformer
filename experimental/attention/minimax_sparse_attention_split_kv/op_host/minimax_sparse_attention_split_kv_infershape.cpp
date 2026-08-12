/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr uint32_t QUERY_INDEX = 0;
static constexpr uint32_t ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t TND_DIM_NUM = 3;
static constexpr int32_t UNKNOWN_DIMS = -2;

static ge::graphStatus InferShapeMinimaxSparseAttentionSplitKv(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("MinimaxSparseAttentionSplitKv", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    gert::Shape *attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);

    if (queryShape->GetDimNum() == 1 && queryShape->GetDim(0) == UNKNOWN_DIMS) {
        attentionOutShape->SetDimNum(1);
        (*attentionOutShape)[0] = UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    if (queryShape->GetDimNum() != TND_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "MinimaxSparseAttentionSplitKv only supports TND layout, queryDims(%zu) must be 3!",
                queryShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    *attentionOutShape = *queryShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMinimaxSparseAttentionSplitKv(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(QUERY_INDEX);
    context->SetOutputDataType(ATTENTION_OUT_INDEX, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MinimaxSparseAttentionSplitKv)
    .InferShape(InferShapeMinimaxSparseAttentionSplitKv)
    .InferDataType(InferDataTypeMinimaxSparseAttentionSplitKv);

}  // namespace ops
