/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <exe_graph/runtime/infer_shape_context.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

namespace ops {
namespace {
constexpr size_t INPUT_PARTIAL_BLOCK = 0;
constexpr size_t OUTPUT_PARTIAL_BLOCK = 0;
constexpr size_t OUTPUT_H = 1;
} // namespace

static ge::graphStatus InferShapeForBlockAttnResUpdate(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BlockAttnResUpdate", "InferShapeContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *partialBlockShape = context->GetInputShape(INPUT_PARTIAL_BLOCK);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialBlockShape);
    gert::Shape *partialBlockRefShape = context->GetOutputShape(OUTPUT_PARTIAL_BLOCK);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialBlockRefShape);
    gert::Shape *hShape = context->GetOutputShape(OUTPUT_H);
    OP_CHECK_NULL_WITH_CONTEXT(context, hShape);
    *partialBlockRefShape = *partialBlockShape;
    *hShape = *partialBlockShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForBlockAttnResUpdate(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BlockAttnResUpdate", "InferDataTypeContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_PARTIAL_BLOCK, context->GetInputDataType(INPUT_PARTIAL_BLOCK));
    context->SetOutputDataType(OUTPUT_H, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlockAttnResUpdate)
    .InferShape(InferShapeForBlockAttnResUpdate)
    .InferDataType(InferDataTypeForBlockAttnResUpdate);
} // namespace ops
