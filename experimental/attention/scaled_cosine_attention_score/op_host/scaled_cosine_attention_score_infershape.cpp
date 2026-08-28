/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr size_t IN_QUERY = 0;
constexpr size_t IN_KEY = 1;
constexpr size_t IN_SCALE = 2;
constexpr size_t OUT_SCORE = 0;

bool KnownMismatch(int64_t lhs, int64_t rhs)
{
    return lhs >= 0 && rhs >= 0 && lhs != rhs;
}

bool IsValidScaleShape(const gert::Shape &scale, int64_t heads)
{
    if (scale.GetDimNum() == 1) {
        return !KnownMismatch(scale.GetDim(0), heads);
    }
    if (scale.GetDimNum() != 3) {
        return false;
    }
    return !KnownMismatch(scale.GetDim(0), heads) && !KnownMismatch(scale.GetDim(1), 1) &&
           !KnownMismatch(scale.GetDim(2), 1);
}
} // namespace

static ge::graphStatus InferShapeScaledCosineAttentionScore(gert::InferShapeContext *context)
{
    const gert::Shape *query = context->GetInputShape(IN_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, query);
    const gert::Shape *key = context->GetInputShape(IN_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, key);
    const gert::Shape *scale = context->GetInputShape(IN_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    gert::Shape *output = context->GetOutputShape(OUT_SCORE);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);

    OP_CHECK_IF(query->GetDimNum() != 4 || key->GetDimNum() != 4,
                OP_LOGE(context, "query/key rank must be 4, got %zu/%zu", query->GetDimNum(), key->GetDimNum()),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < 4; ++i) {
        OP_CHECK_IF(KnownMismatch(query->GetDim(i), key->GetDim(i)),
                    OP_LOGE(context, "query/key dimension %zu mismatch: %ld/%ld", i, query->GetDim(i), key->GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(!IsValidScaleShape(*scale, query->GetDim(1)), OP_LOGE(context, "scale must have shape [H] or [H,1,1]"),
                return ge::GRAPH_FAILED);

    output->SetDimNum(4);
    output->SetDim(0, query->GetDim(0));
    output->SetDim(1, query->GetDim(1));
    output->SetDim(2, query->GetDim(2));
    output->SetDim(3, key->GetDim(2));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeScaledCosineAttentionScore(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OUT_SCORE, context->GetInputDataType(IN_QUERY));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScaledCosineAttentionScore)
    .InferShape(InferShapeScaledCosineAttentionScore)
    .InferDataType(InferDataTypeScaledCosineAttentionScore);
} // namespace ops
