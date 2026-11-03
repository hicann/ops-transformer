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

namespace ops {
static ge::graphStatus InferShapeKvCacheTurboQuant(gert::InferShapeContext* context)
{
    const auto* input = context->GetInputShape(0);
    const auto* attrs = context->GetAttrs();
    const auto* qjlInput = context->GetInputShape(2);
    if (input == nullptr || attrs == nullptr || qjlInput == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (input->GetDimNum() != 3 || qjlInput->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const auto* bits = attrs->GetAttrPointer<int64_t>(0);
    if (bits == nullptr || (*bits != 2 && *bits != 3 && *bits != 4)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tokens = input->GetDim(0);
    const int64_t heads = input->GetDim(1);
    const int64_t headDim = input->GetDim(2);
    const int64_t qjlDim = qjlInput->GetDim(0);
    auto* indexOutput = context->GetOutputShape(0);
    indexOutput->SetDimNum(3);
    indexOutput->SetDim(0, tokens);
    indexOutput->SetDim(1, heads);
    indexOutput->SetDim(2, (headDim * *bits + 7) / 8);
    auto* qjlOutput = context->GetOutputShape(1);
    qjlOutput->SetDimNum(3);
    qjlOutput->SetDim(0, tokens);
    qjlOutput->SetDim(1, heads);
    qjlOutput->SetDim(2, (qjlDim + 7) / 8);
    for (size_t index = 2; index < 4; ++index) {
        auto* output = context->GetOutputShape(index);
        output->SetDimNum(2);
        output->SetDim(0, tokens);
        output->SetDim(1, heads);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeKvCacheTurboQuant(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_UINT8);
    context->SetOutputDataType(1, ge::DT_UINT8);
    const auto* outputDtype = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    const auto metadataType = outputDtype != nullptr && *outputDtype == 1 ? ge::DT_FLOAT16 : ge::DT_BF16;
    context->SetOutputDataType(2, metadataType);
    context->SetOutputDataType(3, metadataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCacheTurboQuant)
    .InferShape(InferShapeKvCacheTurboQuant)
    .InferDataType(InferDataTypeKvCacheTurboQuant);
} // namespace ops
