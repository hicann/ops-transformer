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
static ge::graphStatus InferShapeGroupMemcpy(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const size_t count = context->GetComputeNodeOutputNum();
    if (count == 0) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < count; ++i) {
        const auto* src = context->GetDynamicInputShape(0, i);
        auto* dst = context->GetOutputShape(i);
        if (src == nullptr || dst == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *dst = *src;
    }
    return context->GetDynamicInputShape(0, count) == nullptr ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

static ge::graphStatus InferDataTypeGroupMemcpy(gert::InferDataTypeContext* context)
{
    if (context == nullptr || context->GetComputeNodeOutputNum() == 0) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < context->GetComputeNodeOutputNum(); ++i) {
        const auto dtype = context->GetDynamicInputDataType(0, i);
        if (dtype == ge::DT_UNDEFINED) {
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(i, dtype);
    }
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(GroupMemcpy).InferShape(InferShapeGroupMemcpy).InferDataType(InferDataTypeGroupMemcpy);
} // namespace ops
