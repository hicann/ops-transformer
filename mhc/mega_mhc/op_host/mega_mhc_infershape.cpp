/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "register/op_impl_registry.h"
namespace ops {
namespace {
/** Infer active shapes and retain empty launcher slots for reserved outputs. */
ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* x = context->GetDynamicInputShape(0, 0);
    const auto* residual = context->GetDynamicInputShape(1, 0);
    const auto* post = context->GetDynamicInputShape(2, 0);
    const auto* comb = context->GetDynamicInputShape(7, 0);
    if (x == nullptr || residual == nullptr || post == nullptr || comb == nullptr ||
        (x->GetDimNum() != 2 && x->GetDimNum() != 3)) {
        return ge::GRAPH_FAILED;
    }
    const auto* shifted = context->GetDynamicInputShape(8, 0);
    const bool hasShiftedPrevMix = shifted != nullptr && shifted->GetDimNum() != 0;
    const gert::Shape* shapes[] = {residual, post, comb, hasShiftedPrevMix ? post : nullptr, x, nullptr, nullptr};
    for (uint32_t index = 0; index < 7U; ++index) {
        auto* output = context->GetOutputShape(index);
        if (output == nullptr) {
            if (shapes[index] != nullptr) {
                return ge::GRAPH_FAILED;
            }
            continue;
        }
        if (shapes[index] == nullptr) {
            output->SetDimNum(1);
            output->SetDim(0, 0);
        } else {
            *output = *shapes[index];
        }
    }
    return ge::GRAPH_SUCCESS;
}
/** Match activation outputs to x and use the declared coefficient/quantization types. */
ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto input = context->GetInputDataType(0);
    const ge::DataType types[] = {input, ge::DT_FLOAT,         ge::DT_FLOAT,      ge::DT_FLOAT,
                                  input, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0};
    for (uint32_t index = 0; index < 7U; ++index) {
        context->SetOutputDataType(index, types[index]);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_INFERSHAPE(MegaMhc).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ops
