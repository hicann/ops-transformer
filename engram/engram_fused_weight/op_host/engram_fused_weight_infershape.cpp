/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_fused_weight_infershape.cpp
 * \brief EngramFusedWeight shape/dtype 推导
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;
namespace ops {

graphStatus InferShape4EngramFusedWeight(gert::InferShapeContext* context)
{
    // 输出 weight_fused 与输入 weight_hidden 同 shape。
    const gert::Shape* weightHiddenShape = context->GetInputShape(0);
    gert::Shape* weightFusedShape = context->GetOutputShape(0);
    if (weightHiddenShape == nullptr || weightFusedShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *weightFusedShape = *weightHiddenShape;
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4EngramFusedWeight(gert::InferDataTypeContext* context)
{
    // 输出固定为 FP32 (kernel 内 BF16/FLOAT16 输入先升精度再相乘)。
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramFusedWeight)
    .InferShape(InferShape4EngramFusedWeight)
    .InferDataType(InferDtype4EngramFusedWeight);
} // namespace ops
