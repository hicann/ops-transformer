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

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeFlashAttnGrad(gert::InferShapeContext *context)
{
    auto qShape = context->GetInputShape(0);
    auto kShape = context->GetInputShape(1);
    auto vShape = context->GetInputShape(2);

    auto dqShape = context->GetOutputShape(0);
    auto dkShape = context->GetOutputShape(1);
    auto dvShape = context->GetOutputShape(2);

    *dqShape = *qShape;
    *dkShape = *kShape;
    *dvShape = *vShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeFlashAttnGrad(gert::InferDataTypeContext *context)
{
    auto qDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, qDtype);
    context->SetOutputDataType(1, qDtype);
    context->SetOutputDataType(2, qDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashAttnGrad).InferShape(InferShapeFlashAttnGrad).InferDataType(InferDtypeFlashAttnGrad);
} // namespace ops
