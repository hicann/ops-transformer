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

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeFusedQkvProjection(gert::InferShapeContext *context)
{
    const gert::Shape *hsShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, hsShape);
    auto dimNum = hsShape->GetDimNum();
    OP_CHECK_IF(dimNum < 3, OP_LOGE(context, "hidden_states rank < 3"), return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *qPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *kPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *vPtr = attrs->GetAttrPointer<int64_t>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, qPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, kPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, vPtr);

    int64_t batch = hsShape->GetDim(0);
    int64_t seqLen = hsShape->GetDim(1);
    int64_t qDim = *qPtr;
    int64_t kDim = *kPtr;
    int64_t vDim = *vPtr;

    gert::Shape *qShape = context->GetOutputShape(0);
    gert::Shape *kShape = context->GetOutputShape(1);
    gert::Shape *vShape = context->GetOutputShape(2);

    qShape->SetDimNum(3);
    kShape->SetDimNum(3);
    vShape->SetDimNum(3);
    qShape->SetDim(0, batch);
    qShape->SetDim(1, seqLen);
    qShape->SetDim(2, qDim);
    kShape->SetDim(0, batch);
    kShape->SetDim(1, seqLen);
    kShape->SetDim(2, kDim);
    vShape->SetDim(0, batch);
    vShape->SetDim(1, seqLen);
    vShape->SetDim(2, vDim);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedQkvProjection).InferShape(InferShapeFusedQkvProjection);
} // namespace ops
