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
namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // M,K or K,M
    const gert::Shape *x_shape = context->GetInputShape(0);
    // K,N
    const gert::Shape *weight_shape = context->GetInputShape(1);
    const gert::Shape *group_list_shape = context->GetInputShape(2);
    // check dim
    if (x_shape->GetDimNum() != 2 || weight_shape->GetDimNum() != 2 || group_list_shape->GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "Dim of inputs not right");
        return ge::GRAPH_FAILED;
    }

    bool trans_a = *(context->GetAttrs()->GetAttrPointer<bool>(0));

    auto M = trans_a ? x_shape->GetDim(1) : x_shape->GetDim(0);
    auto N = weight_shape->GetDim(1);
    auto groupNum = group_list_shape->GetDim(0);

    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = gert::Shape({groupNum, M, N});
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GmmKDim).InferShape(InferShape);
} // namespace ge
