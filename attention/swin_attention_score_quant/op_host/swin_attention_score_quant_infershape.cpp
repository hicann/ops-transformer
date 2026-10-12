/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swin_attention_score_quant_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace ops {
static ge::graphStatus InferShapeSwinAttentionScoreQuant(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSwinAttentionScoreQuant(gert::InferDataTypeContext* context)
{
    const ge::DataType x1_data_type = context->GetInputDataType(3);
    ge::graphStatus ret = context->SetOutputDataType(0, x1_data_type);
    return ret;
}

IMPL_OP_INFERSHAPE(SwinAttentionScoreQuant)
    .InferShape(InferShapeSwinAttentionScoreQuant)
    .InferDataType(InferDataTypeSwinAttentionScoreQuant);
} // namespace ops
