/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file lightning_indexer_kl_loss_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeLightningIndexerKLLoss(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeLightningIndexerKLLoss");

    // get input shapes
    const gert::Shape *targetScoreShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetScoreShape);

    // get output shapes
    gert::Shape *lossShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);

    // 填充输出shape大小
    auto targetScoreShapeSize = targetScoreShape->GetDimNum();
    lossShape->SetDimNum(1);
    lossShape->SetDim(0, 1);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeLightningIndexerKLLoss");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexerKLLoss(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeLightningIndexerKLLoss");

    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto inputDataType = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(0, inputDataType);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeLightningIndexerKLLoss");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexerKLLoss)
    .InferShape(InferShapeLightningIndexerKLLoss)
    .InferDataType(InferDataTypeLightningIndexerKLLoss);
} // namespace ops
