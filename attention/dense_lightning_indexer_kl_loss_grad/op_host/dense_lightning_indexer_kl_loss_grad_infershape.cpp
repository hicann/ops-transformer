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
 * \file dense_lightning_indexer_kl_loss_grad_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
enum InputIdx { qEnum = 0, kEnum, wEnum, attnSoftmaxL1NormEnum, softmaxLse };

enum OutputIdx { dqEnum = 0, dkEnum, dwEnum, softmaxOutEnum };

ge::graphStatus InferShapeDenseLightningIndexerKLLossGrad(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Start enter DenseLightningIndexerKLLossGrad infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *qShape = context->GetInputShape(qEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    const gert::Shape *kShape = context->GetInputShape(kEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    const gert::Shape *wShape = context->GetInputShape(wEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    const gert::Shape *softmaxShape = context->GetInputShape(attnSoftmaxL1NormEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxShape);
    const gert::Shape *softmaxLseShape = context->GetInputShape(softmaxLse);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);

    gert::Shape *dqShape = context->GetOutputShape(dqEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, dqShape);
    gert::Shape *dkShape = context->GetOutputShape(dkEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, dkShape);
    gert::Shape *dwShape = context->GetOutputShape(dwEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, dwShape);
    gert::Shape *softmaxOutShape = context->GetOutputShape(softmaxOutEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxOutShape);

    *dqShape = *qShape;
    *dkShape = *kShape;
    *dwShape = *wShape;
    *softmaxOutShape = *softmaxShape;
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeDenseLightningIndexerKLLossGrad(gert::InferDataTypeContext *context)
{
    OP_LOGI(context, "Start enter DenseLightningIndexerKLLossGrad infer dtype impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(dqEnum, context->GetInputDataType(qEnum));
    context->SetOutputDataType(dkEnum, context->GetInputDataType(kEnum));
    context->SetOutputDataType(dwEnum, ge::DT_FLOAT);
    context->SetOutputDataType(softmaxOutEnum, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DenseLightningIndexerKLLossGrad)
    .InferShape(InferShapeDenseLightningIndexerKLLossGrad)
    .InferDataType(InferDataTypeDenseLightningIndexerKLLossGrad);
} // namespace ops
