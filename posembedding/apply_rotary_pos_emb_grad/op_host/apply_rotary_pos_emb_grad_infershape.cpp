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
 * \file apply_rotary_pos_emb_grad_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t IGQ = 0; // grad_query_embed
static constexpr size_t IGK = 1; // grad_key_embed
static constexpr size_t ICS = 2; // cos
static constexpr size_t ISN = 3; // sin
static constexpr size_t OGQ = 0; // grad_query
static constexpr size_t OGK = 1; // grad_key
static constexpr size_t OGC = 2; // grad_cos
static constexpr size_t OGS = 3; // grad_sin
static constexpr int64_t NEG_TWO = -2;

/*
 * @brief: set grad output shape by its same-shape input (tiling 阶段校验全等)
 * @param [in] inputShape: const gert::Shape&, input shape
 * @param [in/out] outputShape: gert::Shape*, output shape
 * 动态 shape 处理: -2 (unknown rank), -1 (unknown dim)
 */
static void SetGradOutputShape(const gert::Shape &inputShape, gert::Shape *outputShape)
{
    if (Ops::Base::IsUnknownRank(inputShape)) {
        outputShape->SetDimNum(1);
        outputShape->SetDim(0, NEG_TWO);
        return;
    }
    *outputShape = inputShape;
}

static ge::graphStatus InferShapeForApplyRotaryPosEmbGrad(gert::InferShapeContext *context)
{
    const gert::Shape *gq = context->GetInputShape(IGQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, gq);
    const gert::Shape *gk = context->GetInputShape(IGK);
    OP_CHECK_NULL_WITH_CONTEXT(context, gk);
    const gert::Shape *cs = context->GetInputShape(ICS);
    OP_CHECK_NULL_WITH_CONTEXT(context, cs);
    const gert::Shape *sn = context->GetInputShape(ISN);
    OP_CHECK_NULL_WITH_CONTEXT(context, sn);

    gert::Shape *ogq = context->GetOutputShape(OGQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, ogq);
    gert::Shape *ogk = context->GetOutputShape(OGK);
    OP_CHECK_NULL_WITH_CONTEXT(context, ogk);
    gert::Shape *ogc = context->GetOutputShape(OGC);
    gert::Shape *ogs = context->GetOutputShape(OGS);

    SetGradOutputShape(*gq, ogq);
    SetGradOutputShape(*gk, ogk);
    if (ogc != nullptr) {
        SetGradOutputShape(*cs, ogc);
    }
    if (ogs != nullptr) {
        SetGradOutputShape(*sn, ogs);
    }
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForApplyRotaryPosEmbGrad(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OGQ, context->GetInputDataType(IGQ));
    context->SetOutputDataType(OGK, context->GetInputDataType(IGK));
    context->SetOutputDataType(OGC, context->GetInputDataType(ICS));
    context->SetOutputDataType(OGS, context->GetInputDataType(ISN));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyRotaryPosEmbGrad)
    .InferShape(InferShapeForApplyRotaryPosEmbGrad)
    .InferDataType(InferDataTypeForApplyRotaryPosEmbGrad);
} // namespace ops
