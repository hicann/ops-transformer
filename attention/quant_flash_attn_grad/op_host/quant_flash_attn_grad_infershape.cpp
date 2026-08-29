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
 * \file quant_flash_attn_grad_infershape.cpp
 * \brief QuantFlashAttnGrad算子InferShape实现
 * 输出shape严格按规格：
 *   - dq: 按layout‑q解析(BSND‑>[B,Sq,Nq,D] / BNSD‑>[B,Nq,Sq,D] / TND‑>[Tq,Nq,D])
 *   - dk: 与dq同形(dk/dv为Q形状)
 *   - dv: 与dq同形
 *   - dsink: 一维 [numHeadsQ]
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr size_t OUTPUT_IDX_DQ = 0;
static constexpr size_t OUTPUT_IDX_DK = 1;
static constexpr size_t OUTPUT_IDX_DV = 2;
static constexpr size_t OUTPUT_IDX_DSINK = 3;

static constexpr size_t LAYOUT_Q = 7;
static constexpr size_t LAYOUT_KV = 8;

ge::graphStatus InferShapeQuantFlashAttnGrad(gert::InferShapeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *qInputLayout = attrs->GetAttrPointer<char>(LAYOUT_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qInputLayout);
    const char *kvInputLayout = attrs->GetAttrPointer<char>(LAYOUT_KV);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvInputLayout);
    std::string qInputLayoutStr = std::string(qInputLayout) == "" ? "empty" : std::string(qInputLayout);
    for (auto &c : qInputLayoutStr) {
        c = toupper(c);
    }
    std::string kvInputLayoutStr = std::string(kvInputLayout) == "" ? "empty" : std::string(kvInputLayout);
    for (auto &c : kvInputLayoutStr) {
        c = toupper(c);
    }
    if ((qInputLayoutStr != "BSND" && qInputLayoutStr != "BNSD") ||
        (kvInputLayoutStr != "BSND" && kvInputLayoutStr != "BNSD") || kvInputLayoutStr != qInputLayoutStr) {
        OP_LOGI(context, "QuantFlashAttnGrad inputLayout error.");
        return GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(OUTPUT_IDX_DQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(OUTPUT_IDX_DK);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape *valueShape = context->GetInputShape(OUTPUT_IDX_DV);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);
    gert::Shape *dqShape = context->GetOutputShape(OUTPUT_IDX_DQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, dqShape);
    *dqShape = *queryShape;
    gert::Shape *dkShape = context->GetOutputShape(OUTPUT_IDX_DK);
    OP_CHECK_NULL_WITH_CONTEXT(context, dkShape);
    *dkShape = *keyShape;
    gert::Shape *dvShape = context->GetOutputShape(OUTPUT_IDX_DV);
    OP_CHECK_NULL_WITH_CONTEXT(context, dvShape);
    *dvShape = *valueShape;
    OP_LOGI(context, "QuantFlashAttnGrad InferShape done. dq dims.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeQuantFlashAttnGrad(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_IDX_DQ, ge::DT_BF16);
    context->SetOutputDataType(OUTPUT_IDX_DK, ge::DT_BF16);
    context->SetOutputDataType(OUTPUT_IDX_DV, ge::DT_BF16);
    context->SetOutputDataType(OUTPUT_IDX_DSINK, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantFlashAttnGrad)
    .InferShape(InferShapeQuantFlashAttnGrad)
    .InferDataType(InferDataTypeQuantFlashAttnGrad);

} // namespace ops
