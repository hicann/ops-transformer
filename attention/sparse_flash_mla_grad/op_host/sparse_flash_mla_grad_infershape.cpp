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
 * \file sparse_flash_mla_grad_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
namespace smlag {

enum class InputIndex : uint32_t {
    QUERY = 0,
    ATTENTION_OUT_GRAD,
    ATTENTION_OUT,
    LSE,
    ORI_KV,
    CMP_KV,
    ORI_SPARSE_INDICES,
    CMP_SPARSE_INDICES,
    CU_SEQLENS_Q,
    CU_SEQLENS_ORI_KV,
    CU_SEQLENS_CMP_KV,
    SEQUSED_Q,
    SEQUSED_ORI_KV,
    SEQUSED_CMP_KV,
    CMP_RESIDUAL_KV,
    ORI_TOPK_LENGTH,
    CMP_TOPK_LENGTH,
    SINKS,
    METADATA
};

enum class OutputIndex : uint32_t {
    DQ = 0,
    DORI_KV,
    DCMP_KV,
    DSINKS,
    ORI_SOFTMAX_L1_NORM,
    CMP_SOFTMAX_L1_NORM
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    CMP_RATIO,
    INPUT_LAYOUT,
    ORI_MASK_MODE,
    CMP_MASK_MODE,
    ORI_WIN_LEFT,
    ORI_WIN_RIGHT,
    LAYOUT_Q,
    LAYOUT_KV
};

ge::graphStatus InferShape4SparseFlashMlaGrad(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashMlaGrad", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferShape4SparseFlashMlaGrad.");

    const gert::Shape *queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    const gert::Shape *oriKvShape = context->GetInputShape(static_cast<size_t>(InputIndex::ORI_KV));
    const gert::Shape *cmpKvShape = context->GetInputShape(static_cast<size_t>(InputIndex::CMP_KV));
    const gert::Shape *sinksShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SINKS));
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, oriKvShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpKvShape);

    auto attrs = context->GetAttrs();
    auto scaleValue = attrs->GetFloat(static_cast<size_t>(AttrIndex::SCALE_VALUE));
    auto cmpRatio = attrs->GetInt(static_cast<size_t>(AttrIndex::CMP_RATIO));
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleValue);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpRatio);

    gert::Shape *dqShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ));
    gert::Shape *dOriKvShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DORI_KV));
    gert::Shape *dCmpKvShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DCMP_KV));
    OP_CHECK_NULL_WITH_CONTEXT(context, dqShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dOriKvShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dCmpKvShape);
    *dqShape = *queryShape;
    *dOriKvShape = *oriKvShape;
    *dCmpKvShape = *cmpKvShape;

    if (sinksShape != nullptr) {
        gert::Shape *dSinksShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DSINKS));
        OP_CHECK_NULL_WITH_CONTEXT(context, dSinksShape);
        *dSinksShape = *sinksShape;
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4SparseFlashMlaGrad(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashMlaGrad", "InferDataTypeContext is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferDataType4SparseFlashMlaGrad.");

    auto dtype = context->GetInputDataType(static_cast<size_t>(InputIndex::QUERY));
    auto sinksDtype = context->GetInputDataType(static_cast<size_t>(InputIndex::SINKS));
    auto softmaxDtype = context->GetInputDataType(static_cast<size_t>(InputIndex::LSE));
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DORI_KV), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DCMP_KV), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DSINKS), sinksDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::ORI_SOFTMAX_L1_NORM), softmaxDtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::CMP_SOFTMAX_L1_NORM), softmaxDtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseFlashMlaGrad)
    .InferShape(InferShape4SparseFlashMlaGrad)
    .InferDataType(InferDataType4SparseFlashMlaGrad);
} // namespace smlag
} // namespace ops
