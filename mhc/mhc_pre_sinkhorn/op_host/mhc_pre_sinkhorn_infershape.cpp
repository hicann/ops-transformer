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
 * \file mhc_pre_sinkhorn_infershape.cpp
 * \brief MhcPreSinkhorn infershape implementation
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t INDEX_X = 0;
constexpr size_t OUTPUT_HIN = 0;
constexpr size_t OUTPUT_HPOST = 1;
constexpr size_t OUTPUT_HRES = 2;
constexpr size_t OUTPUT_HPRE = 3;
constexpr size_t OUTPUT_HC_BEFORE_NORM = 4;
constexpr size_t OUTPUT_INV_RMS = 5;
constexpr size_t OUTPUT_SUM_OUT = 6;
constexpr size_t OUTPUT_NORM_OUT = 7;

constexpr size_t ATTR_NUM_ITERS = 1;
constexpr size_t ATTR_NEED_BACKWARD = 4;

constexpr int64_t DEFAULT_NUM_ITERS = 20;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;
constexpr int64_t EMPTY_DIM = 0;
} // namespace

static void SetShape(gert::Shape *shape, const std::initializer_list<int64_t> &dims)
{
    shape->SetDimNum(dims.size());
    size_t index = 0;
    for (int64_t dim : dims) {
        shape->SetDim(index++, dim);
    }
}

static ge::graphStatus InferShapeForMhcPreSinkhorn(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MhcPreSinkhorn infershape.");
    const gert::Shape *xShape = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const size_t xDimNum = xShape->GetDimNum();
    OP_CHECK_IF((xDimNum != DIM_THREE) && (xDimNum != DIM_FOUR),
                OP_LOGE(context->GetNodeName(), "x dim must be 3 or 4, but got %zu.", xDimNum),
                return ge::GRAPH_FAILED);

    int64_t batch = 1;
    int64_t seqLen = 0;
    int64_t hcMult = 0;
    int64_t hiddenDim = 0;
    bool isTnd = false;
    if (xDimNum == DIM_THREE) {
        seqLen = xShape->GetDim(0);
        hcMult = xShape->GetDim(1);
        hiddenDim = xShape->GetDim(2);
        isTnd = true;
    } else {
        batch = xShape->GetDim(0);
        seqLen = xShape->GetDim(1);
        hcMult = xShape->GetDim(2);
        hiddenDim = xShape->GetDim(3);
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *numItersAttr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ITERS);
    const bool *needBackwardAttr = attrs->GetAttrPointer<bool>(ATTR_NEED_BACKWARD);
    const int64_t numIters = numItersAttr == nullptr ? DEFAULT_NUM_ITERS : *numItersAttr;
    const bool needBackward = needBackwardAttr == nullptr ? true : *needBackwardAttr;
    const int64_t hcMix = hcMult * hcMult + 2 * hcMult;

    gert::Shape *hinShape = context->GetOutputShape(OUTPUT_HIN);
    gert::Shape *hPostShape = context->GetOutputShape(OUTPUT_HPOST);
    gert::Shape *hResShape = context->GetOutputShape(OUTPUT_HRES);
    gert::Shape *hPreShape = context->GetOutputShape(OUTPUT_HPRE);
    gert::Shape *hcBeforeNormShape = context->GetOutputShape(OUTPUT_HC_BEFORE_NORM);
    gert::Shape *invRmsShape = context->GetOutputShape(OUTPUT_INV_RMS);
    gert::Shape *sumOutShape = context->GetOutputShape(OUTPUT_SUM_OUT);
    gert::Shape *normOutShape = context->GetOutputShape(OUTPUT_NORM_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, hinShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hPostShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hResShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hPreShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, hcBeforeNormShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, invRmsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, normOutShape);

    if (isTnd) {
        SetShape(hinShape, {seqLen, hiddenDim});
        SetShape(hPostShape, {seqLen, hcMult});
        SetShape(hResShape, {seqLen, hcMult * hcMult});
        if (needBackward) {
            SetShape(hPreShape, {seqLen, hcMult});
            SetShape(hcBeforeNormShape, {seqLen, hcMix});
            SetShape(invRmsShape, {seqLen, 1});
            SetShape(sumOutShape, {numIters * 2, seqLen, hcMult});
            SetShape(normOutShape, {numIters * 2, seqLen, hcMult, hcMult});
        } else {
            SetShape(hPreShape, {EMPTY_DIM});
            SetShape(hcBeforeNormShape, {EMPTY_DIM});
            SetShape(invRmsShape, {EMPTY_DIM});
            SetShape(sumOutShape, {EMPTY_DIM});
            SetShape(normOutShape, {EMPTY_DIM});
        }
    } else {
        SetShape(hinShape, {batch, seqLen, hiddenDim});
        SetShape(hPostShape, {batch, seqLen, hcMult});
        SetShape(hResShape, {batch, seqLen, hcMult * hcMult});
        if (needBackward) {
            SetShape(hPreShape, {batch, seqLen, hcMult});
            SetShape(hcBeforeNormShape, {batch, seqLen, hcMix});
            SetShape(invRmsShape, {batch, seqLen, 1});
            SetShape(sumOutShape, {numIters * 2, batch, seqLen, hcMult});
            SetShape(normOutShape, {numIters * 2, batch, seqLen, hcMult, hcMult});
        } else {
            SetShape(hPreShape, {EMPTY_DIM});
            SetShape(hcBeforeNormShape, {EMPTY_DIM});
            SetShape(invRmsShape, {EMPTY_DIM});
            SetShape(sumOutShape, {EMPTY_DIM});
            SetShape(normOutShape, {EMPTY_DIM});
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do MhcPreSinkhorn infershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMhcPreSinkhorn(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MhcPreSinkhorn infer datatype.");
    const ge::DataType xDtype = context->GetInputDataType(INDEX_X);
    context->SetOutputDataType(OUTPUT_HIN, xDtype);
    context->SetOutputDataType(OUTPUT_HPOST, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_HRES, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_HPRE, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_HC_BEFORE_NORM, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_INV_RMS, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_SUM_OUT, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_NORM_OUT, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do MhcPreSinkhorn infer datatype.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPreSinkhorn)
    .InferShape(InferShapeForMhcPreSinkhorn)
    .InferDataType(InferDataTypeForMhcPreSinkhorn);

} // namespace ops
