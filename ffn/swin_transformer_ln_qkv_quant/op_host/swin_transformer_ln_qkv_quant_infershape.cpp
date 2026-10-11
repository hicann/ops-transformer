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
 * \file swin_transformer_ln_qkv_quant.cc
 * \brief
 */
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace {
constexpr size_t OUTPUT_CHANNEL = 4;
constexpr int64_t UNKNOWN_DIM_VALUE = -1;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;
} // namespace

using namespace ge;
namespace ops {
inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    OP_CHECK_IF(output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
                return ge::GRAPH_FAILED);

    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus SetUnknownRank(gert::Shape* outShape)
{
    outShape->SetDimNum(0);
    outShape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeSwinTransformerLnQkvQuant(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter SwinTransformerLnQkvQuant infershape impl.");
    // xShape shape : (B, S, H)
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* qShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    gert::Shape* kShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    gert::Shape* vShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);
    uint32_t inputSize = 1;
    ge::graphStatus ret;

    if (xShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Input shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& xShapeRef = *xShape;

    if (Ops::Base::IsUnknownRank(xShapeRef)) {
        OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to -2");
        ret = SetUnknownRank(qShape);
        if (ret != ge::GRAPH_SUCCESS || (SetUnknownRank(kShape) != ge::GRAPH_SUCCESS) ||
            (SetUnknownRank(vShape) != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    for (uint32_t dimIdx = 0; dimIdx < xShape->GetDimNum(); dimIdx++) {
        auto shapeValue = xShape->GetDim(dimIdx);
        if (shapeValue == UNKNOWN_DIM_VALUE_) {
            ret = SetAllUnknownDim(OUTPUT_CHANNEL, qShape);
            if ((ret != ge::GRAPH_SUCCESS) || (SetAllUnknownDim(OUTPUT_CHANNEL, kShape) != ge::GRAPH_SUCCESS) ||
                (SetAllUnknownDim(OUTPUT_CHANNEL, vShape) != ge::GRAPH_SUCCESS)) {
                return ge::GRAPH_FAILED;
            }
            return ge::GRAPH_SUCCESS;
        }
        inputSize *= shapeValue;
    }

    OP_LOGW(context->GetNodeName(), "inputSize: %d", inputSize);
    qShape->SetDimNum(OUTPUT_CHANNEL);
    kShape->SetDimNum(OUTPUT_CHANNEL);
    vShape->SetDimNum(OUTPUT_CHANNEL);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* headNum = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNum);
    const auto* seqLength = attrs->GetAttrPointer<int64_t>(1);
    const auto* hWinSize = attrs->GetAttrPointer<int64_t>(5); // attr param 5 is hWin
    const auto* wWinSize = attrs->GetAttrPointer<int64_t>(6); //  attr params 6 is wWin

    const uint32_t outputChannel1 = *headNum;
    const uint32_t outputChannel2 = (*hWinSize) * (*wWinSize);
    const uint32_t outputChannel3 = *seqLength;
    const uint32_t outputChannel0 = inputSize / (outputChannel1 * outputChannel2 * outputChannel3);
    const uint32_t outputDim0 = 0;
    const uint32_t outputDim1 = 1;
    const uint32_t outputDim2 = 2;
    const uint32_t outputDim3 = 3;
    OP_LOGW(context->GetNodeName(), "headNum: %ld, hWinSize is %ld, wWinSize is %ld, outputDim0 is %d, \
    outputDim1 is %d,outputDim2 is %d,outputDim3 is %d",
            *headNum, *hWinSize, *wWinSize, outputChannel0, outputChannel1, outputChannel2, outputChannel3);
    qShape->SetDim(outputDim0, outputChannel0);
    qShape->SetDim(outputDim1, outputChannel1);
    qShape->SetDim(outputDim2, outputChannel2);
    qShape->SetDim(outputDim3, outputChannel3);
    *kShape = *qShape;
    *vShape = *qShape;
    OP_LOGI(context->GetNodeName(), "SwinTransformerLnQKV infershape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSwinTransformerLnQkvQuant(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter SwinTransformerLnQkvQuant inferDataType impl.");
    const ge::DataType dtype = context->GetInputDataType(0);
    // q_out, outidx:0
    ge::graphStatus ret0 = context->SetOutputDataType(0, dtype);
    // k_out, outidx:1
    context->SetOutputDataType(1, dtype);
    // v_out, outidx:2
    context->SetOutputDataType(2, dtype);
    OP_LOGI(context->GetNodeName(), "SwinTransformerLnQkvQuant inferDataType end.");
    return ret0;
}

IMPL_OP_INFERSHAPE(SwinTransformerLnQkvQuant)
    .InferShape(InferShapeSwinTransformerLnQkvQuant)
    .InferDataType(InferDataTypeSwinTransformerLnQkvQuant);
} // namespace ops
