/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file compressor_grad_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"
#include "log/log.h"

using namespace ge;

namespace ops {
    // INPUT
    constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
    constexpr uint32_t WEIGHT_KV_INPUT_INDEX = 1;
    constexpr uint32_t WEIGHT_WGATE_INPUT_INDEX = 2;
    constexpr uint32_t D_CMP_KV_INPUT_INDEX = 3;
    constexpr uint32_t SOFTMAX_SCORE_INPUT_INDEX = 4;
    constexpr uint32_t KV_INPUT_INDEX = 5;

    // INPUT(OPTION)
    constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 6;
    constexpr uint32_t SEQ_USED_INPUT_INDEX = 7;
    constexpr uint32_t START_POS_INPUT_INDEX = 8;

    // ATTR
    constexpr uint32_t CMP_RATIO_ATTR_INDEX = 0;
    constexpr uint32_t COFF_ATTR_INDEX = 1;

    // OUTPUT
    constexpr uint32_t D_X_OUTPUT_INDEX = 0;
    constexpr uint32_t D_WKV_OUTPUT_INDEX = 1;
    constexpr uint32_t D_WGATE_OUTPUT_INDEX = 2;
    constexpr uint32_t D_APE_OUTPUT_INDEX = 3;

    // ATTR DEFAULT VALUE
    constexpr uint32_t CMP_RATIO_VALUE = 4;
    constexpr uint32_t COFF_VALUE = 1;

struct CompressorGradProtoShapeParam {
    bool isBsMerge { false };
    int64_t B { 0 };
    int64_t T { 0 };
    int64_t S { 0 };
    int64_t Sr { 0 };
    int64_t H { 0 };
    int64_t D { 0 };
};

// tmp
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;

ge::graphStatus GetCompressorGradShapeDim(const gert::InferShapeContext* context,
                                          CompressorGradProtoShapeParam &shapeParam)
{
    auto xShape = context->GetRequiredInputShape(TOKEN_X_INPUT_INDEX);      // (B, S, H) | (T, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto wkvShape = context->GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);  // (coff * D, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, wkvShape);
    auto wgateShape = context->GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);  // (coff * D, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, wgateShape);

    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t *cmpRatioPtr = attr->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_INDEX);
    int64_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    const int64_t *coffPtr = attr->GetAttrPointer<int64_t>(COFF_ATTR_INDEX);
    int64_t coff = (coffPtr != nullptr) ? *coffPtr : COFF_VALUE;

    if (xShape->GetDimNum() == DIM_NUM_3) {                // BS
        shapeParam.isBsMerge = false;
        shapeParam.B = xShape->GetDim(DIM_INDEX_0);
        shapeParam.S = xShape->GetDim(DIM_INDEX_1);
        shapeParam.H = xShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else {                                                    // T
        shapeParam.isBsMerge = true;
        auto cuSeqlensShape = context->GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
        shapeParam.T = xShape->GetDim(DIM_INDEX_0);
        shapeParam.H = xShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.D = wkvShape->GetDim(DIM_INDEX_0) / coff;

    return GRAPH_SUCCESS;
}

ge::graphStatus SetCompressorGradShapeDim(const CompressorGradProtoShapeParam &shapeParam,
                                          gert::InferShapeContext* context)
{
    // d_x: 与 x 同形状 (B, S, H) | (T, H)
    auto dxShape = context->GetOutputShape(D_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);

    // d_wkv: (coff * D, H)
    auto dWkvShape = context->GetOutputShape(D_WKV_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dWkvShape);

    auto dWgateShape = context->GetOutputShape(D_WGATE_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dWgateShape);

    auto dApeShape = context->GetOutputShape(D_APE_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dApeShape);

    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t *cmpRatioPtr = attr->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_INDEX);
    int64_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    const int64_t *coffPtr = attr->GetAttrPointer<int64_t>(COFF_ATTR_INDEX);
    int64_t coff = (coffPtr != nullptr) ? *coffPtr : COFF_VALUE;
    // Set output shape
    if (!shapeParam.isBsMerge) {
        dxShape->SetDimNum(DIM_NUM_3);                   // (B, S, H)
        dxShape->SetDim(DIM_INDEX_0, shapeParam.B);
        dxShape->SetDim(DIM_INDEX_1, shapeParam.S);
        dxShape->SetDim(DIM_INDEX_2, shapeParam.H);
    } else {
        dxShape->SetDimNum(DIM_NUM_2);                   // (T, H)
        dxShape->SetDim(DIM_INDEX_0, shapeParam.T);
        dxShape->SetDim(DIM_INDEX_1, shapeParam.H);
    }

    dWkvShape->SetDimNum(DIM_NUM_2);                   // (coff*D, H)
    dWkvShape->SetDim(DIM_INDEX_0, shapeParam.D * coff);                   // (coff*D, H)
    dWkvShape->SetDim(DIM_INDEX_1, shapeParam.H);                   // (coff*D, H)

    dWgateShape->SetDimNum(DIM_NUM_2);                   // (coff*D, H)
    dWgateShape->SetDim(DIM_INDEX_0, shapeParam.D * coff);                   // (coff*D, H)
    dWgateShape->SetDim(DIM_INDEX_1, shapeParam.H);                   // (coff*D, H)

    dApeShape->SetDimNum(DIM_NUM_2);                   // (cmpRatio, coff*D)
    dApeShape->SetDim(DIM_INDEX_0, cmpRatio);                   // (cmpRatio, coff*D)
    dApeShape->SetDim(DIM_INDEX_1, shapeParam.D * coff);                   // (cmpRatio, coff*D)

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeCompressorGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr,
               OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "context", "is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter CompressorGrad inferDataType impl.");

    context->SetOutputDataType(D_X_OUTPUT_INDEX, context->GetRequiredInputDataType(TOKEN_X_INPUT_INDEX));
    context->SetOutputDataType(D_WKV_OUTPUT_INDEX, context->GetRequiredInputDataType(TOKEN_X_INPUT_INDEX));
    context->SetOutputDataType(D_WGATE_OUTPUT_INDEX, context->GetRequiredInputDataType(TOKEN_X_INPUT_INDEX));
    context->SetOutputDataType(D_APE_OUTPUT_INDEX, context->GetRequiredInputDataType(SOFTMAX_SCORE_INPUT_INDEX));

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeCompressorGrad(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr,
               OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "context", "is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter CompressorGrad infershape impl.");

    CompressorGradProtoShapeParam shapeParam {};
    auto apiRet = GetCompressorGradShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "context", "get input shape failed"),
                return ge::GRAPH_FAILED);

    apiRet = SetCompressorGradShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "context", "set output shape failed"),
                return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CompressorGrad).InferShape(InferShapeCompressorGrad).InferDataType(InferDataTypeCompressorGrad);
}  // namespace ops