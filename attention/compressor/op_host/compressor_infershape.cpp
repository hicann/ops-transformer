/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
    // INPUT
    constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
    constexpr uint32_t WEIGHT_KV_INPUT_INDEX = 1;
    constexpr uint32_t WEIGHT_WGATE_INPUT_INDEX = 2;

    constexpr uint32_t STATE_CACHE_INPUT_INDEX = 3;

    constexpr uint32_t APE_INPUT_INDEX = 4;

    // INPUT(OPTION)
    constexpr uint32_t STATE_BLOCK_TABLE_INPUT_INDEX = 5;

    constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 6;
    constexpr uint32_t SEQ_USED_INPUT_INDEX = 7;
    constexpr uint32_t START_POS_INPUT_INDEX = 8;

    // ATTR
    constexpr uint32_t CMP_RATIO_ATTR_INDEX = 0;
    constexpr uint32_t COFF_ATTR_INDEX = 1;
    constexpr uint32_t CACHE_MODE_ATTR_INDEX = 2;
    constexpr uint32_t STATE_CACHE_STRIDE_DIM0_ATTR_INDEX = 3;

    // OUTPUT
    constexpr uint32_t CMP_KV_OUTPUT_INDEX = 0;
    constexpr uint32_t SOFTMAX_SCORE_OUTPUT_INDEX = 2;
    constexpr uint32_t KV_OUTPUT_INDEX = 3;

    // ATTR DEFAULT VALUE
    constexpr uint32_t CMP_RATIO_VALUE = 4;
    constexpr uint32_t COFF_VALUE = 1;

struct CompressorProtoShapeParam {
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

ge::graphStatus GetCompressorShapeDim(const gert::InferShapeContext* context, CompressorProtoShapeParam &shapeParam)
{
    auto xShape = context->GetRequiredInputShape(TOKEN_X_INPUT_INDEX);      // (B, S, H) | (T, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto wkvShape = context->GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);  // (coff * D, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, wkvShape);
    auto wgateShape = context->GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);  // (coff * D, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, wgateShape);
    // (block_num, block_size, 2 * coff * D) | (B, token_size, 2 * coff * D)
    auto stateCacheShape = context->GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateCacheShape);

    auto apeShape = context->GetRequiredInputShape(APE_INPUT_INDEX);    // (r, coff * D)
    OP_CHECK_NULL_WITH_CONTEXT(context, apeShape);

    // (B, sMax/block_size) | (B, )
    auto stateBlockTableShape = context->GetOptionalInputShape(STATE_BLOCK_TABLE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateBlockTableShape);
    if (xShape->GetDimNum() == DIM_NUM_2) {
        auto cuSeqlensShape = context->GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);    // (B+1,)
        OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqlensShape);
    }
    auto seqUsedShape = context->GetOptionalInputShape(SEQ_USED_INPUT_INDEX);    // (B,)
    OP_CHECK_NULL_WITH_CONTEXT(context, seqUsedShape);
    auto startPosShape = context->GetOptionalInputShape(START_POS_INPUT_INDEX);    // (B,)
    OP_CHECK_NULL_WITH_CONTEXT(context, startPosShape);

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
        shapeParam.Sr = (xShape->GetDim(DIM_INDEX_1) + cmpRatio - 1) / cmpRatio;
        shapeParam.H = xShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else {                                                    // T
        shapeParam.isBsMerge = true;
        auto cuSeqlensShape = context->GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
        shapeParam.Sr = std::min(xShape->GetDim(DIM_INDEX_0),
            xShape->GetDim(DIM_INDEX_0) / cmpRatio+ cuSeqlensShape->GetDim(DIM_INDEX_0) - 1);
        shapeParam.T = xShape->GetDim(DIM_INDEX_0);
        shapeParam.H = xShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.D = wkvShape->GetDim(DIM_INDEX_0) / coff;

    return GRAPH_SUCCESS;
}

ge::graphStatus SetCompressorShapeDim(const CompressorProtoShapeParam &shapeParam, gert::InferShapeContext* context)
{
    // query: (B, S, N, Hckv) | (T, N, Hckv)
    auto cmpKvShape = context->GetOutputShape(CMP_KV_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpKvShape);
    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t *cmpRatioPtr = attr->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_INDEX);
    int64_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    const int64_t *coffPtr = attr->GetAttrPointer<int64_t>(COFF_ATTR_INDEX);
    int64_t coff = (coffPtr != nullptr) ? *coffPtr : COFF_VALUE;
    int64_t extraDim = coff * cmpRatio;

    // Set output shape
    if (!shapeParam.isBsMerge) {
        cmpKvShape->SetDimNum(DIM_NUM_3);                   // (B, Sr, H)
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.B);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_2, shapeParam.D);

        auto softmaxScoreShape = context->GetOutputShape(SOFTMAX_SCORE_OUTPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, softmaxScoreShape);
        softmaxScoreShape->SetDimNum(DIM_NUM_4);        // (B, Sr, coff*cmpRatio, D)
        softmaxScoreShape->SetDim(DIM_INDEX_0, shapeParam.B);
        softmaxScoreShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        softmaxScoreShape->SetDim(DIM_INDEX_2, extraDim);
        softmaxScoreShape->SetDim(DIM_INDEX_3, shapeParam.D);

        auto kvShape = context->GetOutputShape(KV_OUTPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, kvShape);
        kvShape->SetDimNum(DIM_NUM_4);                  // (B, Sr, coff*cmpRatio, D)
        kvShape->SetDim(DIM_INDEX_0, shapeParam.B);
        kvShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        kvShape->SetDim(DIM_INDEX_2, extraDim);
        kvShape->SetDim(DIM_INDEX_3, shapeParam.D);
    } else {
        cmpKvShape->SetDimNum(DIM_NUM_2);                   // (T, N, Hckv)
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.D);

        auto softmaxScoreShape = context->GetOutputShape(SOFTMAX_SCORE_OUTPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, softmaxScoreShape);
        softmaxScoreShape->SetDimNum(DIM_NUM_3);        // (Sr, coff*cmpRatio, D)
        softmaxScoreShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        softmaxScoreShape->SetDim(DIM_INDEX_1, extraDim);
        softmaxScoreShape->SetDim(DIM_INDEX_2, shapeParam.D);

        auto kvShape = context->GetOutputShape(KV_OUTPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, kvShape);
        kvShape->SetDimNum(DIM_NUM_3);                  // (Sr, coff*cmpRatio, D)
        kvShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        kvShape->SetDim(DIM_INDEX_1, extraDim);
        kvShape->SetDim(DIM_INDEX_2, shapeParam.D);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeCompressor(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr,
               OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("Compressor", "context", "is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter Compressor inferDataType impl.");

    context->SetOutputDataType(CMP_KV_OUTPUT_INDEX, context->GetRequiredInputDataType(TOKEN_X_INPUT_INDEX));
    context->SetOutputDataType(SOFTMAX_SCORE_OUTPUT_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(KV_OUTPUT_INDEX, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeCompressor(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr,
               OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("Compressor", "context", "is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter Compressor infershape impl.");

    CompressorProtoShapeParam shapeParam {};
    auto apiRet = GetCompressorShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "context", "get input shape failed"),
                return ge::GRAPH_FAILED);

    apiRet = SetCompressorShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "context", "set output shape failed"),
                return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Compressor).InferShape(InferShapeCompressor).InferDataType(InferDataTypeCompressor);
}  // namespace ops