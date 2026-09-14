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
#include <cstdint>
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
constexpr uint32_t X_DESCALE_INPUT_INDEX = 5;
constexpr uint32_t WKV_DESCALE_INPUT_INDEX = 6;
constexpr uint32_t WGATE_DESCALE_INPUT_INDEX = 7;
constexpr uint32_t STATE_BLOCK_TABLE_INPUT_INDEX = 8;
constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 9;
constexpr uint32_t SEQ_USED_INPUT_INDEX = 10;
constexpr uint32_t START_POS_INPUT_INDEX = 11;

// ATTR
constexpr uint32_t QUANT_MODE_ATTR_INDEX = 0;
constexpr uint32_t CMP_RATIO_ATTR_INDEX = 1;
constexpr uint32_t COFF_ATTR_INDEX = 2;
constexpr uint32_t CACHE_MODE_ATTR_INDEX = 3;

// OUTPUT
constexpr uint32_t CMP_KV_OUTPUT_INDEX = 0;

// ATTR RANGE
constexpr int64_t CMP_RATIO_MIN = 2;
constexpr int64_t CMP_RATIO_MAX = 128;
constexpr int64_t COFF_VALUE_1 = 1;
constexpr int64_t COFF_VALUE_2 = 2;
constexpr int64_t COFF_DEFAULT = 1;
constexpr uint32_t DIM_NUM_0 = 0;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;

struct QuantCompressorProtoShapeParam {
    bool isBsMerge{false};
    int64_t B{0};
    int64_t T{0};
    int64_t S{0};
    int64_t Sr{0};
    int64_t H{0};
    int64_t D{0};
};

ge::graphStatus GetQuantCompressorShapeDim(const gert::InferShapeContext *context,
                                           QuantCompressorProtoShapeParam &shapeParam)
{
    auto xShape = context->GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto wkvShape = context->GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wkvShape);

    auto xDim = xShape->GetDimNum();
    OP_CHECK_IF((xDim != DIM_NUM_2 && xDim != DIM_NUM_3),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xDim) + "D", "2D or 3D"),
                return ge::GRAPH_FAILED);

    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t *cmpRatioPtr = attr->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_INDEX);
    OP_CHECK_IF((cmpRatioPtr == nullptr),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "cmp_ratio", "attr is required"),
                return ge::GRAPH_FAILED);
    int64_t cmpRatio = *cmpRatioPtr;
    OP_CHECK_IF((cmpRatio < CMP_RATIO_MIN || cmpRatio > CMP_RATIO_MAX),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "cmp_ratio", std::to_string(cmpRatio),
                                                      "cmp_ratio should be within [" + std::to_string(CMP_RATIO_MIN) +
                                                          ", " + std::to_string(CMP_RATIO_MAX) + "]"),
                return ge::GRAPH_FAILED);
    const int64_t *coffPtr = attr->GetAttrPointer<int64_t>(COFF_ATTR_INDEX);
    int64_t coff = (coffPtr != nullptr) ? *coffPtr : COFF_DEFAULT;
    OP_CHECK_IF((coff != COFF_VALUE_1 && coff != COFF_VALUE_2),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coff", std::to_string(coff),
                                                      "coff should be 1 or 2"),
                return ge::GRAPH_FAILED);

    if (xShape->GetDimNum() == DIM_NUM_3) {
        shapeParam.isBsMerge = false;
        shapeParam.B = xShape->GetDim(DIM_INDEX_0);
        shapeParam.S = xShape->GetDim(DIM_INDEX_1);
        shapeParam.Sr = (xShape->GetDim(DIM_INDEX_1) > DIM_NUM_0) ?
                            ((xShape->GetDim(DIM_INDEX_1) - DIM_NUM_1) / cmpRatio + DIM_NUM_1) :
                            DIM_NUM_0;
        shapeParam.H = xShape->GetDim(DIM_INDEX_2);
    } else {
        shapeParam.isBsMerge = true;
        shapeParam.T = xShape->GetDim(DIM_INDEX_0);
        shapeParam.H = xShape->GetDim(DIM_INDEX_1);
        auto cuSeqlensShape = context->GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
        OP_CHECK_IF((cuSeqlensShape == nullptr),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "cu_seqlens",
                                                             "is null but required for TH layout"),
                    return ge::GRAPH_FAILED);
        int64_t cuSeqLenDim0 = cuSeqlensShape->GetDim(DIM_INDEX_0);
        OP_CHECK_IF((cuSeqLenDim0 < DIM_NUM_1),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "cu_seqlens",
                                                          std::to_string(cuSeqLenDim0), "dim0 must be positive"),
                    return ge::GRAPH_FAILED);
        int64_t Bsize = cuSeqLenDim0 - DIM_NUM_1;
        int64_t tDivR = shapeParam.T / cmpRatio;
        // Bsize 已由上方校验保证非负，此处仅保护 tDivR + Bsize 不发生加法溢出
        if (tDivR <= INT64_MAX - Bsize) {
            shapeParam.Sr = std::min(shapeParam.T, tDivR + Bsize);
        } else {
            shapeParam.Sr = shapeParam.T;
        }
    }

    shapeParam.D = wkvShape->GetDim(DIM_INDEX_0) / coff;
    OP_CHECK_IF((shapeParam.D <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "headDim (wkv.dim0 / coff)",
                                                      std::to_string(shapeParam.D), "must be positive"),
                return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

ge::graphStatus SetQuantCompressorShapeDim(const QuantCompressorProtoShapeParam &shapeParam,
                                           gert::InferShapeContext *context)
{
    auto cmpKvShape = context->GetOutputShape(CMP_KV_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpKvShape);
    if (!shapeParam.isBsMerge) {
        cmpKvShape->SetDimNum(DIM_NUM_3);
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.B);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_2, shapeParam.D);
    } else {
        cmpKvShape->SetDimNum(DIM_NUM_2);
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.D);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeQuantCompressor(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantCompressor", "context", "is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter QuantCompressor inferDataType impl.");
    context->SetOutputDataType(CMP_KV_OUTPUT_INDEX, ge::DT_BF16);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeQuantCompressor(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantCompressor", "context", "is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter QuantCompressor infershape impl.");

    QuantCompressorProtoShapeParam shapeParam{};
    auto apiRet = GetQuantCompressorShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "context", "get input shape failed"),
                return ge::GRAPH_FAILED);

    apiRet = SetQuantCompressorShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "context", "set output shape failed"),
                return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantCompressor).InferShape(InferShapeQuantCompressor).InferDataType(InferDataTypeQuantCompressor);
} // namespace ops
