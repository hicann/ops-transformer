/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file quant_flash_attention_score_grad_tiling_common_regbase.cpp
 * \brief
 */

#include "quant_flash_attention_score_grad_tiling_common_regbase.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {
namespace QuantFag {

ge::graphStatus CheckSoftmaxLseShape(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, bool isQuant)
{
    auto softmaxLseShape = context->GetInputShape(static_cast<size_t>(InputIndex::SOFTMAX_LSE));
    if (softmaxLseShape == nullptr) {
        OP_LOGE(context, "CheckSoftmaxLseShape softmaxLse is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto softmaxLseShapeDim = softmaxLseShape->GetStorageShape().GetDimNum();
    if (softmaxLseShapeDim != 3) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("QuantFlashAttentionScoreGrad", "softmaxLse",
                                                 std::to_string(softmaxLseShapeDim).c_str(),
                                                 "The shape dim of softmaxLse must be 3");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxLseShape->GetStorageShape().GetDim(0); // b
    auto dim1 = softmaxLseShape->GetStorageShape().GetDim(1); // n1
    auto dim2 = softmaxLseShape->GetStorageShape().GetDim(2); // s1
    std::string reasonMsg = "The shape of softmaxLse must be [" + std::to_string(b) + "," + std::to_string(n1) + "," +
                            std::to_string(s1) + "]";
    OP_CHECK_IF((dim0 != b || dim1 != n1 || dim2 != s1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("QuantFlashAttentionScoreGrad", "softmaxLse",
                                                       Ops::Base::ToString(softmaxLseShape->GetStorageShape()).c_str(),
                                                       reasonMsg.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInShape(gert::TilingContext *context)
{
    auto attentionInShape = context->GetInputShape(static_cast<size_t>(InputIndex::ATTN_OUT));
    if (attentionInShape == nullptr) {
        OP_LOGE(context, "CheckAttentionInShape attn_out is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
    auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
    if (attentionInShapeDim != queryShapeDim) {
        std::string shapeDimMsg = std::to_string(queryShapeDim) + ", " + std::to_string(attentionInShapeDim);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("QuantFlashAttentionScoreGrad", "query, attentionInOptional",
                                                 shapeDimMsg.c_str(),
                                                 "The shape dims of query and attentionInOptional must be same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeValid(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, int64_t d)
{
    auto isShapeInValid = (b == 0 || n1 == 0 || s1 == 0 || d == 0);
    std::string shapeMsg =
        std::to_string(b) + "," + std::to_string(n1) + "," + std::to_string(s1) + "," + std::to_string(d);
    OP_CHECK_IF(isShapeInValid,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("QuantFlashAttentionScoreGrad", "query", shapeMsg.c_str(),
                                                      "All axes of query must be postitive numbers"),
                return ge::GRAPH_FAILED);

    auto queryType = context->GetInputDesc(0)->GetDataType();
    bool isQuant = queryType == ge::DT_HIFLOAT8;
    auto ret = CheckSoftmaxLseShape(context, b, n1, s1, isQuant);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "CheckShapeValid CheckSoftmaxLseShape error");
        return ret;
    }
    ret = CheckAttentionInShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "CheckShapeValid CheckAttentionInShape error");
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttenMaskShape(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    // check atten_mask shape when enable atten_mask_compress
    if (fBaseParams.attenMaskCompressMode == 0) {
        bool invalid =
            fBaseParams.attenMaskOptional != EMPTY_TENSOR && fBaseParams.layoutType != INPUT_FORMAT_TND &&
            (static_cast<int64_t>(fBaseParams.attenMaskS1Size) * static_cast<int64_t>(fBaseParams.attenMaskS2Size) <
             static_cast<int64_t>(fBaseParams.s1) * static_cast<int64_t>(fBaseParams.s2));
        if (invalid) {
            std::string shapeSizeMsg =
                std::to_string(fBaseParams.attenMaskS1Size) + " *" + std::to_string(fBaseParams.attenMaskS2Size);
            std::string reasonMsg = "When attenMaskOptional is not empty and inputLayout is not TND, "
                                    "the shape size of attenMaskOptional cannot be less than" +
                                    std::to_string(fBaseParams.s1) + " *" + std::to_string(fBaseParams.s2);
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("QuantFlashAttentionScoreGrad", "attenMaskOptional",
                                                      shapeSizeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (fBaseParams.attenMaskCompressMode == static_cast<uint32_t>(AttenMaskCompressMode::PREFIX_COMPRESS_MODE)) {
        if (fBaseParams.attenMaskS1Size != PREFIX_COMPRESS_S1_SIZE ||
            fBaseParams.attenMaskS2Size != ATTEN_MASK_COMPRESS_LIMIT) {
            std::string shapeMsg =
                std::to_string(fBaseParams.attenMaskS1Size) + ", " + std::to_string(fBaseParams.attenMaskS2Size);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "QuantFlashAttentionScoreGrad", "attenMaskOptional", shapeMsg.c_str(),
                "When sparseMode is 6, the shape of attenMaskOptional must be [3072, 2048]");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (fBaseParams.attenMaskS1Size != fBaseParams.attenMaskS2Size) {
        std::string shapeMsg =
            std::to_string(fBaseParams.attenMaskS1Size) + ", " + std::to_string(fBaseParams.attenMaskS2Size);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("QuantFlashAttentionScoreGrad", "attenMaskOptional", shapeMsg.c_str(),
                                              "Sq of attenMaskOptional must be equal to Skv of attenMaskOptional");
        return ge::GRAPH_FAILED;
    }

    if (fBaseParams.attenMaskS2Size != ATTEN_MASK_COMPRESS_LIMIT) {
        std::string shapeMsg =
            std::to_string(fBaseParams.attenMaskS1Size) + ", " + std::to_string(fBaseParams.attenMaskS2Size);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("QuantFlashAttentionScoreGrad", "attenMaskOptional", shapeMsg.c_str(),
                                              "Skv of attenMaskOptional must be equal to 2048");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantScaleShapeValidCheck(gert::TilingContext *context_, const FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    auto deqScaleQShape = context_->GetInputShape(static_cast<size_t>(InputIndex::D_SCALE_Q));
    auto deqScaleKShape = context_->GetInputShape(static_cast<size_t>(InputIndex::D_SCALE_K));
    auto deqScaleVShape = context_->GetInputShape(static_cast<size_t>(InputIndex::D_SCALE_V));
    auto deqScaleDyShape = context_->GetInputShape(static_cast<size_t>(InputIndex::D_SCALE_DOUT));
    if (deqScaleQShape != nullptr && deqScaleKShape != nullptr && deqScaleVShape != nullptr &&
        deqScaleDyShape != nullptr) {
        auto deqScaleQStorageShape = deqScaleQShape->GetStorageShape();
        auto deqScaleKStorageShape = deqScaleKShape->GetStorageShape();
        auto deqScaleVStorageShape = deqScaleVShape->GetStorageShape();
        auto deqScaleDyStorageShape = deqScaleDyShape->GetStorageShape();
        int64_t deqScaleQDimNum = deqScaleQStorageShape.GetDimNum();
        int64_t deqScaleKDimNum = deqScaleKStorageShape.GetDimNum();
        int64_t deqScaleVDimNum = deqScaleVStorageShape.GetDimNum();
        int64_t deqScaleDyDimNum = deqScaleDyStorageShape.GetDimNum();
        if (deqScaleQDimNum == 1) {
            int64_t deqScaleQDim0 = deqScaleQStorageShape.GetDim(INPUT_DIM_0);
            OP_CHECK_IF(
                (deqScaleQDim0 != 1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    "QuantFlashAttentionScoreGrad", "deqScaleQ", Ops::Base::ToString(deqScaleQStorageShape).c_str(),
                    "When the dType of query is HIFLOAT8, the shape of deqScaleQ must be [1]"),
                return ge::GRAPH_FAILED);
        } else {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "QuantFlashAttentionScoreGrad", "deqScaleQ", Ops::Base::ToString(deqScaleQStorageShape).c_str(),
                "When the dType of query is HIFLOAT8, the shape of deqScaleQ must be [1]");
            return ge::GRAPH_FAILED;
        }

        if (deqScaleKDimNum == 1) {
            int64_t deqScaleKDim0 = deqScaleKStorageShape.GetDim(INPUT_DIM_0);
            OP_CHECK_IF(
                (deqScaleKDim0 != 1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    "QuantFlashAttentionScoreGrad", "deqScaleK", Ops::Base::ToString(deqScaleKStorageShape).c_str(),
                    "When the dType of query is HIFLOAT8, the shape of deqScaleK must be [1]"),
                return ge::GRAPH_FAILED);
        } else {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "QuantFlashAttentionScoreGrad", "deqScaleK", Ops::Base::ToString(deqScaleKStorageShape).c_str(),
                "When the dType of query is HIFLOAT8, the shape of deqScaleK must be [1]");
            return ge::GRAPH_FAILED;
        }
        if (deqScaleVDimNum == 1) {
            int64_t deqScaleVDim0 = deqScaleVStorageShape.GetDim(INPUT_DIM_0);
            OP_CHECK_IF(
                (deqScaleVDim0 != 1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    "QuantFlashAttentionScoreGrad", "deqScaleV", Ops::Base::ToString(deqScaleVStorageShape).c_str(),
                    "When the dType of query is HIFLOAT8, the shape of deqScaleV must be [1]"),
                return ge::GRAPH_FAILED);
        } else {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "QuantFlashAttentionScoreGrad", "deqScaleV", Ops::Base::ToString(deqScaleVStorageShape).c_str(),
                "When the dType of query is HIFLOAT8, the shape of deqScaleV must be [1]");
            return ge::GRAPH_FAILED;
        }
        if (deqScaleDyDimNum == 1) {
            int64_t deqScaleDyDim0 = deqScaleDyStorageShape.GetDim(INPUT_DIM_0);
            OP_CHECK_IF(
                (deqScaleDyDim0 != 1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    "QuantFlashAttentionScoreGrad", "deqScaleDy", Ops::Base::ToString(deqScaleDyStorageShape).c_str(),
                    "When the dType of query is HIFLOAT8, the shape of deqScaleDy must be [1]"),
                return ge::GRAPH_FAILED);
        } else {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "QuantFlashAttentionScoreGrad", "deqScaleDy", Ops::Base::ToString(deqScaleDyStorageShape).c_str(),
                "When the dType of query is HIFLOAT8, the shape of deqScaleDy must be [1]");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(context_, "q_descale、k_descal、v_descale、do_descale can not be nullptr");
        return ge::GRAPH_FAILED;
    }

    // new intercept
    if (fBaseParams.queryType == ge::DT_HIFLOAT8) {
        std::string shapeMsg = std::to_string(fBaseParams.b) + std::to_string(fBaseParams.n1) +
                               std::to_string(fBaseParams.s1) + std::to_string(fBaseParams.d);
        OP_CHECK_IF(fBaseParams.d != ALIGN128,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        "QuantFlashAttentionScoreGrad", "query", shapeMsg.c_str(),
                        "When the dType of query is HIFLOAT8, d of query must be equal to 128"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(fBaseParams.n1 != fBaseParams.n2,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        "QuantFlashAttentionScoreGrad", "query", shapeMsg.c_str(),
                        "When the dType of query is HIFLOAT8, n of query must be equal to n of keyIn"),
                    return ge::GRAPH_FAILED);
        auto deqScaleDsShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_DS_IDX));
        auto deqScalePShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_P_IDX));
        bool tmpDsNull = deqScaleDsShape == nullptr;
        bool tmpPNull = deqScalePShape == nullptr;
        OP_LOGD(context_, "tmpDsNull = %d, tmpPNull = %d.", tmpDsNull, tmpPNull);
        OP_CHECK_IF((deqScaleDsShape == nullptr || deqScalePShape == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT("QuantFlashAttentionScoreGrad", "dsScale, pScale"),
                    return ge::GRAPH_FAILED);
        auto deqScaleDsStorageShape = deqScaleDsShape->GetStorageShape();
        auto deqScalePStorageShape = deqScalePShape->GetStorageShape();
        int64_t deqScaleDsDimNum = deqScaleDsStorageShape.GetDimNum();
        int64_t deqScalePDimNum = deqScalePStorageShape.GetDimNum();
        if (deqScaleDsDimNum == 1) {
            int64_t deqScaleDsDim0 = deqScaleDsStorageShape.GetDim(INPUT_DIM_0);
            OP_CHECK_IF(
                (deqScaleDsDim0 != 1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    "QuantFlashAttentionScoreGrad", "dsScale", Ops::Base::ToString(deqScaleDsStorageShape).c_str(),
                    "When the dType of query is HIFLOAT8, the shape of dsScale must be [1]"),
                return ge::GRAPH_FAILED);
        } else {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "QuantFlashAttentionScoreGrad", "dsScale", Ops::Base::ToString(deqScaleDsStorageShape).c_str(),
                "When the dType of query is HIFLOAT8, the shape of dsScale must be [1]");
            return ge::GRAPH_FAILED;
        }
        if (deqScalePDimNum == 1) {
            int64_t deqScalePDim0 = deqScalePStorageShape.GetDim(INPUT_DIM_0);
            OP_CHECK_IF(
                (deqScalePDim0 != 1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    "QuantFlashAttentionScoreGrad", "pScale", Ops::Base::ToString(deqScalePStorageShape).c_str(),
                    "When the dType of query is HIFLOAT8, the shape of pScale must be [1]"),
                return ge::GRAPH_FAILED);
        } else {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "QuantFlashAttentionScoreGrad", "pScale", Ops::Base::ToString(deqScalePStorageShape).c_str(),
                "When the dType of query is HIFLOAT8, the shape of pScale must be [1]");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void JudgeIsNeedDeter(FuzzyBaseInfoParamsRegbase &fBaseParams, std::array<int64_t, CORE_LIST_NUM> &dqOffset,
                      std::array<int64_t, CORE_LIST_NUM> &dkDvOffset, std::array<int64_t, CORE_LIST_NUM> &dqOffsetpre,
                      std::array<int64_t, CORE_LIST_NUM> &dkDvOffsetpre, int64_t calcNum, bool &noNeedDeter,
                      bool &dqNeedDeterpre, bool &dkDvNeedDeterpre)
{
    bool dqNeedDeter = false;
    bool dkDvNeedDeter = false;
    for (uint16_t i = 0; i < fBaseParams.blockOuter - 1; i++) {
        for (uint16_t j = i + 1; j < fBaseParams.blockOuter; j++) {
            if (!dqNeedDeter && dqOffset[i] == dqOffset[j] && dqOffset[i] != OUTINDEX) {
                dqNeedDeter = true;
            }
            if (!dkDvNeedDeter && dkDvOffset[i] == dkDvOffset[j] && dkDvOffset[i] != OUTINDEX) {
                dkDvNeedDeter = true;
            }
        }
    }
    if (calcNum != 0 && ((!dqNeedDeter && dqNeedDeterpre) || (!dkDvNeedDeter && dkDvNeedDeterpre))) {
        for (uint16_t i = 0; i < fBaseParams.blockOuter; i++) {
            for (uint16_t j = 0; j < fBaseParams.blockOuter; j++) {
                if (!dqNeedDeter && dqNeedDeterpre && dqOffset[i] == dqOffsetpre[j] && dqOffset[i] != OUTINDEX) {
                    dqNeedDeter = true;
                }
                if (!dkDvNeedDeter && dkDvNeedDeterpre && dkDvOffset[i] == dkDvOffsetpre[j] &&
                    dkDvOffset[i] != OUTINDEX) {
                    dkDvNeedDeter = true;
                }
            }
        }
    }

    dqNeedDeterpre = dqNeedDeter;
    dkDvNeedDeterpre = dkDvNeedDeter;

    for (uint16_t i = 0; i < fBaseParams.blockOuter; i++) {
        dqOffsetpre[i] = dqOffset[i];
        dkDvOffsetpre[i] = dkDvOffset[i];
    }
    noNeedDeter = noNeedDeter && !dqNeedDeter && !dkDvNeedDeter;
    // caculate index and position
    int64_t index = calcNum / 64;
    int64_t bitPosition = calcNum % 64;
    if (index >= 0 && index < INT64_NUM) {
        if (dqNeedDeter) {
            fBaseParams.dqIsNeedDeter[index] |= (1ULL << bitPosition);
        } else {
            fBaseParams.dqIsNeedDeter[index] &= ~(1ULL << bitPosition);
        }
        if (dkDvNeedDeter) {
            fBaseParams.dkDvIsNeedDeter[index] |= (1ULL << bitPosition);
        } else {
            fBaseParams.dkDvIsNeedDeter[index] &= ~(1ULL << bitPosition);
        }
    } else {
        OP_LOGI("JudgeIsNeedDeter", "calcNum = %ld out of bounds", calcNum);
    }
}

void GetOffset(FuzzyBaseInfoParamsRegbase &fBaseParams, int64_t &currentDqOffset, int64_t &currentDkDvOffset,
               int64_t blockIdx)
{
    int64_t boIdx = 0;
    int64_t bDimTail = 0;
    int64_t n2oIdx = 0;
    int64_t n2DimTail = 0;
    int64_t goIdx = 0;
    int64_t gDimTail = 0;
    int64_t s2oIdx = 0;
    int64_t s1oIdx = 0;

    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t s2Offset = 0;

    boIdx = blockIdx / (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
    bDimTail = blockIdx % (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
    n2oIdx = bDimTail / (fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
    n2DimTail = bDimTail % (fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
    goIdx = n2DimTail / (fBaseParams.s1Outer * fBaseParams.s2Outer);
    gDimTail = n2DimTail % (fBaseParams.s1Outer * fBaseParams.s2Outer);
    s2oIdx = gDimTail / fBaseParams.s1Outer;
    s1oIdx = gDimTail % fBaseParams.s1Outer;
    // caculate dq offset
    if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {
        bOffset = boIdx * (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d);
        n2Offset = n2oIdx * (fBaseParams.g * fBaseParams.s1 * fBaseParams.d);
        gOffset = goIdx * (fBaseParams.s1 * fBaseParams.d);
        s1Offset = s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * fBaseParams.d;
    } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD) {
        bOffset = boIdx * (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d);
        s1Offset = s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
        n2Offset = n2oIdx * (fBaseParams.g * fBaseParams.d);
        gOffset = goIdx * fBaseParams.d;
    }
    currentDqOffset = bOffset + n2Offset + gOffset + s1Offset;
    // caculate dk dv offset
    if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {
        bOffset = boIdx * (fBaseParams.n2 * fBaseParams.s2 * fBaseParams.d);
        n2Offset = n2oIdx * (fBaseParams.s2 * fBaseParams.d);
        s2Offset = s2oIdx * fBaseParams.s2Inner * fBaseParams.d;
    } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD) {
        bOffset = boIdx * (fBaseParams.n2 * fBaseParams.s2 * fBaseParams.d);
        s2Offset = s2oIdx * fBaseParams.s2Inner * (fBaseParams.n2 * fBaseParams.d);
        n2Offset = n2oIdx * fBaseParams.d;
    }
    currentDkDvOffset = bOffset + n2Offset + s2Offset;
}

void PrintShapeInfo(gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    OP_LOGI(context_,
            "FAG s1s2_bn2gs1s2 with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] d[%ld] preToken[%ld] nextToken[%ld]!",
            fBaseParams.b, fBaseParams.n2, fBaseParams.g, fBaseParams.s1, fBaseParams.s2, fBaseParams.d,
            fBaseParams.s1Token, fBaseParams.s2Token);
}

bool CheckSparseModeValue(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.sparseMode > static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("QuantFlashAttentionScoreGrad", "sparseMode",
                                              std::to_string(fBaseParams.sparseMode).c_str(),
                                              "the value of sparseMode cannot be greater than 6");
        return false;
    }
    return true;
}

bool CheckPrefixNExist(FuzzyBaseInfoParamsRegbase &fBaseParams, const int64_t bIdx, const int64_t prefixN,
                       std::vector<std::vector<std::pair<int64_t, int64_t>>> &s1ValidIdx)
{
    for (int64_t i = 0; i < bIdx; ++i) {
        if (fBaseParams.prefixN[i] == prefixN) {
            OP_LOGD("Sparse", "prefixN of bIdx[%ld] and bIdx[%ld] is same as %ld", i, bIdx, prefixN);
            s1ValidIdx[bIdx].assign(s1ValidIdx[i].begin(), s1ValidIdx[i].end());
            return true;
        }
    }
    return false;
}

void CalcleBandDeterParam(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    int64_t m{fBaseParams.s1Outer}, n{fBaseParams.s2Outer}, k{static_cast<int64_t>(fBaseParams.aicNum)},
        b{fBaseParams.b * fBaseParams.n2};
    int64_t actualCalcS1Token{fBaseParams.s1Token}, actualCalcS2Token{fBaseParams.s2Token};
    int64_t p = CeilDivideBy(actualCalcS1Token, fBaseParams.s1Inner * fBaseParams.s1CvRatio) + 1;
    int64_t q = CeilDivideBy(actualCalcS2Token, fBaseParams.s2Inner * fBaseParams.s2CvRatio) + 1;

    q = q > n ? n : q;
    p = p > m ? m : p;

    // 负数场景变换
    if (q < 0) {
        m = m + q;
        p = p + q;
        q = 1;
    } else if (p < 0) {
        n = n + p;
        q = p + q;
        p = 1;
    }

    int64_t b1 = b / k;
    int64_t b2 = b % k;
    int64_t L1, L2, L3, n_seg;
    if (p + q > m) {
        L1 = m - p;
        L2 = p + q - m;
        L3 = std::min(m - 1, n - q);
        n_seg = L1 + L2 + L3;
    } else {
        L1 = q - 1;
        L2 = std::min(n - q + 1, m + NUM_TWO - p - q);
        L3 = std::max(static_cast<int64_t>(0), std::min(p + n - m - 1, p + q - NUM_TWO));
        if (L3 == 0) {
            m = p + q + L2 - NUM_TWO;
        }
        n_seg = L1 + L2 + L3;
    }
    int64_t r1 = (m * n_seg - (m - p) * (m - p + 1) / NUM_TWO - (n_seg - q) * (n_seg - q + 1) / NUM_TWO) * b1;
    int64_t r2 = 0;
    if (b2 > 0) {
        if (p + q > m) {
            r2 = std::max(m * CeilDivideBy((n * b2), std::min(k, b2 * m)), n);
        } else {
            r2 = std::max(CeilDivideBy((n * b2), k) * (p + q - 1), n);
        }
    }
    fBaseParams.deterMaxRound = r1 + r2;
}

void CalcleCausalDeterParam(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    int64_t m{fBaseParams.s1Outer}, n{fBaseParams.s2Outer}, k{static_cast<int64_t>(fBaseParams.aicNum)},
        b{fBaseParams.b * fBaseParams.n2};
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) && m > n) {
        int64_t skipM = (fBaseParams.s1 - fBaseParams.s2) / (fBaseParams.s1Inner * fBaseParams.s1CvRatio);
        m -= skipM;
    } else if ((fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) ||
                fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL)) &&
               n > m) {
        n = m;
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) && m < n) {
        fBaseParams.deterSparseType = static_cast<uint32_t>(DeterSparseType::DETER_BAND);
        return;
    }

    int64_t bTail = b % k;
    int64_t rUpper = b / k * (n * m - n * (n - 1) / MULT_BASE);
    int64_t t = n / k;
    int64_t ell = n % k;
    int64_t t1 = n / (MULT_BASE * k);
    int64_t n1 = t * k;

    if (fBaseParams.g != 1) {
        rUpper += (MULT_BASE * m - n1 + 1) * t * (bTail / MULT_BASE);
    } else {
        rUpper += bTail * (n1 * m - n1 * (n1 - 1) / MULT_BASE) / k;
    }
    if (bTail % MULT_BASE == 1) {
        if ((t % MULT_BASE) == 1) {
            int64_t m1 = m - t1 * MULT_BASE * k;
            if (ell == 0) {
                int64_t rm3 = (fBaseParams.g != 1) ? (m + m1 + 1) * t1 : 0;
                rUpper += m1 + rm3;
            } else {
                int64_t rm3 = (fBaseParams.g != 1) ? (m + m1 + 1) * t1 : 0;
                rUpper += std::max(m1, MULT_BASE * m1 - MULT_BASE * k + 1) + rm3;
            }
            bTail = bTail - 1;
        } else {
            rUpper += (NUM_TWO * m - n1 + 1) * t / MULT_BASE;
        }
    }

    int64_t ell1, L;
    if (ell % MULT_BASE == 0) {
        ell1 = ell / MULT_BASE;
        L = MULT_BASE * (m - n) + ell + 1;
    } else {
        ell1 = ell / MULT_BASE + 1;
        L = MULT_BASE * (m - n) + ell;
    }
    rUpper += CeilDivideBy(ell1 * bTail, k) * L;
    rUpper *= fBaseParams.g;
    fBaseParams.deterMaxRound = rUpper;
}

void SetSparsePrefixBlockInterval(const FuzzyBaseInfoParamsRegbase &fBaseParams, int64_t bIdx, int64_t nIdx,
                                  std::vector<std::vector<std::pair<int64_t, int64_t>>> &s1ValidIdx,
                                  int64_t (&blockStarts)[CORE_LIST_NUM], int64_t (&blockEnds)[CORE_LIST_NUM],
                                  uint32_t &coreNum, int64_t &tmepBlock)
{
    for (int64_t gIdx = 0; gIdx < fBaseParams.g; ++gIdx) {
        for (int64_t s2Idx = 0; s2Idx < fBaseParams.s2Outer; ++s2Idx) {
            tmepBlock += s1ValidIdx[bIdx][s2Idx].first;
            while (tmepBlock >= fBaseParams.blockFactor && coreNum < CORE_LIST_NUM - 1) {
                blockEnds[coreNum++] =
                    (((bIdx * fBaseParams.n2 + nIdx) * fBaseParams.g + gIdx) * fBaseParams.s2Outer + s2Idx) *
                        fBaseParams.s1Outer +
                    fBaseParams.s1Outer - (tmepBlock - fBaseParams.blockFactor);
                blockStarts[coreNum] = blockEnds[coreNum - 1];
                tmepBlock = tmepBlock - fBaseParams.blockFactor;
            }
        }
    }
    return;
}

std::pair<uint32_t, uint32_t> GetS1S2TemplateType(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.queryType == ge::DT_HIFLOAT8) {
        fBaseParams.s1TemplateType = ConstAxisTemplateNum::NUM512;
        fBaseParams.s2TemplateType = ConstAxisTemplateNum::NUM512;
        return std::make_pair(static_cast<uint32_t>(ConstAxisTemplateNum::NUM512),
                              static_cast<uint32_t>(ConstAxisTemplateNum::NUM512));
    } else if ((AlignTo(fBaseParams.s1, static_cast<int64_t>(ConstAxisTemplateNum::NUM16)) >
                    static_cast<int64_t>(ConstAxisTemplateNum::NUM16) ||
                AlignTo(fBaseParams.s2, static_cast<int64_t>(ConstAxisTemplateNum::NUM16)) >
                    static_cast<int64_t>(ConstAxisTemplateNum::NUM16)) &&
               AlignTo(fBaseParams.s1, static_cast<int64_t>(ConstAxisTemplateNum::NUM16)) *
                       AlignTo(fBaseParams.s2, static_cast<int64_t>(ConstAxisTemplateNum::NUM16)) >=
                   static_cast<int64_t>(ConstAxisTemplateNum::NUM128) *
                       static_cast<int64_t>(ConstAxisTemplateNum::NUM128)) {
        fBaseParams.s1TemplateType = ConstAxisTemplateNum::NUM128;
        fBaseParams.s2TemplateType = ConstAxisTemplateNum::NUM128;
        return std::make_pair(static_cast<uint32_t>(ConstAxisTemplateNum::NUM128),
                              static_cast<uint32_t>(ConstAxisTemplateNum::NUM128));
    }
    return std::make_pair(static_cast<uint32_t>(ConstAxisTemplateNum::NUM128),
                          static_cast<uint32_t>(ConstAxisTemplateNum::NUM128));
}

uint32_t GetDTemplateType(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM64)) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM64;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM64);
    } else if (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM128)) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM128;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM128);
    } else if (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM192)) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM192;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM192);
    } else if (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM256)) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM256;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM256);
    } else if (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM768)) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM768;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM768);
    }
    return static_cast<uint32_t>(ConstAxisTemplateNum::NUM768);
}

void GetCommS1S2OuterInfo(FuzzyBaseInfoParamsRegbase &fBaseParams, const int64_t prefixN,
                          std::vector<std::pair<int64_t, int64_t>> &s1ValidIdx)
{
    for (int64_t i = 0; i < fBaseParams.s2Outer; i++) {
        int64_t s1Start = 0;
        int64_t cvS2Idx = i * fBaseParams.cvS2Inner;
        if (cvS2Idx >= prefixN) {
            int64_t deltaS1S2 = static_cast<int64_t>(fBaseParams.s1) - static_cast<int64_t>(fBaseParams.s2);
            s1Start = std::min(static_cast<int64_t>(cvS2Idx) + deltaS1S2, static_cast<int64_t>(fBaseParams.s1));
        }

        s1ValidIdx[i].first = (static_cast<int64_t>(AlignTo(fBaseParams.s1, fBaseParams.s1CvInner)) - s1Start +
                               static_cast<int64_t>(fBaseParams.s1CvInner) - 1) /
                              static_cast<int64_t>(fBaseParams.s1CvInner);
        if (i == 0) {
            s1ValidIdx[i].second = s1ValidIdx[i].first;
        } else {
            s1ValidIdx[i].second = s1ValidIdx[i - 1].second + s1ValidIdx[i].first;
        }
    }
}

void GetCommonS1S2OuterIndex(const FuzzyBaseInfoParamsRegbase &fBaseParams, int64_t (*parseInfo)[ARRAY_LENGTH],
                             int64_t gTail, int64_t &s1oIdx, int64_t &s2oIdx)
{
    int64_t preSize = 0;
    int64_t nextSize = 0;
    for (int64_t i = 0; i < fBaseParams.s2Outer; i++) {
        if (gTail >= preSize) {
            nextSize = parseInfo[i][LENGTH_IDX];
            if (gTail < nextSize) {
                s2oIdx = i;
                s1oIdx = parseInfo[i][BEGIN_IDX] + gTail - preSize - 1;
                OP_LOGD("Sparse", " s1oIdx = %ld, s2oIdx = %ld, preSize = %ld, nextSize = %ld", s1oIdx, s2oIdx, preSize,
                        nextSize);
                break;
            }
            preSize = parseInfo[i][LENGTH_IDX];
        }
    }
}

void CalcleActualToken(FuzzyBaseInfoParamsRegbase &fBaseParams, int64_t batchIdx, int64_t &actualCalcS1Token,
                       int64_t &actualCalcS2Token)
{
    int64_t actualS1Len = fBaseParams.actualSeqQlen[batchIdx];
    int64_t actualS2Len = fBaseParams.actualSeqKvlen[batchIdx];
    // 对unpad场景的token值做二次校正
    // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
    actualCalcS1Token = fBaseParams.s1Token;
    actualCalcS2Token = fBaseParams.s2Token;
    if ((fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) &&
         batchIdx != fBaseParams.bandIdx) ||
        (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL) &&
         batchIdx != fBaseParams.bandIdx)) {
        actualCalcS1Token = INT32_MAX;
        actualCalcS2Token = 0;
    }
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) ||
        (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL) &&
         batchIdx == fBaseParams.bandIdx)) {
        actualCalcS1Token = actualCalcS1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = actualCalcS2Token - actualS1Len + actualS2Len;
    }
}

ge::graphStatus ProcessOptionalInput(gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    fBaseParams.qSize =
        static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d;
    fBaseParams.kSize = static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * 1 * fBaseParams.s2 * fBaseParams.d;
    fBaseParams.vSize = static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * 1 * fBaseParams.s2 * fBaseParams.d1;
    fBaseParams.dropMaskSize =
        static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g * fBaseParams.s2 * fBaseParams.s1;

    // mBaseParams is used for matmal tiling module
    auto queryType = context_->GetInputDesc(0)->GetDataType();
    fBaseParams.queryType = queryType;
    fBaseParams.calTypeSize = FP32_BYTES;

    fBaseParams.scaleValue =
        *(context_->GetAttrs()->GetAttrPointer<float>(static_cast<size_t>(AttrIndex::SCALE_VALUE)));
    auto metadataShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::METADATA));
    if (metadataShape != nullptr && metadataShape->GetStorageShape().GetDimNum() == ATTEN_MASK_DIM_LENGTH_2) {
        fBaseParams.metadataLen = metadataShape->GetStorageShape().GetDim(1);
    }
    fBaseParams.keepProb = 1;
    fBaseParams.dropoutIsDivisibleBy8 = 1.0;
    auto hasSequsedQ = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SEQUSED_Q));
    auto hasSequsedKV = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SEQUSED_KV));
    if (hasSequsedQ != nullptr) {
        fBaseParams.hasSequsedQ = true;
    }
    if (hasSequsedKV != nullptr) {
        fBaseParams.hasSequsedKV = true;
    }
    auto sink = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SINK_IDX));
    if (sink != nullptr) {
        fBaseParams.hasSink = true;
    }
    PrintShapeInfo(context_, fBaseParams);
    auto ret = ProcessQuantInfo(context_, fBaseParams);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "ProcessOptionalInput ProcessQuantInfo error");
        return ret;
    }

    // token_info
    fBaseParams.s1Token = *(context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::WIN_LEFT)));
    fBaseParams.s2Token = *(context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::WIN_RIGHT)));

    ret = ProcessSparseModeInfo(context_, fBaseParams);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "ProcessOptionalInput ProcessSparseModeInfo error");
        return ret;
    }
    ret = ProcessTokensInfo(fBaseParams);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "ProcessOptionalInput ProcessTokensInfo error");
        return ret;
    }

    fBaseParams.isSparse = SetSparseParams(context_, fBaseParams);
    OP_LOGD("Sparse FLAG", "FAG Us1s2Bbn2gs1s2 sparse mode = %u, sparse %s.", fBaseParams.sparseMode,
            fBaseParams.isSparse ? "enable" : "disable");
    if (fBaseParams.isSparse == false && fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        OP_LOGE(context_, "Sparse capability must be supported under prefix compress mode, pls check input params");
        return ge::GRAPH_FAILED;
    }
    if (fBaseParams.isSparse == false && fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX)) {
        // 与71处理逻辑保持一致
        OP_LOGD("Sparse FLAG", "Set sparse_mode from PREFIX to ALL_MASK because of empty or nullptr prefixN.");
        fBaseParams.sparseMode = static_cast<uint32_t>(SparseMode::ALL_MASK);
    }

    if (CheckAttenMaskShape(fBaseParams) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "ProcessOptionalInput CheckAttenMaskShape error");
        return ge::GRAPH_FAILED;
    }

    return CheckShapeValid(context_, fBaseParams.b, fBaseParams.n1, fBaseParams.s1, fBaseParams.d);
}

ge::graphStatus QuantScaleDtypeValidCheck(gert::TilingContext *context_, const FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    auto yInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::ATTN_OUT));
    auto deqScaleQInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_Q));
    auto deqScaleKInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_K));
    auto deqScaleVInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_V));
    auto deqScaleDyInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_DOUT));
    auto deqScaleDsInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_DS_IDX));
    auto deqScalePInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_P_IDX));
    if (yInput != nullptr) {
        auto yInputDtype = yInput->GetDataType();
        bool isYInputNotValid = (fBaseParams.queryType == ge::DT_HIFLOAT8 && yInputDtype != ge::DT_BF16);
        OP_CHECK_IF(isYInputNotValid,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        "QuantFlashAttentionScoreGrad", "attentionInOptional",
                        ge::TypeUtils::DataTypeToSerialString(yInputDtype).c_str(),
                        "When the dType of query is HIFLOAT8, the dtype of attentionInOptional must be BFLOAT16"),
                    return ge::GRAPH_FAILED);
    }
    if (deqScaleQInput != nullptr && deqScaleKInput != nullptr && deqScaleVInput != nullptr &&
        deqScaleDyInput != nullptr && deqScaleDsInput != nullptr && deqScalePInput != nullptr) {
        auto deqScaleQDtype = deqScaleQInput->GetDataType();
        auto deqScaleKDtype = deqScaleKInput->GetDataType();
        auto deqScaleVDtype = deqScaleVInput->GetDataType();
        auto deqScaleDyDtype = deqScaleDyInput->GetDataType();
        auto deqScaleDsDtype = deqScaleDsInput->GetDataType();
        auto deqScalePDtype = deqScalePInput->GetDataType();
        std::string dtypesMsg = ge::TypeUtils::DataTypeToSerialString(deqScaleQDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(deqScaleKDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(deqScaleVDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(deqScaleDyDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(deqScaleDsDtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(deqScalePDtype);
        OP_CHECK_IF(
            deqScaleQDtype != ge::DT_FLOAT || deqScaleKDtype != ge::DT_FLOAT || deqScaleVDtype != ge::DT_FLOAT ||
                deqScaleDyDtype != ge::DT_FLOAT || deqScaleDsDtype != ge::DT_FLOAT || deqScalePDtype != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                "QuantFlashAttentionScoreGrad", "dScaleQ, dScaleK, dScaleV, dScaleDy, dsScale, pScale",
                dtypesMsg.c_str(), "The dtypes of dScaleQ, dScaleK, dScaleV, dScaleDy, dsScale pScale must be FLOAT32"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantShapeValidCheck(gert::TilingContext *context_, const FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.queryType == ge::DT_HIFLOAT8) {
        auto queryShape = context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
        auto keyShape = context_->GetInputShape(static_cast<size_t>(InputIndex::KEY));
        auto valueShape = context_->GetInputShape(static_cast<size_t>(InputIndex::VALUE));
        auto dyShape = context_->GetInputShape(static_cast<size_t>(InputIndex::DOUT));
        auto attentionInShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::ATTN_OUT));
        if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr || dyShape == nullptr ||
            attentionInShape == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT("QuantFlashAttentionScoreGrad", "query, keyIn, value, dy, attentionIn");
            return ge::GRAPH_FAILED;
        }
        // 校验query, key, value, dy, attn_out的维度必须为4维
        auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
        auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
        auto dyShapeDim = dyShape->GetStorageShape().GetDimNum();
        auto keyShapeDim = keyShape->GetStorageShape().GetDimNum();
        auto valueShapeDim = valueShape->GetStorageShape().GetDimNum();
        constexpr int64_t EXPECTED_DIM_NUM = 4;
        if (attentionInShapeDim != EXPECTED_DIM_NUM || queryShapeDim != EXPECTED_DIM_NUM ||
            dyShapeDim != EXPECTED_DIM_NUM || keyShapeDim != EXPECTED_DIM_NUM || valueShapeDim != EXPECTED_DIM_NUM) {
            std::string dimMsg = "{" + std::to_string(attentionInShapeDim) + ", " + std::to_string(queryShapeDim) +
                                 ", " + std::to_string(dyShapeDim) + ", " + std::to_string(keyShapeDim) + ", " +
                                 std::to_string(valueShapeDim) + "}";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("QuantFlashAttentionScoreGrad",
                                                     "query, keyIn, value, dy, attentionIn", dimMsg.c_str(),
                                                     "The shape dim of query, keyIn, value, dy, attentionIn "
                                                     "must be 4");
            return ge::GRAPH_FAILED;
        }
        for (uint32_t dimIdx = 0; dimIdx < queryShapeDim; dimIdx++) {
            if ((queryShape->GetStorageShape().GetDim(dimIdx) != dyShape->GetStorageShape().GetDim(dimIdx)) ||
                (queryShape->GetStorageShape().GetDim(dimIdx) != attentionInShape->GetStorageShape().GetDim(dimIdx))) {
                std::string shapesMsg = "{" + Ops::Base::ToString(queryShape->GetStorageShape()) + ", " +
                                        Ops::Base::ToString(dyShape->GetStorageShape()) + ", " +
                                        Ops::Base::ToString(attentionInShape->GetStorageShape()) + "}";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("QuantFlashAttentionScoreGrad", "query, dy, attentionIn",
                                                       shapesMsg.c_str(),
                                                       "When the dtype of query is HIFLOAT8, "
                                                       "all axes of query, dy, attentionIn must be same");
                return ge::GRAPH_FAILED;
            }
            if ((keyShape->GetStorageShape().GetDim(dimIdx) != valueShape->GetStorageShape().GetDim(dimIdx))) {
                std::string shapesMsg = "{" + Ops::Base::ToString(keyShape->GetStorageShape()) + ", " +
                                        Ops::Base::ToString(valueShape->GetStorageShape()) + "}";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    "QuantFlashAttentionScoreGrad", "keyIn, value", shapesMsg.c_str(),
                    "When the dtype of query is HIFLOAT8, all axes of keyIn and value must be same");
                return ge::GRAPH_FAILED;
            }
        }
        if (queryShape->GetStorageShape().GetDim(0) != keyShape->GetStorageShape().GetDim(0) ||
            queryShape->GetStorageShape().GetDim(3) != keyShape->GetStorageShape().GetDim(3)) {
            std::string shapesMsg = "{" + Ops::Base::ToString(keyShape->GetStorageShape()) + ", " +
                                    Ops::Base::ToString(queryShape->GetStorageShape()) + "}";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "QuantFlashAttentionScoreGrad", "keyIn, query", shapesMsg.c_str(),
                "When the dtype of query is HIFLOAT8, b of keyIn and query must be same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ProcessQuantInfo(gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    DetermineMode(fBaseParams);
    if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
        fBaseParams.queryType == ge::DT_UINT8 || fBaseParams.queryType == ge::DT_INT8 ||
        fBaseParams.queryType == ge::DT_QINT8) {
        auto queryDType = context_->GetInputDesc(0)->GetDataType();
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("QuantFlashAttentionScoreGrad", "query",
                                               ge::TypeUtils::DataTypeToSerialString(queryDType).c_str(),
                                               "The dtype of query must be HIFLOAT8");
        return ge::GRAPH_FAILED;
    }
    // hifp8 shape whitelist
    auto quantShapeRet = QuantShapeValidCheck(context_, fBaseParams);
    if (quantShapeRet != ge::GRAPH_SUCCESS) {
        return quantShapeRet;
    }
    fBaseParams.outDtype = fBaseParams.inputDtype;
    if (context_->GetAttrs()->GetAttrNum() > OUTDTYPE_ATTR_IDX && (fBaseParams.queryType == ge::DT_HIFLOAT8)) {
        int64_t outDType = *(context_->GetAttrs()->GetAttrPointer<int>(OUTDTYPE_ATTR_IDX));
        if (outDType == 1) {
            fBaseParams.outDtype = DtypeEnum::BFLOAT16;
        } else {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                "QuantFlashAttentionScoreGrad", "outDType", "Non-BFLOAT16",
                "When the dtype of query is HIFLOAT8, the dtype of outDType must be BFLOAT16");
            return ge::GRAPH_FAILED;
        }
    } else {
        // 非FP8场景无需check
        return ge::GRAPH_SUCCESS;
    }
    auto quantScaleShapeCheckRet = QuantScaleShapeValidCheck(context_, fBaseParams);
    if (quantScaleShapeCheckRet != ge::GRAPH_SUCCESS) {
        return quantScaleShapeCheckRet;
    }
    auto quantScaleDtypeCheckRet = QuantScaleDtypeValidCheck(context_, fBaseParams);
    if (quantScaleDtypeCheckRet != ge::GRAPH_SUCCESS) {
        return quantScaleDtypeCheckRet;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ProcessSparseModeInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    // 新增SPARSE_MODE属性，上库兼容处理
    auto attrs = context_->GetAttrs();
    fBaseParams.sparseMode = static_cast<uint32_t>(SparseMode::NO_MASK);
    if (attrs->GetAttrNum() > static_cast<size_t>(AttrIndex::SPARSE_MODE)) {
        fBaseParams.sparseMode = *(attrs->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SPARSE_MODE))); // 7
    }
    auto attnMaskShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::ATTN_MASK));
    if (attnMaskShape != nullptr) {
        fBaseParams.hasAttnMask = true;
        // 校验attnMask的shape必须为[2048, 2048]
        auto attnMaskShapeDim = attnMaskShape->GetStorageShape().GetDimNum();
        if (attnMaskShapeDim != ATTEN_MASK_DIM_LENGTH_2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("QuantFlashAttentionScoreGrad", "attnMask",
                                                     std::to_string(attnMaskShapeDim).c_str(),
                                                     "The shape dim of attnMask must be 2");
            return ge::GRAPH_FAILED;
        }
        auto attnMaskDim0 = attnMaskShape->GetStorageShape().GetDim(0);
        auto attnMaskDim1 = attnMaskShape->GetStorageShape().GetDim(1);
        std::string attnMaskShapeMsg = std::to_string(attnMaskDim0) + ", " + std::to_string(attnMaskDim1);
        std::string attnMaskReasonMsg = "The shape of attnMask must be [" + std::to_string(ATTEN_MASK_COMPRESS_LIMIT) +
                                        ", " + std::to_string(ATTEN_MASK_COMPRESS_LIMIT) + "]";
        OP_CHECK_IF((attnMaskDim0 != static_cast<int64_t>(ATTEN_MASK_COMPRESS_LIMIT) ||
                     attnMaskDim1 != static_cast<int64_t>(ATTEN_MASK_COMPRESS_LIMIT)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("QuantFlashAttentionScoreGrad", "attnMask",
                                                           attnMaskShapeMsg.c_str(), attnMaskReasonMsg.c_str()),
                    return ge::GRAPH_FAILED);
    }
    // 校验sparseMode与attnMask的匹配关系
    // sparseMode为0(NO_MASK)时，attnMask必须为空
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) && attnMaskShape != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("QuantFlashAttentionScoreGrad", "sparseMode",
                                              std::to_string(fBaseParams.sparseMode).c_str(),
                                              "when sparseMode is 0(NO_MASK), attnMask must be empty");
        return ge::GRAPH_FAILED;
    }
    // sparseMode为3(RIGHT_DOWN_CAUSAL)或4(BAND)时，attnMask不能为空
    if ((fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
         fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) &&
        attnMaskShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("QuantFlashAttentionScoreGrad", "sparseMode",
                                              std::to_string(fBaseParams.sparseMode).c_str(),
                                              "when sparseMode is 3(RIGHT_DOWN_CAUSAL) or 4(BAND), "
                                              "attnMask must not be empty");
        return ge::GRAPH_FAILED;
    }
    if (!CheckSparseModeValue(fBaseParams)) {
        OP_LOGE(context_, "ProcessSparseModeInfo CheckSparseModeValue error");
        return ge::GRAPH_FAILED;
    }
    fBaseParams.attenMaskCompressMode = 0;
    fBaseParams.attenMaskOptional = EMPTY_TENSOR;
    return ge::GRAPH_SUCCESS;
}

// 以下场景对外部输入token屏蔽，重新设置token值并做校验
ge::graphStatus ProcessTokensInfo(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    OP_LOGD("ProcessTokensInfo", " Before correction ,the value of s1Token = %ld and the value of s2Token %ld.",
            fBaseParams.s1Token, fBaseParams.s2Token);

    // 自动校正left和right causal的token值，token信息仅用于sparse分核计算
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        fBaseParams.s1Token = INT32_MAX;
        fBaseParams.s2Token = 0;
    }

    // 对pad场景做校正
    // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
    if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
        (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
         fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND))) {
        fBaseParams.s1Token = fBaseParams.s1Token + fBaseParams.s1 - fBaseParams.s2;
        fBaseParams.s2Token = fBaseParams.s2Token - fBaseParams.s1 + fBaseParams.s2;
    }

    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK) ||
        fBaseParams.attenMaskOptional == EMPTY_TENSOR) {
        fBaseParams.s1Token = INT32_MAX;
        fBaseParams.s2Token = INT32_MAX;
    }

    OP_LOGD("ProcessTokensInfo", " the corrected s1Token = %ld, s2Token %ld.", fBaseParams.s1Token,
            fBaseParams.s2Token);

    // 1  2  3  5  6  不校验
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        return ge::GRAPH_SUCCESS;
    }

    // 校验pad场景token是否合法
    if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
        (-fBaseParams.s1Token > int64_t(fBaseParams.s2) || -fBaseParams.s2Token > int64_t(fBaseParams.s1) ||
         (fBaseParams.s1Token + fBaseParams.s2Token) < 0)) {
        std::string valueMsg = "{" + std::to_string(fBaseParams.s1Token) + ", " + std::to_string(fBaseParams.s2Token) +
                               ", " + std::to_string(fBaseParams.s1Token + fBaseParams.s2Token) + "}";
        std::string reasonMsg = "When inputLayout is TND, the valud of nextTokens, preTokens, nextToKens + preTokens "
                                "cannot be less than {" +
                                std::to_string(int64_t(-fBaseParams.s2)) + ", " +
                                std::to_string(int64_t(-fBaseParams.s1)) + ", 0}";
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("QuantFlashAttentionScoreGrad", "nextTokens, preTokens, nextToKens",
                                               valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // 校验unpad场景token是否合法   0  4  7  8
    return ge::GRAPH_SUCCESS;
}

bool SetSparseParams(gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK) ||
        fBaseParams.attenMaskOptional == EMPTY_TENSOR) {
        OP_LOGD("SetSparseParams ", " in the ALL_MASK or attenMask is none scenario,isSparse is false");
        return false;
    }

    // 兼容老版本，未配置sparseMode或配置sparseMode为0的处理
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK)) {
        if (int64_t(fBaseParams.s1) > fBaseParams.s1Token ||
            int64_t(fBaseParams.s2) > fBaseParams.s2Token) { // band场景，包含causal
            OP_LOGD("SetSparseParams ", " in the NONE_MASK  and token is band scenario,isSparse is true ");
            return true;
        } else {
            OP_LOGD("SetSparseParams ", " in the NONE_MASK  and token is not band scenario,isSparse is false");
            return false;
        }
    }

    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        OP_LOGD("SetSparseParams ", " in the LEFT_UP_CAUSAL  or RIGHT_DOWN_CAUSAL scenario,isSparse is true");
        return true;
    }

    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) &&
        (int64_t(fBaseParams.s1) > fBaseParams.s1Token || int64_t(fBaseParams.s2) > fBaseParams.s2Token)) {
        OP_LOGD("SetSparseParams ", " in the BAND  and token is band scenario,isSparse is true ");
        return true;
    }

    OP_LOGD("SetSparseParams ", " no scenario is hit, isSparse is false ");
    return false;
}

void DetermineMode(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.queryType == ge::DT_HIFLOAT8) {
        fBaseParams.inputDtype = static_cast<optiling::DtypeEnum>(DTYPE_ENUM_INDEX_6); // DtypeEnum::HIFLOAT8
    } else {
        fBaseParams.inputDtype = DtypeEnum::FLOAT16_PRECISION;
    }
}
} // namespace QuantFag
} // namespace optiling
