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
 * \file flash_attention_score_grad_tiling_common_regbase.cpp
 * \brief
 */

#include "flash_attention_score_grad_tiling_common_regbase.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {
namespace fag {

ge::graphStatus CheckSoftmaxMaxShape(const gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1,
                                     bool isQuant)
{
    auto softmaxMaxShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    if (softmaxMaxShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxMaxShapeDim = softmaxMaxShape->GetStorageShape().GetDimNum();
    if (softmaxMaxShapeDim != 4) { // softmaxMax only support 4 dimensions
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("FlashAttentionScoreGrad", "softmaxMaxOptional",
                                                 std::to_string(softmaxMaxShapeDim).c_str(),
                                                 "The shape dim of softmaxMaxOptional must be 4");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxMaxShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = softmaxMaxShape->GetStorageShape().GetDim(1); // 1:n1
    auto dim2 = softmaxMaxShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = softmaxMaxShape->GetStorageShape().GetDim(3); // 3:8
    int64_t validDim3 = isQuant ? 1 : 8;
    std::string reasonMsg = "The shape of softmaxMaxOptional must be [" + std::to_string(b) + ", " +
                            std::to_string(n1) + ", " + std::to_string(s1) + ", " + std::to_string(validDim3) + "]";
    OP_CHECK_IF((dim0 != b || dim1 != n1 || dim2 != s1 || dim3 != validDim3),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "softmaxMaxOptional",
                                                       Ops::Base::ToString(softmaxMaxShape->GetStorageShape()).c_str(),
                                                       reasonMsg.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxSumShape(const gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1,
                                     bool isQuant)
{
    auto softmaxSumShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    if (softmaxSumShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxSumShapeDim = softmaxSumShape->GetStorageShape().GetDimNum();
    if (softmaxSumShapeDim != 4) { // softmaxSum only support 4 dimensions
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("FlashAttentionScoreGrad", "softmaxSumOptional",
                                                 std::to_string(softmaxSumShapeDim).c_str(),
                                                 "The shape dim of softmaxSumOptional must be 4");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxSumShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = softmaxSumShape->GetStorageShape().GetDim(1); // 1:n1
    auto dim2 = softmaxSumShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = softmaxSumShape->GetStorageShape().GetDim(3); // 3:8
    int64_t validDim3 = isQuant ? 1 : 8;
    std::string reasonMsg = "The shape of softmaxSumOptional must be [" + std::to_string(b) + ", " +
                            std::to_string(n1) + ", " + std::to_string(s1) + ", " + std::to_string(validDim3) + "]";
    OP_CHECK_IF((dim0 != b || dim1 != n1 || dim2 != s1 || dim3 != validDim3),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "softmaxSumOptional",
                                                       Ops::Base::ToString(softmaxSumShape->GetStorageShape()).c_str(),
                                                       reasonMsg.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxMaxSumTndShape(const gert::TilingContext *context, int64_t t1, int64_t n1)
{
    if (context->GetAttrs()->GetAttrNum() > static_cast<size_t>(AttrIndex::TND_SOFTMAX_IN)) {
        // read 13th attr softmax_out_layout
        const char *softmaxInLayout =
            context->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::TND_SOFTMAX_IN));
        if (strcmp(softmaxInLayout, "same_as_input") != 0) {
            // check whether softmax_out_layout is TND
            return ge::GRAPH_SUCCESS;
        }
    } else {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxSumShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    auto softmaxMaxShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    if (softmaxMaxShape != nullptr) {
        auto softmaxMaxShapeDim = softmaxMaxShape->GetStorageShape().GetDimNum();
        if (softmaxMaxShapeDim != DIM_3) { // softmax TND only support 3 dimensions
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("FlashAttentionScoreGrad", "softmaxMaxOptional",
                                                     std::to_string(softmaxMaxShapeDim).c_str(),
                                                     "The shape dim of softmaxMaxOptional must be 3");
            return ge::GRAPH_FAILED;
        }
        auto dim0 = softmaxMaxShape->GetStorageShape().GetDim(DIM_0); // 0:t1
        auto dim1 = softmaxMaxShape->GetStorageShape().GetDim(DIM_1); // 1:n1
        auto dim2 = softmaxMaxShape->GetStorageShape().GetDim(DIM_2); // 2:8
        std::string reasonMsg = "The shape of softmaxMaxOptional must be [" + std::to_string(t1) + ", " +
                                std::to_string(n1) + ", " + std::to_string(BIT_NUMS) + "]";
        OP_CHECK_IF((dim0 != t1 || dim1 != n1 || dim2 != BIT_NUMS),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "softmaxMaxOptional",
                        Ops::Base::ToString(softmaxMaxShape->GetStorageShape()).c_str(), reasonMsg.c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (softmaxSumShape != nullptr) {
        auto softmaxSumShapeDim = softmaxSumShape->GetStorageShape().GetDimNum();
        if (softmaxSumShapeDim != DIM_3) { // softmax TND only support 3 dimensions
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("FlashAttentionScoreGrad", "softmaxSumOptional",
                                                     std::to_string(softmaxSumShapeDim).c_str(),
                                                     "The shape dim of softmaxSumOptional must be 3");
            return ge::GRAPH_FAILED;
        }
        auto dim0 = softmaxSumShape->GetStorageShape().GetDim(DIM_0); // 0:t1
        auto dim1 = softmaxSumShape->GetStorageShape().GetDim(DIM_1); // 1:n1
        auto dim2 = softmaxSumShape->GetStorageShape().GetDim(DIM_2); // 2:8
        std::string reasonMsg = "The shape of softmaxSumOptional must be [" + std::to_string(t1) + ", " +
                                std::to_string(n1) + ", " + std::to_string(BIT_NUMS) + "]";
        OP_CHECK_IF((dim0 != t1 || dim1 != n1 || dim2 != BIT_NUMS),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad", "softmaxSumOptional",
                        Ops::Base::ToString(softmaxSumShape->GetStorageShape()).c_str(), reasonMsg.c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInShape(const gert::TilingContext *context)
{
    auto attentionInShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::ATTENTION_IN));
    if (attentionInShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
    auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
    if (attentionInShapeDim != queryShapeDim) {
        std::string shapeDimMsg = std::to_string(queryShapeDim) + ", " + std::to_string(attentionInShapeDim);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("FlashAttentionScoreGrad", "query, attentionInOptional",
                                                 shapeDimMsg.c_str(),
                                                 "The shape dims of query and attentionInOptional must be same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxDtype(const gert::TilingContext *context)
{
    auto softmaxMax = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    auto softmaxSum = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    OP_CHECK_IF(softmaxMax == nullptr || softmaxSum == nullptr,
                OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "softmaxMaxOptional or softmaxSumOptional"),
                return ge::GRAPH_FAILED);

    auto softmaxMaxType = softmaxMax->GetDataType();
    auto softmaxSumType = softmaxSum->GetDataType();

    bool softmaxTypeCheck = (softmaxMaxType == softmaxSumType) && (softmaxMaxType == ge::DT_FLOAT);
    std::string dtypeMsg = "{" + ge::TypeUtils::DataTypeToSerialString(softmaxMaxType) + ", " +
                           ge::TypeUtils::DataTypeToSerialString(softmaxSumType) + "}";
    OP_CHECK_IF(softmaxTypeCheck != true,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    "FlashAttentionScoreGrad", "softmaxMaxOptional, softmaxSumOptional", dtypeMsg.c_str(),
                    "The dtype of softmaxMaxOptional and softmaxSumOptional must be FLOAT32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInDtype(const gert::TilingContext *context)
{
    auto query = context->GetInputDesc(static_cast<size_t>(InputIndex::QUERY));
    auto attentionIn = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::ATTENTION_IN));
    OP_CHECK_IF(query == nullptr || attentionIn == nullptr,
                OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query or attentionInOptional"),
                return ge::GRAPH_FAILED);

    auto queryType = query->GetDataType();
    auto attentionInType = attentionIn->GetDataType();

    if (queryType == ge::DT_HIFLOAT8) {
        return ge::GRAPH_SUCCESS;
    }

    std::string dtypeMsg = "{" + ge::TypeUtils::DataTypeToSerialString(queryType) + ", " +
                           ge::TypeUtils::DataTypeToSerialString(attentionInType) + "}";
    OP_CHECK_IF(queryType != attentionInType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("FlashAttentionScoreGrad", "query, attentionInOptional",
                                                       dtypeMsg.c_str(),
                                                       "The dtype of query and attentionInOptional must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeValid(const gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, int64_t d)
{
    auto isShapeInValid = (b == 0 || n1 == 0 || s1 == 0 || d == 0);
    std::string shapeMsg =
        std::to_string(b) + ", " + std::to_string(n1) + ", " + std::to_string(s1) + ", " + std::to_string(d);
    OP_CHECK_IF(isShapeInValid,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("FlashAttentionScoreGrad", "query", shapeMsg.c_str(),
                                                      "All axes of query must be postitive numbers"),
                return ge::GRAPH_FAILED);

    auto queryType = context->GetInputDesc(0)->GetDataType();
    bool isQuant = queryType == ge::DT_HIFLOAT8;
    auto ret = CheckSoftmaxMaxShape(context, b, n1, s1, isQuant);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckSoftmaxSumShape(context, b, n1, s1, isQuant);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionInShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndShapeValid(const gert::TilingContext *context, int64_t t1, int64_t n1, int64_t d)
{
    if (context == nullptr) {
        OP_LOGE(context, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto isShapeInValid = (t1 == 0 || n1 == 0 || d == 0);
    std::string shapeMsg = std::to_string(t1) + ", " + std::to_string(n1) + ", " + std::to_string(d);
    OP_CHECK_IF(isShapeInValid,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("FlashAttentionScoreGrad", "query", shapeMsg.c_str(),
                                                      "All axes of query must be postitive numbers"),
                return ge::GRAPH_FAILED);

    auto ret = CheckAttentionInShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckSoftmaxMaxSumTndShape(context, t1, n1);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeValid(const gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE(context, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto ret = CheckSoftmaxDtype(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionInDtype(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttenMaskShape(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    // check atten_mask shape when enable atten_mask_compress
    if (fBaseParams.attenMaskCompressMode == 0) {
        constexpr int64_t BEST_BASIC_BLOCK_NUM = ALIGN64 * ALIGN128;
        constexpr uint32_t MAX_KV_SEQLEN = 1536;
        auto bestBasicBlockNum = fBaseParams.s1 >= FP32_BYTES ? (BEST_BASIC_BLOCK_NUM / FP32_BYTES * (FP32_BYTES - 1)) :
                                                                BEST_BASIC_BLOCK_NUM;
        bool ngs1s2BIsAble =
            (fBaseParams.n2 * fBaseParams.g *
             AlignTo(fBaseParams.s1, static_cast<int64_t>(ConstAxisTemplateNum::NUM16)) *
             AlignTo(fBaseParams.s2, static_cast<int64_t>(ConstAxisTemplateNum::NUM16))) <= bestBasicBlockNum;
        bool isBCaple = fBaseParams.attenMaskOptional != EMPTY_TENSOR && fBaseParams.layoutType != INPUT_FORMAT_TND &&
                        fBaseParams.queryType != ge::DT_FLOAT && fBaseParams.d == fBaseParams.d1 &&
                        fBaseParams.pseType == static_cast<uint32_t>(PseType::PSE_OUTER_ADD_MUL_TYPE) && ngs1s2BIsAble;
        int64_t dTypeSize = fBaseParams.queryType == ge::DT_BF16 ?
                                ge::GetSizeByDataType(static_cast<ge::DataType>(fBaseParams.queryType)) * NUM_TWO :
                                ge::GetSizeByDataType(static_cast<ge::DataType>(fBaseParams.queryType));
        int64_t s2Align =
            AlignTo(fBaseParams.s2 * ge::GetSizeByDataType(static_cast<ge::DataType>(fBaseParams.queryType)),
                    static_cast<int64_t>(ConstAxisTemplateNum::NUM32)) /
            ge::GetSizeByDataType(static_cast<ge::DataType>(fBaseParams.queryType));
        int64_t s2AlignVecSize = AlignTo(s2Align * dTypeSize, static_cast<int64_t>(ConstAxisTemplateNum::NUM32));
        bool ngs1s2BnIsAble = ((fBaseParams.g * fBaseParams.s1 * s2AlignVecSize) != 0) && (fBaseParams.g == 1) &&
                              (s2AlignVecSize <= MAX_KV_SEQLEN) &&
                              ((fBaseParams.g * fBaseParams.s1 * s2AlignVecSize) <= ALIGN64 * ALIGN128 * FP32_BYTES);
        bool isBNCaple = fBaseParams.attenMaskOptional != EMPTY_TENSOR && fBaseParams.layoutType != INPUT_FORMAT_TND &&
                         fBaseParams.queryType != ge::DT_FLOAT && fBaseParams.d == fBaseParams.d1 &&
                         fBaseParams.pseType == static_cast<uint32_t>(PseType::PSE_OUTER_ADD_MUL_TYPE) &&
                         ngs1s2BnIsAble;
        bool invalid =
            (static_cast<int64_t>(fBaseParams.attenMaskS1Size) * static_cast<int64_t>(fBaseParams.attenMaskS2Size) <
             static_cast<int64_t>(fBaseParams.s1) * static_cast<int64_t>(fBaseParams.s2));
        bool attenMaskInvalid =
            (static_cast<int64_t>(fBaseParams.attenMaskS1Size) != static_cast<int64_t>(fBaseParams.s1)) ||
            (static_cast<int64_t>(fBaseParams.attenMaskS2Size) != static_cast<int64_t>(fBaseParams.s2));
        if ((isBNCaple || isBCaple) && invalid) {
            std::string shapeSizeMsg =
                std::to_string(fBaseParams.attenMaskS1Size) + " * " + std::to_string(fBaseParams.attenMaskS2Size);
            std::string reasonMsg = "When attenMaskOptional is not empty and inputLayout is not TND, "
                                    "the shape size of attenMaskOptional cannot be less than " +
                                    std::to_string(fBaseParams.s1) + " * " + std::to_string(fBaseParams.s2);
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("FlashAttentionScoreGrad", "attenMaskOptional",
                                                      shapeSizeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        } else if (fBaseParams.attenMaskOptional != EMPTY_TENSOR && fBaseParams.layoutType != INPUT_FORMAT_TND &&
                   attenMaskInvalid && (!(isBNCaple || isBCaple))) {
            std::string shapeMsg =
                std::to_string(fBaseParams.attenMaskS1Size) + ", " + std::to_string(fBaseParams.attenMaskS2Size);
            std::string reasonMsg = "When attenMaskOptional is not empty tensor and inputLayout is not TND, "
                                    "the shape of attenMaskOptional must be [" +
                                    std::to_string(fBaseParams.s1) + ", " + std::to_string(fBaseParams.s2) + "]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("FlashAttentionScoreGrad", "attenMaskOptional", shapeMsg.c_str(),
                                                  reasonMsg.c_str());
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
                "FlashAttentionScoreGrad", "attenMaskOptional", shapeMsg.c_str(),
                "When sparseMode is 6, the shape of attenMaskOptional must be [3072, 2048]");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (fBaseParams.attenMaskS1Size != fBaseParams.attenMaskS2Size) {
        std::string shapeMsg =
            std::to_string(fBaseParams.attenMaskS1Size) + ", " + std::to_string(fBaseParams.attenMaskS2Size);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("FlashAttentionScoreGrad", "attenMaskOptional", shapeMsg.c_str(),
                                              "Sq of attenMaskOptional must be equal to Skv of attenMaskOptional");
        return ge::GRAPH_FAILED;
    }

    if (fBaseParams.attenMaskS2Size != ATTEN_MASK_COMPRESS_LIMIT) {
        std::string shapeMsg =
            std::to_string(fBaseParams.attenMaskS1Size) + ", " + std::to_string(fBaseParams.attenMaskS2Size);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("FlashAttentionScoreGrad", "attenMaskOptional", shapeMsg.c_str(),
                                              "Skv of attenMaskOptional must be equal to 2048");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool CheckIsLargeInvalidBlk(const FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if ((fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL)) &&
        (fBaseParams.s1Outer >= 0 && fBaseParams.s2Outer >= 0) && (fBaseParams.s1Outer < fBaseParams.s2Outer) &&
        (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM256))) {
        return (fBaseParams.s2Outer - fBaseParams.s1Outer) * fBaseParams.s1Outer >= LARGE_INVALID_NUM;
    }
    return false;
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

    // Adjacent rounds are pipelined. Even if neither round has an intra-round conflict, two different cores can
    // atomically accumulate to the same output across the round boundary. Mark both rounds as deterministic so
    // that the earlier normal atomic path cannot overlap the later round.
    bool dqCrossRoundConflict = false;
    bool dkDvCrossRoundConflict = false;
    if (calcNum != 0) {
        for (uint16_t i = 0; i < fBaseParams.blockOuter && (!dqCrossRoundConflict || !dkDvCrossRoundConflict); i++) {
            for (uint16_t j = 0; j < fBaseParams.blockOuter; j++) {
                if (i == j) {
                    continue;
                }
                if (!dqCrossRoundConflict && dqOffset[i] == dqOffsetpre[j] && dqOffset[i] != OUTINDEX) {
                    dqCrossRoundConflict = true;
                }
                if (!dkDvCrossRoundConflict && dkDvOffset[i] == dkDvOffsetpre[j] && dkDvOffset[i] != OUTINDEX) {
                    dkDvCrossRoundConflict = true;
                }
            }
        }

        int64_t previousIndex = (calcNum - 1) / ALIGN64;
        int64_t previousBitPosition = (calcNum - 1) % ALIGN64;
        dqNeedDeter = dqNeedDeter || dqCrossRoundConflict;
        dkDvNeedDeter = dkDvNeedDeter || dkDvCrossRoundConflict;
        if (previousIndex >= 0 && previousIndex < INT64_NUM) {
            if (dqCrossRoundConflict) {
                fBaseParams.dqIsNeedDeter[previousIndex] |= (1ULL << previousBitPosition);
            }
            if (dkDvCrossRoundConflict) {
                fBaseParams.dkDvIsNeedDeter[previousIndex] |= (1ULL << previousBitPosition);
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
    int64_t index = calcNum / ALIGN64;
    int64_t bitPosition = calcNum % ALIGN64;
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
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        int64_t resbaseIdx = blockIdx;
        for (int64_t bIdx = 0; bIdx < fBaseParams.b; bIdx++) {
            int64_t actualS1Len = fBaseParams.actualSeqQlen[bIdx];
            int64_t actualS2Len = fBaseParams.actualSeqKvlen[bIdx];
            int64_t s1OuterTmp = (actualS1Len + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
            int64_t s2OuterTmp = (actualS2Len + fBaseParams.s2Inner - 1) / fBaseParams.s2Inner;
            int64_t totalBaseIdx = fBaseParams.n2 * fBaseParams.g * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                boIdx = bIdx;
                bDimTail = resbaseIdx;
                n2oIdx = bDimTail / (fBaseParams.g * s1OuterTmp * s2OuterTmp);
                n2DimTail = bDimTail % (fBaseParams.g * s1OuterTmp * s2OuterTmp);
                goIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
                gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
                s2oIdx = gDimTail / s1OuterTmp;
                s1oIdx = gDimTail % s1OuterTmp;
                break;
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
        // caculate dq offset
        for (int64_t bIdx = 0; bIdx < boIdx; bIdx++) {
            bOffset += fBaseParams.actualSeqQlen[bIdx] * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
        }
        s1Offset = s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
        n2Offset = n2oIdx * fBaseParams.g * fBaseParams.d;
        gOffset = goIdx * fBaseParams.d;
        currentDqOffset = bOffset + n2Offset + gOffset + s1Offset;
        // caculate dk dv offset
        bOffset = 0;
        for (int64_t bIdx = 0; bIdx < boIdx; bIdx++) {
            bOffset += fBaseParams.actualSeqKvlen[bIdx] * (fBaseParams.n2 * fBaseParams.d);
        }
        s2Offset = s2oIdx * fBaseParams.s2Inner * fBaseParams.n2 * fBaseParams.d;
        n2Offset = n2oIdx * fBaseParams.d;
        currentDkDvOffset = bOffset + n2Offset + s2Offset;
    } else {
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
        } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) {
            s1Offset = s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT *
                       (fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.g * fBaseParams.d);
            gOffset = goIdx * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d);
            s1Offset =
                s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.g * fBaseParams.d);
            gOffset = goIdx * fBaseParams.d;
        }
        currentDqOffset = bOffset + n2Offset + gOffset + s1Offset;
        // caculate dk dv offset
        if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.s2 * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.s2 * fBaseParams.d);
            s2Offset = s2oIdx * fBaseParams.s2Inner * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) {
            s2Offset = s2oIdx * fBaseParams.s2Inner * (fBaseParams.b * fBaseParams.n2 * fBaseParams.d);
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.d);
            n2Offset = n2oIdx * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.s2 * fBaseParams.d);
            s2Offset = s2oIdx * fBaseParams.s2Inner * (fBaseParams.n2 * fBaseParams.d);
            n2Offset = n2oIdx * fBaseParams.d;
        }
        currentDkDvOffset = bOffset + n2Offset + s2Offset;
    }
}

int64_t GetTotalPerBatchNum(FuzzyBaseInfoParamsRegbase &fBaseParams, uint8_t sparseType)
{
    int64_t totalPerBatchNum = 0;
    if (sparseType == static_cast<uint8_t>(SparseType::DENSE)) {
        totalPerBatchNum = fBaseParams.s1Outer * fBaseParams.s2Outer;
    } else if (sparseType == static_cast<uint8_t>(SparseType::CASUAL)) {
        if (fBaseParams.s1 < fBaseParams.s2) {
            if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
                totalPerBatchNum =
                    ((((fBaseParams.s2Outer << 1) - fBaseParams.s1Outer + 1) * fBaseParams.s1Outer) >> 1) +
                    (fBaseParams.s1Outer - 1);
            } else {
                totalPerBatchNum = (((fBaseParams.s1Outer << 1) - fBaseParams.s1Outer + 1) * fBaseParams.s1Outer) >> 1;
            }
        } else {
            totalPerBatchNum = (((fBaseParams.s1Outer << 1) - fBaseParams.s2Outer + 1) * fBaseParams.s2Outer) >> 1;
        }
    } else if (sparseType == static_cast<uint8_t>(SparseType::BAND)) {
        int64_t p = CeilDivideBy(fBaseParams.s1Token, static_cast<int64_t>(fBaseParams.s1TemplateType));
        int64_t q = CeilDivideBy(fBaseParams.s2Token, static_cast<int64_t>(fBaseParams.s2TemplateType));
        for (int64_t s2oIdx = 0; s2oIdx < fBaseParams.s2Outer; s2oIdx++) {
            int64_t xMin = (s2oIdx - q) > 0 ? (s2oIdx - q) : 0;
            int64_t xMax = (fBaseParams.s1Outer - 1) > (s2oIdx + p) ? (s2oIdx + p) : (fBaseParams.s1Outer - 1);
            int64_t length = (xMax >= xMin) ? (xMax - xMin + 1) : 0;
            if (length > 0) {
                totalPerBatchNum += length;
            }
        }
    }
    return totalPerBatchNum;
}

void PrintShapeInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    OP_LOGI(context_,
            "FAG s1s2_bn2gs1s2 with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] d[%ld] preToken[%ld] nextToken[%ld]!",
            fBaseParams.b, fBaseParams.n2, fBaseParams.g, fBaseParams.s1, fBaseParams.s2, fBaseParams.d,
            fBaseParams.s1Token, fBaseParams.s2Token);
}

bool CheckSparseModeValue(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.sparseMode > static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "sparseMode",
                                              std::to_string(fBaseParams.sparseMode).c_str(),
                                              "the value of sparseMode cannot be greater than 6");
        return false;
    }
    return true;
}

bool CheckVarLenSparseModeValue(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
        fBaseParams.sparseMode > static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "FlashAttentionScoreGrad", "sparseMode", std::to_string(fBaseParams.sparseMode).c_str(),
            "When inputLayout is TND, the value of sparseMode cannot be 5 or greater than 8");
        return false;
    }
    return true;
}

ge::graphStatus CheckUnpadTokensInfo(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    // 7  8
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[fBaseParams.bandIdx];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[fBaseParams.bandIdx];
        if (-fBaseParams.s1Token > actualS1Len || -fBaseParams.s2Token > actualS2Len ||
            (fBaseParams.s1Token + fBaseParams.s2Token) <= 0) {
            std::string s1s2ValueMsg = "{" + std::to_string(fBaseParams.s1Token) + ", " +
                                       std::to_string(fBaseParams.s2Token) + ", " +
                                       std::to_string(fBaseParams.s1Token + fBaseParams.s2Token) + "}";
            std::string s1s2ReasonMsg = "When inputLayout is TND and sparseMode is 7 or 8, the values of nextTokens, "
                                        "preTokens cannot be less than " +
                                        std::to_string(-actualS1Len) + ", " + std::to_string(-actualS2Len) +
                                        ", the value of nextTokens + preTokens must be greater than 0";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("FlashAttentionScoreGrad",
                                                   "nextTokens, preTokens, nextTokens+preTokens", s1s2ValueMsg.c_str(),
                                                   s1s2ReasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    // 0  4
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];
        if (actualS1Len == 0 || actualS2Len == 0) {
            continue;
        }
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK)) {
            if (-fBaseParams.s1Token > actualS2Len || -fBaseParams.s2Token > actualS1Len ||
                (fBaseParams.s1Token + fBaseParams.s2Token) <= 0) {
                std::string valuesMsg = "{" + std::to_string(fBaseParams.s1Token) + ", " +
                                        std::to_string(fBaseParams.s2Token) + ", " +
                                        std::to_string(fBaseParams.s1Token + fBaseParams.s2Token) + "}";
                std::string reasonMsg = "When inputLayout is TND and sparseMode is 0, the values of nextTokens, "
                                        "preTokens cannot be less than " +
                                        std::to_string(-actualS2Len) + ", " + std::to_string(-actualS1Len) +
                                        ", the value of nextTokens + preTokens must be greater than 0";
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("FlashAttentionScoreGrad",
                                                       "nextTokens, preTokens, nextTokens+preTokens", valuesMsg.c_str(),
                                                       reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) {
            if (-fBaseParams.s1Token > actualS1Len || -fBaseParams.s2Token > actualS2Len ||
                (fBaseParams.s1Token + fBaseParams.s2Token) <= 0) {
                std::string valuesMsg = "{" + std::to_string(fBaseParams.s1Token) + ", " +
                                        std::to_string(fBaseParams.s2Token) + ", " +
                                        std::to_string(fBaseParams.s1Token + fBaseParams.s2Token) + "}";
                std::string reasonMsg = "When inputLayout is TND and sparseMode is 4, the values of nextTokens, "
                                        "preTokens cannot be less than " +
                                        std::to_string(-actualS1Len) + ", " + std::to_string(-actualS2Len) +
                                        ", the value of nextTokens + preTokens must be greater than 0";
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("FlashAttentionScoreGrad",
                                                       "nextTokens, preTokens, nextTokens+preTokens", valuesMsg.c_str(),
                                                       reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

int64_t FindBandIdx(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND)) {
        for (int64_t i = fBaseParams.b - 1; i >= 0; i--) {
            if (fBaseParams.actualSeqQlen[i] != 0) {
                return i;
            }
        }
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        for (int64_t i = 0; i < fBaseParams.b; i++) {
            if (fBaseParams.actualSeqQlen[i] != 0) {
                return i;
            }
        }
    }
    return 0;
}

bool IsNewDeter(const FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    return fBaseParams.deterSparseType >= static_cast<uint32_t>(DeterSparseType::DETER_DENSE) &&
           fBaseParams.deterSparseType <= static_cast<uint32_t>(DeterSparseType::DETER_BAND) &&
           (fBaseParams.layoutType == INPUT_FORMAT_TND);
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
    if (fBaseParams.queryType == ge::DT_FLOAT && fBaseParams.d > static_cast<uint32_t>(ConstAxisTemplateNum::NUM256)) {
        fBaseParams.s1TemplateType = ConstAxisTemplateNum::NUM64;
        fBaseParams.s2TemplateType = ConstAxisTemplateNum::NUM128;
        return std::make_pair(static_cast<uint32_t>(ConstAxisTemplateNum::NUM64),
                              static_cast<uint32_t>(ConstAxisTemplateNum::NUM128));
    } else if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN) {
        // FP8场景基本块修改
        fBaseParams.s1TemplateType = ConstAxisTemplateNum::NUM64;
        fBaseParams.s2TemplateType = ConstAxisTemplateNum::NUM256;
        return std::make_pair(static_cast<uint32_t>(ConstAxisTemplateNum::NUM64),
                              static_cast<uint32_t>(ConstAxisTemplateNum::NUM256));
    } else if (fBaseParams.queryType == ge::DT_HIFLOAT8) {
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
    if (fBaseParams.hasRope) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM192;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM192);
    }
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

ge::graphStatus ProcessSinkInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    auto sinkShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SINK_IDX));
    auto dsinkShape = context_->GetOutputShape(DSINKOUT_IDX);
    if (sinkShape == nullptr || sinkShape->GetStorageShape().GetDimNum() == 0) {
        OP_LOGD(context_, "ProcessSinkInfo, sinkShape is null : %d", sinkShape == nullptr);
        fBaseParams.sinkOptional = EMPTY_TENSOR;
        return ge::GRAPH_SUCCESS;
    }
    std::string sinkReasonMsg = "The shape of sinkInOptional must be [" + std::to_string(fBaseParams.n1) + "]";
    OP_CHECK_IF(
        (sinkShape->GetStorageShape().GetDimNum() != 1 || sinkShape->GetStorageShape().GetDim(0) != fBaseParams.n1),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "sinkInOptional",
                                               Ops::Base::ToString(sinkShape->GetStorageShape()).c_str(),
                                               sinkReasonMsg.c_str()),
        return ge::GRAPH_FAILED);
    std::string dsinkReasonMsg = "The shape of dsinkOut must be [" + std::to_string(fBaseParams.n1) + "]";
    OP_CHECK_IF(
        (dsinkShape->GetStorageShape().GetDimNum() != 1 || dsinkShape->GetStorageShape().GetDim(0) != fBaseParams.n1),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "dsinkOut",
                                               Ops::Base::ToString(dsinkShape->GetStorageShape()).c_str(),
                                               dsinkReasonMsg.c_str()),
        return ge::GRAPH_FAILED);
    auto sinkInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::SINK_IDX));
    if (sinkInput != nullptr) {
        auto sinkDtype = sinkInput->GetDataType();
        OP_CHECK_IF(sinkDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("FlashAttention", "sinkInOptional",
                                                          ge::TypeUtils::DataTypeToSerialString(sinkDtype).c_str(),
                                                          "The dtype of sinkInOptional must be FLOAT32"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(!(fBaseParams.queryType == ge::DT_FLOAT16 || fBaseParams.queryType == ge::DT_BF16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    "FlashAttention", "query", "Non-FLOAT16 or Non-BFLOAT16",
                    "When optional parameter sinkInOptional exists, the dtype of query must be FLOAT16 or BFLOAT16"),
                return ge::GRAPH_FAILED);
    fBaseParams.sinkOptional = NORMAL_TENSOR;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ProcessOptionalInput(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    const char *inputLayout = context_->GetAttrs()->GetAttrPointer<char>(LAYOUT_ATTR_IDX);
    if (strcmp(inputLayout, "TND") == 0) {
        fBaseParams.qSize = static_cast<uint64_t>(fBaseParams.t1) * fBaseParams.n2 * fBaseParams.g * fBaseParams.d;
        fBaseParams.kSize = static_cast<uint64_t>(fBaseParams.t2) * fBaseParams.n2 * 1 * fBaseParams.d;
        fBaseParams.vSize = static_cast<uint64_t>(fBaseParams.t2) * fBaseParams.n2 * 1 * fBaseParams.d1;
        fBaseParams.dropMaskSize = static_cast<uint64_t>(fBaseParams.n2) * fBaseParams.g * fBaseParams.sumS1S2Product;
    } else {
        fBaseParams.qSize =
            static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d;
        fBaseParams.kSize = static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * 1 * fBaseParams.s2 * fBaseParams.d;
        fBaseParams.vSize = static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * 1 * fBaseParams.s2 * fBaseParams.d1;
        fBaseParams.dropMaskSize =
            static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g * fBaseParams.s2 * fBaseParams.s1;
    }

    // process tnd softmax layout
    if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(AttrIndex::TND_SOFTMAX_IN)) {
        const char *tndMaxSumLayout =
            context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::TND_SOFTMAX_IN));
        if (strcmp(inputLayout, "TND") == 0 && strcmp(tndMaxSumLayout, "same_as_input") == 0) {
            fBaseParams.tndMaxSumLayout = 1;
        }
        OP_LOGD("TND Max Sum Layout", "FAG tndMaxSumLayout = %d, tndMaxSumLayout = %s.", fBaseParams.tndMaxSumLayout,
                tndMaxSumLayout);
    }

    // mBaseParams is used for matmal tiling module
    auto queryType = context_->GetInputDesc(0)->GetDataType();
    fBaseParams.queryType = queryType;
    fBaseParams.calTypeSize = FP32_BYTES;

    fBaseParams.scaleValue = *(context_->GetAttrs()->GetAttrPointer<float>(0));
    fBaseParams.keepProb = *(context_->GetAttrs()->GetAttrPointer<float>(1));

    fBaseParams.dropoutIsDivisibleBy8 = 1;
    if (fBaseParams.keepProb < 1) {
        ProcessDropoutIsDivisibleBy8(context_, fBaseParams);
    }

    auto ret = ProcessDropoutInfo(context_, fBaseParams);

    PrintShapeInfo(context_, fBaseParams);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ProcessQuantInfo(context_, fBaseParams);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // token_info
    fBaseParams.s1Token = *(context_->GetAttrs()->GetAttrPointer<int64_t>(PRE_TOKEN_ATTR_IDX));
    fBaseParams.s2Token = *(context_->GetAttrs()->GetAttrPointer<int64_t>(NEXT_TOKEN_ATTR_IDX));

    ret = ProcessSparseModeInfo(context_, fBaseParams);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ProcessTokensInfo(fBaseParams);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    SetQKVStartIdx(context_, fBaseParams);
    ret = ProcessPseInfo(context_, fBaseParams, inputLayout);
    if (ret != ge::GRAPH_SUCCESS) {
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

    ret = ProcessSinkInfo(context_, fBaseParams);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (CheckAttenMaskShape(fBaseParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckDtypeValid(context_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return (strcmp(inputLayout, "TND") == 0) ?
               CheckTndShapeValid(context_, fBaseParams.t1, fBaseParams.n1, fBaseParams.d) :
               CheckShapeValid(context_, fBaseParams.b, fBaseParams.n1, fBaseParams.s1, fBaseParams.d);
}

void ProcessDropoutIsDivisibleBy8(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    const char *inputLayout = context_->GetAttrs()->GetAttrPointer<char>(LAYOUT_ATTR_IDX);
    if (strcmp(inputLayout, "TND") == 0) {
        for (int64_t i = 0; i < fBaseParams.b; i++)
            if (fBaseParams.actualSeqKvlen[i] % BIT_NUMS != 0) {
                fBaseParams.dropoutIsDivisibleBy8 = 0;
                break;
            }
    } else {
        if (fBaseParams.s2 % BIT_NUMS != 0) {
            fBaseParams.dropoutIsDivisibleBy8 = 0;
        }
    }
    return;
}

ge::graphStatus ProcessDropoutInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    bool hasDrop = fBaseParams.keepProb < 1;
    // dropout mask
    fBaseParams.keepProbUint8 = static_cast<int64_t>(fBaseParams.keepProb * UINT8_MAX);
    if (context_->GetAttrs()->GetAttrNum() > SEED_ATTR_IDX) {
        fBaseParams.seed = *(context_->GetAttrs()->GetAttrPointer<int>(SEED_ATTR_IDX));
    }
    if (context_->GetAttrs()->GetAttrNum() > OFFSET_ATTR_IDX) {
        fBaseParams.offset = *(context_->GetAttrs()->GetAttrPointer<int>(OFFSET_ATTR_IDX));
    }
    auto dropMask = context_->GetOptionalInputDesc(DROP_MASK_IDX);
    auto dropMaskShape = context_->GetOptionalInputShape(DROP_MASK_IDX);
    if (dropMask != nullptr && dropMaskShape != nullptr && dropMaskShape->GetStorageShape().GetDimNum() != 0) {
        if (!hasDrop) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "FlashAttentionScoreGrad", "keepProb", std::to_string(fBaseParams.keepProb).c_str(),
                "When optional parameter dropMaskOptional exists, the value of keepProb must be less than 1");
            return ge::GRAPH_FAILED;
        }
        auto dropMaskDType = dropMask->GetDataType();
        OP_CHECK_IF(dropMaskDType != ge::DT_UINT8,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("FlashAttentionScoreGrad", "dropMaskOptional",
                                                          ge::TypeUtils::DataTypeToSerialString(dropMaskDType).c_str(),
                                                          "The dtype of dropMaskOptional must be uint8"),
                    return ge::GRAPH_FAILED);
        uint64_t dropMaskDim = dropMaskShape->GetStorageShape().GetDimNum();
        uint64_t dropMaskShapeSize = 1;
        for (uint64_t i = 0; i < dropMaskDim; ++i) {
            uint64_t dimValue = dropMaskShape->GetStorageShape().GetDim(i);
            dropMaskShapeSize *= dimValue;
        }
        auto shapeSize = AlignUp(fBaseParams.dropMaskSize, static_cast<uint64_t>(BIT_NUMS)) / BIT_NUMS;
        if (dropMaskShapeSize < shapeSize) {
            std::string reasonMsg =
                "The shape size of dropMaskOptional cannot be less than " + std::to_string(shapeSize);
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("FlashAttentionScoreGrad", "dropMaskOptional",
                                                      std::to_string(dropMaskShapeSize).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        fBaseParams.dropMaskOuter = true;
    } else {
        fBaseParams.dropMaskOuter = false;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ProcessQuantInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    DetermineMode(fBaseParams);
    if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
        fBaseParams.queryType == ge::DT_UINT8 || fBaseParams.queryType == ge::DT_INT8 ||
        fBaseParams.queryType == ge::DT_QINT8 || fBaseParams.queryType == ge::DT_HIFLOAT8) {
        auto queryDType = context_->GetInputDesc(0)->GetDataType();
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("FlashAttentionScoreGrad", "query",
                                               ge::TypeUtils::DataTypeToSerialString(queryDType).c_str(),
                                               "The dtype of query is invalid");
        return ge::GRAPH_FAILED;
    }

    if (fBaseParams.queryType != ge::DT_HIFLOAT8) {
        auto deqScaleQShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_Q));
        auto deqScaleKShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_K));
        auto deqScaleVShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_V));
        auto deqScaleDyShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_DY));
        std::string dScaleShapesMsg = "{";
        dScaleShapesMsg +=
            deqScaleQShape != nullptr ? Ops::Base::ToString(deqScaleQShape->GetStorageShape()) + ", " : "nullptr, ";
        dScaleShapesMsg +=
            deqScaleKShape != nullptr ? Ops::Base::ToString(deqScaleKShape->GetStorageShape()) + ", " : "nullptr, ";
        dScaleShapesMsg +=
            deqScaleVShape != nullptr ? Ops::Base::ToString(deqScaleVShape->GetStorageShape()) + ", " : "nullptr, ";
        dScaleShapesMsg +=
            deqScaleDyShape != nullptr ? Ops::Base::ToString(deqScaleDyShape->GetStorageShape()) + "}" : "nullptr}";
        OP_CHECK_IF((deqScaleQShape != nullptr && deqScaleQShape->GetStorageShape().GetDimNum() != 0) ||
                        (deqScaleKShape != nullptr && deqScaleKShape->GetStorageShape().GetDimNum() != 0) ||
                        (deqScaleVShape != nullptr && deqScaleVShape->GetStorageShape().GetDimNum() != 0) ||
                        (deqScaleDyShape != nullptr && deqScaleDyShape->GetStorageShape().GetDimNum() != 0),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        "FlashAttentionScoreGrad",
                        "dScaleQOptional, dScaleKOptional, dScaleVOptional, dScaleDyOptional", dScaleShapesMsg.c_str(),
                        "When the dType of input is not HIFLOAT8, "
                        "dScaleQOptional, dScaleKOptional, dScaleVOptional and dScaleDyOptional should be nullptr now"),
                    return ge::GRAPH_FAILED);
    }

    fBaseParams.outDtype = fBaseParams.inputDtype;
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

    if (SupportTrans2BS2N2GD(fBaseParams)) {
        fBaseParams.layoutType = INPUT_FORMAT_BS2N2GD;
        OP_LOGD("inputLayout = TND, but all s1 s2 same, inputLayout set BSND");
    }

    if (!(fBaseParams.layoutType == INPUT_FORMAT_TND ? CheckVarLenSparseModeValue(fBaseParams) :
                                                       CheckSparseModeValue(fBaseParams))) {
        return ge::GRAPH_FAILED;
    }
    fBaseParams.attenMaskCompressMode = 0;
    auto attenMaskShape = context_->GetOptionalInputShape(INPUT_IDX_ATTEN_MASK);
    if (attenMaskShape == nullptr || attenMaskShape->GetStorageShape().GetDimNum() == 0) {
        fBaseParams.attenMaskOptional = EMPTY_TENSOR;
        return ge::GRAPH_SUCCESS;
    }
    fBaseParams.attenMaskOptional = NORMAL_TENSOR;
    auto storageShape = attenMaskShape->GetStorageShape();
    size_t dimNum = storageShape.GetDimNum();
    auto ret = SetAttenMaskShapeType(fBaseParams, attenMaskShape, dimNum);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    fBaseParams.attenMaskS2Size = storageShape.GetDim(dimNum - LAST_AXIS_IDX);
    fBaseParams.attenMaskS1Size = storageShape.GetDim(dimNum - SEC_LAST_AXIS_IDX);

    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::BAND_EQUAL_S_MODE);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::PREFIX_COMPRESS_MODE);
        fBaseParams.attenMaskS2Size = ATTEN_MASK_COMPRESS_LIMIT;
        fBaseParams.attenMaskS1Size = PREFIX_COMPRESS_S1_SIZE;
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::RIGHT_DOWN_CASUAL_BAND_MODE);
        fBaseParams.attenMaskS2Size = ATTEN_MASK_COMPRESS_LIMIT;
        fBaseParams.attenMaskS1Size = ATTEN_MASK_COMPRESS_LIMIT;
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::BAND_LEFT_UP_CASUAL_MODE);
        fBaseParams.attenMaskS2Size = ATTEN_MASK_COMPRESS_LIMIT;
        fBaseParams.attenMaskS1Size = ATTEN_MASK_COMPRESS_LIMIT;
    }

    auto attenMask = context_->GetOptionalInputDesc(INPUT_IDX_ATTEN_MASK);
    if (attenMask != nullptr) {
        auto attenMaskType = attenMask->GetDataType();
        OP_CHECK_IF(attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("FlashAttentionScoreGrad", "attenMaskOptional",
                                                          ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str(),
                                                          "The dtype of attenMaskOptional must be bool or uint8"),
                    return ge::GRAPH_FAILED);
        fBaseParams.attenMaskDtype = static_cast<uint32_t>(AttenDataType::ATTEN_MASK_TYPE_U8_BOOL);
    }

    fBaseParams.bandIdx = FindBandIdx(fBaseParams);
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
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("FlashAttentionScoreGrad", "nextTokens, preTokens, nextToKens",
                                               valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // 校验unpad场景token是否合法   0  4  7  8
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        auto ret = CheckUnpadTokensInfo(fBaseParams);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void SetQKVStartIdx(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    fBaseParams.qStartIdx = 0;
    fBaseParams.kvStartIdx = 0;

    auto qStartIdxTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::Q_START_IDX));
    if (qStartIdxTensor != nullptr) {
        auto &qStartIdxShape = qStartIdxTensor->GetShape().GetStorageShape();
        if (qStartIdxShape.GetDimNum() >= 1 && qStartIdxShape.GetDim(0) != 0) {
            /* Get Data from tensor. */
            const int64_t *value = qStartIdxTensor->GetData<int64_t>();
            if (value != nullptr) {
                fBaseParams.qStartIdx = value[0];
                OP_LOGD(context_, "[%s]SetQKVStartIdx qStartIdx: %ld", "FlashAttentionScoreGradTilingS1s2Bn2gs1s2",
                        fBaseParams.qStartIdx);
            }
        }
    }

    auto kvStartIdxTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::KV_START_IDX));
    if (kvStartIdxTensor != nullptr) {
        auto &kvStartIdxShape = kvStartIdxTensor->GetShape().GetStorageShape();
        if (kvStartIdxShape.GetDimNum() >= 1 && kvStartIdxShape.GetDim(0) != 0) {
            /* Get Data from tensor. */
            const int64_t *kvValue = kvStartIdxTensor->GetData<int64_t>();
            if (kvValue != nullptr) {
                fBaseParams.kvStartIdx = kvValue[0];
                OP_LOGD(context_, "[%s]SetQKVStartIdx kvStartIdx: %ld", "FlashAttentionScoreGradTilingS1s2Bn2gs1s2",
                        fBaseParams.kvStartIdx);
            }
        }
    }
}

ge::graphStatus ProcessPseNormal(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams,
                                 const char *inputLayout)
{
    auto pseShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::PSE_SHIFT));
    auto pseShapeDim = pseShape->GetStorageShape().GetDimNum();
    if (pseShapeDim != PSE_NORMAL_SHAPE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("FlashAttentionScoreGrad", "pseShiftOptional",
                                                 std::to_string(pseShapeDim).c_str(),
                                                 "The shape dim of pseShiftOptional must be 4");
        return ge::GRAPH_PARAM_INVALID;
    }

    auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
    auto dim1 = pseShape->GetStorageShape().GetDim(DIM_1);
    auto dim2 = pseShape->GetStorageShape().GetDim(DIM_2);
    auto dim3 = pseShape->GetStorageShape().GetDim(DIM_3);

    bool isBN1S = (dim0 == fBaseParams.b && dim1 == fBaseParams.n1 && dim2 == 1 && dim3 == fBaseParams.s2);
    bool isBNSS = (dim0 == fBaseParams.b && dim1 == fBaseParams.n1 && dim2 == fBaseParams.s1 && dim3 == fBaseParams.s2);
    bool is1NSS = (dim0 == 1 && dim1 == fBaseParams.n1 && dim2 == fBaseParams.s1 && dim3 == fBaseParams.s2);
    bool isAlibiPse = (dim1 == fBaseParams.n1 && dim2 == MAX_BASIC_BLOCK_SIZE && dim3 == fBaseParams.s2);
    bool isPse = (fBaseParams.s1 == fBaseParams.s2 && fBaseParams.s1 >= MAX_BASIC_BLOCK_SIZE &&
                  int64_t(fBaseParams.s1) <= fBaseParams.s1Token && fBaseParams.s2Token == 0);
    bool isTnd = strcmp(inputLayout, "TND") == 0;
    bool isTndPse = isTnd && int64_t(fBaseParams.s1) <= fBaseParams.s1Token && fBaseParams.s2Token == 0;
    bool isAlibi1NHS = isPse && isAlibiPse && (dim0 == 1);
    bool isAlibiBNHS = isPse && isAlibiPse && (dim0 == fBaseParams.b);
    bool isTndAlibiPse1NHS = isTndPse && isAlibiPse && (dim0 == 1);
    bool isTndAlibiPseBNHS = isTndPse && isAlibiPse && (dim0 == fBaseParams.b);

    if (isTndAlibiPse1NHS) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_1NHS);
    } else if (isTndAlibiPseBNHS) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNHS);
    } else if (isBN1S && !isTnd) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BN1S);
    } else if (isBNSS && !isTnd) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNSS);
    } else if (is1NSS && !isTnd) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_1NSS);
    } else if (isAlibi1NHS) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_1NHS);
    } else if (isAlibiBNHS) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNHS);
    } else {
        if (isTndPse) {
            std::string shapeMsg = std::to_string(dim0) + ", " + std::to_string(dim1) + ", " + std::to_string(dim2) +
                                   ", " + std::to_string(dim3);
            std::string reasonMsg = "The shape of pseShiftOptional must be [1, " + std::to_string(fBaseParams.n1) +
                                    ", 1024, " + std::to_string(fBaseParams.s2) + "] or [" +
                                    std::to_string(fBaseParams.b) + ", " + std::to_string(fBaseParams.n1) + ", 1024, " +
                                    std::to_string(fBaseParams.s2) + "]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("FlashAttentionScoreGrad", "pseShiftOptional", shapeMsg.c_str(),
                                                  reasonMsg.c_str());
        } else if (isPse && !isTnd) {
            std::string shapeMsg = std::to_string(dim0) + ", " + std::to_string(dim1) + ", " + std::to_string(dim2) +
                                   ", " + std::to_string(dim3);
            std::string reasonMsg = "The shape of pseShiftOptional must be [1, " + std::to_string(fBaseParams.n1) +
                                    ", 1024, " + std::to_string(fBaseParams.s2) + "] or [" +
                                    std::to_string(fBaseParams.b) + ", " + std::to_string(fBaseParams.n1) + ", 1024, " +
                                    std::to_string(fBaseParams.s2) + "] or [" + std::to_string(fBaseParams.b) + ", " +
                                    std::to_string(fBaseParams.n1) + ", 1, " + std::to_string(fBaseParams.s2) +
                                    "] or [" + std::to_string(fBaseParams.b) + ", " + std::to_string(fBaseParams.n1) +
                                    ", " + std::to_string(fBaseParams.s1) + ", " + std::to_string(fBaseParams.s2) +
                                    "] or [1, " + std::to_string(fBaseParams.n1) + ", " +
                                    std::to_string(fBaseParams.s1) + ", " + std::to_string(fBaseParams.s2) + "]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("FlashAttentionScoreGrad", "pseShiftOptional", shapeMsg.c_str(),
                                                  reasonMsg.c_str());
        } else if (!isTnd) {
            std::string shapeMsg = std::to_string(dim0) + ", " + std::to_string(dim1) + ", " + std::to_string(dim2) +
                                   ", " + std::to_string(dim3);
            std::string reasonMsg = "The shape of pseShiftOptional must be [" + std::to_string(fBaseParams.b) + ", " +
                                    std::to_string(fBaseParams.n1) + ", 1, " + std::to_string(fBaseParams.s2) +
                                    "] or [" + std::to_string(fBaseParams.b) + ", " + std::to_string(fBaseParams.n1) +
                                    ", " + std::to_string(fBaseParams.s1) + ", " + std::to_string(fBaseParams.s2) +
                                    "] or [1, " + std::to_string(fBaseParams.n1) + ", " +
                                    std::to_string(fBaseParams.s1) + ", " + std::to_string(fBaseParams.s2) + "]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("FlashAttentionScoreGrad", "pseShiftOptional", shapeMsg.c_str(),
                                                  reasonMsg.c_str());
        } else {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "FlashAttentionScoreGrad", "pseShiftOptional", Ops::Base::ToString(pseShape->GetStorageShape()),
                "When inputLayout is TND, Sq of pseShiftOptional cannot be less than 1024 and must be equal to Skv");
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "FlashAttentionScoreGrad", "sparseMode", std::to_string(fBaseParams.sparseMode),
                "When inputLayout is TND and pseShiftOptional is supported, the sparseMode must be 0 or 2 or 3");
        }
        return ge::GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ProcessPseInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams,
                               const char *inputLayout)
{
    if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(AttrIndex::PSETYPE)) {
        fBaseParams.pseType =
            *(context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::PSETYPE))); // 8
        if (fBaseParams.pseType >= static_cast<uint32_t>(PseType::PSE_INVALID_TYPE)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "pseType",
                                                  std::to_string(fBaseParams.pseType).c_str(),
                                                  "The value of pseType must be less than 4");
            return ge::GRAPH_FAILED;
        }
    }

    auto pseShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::PSE_SHIFT));
    if (pseShape == nullptr || pseShape->GetStorageShape().GetDimNum() == 0) {
        fBaseParams.pseOptional = EMPTY_TENSOR;
        return ge::GRAPH_SUCCESS;
    }

    fBaseParams.pseOptional = NORMAL_TENSOR;
    auto pse = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::PSE_SHIFT));
    if (fBaseParams.pseType == static_cast<uint32_t>(PseType::PSE_OUTER_MUL_ADD_TYPE) ||
        fBaseParams.pseType == static_cast<uint32_t>(PseType::PSE_OUTER_ADD_MUL_TYPE)) {
        if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
            fBaseParams.queryType == ge::DT_HIFLOAT8) {
            bool pseTypeCheckResult = (fBaseParams.outDtype == DtypeEnum::FLOAT16_PRECISION) ?
                                          (pse->GetDataType() == ge::DT_FLOAT16) :
                                          (pse->GetDataType() == ge::DT_BF16);
            OP_CHECK_IF(!pseTypeCheckResult,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            "FlashAttentionScoreGrad", "pseShifOptional",
                            ge::TypeUtils::DataTypeToSerialString(pse->GetDataType()).c_str(),
                            "The dtype of pseShifOptional must be the same as dtype of output"),
                        return ge::GRAPH_FAILED);
        } else {
            std::string dTypesMsg = ge::TypeUtils::DataTypeToSerialString(pse->GetDataType()) + ", " +
                                    ge::TypeUtils::DataTypeToSerialString(
                                        context_->GetInputDesc(static_cast<size_t>(InputIndex::QUERY))->GetDataType());
            OP_CHECK_IF(
                pse->GetDataType() != context_->GetInputDesc(static_cast<size_t>(InputIndex::QUERY))->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("FlashAttentionScoreGrad", "pseShifOptional, query",
                                                       dTypesMsg.c_str(),
                                                       "The dtypes of pseShifOptional and query must be the same"),
                return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(
            pse->GetDataType() != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("FlashAttentionScoreGrad", "pseShifOptional",
                                                   ge::TypeUtils::DataTypeToSerialString(pse->GetDataType()).c_str(),
                                                   "The dtype of pseShifOptional must be FLOAT32"),
            return ge::GRAPH_FAILED);
    }

    auto pseShapeDim = pseShape->GetStorageShape().GetDimNum();
    bool isTnd = strcmp(inputLayout, "TND") == 0;
    if (fBaseParams.pseType == static_cast<uint32_t>(PseType::PSE_INNER_MUL_ADD_TYPE) ||
        fBaseParams.pseType == static_cast<uint32_t>(PseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
        auto ret = ProcessInnerPseInfo(context_, fBaseParams, pseShapeDim);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    } else if (pseShapeDim == PSE_DIM_NUM_1 && isTnd) {
        auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
        bool isTndPseBN1S = isTnd && (dim0 == fBaseParams.t2 * fBaseParams.n1);
        bool isTndPseBNSS = isTnd && (dim0 == fBaseParams.sumS1S2Product * fBaseParams.n1);
        if (isTndPseBN1S) {
            fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BN1S);
        } else if (isTndPseBNSS) {
            fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNSS);
        } else {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "FlashAttentionScoreGrad", "pseShifOptional", Ops::Base::ToString(pseShape->GetStorageShape()).c_str(),
                "When pseType is 0 or 1 and inputLayout is TND, the shape of pseShifOptional is unspported");
            return ge::GRAPH_FAILED;
        }
    } else {
        auto ret = ProcessPseNormal(context_, fBaseParams, inputLayout);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }
    SetPseLayout(fBaseParams);
    return ge::GRAPH_SUCCESS;
}

void SetPseLayout(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_1NSS) ||
        fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNHS) ||
        fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_1NHS) ||
        fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNSS)) {
        fBaseParams.pseLayoutType = static_cast<uint32_t>(PseLayoutType::pseS1S2);
    } else if (fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_B_N2_G_SLOPE)) {
        fBaseParams.pseLayoutType = static_cast<uint32_t>(PseLayoutType::pseSlopeBn);
    } else if (fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_1_N2_G_SLOPE)) {
        fBaseParams.pseLayoutType = static_cast<uint32_t>(PseLayoutType::pseSlopeN);
    } else if (fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BN1S)) {
        fBaseParams.pseLayoutType = static_cast<uint32_t>(PseLayoutType::pse1S2);
    }
}

bool SetSparseParams(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        return SetPrefixSparseParams(context_, fBaseParams);
    }

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

void SetSplitAxis(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams,
                  const TndBaseInfo &tndBaseInfo)
{
    fBaseParams.isBn2 = (fBaseParams.s1 <= BN2_MAX_S && fBaseParams.s2 <= BN2_MAX_S) &&
                        (fBaseParams.n1 == fBaseParams.n2) && (fBaseParams.d <= BN2_MAX_D) &&
                        (fBaseParams.queryType != ge::DT_FLOAT) &&
                        !(fBaseParams.queryType == ge::DT_FLOAT8_E5M2 ||
                          fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN || fBaseParams.queryType == ge::DT_HIFLOAT8) &&
                        (fBaseParams.tailZeroCount == 0 && !tndBaseInfo.isSeqExistZero);

    bool bnLimit = ((fBaseParams.b * fBaseParams.n1) >= BN2_MULTIBLK_BN_256) ||
                   ((fBaseParams.b * fBaseParams.n1) >= BN2_MULTIBLK_BN_128 && (fBaseParams.s1 % ALIGN128 == 0) &&
                    (fBaseParams.s2 % ALIGN128 == 0));
    bool bnSparseLimit = bnLimit && (fBaseParams.layoutType != INPUT_FORMAT_TND) &&
                         (fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::PREFIX)) &&
                         (fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS));
    fBaseParams.isBn2MultiBlk =
        bnSparseLimit && (fBaseParams.s1 > BN2_MAX_S || fBaseParams.s2 > BN2_MAX_S) &&
        (fBaseParams.s1 <= BN2_MULTIBLK_SEQ && fBaseParams.s2 <= BN2_MULTIBLK_SEQ) &&
        (fBaseParams.n1 == fBaseParams.n2) && fBaseParams.d <= BN2_MAX_D && (fBaseParams.queryType != ge::DT_FLOAT) &&
        (fBaseParams.d == fBaseParams.d1) &&
        !(fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
          fBaseParams.queryType == ge::DT_HIFLOAT8) &&
        !fBaseParams.hasRope;
    fBaseParams.isBn2 = fBaseParams.isBn2MultiBlk ? true : fBaseParams.isBn2; // 多基本块场景是原始bn2的子集
    if (fBaseParams.isBn2 && !fBaseParams.isBn2MultiBlk) {
        fBaseParams.isDeterministic = false;
        if ((fBaseParams.layoutType == INPUT_FORMAT_TND && fBaseParams.d > ALIGN128) || fBaseParams.dropMaskOuter) {
            fBaseParams.isBn2 = false;
            fBaseParams.isDeterministic = (context_->GetDeterministic() == 1);
        }
    }
    if (fBaseParams.isBn2MultiBlk) {
        fBaseParams.isDeterministic = false;
        if (fBaseParams.dropMaskOuter) {
            fBaseParams.isBn2 = false;
            fBaseParams.isBn2MultiBlk = false;
            fBaseParams.isDeterministic = (context_->GetDeterministic() == 1);
        }
    }

    bool bn2S2NotTndLimit = (fBaseParams.s1 < fBaseParams.s2) && (fBaseParams.s2 <= BN2S2_MAX_S) &&
                            (fBaseParams.s2 - fBaseParams.s1 >= BN2_MAX_S) && (fBaseParams.d <= BN2S2_WRITE_UB_D) &&
                            (!fBaseParams.isSparse) && (!fBaseParams.isDeterministic);
    bool bn2S2RouteLimit =
        !fBaseParams.hasRope && fBaseParams.d <= BN2_MAX_D &&
        (fBaseParams.layoutType == INPUT_FORMAT_TND || (fBaseParams.isAllSame && !fBaseParams.isDeterministic) ||
         bn2S2NotTndLimit) &&
        (fBaseParams.keepProb >= 1 ||
         (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM128) && fBaseParams.keepProb < 1)) &&
        (fBaseParams.n1 == fBaseParams.n2) && (fBaseParams.queryType != ge::DT_FLOAT) &&
        !(fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
          fBaseParams.queryType == ge::DT_HIFLOAT8);

    if (!fBaseParams.isBn2 && bn2S2RouteLimit) {
        fBaseParams.layoutType = fBaseParams.isAllSame ? INPUT_FORMAT_TND : fBaseParams.layoutType;
        fBaseParams.splitAxis = SplitAxisEnum::BN2S2;
    } else if (fBaseParams.isBn2) {
        fBaseParams.splitAxis = SplitAxisEnum::BN2;
    } else {
        fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
    }
}

void DetermineMode(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    // 当前fp16都走高精度
    if (fBaseParams.queryType == ge::DT_FLOAT) {
        fBaseParams.inputDtype = DtypeEnum::FLOAT32;
    } else if (fBaseParams.queryType == ge::DT_BF16) {
        fBaseParams.inputDtype = DtypeEnum::BFLOAT16;
    } else if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2) {
        fBaseParams.inputDtype = static_cast<optiling::DtypeEnum>(DTYPE_ENUM_INDEX_4); // DtypeEnum::FLOAT8_E5M2
    } else if (fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN) {
        fBaseParams.inputDtype = static_cast<optiling::DtypeEnum>(DTYPE_ENUM_INDEX_5); // DtypeEnum::FLOAT8_E4M3
    } else if (fBaseParams.queryType == ge::DT_HIFLOAT8) {
        fBaseParams.inputDtype = static_cast<optiling::DtypeEnum>(DTYPE_ENUM_INDEX_6); // DtypeEnum::HIFLOAT8
    } else {
        fBaseParams.inputDtype = DtypeEnum::FLOAT16_PRECISION;
    }
}

bool SupportTrans2BS2N2GD(const FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    return (fBaseParams.sparseMode <= static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) && fBaseParams.isAllSame &&
           (fBaseParams.layoutType == INPUT_FORMAT_TND);
}

ge::graphStatus SetAttenMaskShapeType(FuzzyBaseInfoParamsRegbase &fBaseParams, const gert::StorageShape *attenMaskShape,
                                      size_t dimNum)
{
    if (dimNum == ATTEN_MASK_DIM_LENGTH_2) {
        fBaseParams.attenMaskShapeType = static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKS1S2);
    } else if (dimNum == ATTEN_MASK_DIM_LENGTH_4) {
        auto dim0 = attenMaskShape->GetStorageShape().GetDim(ATTEN_MASK_SHAPE_DIMS_0);
        auto dim1 = attenMaskShape->GetStorageShape().GetDim(ATTEN_MASK_SHAPE_DIMS_1);
        if ((dim0 == fBaseParams.b) && (dim1 == fBaseParams.n2 * fBaseParams.g)) {
            fBaseParams.attenMaskShapeType = static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKBN2GS1S2);
        } else if ((dim0 == fBaseParams.b) && (dim1 == 1)) {
            fBaseParams.attenMaskShapeType = static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKBS1S2);
        } else if ((dim0 == 1) && (dim1 == 1)) {
            fBaseParams.attenMaskShapeType = static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKS1S2);
        } else {
            std::string reasonMsg = "The shape of attenMaskOptional must be [" + std::to_string(fBaseParams.b) + ", " +
                                    std::to_string(fBaseParams.n2 * fBaseParams.g) + ", " +
                                    std::to_string(fBaseParams.s1) + std::to_string(fBaseParams.s2) + "] or [" +
                                    std::to_string(fBaseParams.b) + ", 1, " + std::to_string(fBaseParams.s1) +
                                    std::to_string(fBaseParams.s2) + "] or [1, 1, " + std::to_string(fBaseParams.s1) +
                                    std::to_string(fBaseParams.s2) + "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "attenMaskOptional",
                                                   Ops::Base::ToString(attenMaskShape->GetStorageShape()).c_str(),
                                                   reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("FlashAttentionScoreGrad", "attenMaskOptional",
                                                 std::to_string(dimNum).c_str(),
                                                 "The shape dim of attenMaskOptional must be 2 or 4");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ProcessInnerPseInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams,
                                    size_t pseShapeDim)
{
    auto pseShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::PSE_SHIFT));
    // sparse mode 7 不支持 pse inner
    OP_CHECK_IF(fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "pseShifOptional",
                                                      std::to_string(fBaseParams.pseType).c_str(),
                                                      "When sparseMode is 7, the value of pseType cannot be 2 or 3"),
                return ge::GRAPH_FAILED);
    // sparse mode 8 支持pse inner的条件
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        auto ret = ProcessPseSparseMode8(fBaseParams);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }
    // 输入为[N1]或者[B, N1]
    if (pseShapeDim == PSE_DIM_NUM_1) {
        auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
        std::string reasonMsg = "The shape of pseShifOptional must be [" + std::to_string(fBaseParams.n1) + "]";
        OP_CHECK_IF(dim0 != fBaseParams.n1,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "pseShifOptional",
                                                           Ops::Base::ToString(pseShape->GetStorageShape()).c_str(),
                                                           reasonMsg.c_str()),
                    return ge::GRAPH_FAILED);
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_1_N2_G_SLOPE);
    } else if (pseShapeDim == PSE_DIM_NUM_2) {
        auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
        auto dim1 = pseShape->GetStorageShape().GetDim(DIM_1);
        std::string reasonMsg = "The shape of pseShifOptional must be [" + std::to_string(fBaseParams.b) + ", " +
                                std::to_string(fBaseParams.n1) + "]";
        OP_CHECK_IF(dim0 != fBaseParams.b || dim1 != fBaseParams.n1,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "pseShifOptional",
                                                           Ops::Base::ToString(pseShape->GetStorageShape()).c_str(),
                                                           reasonMsg.c_str()),
                    return ge::GRAPH_FAILED);
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_B_N2_G_SLOPE);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("FlashAttentionScoreGrad", "pseShifOptional",
                                                 std::to_string(pseShapeDim).c_str(),
                                                 "The shape dim of pseShifOptional must be 1 or 2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ProcessPseSparseMode8(FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    for (int64_t boIdx = 0; boIdx < fBaseParams.b; boIdx++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[boIdx];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[boIdx];
        if (boIdx == 0) {
            if (actualS1Len - actualS2Len + fBaseParams.qStartIdx - fBaseParams.kvStartIdx == 0) {
                continue;
            } else {
                std::string valuesMsg = "{" + std::to_string(actualS1Len) + ", " + std::to_string(actualS2Len) + ", " +
                                        std::to_string(fBaseParams.qStartIdx) + ", " +
                                        std::to_string(fBaseParams.kvStartIdx) + "}";
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    "FlashAttentionScoreGrad",
                    "actualSeqQLenOptional, actualSeqKLenOptional, qStartIdxOptional, kvStartIdxOptional",
                    valuesMsg.c_str(),
                    "When pseTye is 2 or 3 and sparseMode is 8, the value of "
                    "actualSeqQLenOptional - actualSeqKLenOptional + qStartIdxOptional - kvStartIdxOptional must be 0");
                return ge::GRAPH_FAILED;
            }
        }
        if (actualS1Len != actualS2Len) {
            std::string valuesMsg = "{" + std::to_string(actualS1Len) + ", " + std::to_string(actualS2Len) + "}";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                "FlashAttentionScoreGrad", "actualSeqQLenOptional, actualSeqKLenOptional", valuesMsg.c_str(),
                "When pseType is 2 or 3 and "
                "sparseMode is 8, the value of actualSeqQLenOptional must be equal to value of actualSeqKLenOptional");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool SetPrefixSparseParams(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams)
{
    auto prefixNTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::PREFIX_N));
    if (prefixNTensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "prefixOptional");
        return false;
    }

    auto &prefixShape = prefixNTensor->GetShape().GetStorageShape();
    if (prefixShape.GetDimNum() != 1 || prefixShape.GetDim(0) != fBaseParams.b) {
        std::string reasonMsg =
            "When sparseMode is 5, the shape of prefixOptional must be [" + std::to_string(fBaseParams.b) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "prefixOptional",
                                               Ops::Base::ToString(prefixShape).c_str(), reasonMsg.c_str());
        return false;
    }

    std::vector<int64_t> prefixN;
    const int64_t *value = prefixNTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "prefixOptional");
        return false;
    }
    const int64_t shapeSize = prefixNTensor->GetShapeSize();
    for (int64_t i = 0; i < shapeSize; i++) {
        prefixN.push_back(value[i]);
    }

    if (static_cast<int64_t>(prefixN.size()) == fBaseParams.b && prefixN.size() <= BATCH_MAX_SIZE) {
        std::copy(prefixN.begin(), prefixN.end(), fBaseParams.prefixN);
        return true;
    } else {
        std::string reasonMsg = "When sparseMode is 5, the shape size of prefixOptional should be equal to " +
                                std::to_string(fBaseParams.b) + " and cannot be greater than 2048";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("FlashAttentionScoreGrad", "prefixOptional",
                                                  std::to_string(prefixN.size()).c_str(), reasonMsg.c_str());
        return false;
    }
}

} // namespace fag
} // namespace optiling
