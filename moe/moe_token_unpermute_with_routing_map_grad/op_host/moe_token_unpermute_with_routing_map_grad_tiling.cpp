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
 * \file moe_token_unpermute_with_routing_map_grad_tiling.cpp
 * \brief
 */
#include <iostream>
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "moe_token_unpermute_with_routing_map_grad_tiling.h"

namespace optiling {

constexpr int64_t BLOCK_SIZE_32 = 32;
constexpr uint32_t FP32_BLOCK_NUM = 8;
constexpr uint32_t H_BUFFER_NUM = 8;
constexpr uint32_t BLOCK_SIZE_512 = 512;
constexpr uint32_t MAX_TOP_K = 512;
constexpr uint32_t REDUCESUM_ONEREPEAT_CALNUM = 64; // 256 / sizeof(float);
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t INDICES_RESERVE_MAX_NUM = 256;
constexpr int64_t H_LOOP_ALIGN_BUFFER_LENGTH = 512;
constexpr int64_t H_BUFFER_NUM_PAD_FALSE = 3;
constexpr int64_t H_FP32_BUFFER_NUM_PAD_FALSE = 3;
constexpr int64_t SIZE_OF_FLOAT = 4;
constexpr uint32_t MIN_SPILT_H_SIZE = 1;
constexpr size_t ATTR_DROP_AND_PAD_IDX = 0;
constexpr size_t INPUT_UNPERMUTEDOUTPUTD_IDX = 0;
constexpr size_t INPUT_SORTED_TWICE_IDX = 1;
constexpr size_t INPUT_SORTED_TWICE_INDICES_IDX = 2;
constexpr size_t INPUT_ROUTING_MAP_IDX = 3;
constexpr size_t INPUT_PERMUTED_TOKENS_IDX = 4;
constexpr size_t INPUT_PROB_IDX = 5;
constexpr size_t ATTR_PADDEDMODE_IDX = 0;
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr uint32_t UNPERMUTE_GRAD_DIM_NUM = 2;
constexpr uint32_t INDEX_DIM_NUM = 1;
constexpr size_t PROB_NONE_PERLOOP_NUM = 1;
constexpr size_t INDICES_FP32_BUFFER_NUM = 2;

template <typename T>
inline auto AlignUp(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T>
inline auto AlignDown(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : ((num) / (rnd) * (rnd)));
}

template <typename T>
inline auto CeilDiv(T num, T div) -> T
{
    return (((div) == 0) ? 0 : (((num) + (div)-1) / (div)));
}

inline int64_t GetDiv(int64_t num, int64_t div)
{
    return div == 0 ? 0 : num / div;
}

inline int64_t GetRem(int64_t num, int64_t div)
{
    return div == 0 ? 0 : num % div;
}

inline uint32_t GetLengthByType(int32_t dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
        case ge::DT_BF16:
            return sizeof(int16_t);
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return sizeof(int32_t);
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return sizeof(int64_t);
        default:
            return 0;
    }
}

inline int64_t GetGreatestDivisor(int64_t num, int64_t div)
{
    if (div == 0) {
        return -1;
    }
    while (num % div != 0) {
        div--;
        if (div == 0) {
            return -1;
        }
    }
    return div;
}

static void MoeTokenUnpermuteWithRoutingMapGradPrintParam(gert::TilingContext* context,
                                                          MoeTokenUnpermuteWithRoutingMapGradTilingData& tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName,
            ">>>>>>>>>>>>>>> Start to print MoeTokenUnpermuteWithRoutingMapGrad tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== tokensNum:                %ld", tiling.get_tokensNum());
    OP_LOGD(nodeName, "=== topK:                     %ld", tiling.get_topK());
    OP_LOGD(nodeName, "=== capacity:                 %ld", tiling.get_capacity());
    OP_LOGD(nodeName, "=== numExpert:                %ld", tiling.get_numExpert());
    OP_LOGD(nodeName, "=== hiddenSize:               %ld", tiling.get_hiddenSize());
    OP_LOGD(nodeName, "=== numOutTokens:             %ld", tiling.get_numOutTokens());
    OP_LOGD(nodeName, "=== formerCoreNum:            %ld", tiling.get_formerCoreNum());
    OP_LOGD(nodeName, "=== tailCoreNum:              %ld", tiling.get_tailCoreNum());
    OP_LOGD(nodeName, "=== tokenNumEachCore:         %ld", tiling.get_tokenNumEachCore());
    OP_LOGD(nodeName, "=== tokenNumTailCore:         %ld", tiling.get_tokenNumTailCore());
    OP_LOGD(nodeName, "=== rowIdMapEachCore:         %ld", tiling.get_rowIdMapEachCore());
    OP_LOGD(nodeName, "=== rowIdMapTailCore:         %ld", tiling.get_rowIdMapTailCore());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>>       in core information      <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== hiddenSizeAlign:          %ld", tiling.get_hiddenSizeAlign());
    OP_LOGD(nodeName, "=== hiddenSizeLoopTimes:      %ld", tiling.get_hiddenSizeLoopTimes());
    OP_LOGD(nodeName, "=== hiddenSizeLoopTimesAlign: %ld", tiling.get_hiddenSizeLoopTimesAlign());
    OP_LOGD(nodeName, "=== hiddenSizeTail:           %ld", tiling.get_hiddenSizeTail());
    OP_LOGD(nodeName, "=== inputReserveNum:          %ld", tiling.get_inputReserveNum());
    OP_LOGD(nodeName, "=== indicesReserveNum:        %ld", tiling.get_indicesReserveNum());
    OP_LOGD(nodeName, "=== indicesReserveNumAlign:   %ld", tiling.get_indicesReserveNumAlign());
    OP_LOGD(nodeName, "=== totalUbSize:              %ld", tiling.get_totalUbSize());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print MoeTokenUnpermuteWithRoutingMapGrad tiling data end <<<<<<<<<<<<<<<<");
}

static ge::graphStatus TilingForProbIsNone(const gert::TilingContext* context,
                                           MoeTokenUnpermuteWithRoutingMapGradTilingData& tiling, bool paddedMode)
{
    // 核间切分策略
    int64_t numOutTokens = tiling.get_numOutTokens();
    int64_t totalCoreNum = static_cast<int64_t>(context->GetBlockDim());
    int64_t rowIdMapTailCore = GetDiv(numOutTokens, totalCoreNum);
    int64_t formerCoreNum = GetRem(numOutTokens, totalCoreNum);
    int64_t rowIdMapEachCore = formerCoreNum == 0 ? rowIdMapTailCore : rowIdMapTailCore + 1;
    int64_t tailCoreNum = totalCoreNum - formerCoreNum;
    tiling.set_formerCoreNum(formerCoreNum);
    tiling.set_tailCoreNum(tailCoreNum);
    tiling.set_rowIdMapEachCore(rowIdMapEachCore);
    tiling.set_rowIdMapTailCore(rowIdMapTailCore);
    if (!paddedMode) {
        tiling.set_topK(GetDiv(numOutTokens, tiling.get_tokensNum()));
    }
    // 核内切分策略
    int64_t hiddenSize = tiling.get_hiddenSize();
    auto tokensDtype = context->GetInputDesc(INPUT_UNPERMUTEDOUTPUTD_IDX)->GetDataType();
    int64_t inputTypeLength = GetLengthByType(tokensDtype);
    if (inputTypeLength == 0) {
        OP_LOGE(context->GetNodeName(), "The Dtype size of unpermuted_tokens_grad is 0.");
        return ge::GRAPH_FAILED;
    }
    int64_t inputBlockAlignEleNum = BLOCK_SIZE_32 / inputTypeLength;
    int64_t totalUbSize = tiling.get_totalUbSize();
    int64_t hiddenSizeAlign = AlignUp<int64_t>(hiddenSize, inputBlockAlignEleNum);
    int64_t hiddenSizeTmpMax = totalUbSize / (BUFFER_NUM * inputTypeLength);

    if (hiddenSizeTmpMax < hiddenSizeAlign) {
        hiddenSizeAlign = AlignDown<int64_t>(hiddenSizeTmpMax, inputBlockAlignEleNum);
    }
    int64_t hiddenSizeLoopTimes = CeilDiv<int64_t>(hiddenSize, hiddenSizeAlign);
    int64_t hiddenSizeTail = hiddenSize - (hiddenSizeLoopTimes - 1) * hiddenSizeAlign;
    tiling.set_hiddenSizeAlign(hiddenSizeAlign);
    tiling.set_hiddenSizeLoopTimes(hiddenSizeLoopTimes);
    tiling.set_hiddenSizeTail(hiddenSizeTail);
    tiling.set_inputReserveNum(PROB_NONE_PERLOOP_NUM);
    tiling.set_indicesReserveNum(PROB_NONE_PERLOOP_NUM); // indicesNumPerLoop
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForProbNotNonePadTrue(const gert::TilingContext* context,
                                                   MoeTokenUnpermuteWithRoutingMapGradTilingData& tiling)
{
    // 核间切分策略
    int64_t numOutTokens = tiling.get_numOutTokens();
    int64_t totalCoreNum = static_cast<int64_t>(context->GetBlockDim());
    int64_t rowIdMapTailCore = GetDiv(numOutTokens, totalCoreNum);
    int64_t formerCoreNum = GetRem(numOutTokens, totalCoreNum);
    int64_t rowIdMapEachCore = formerCoreNum == 0 ? rowIdMapTailCore : rowIdMapTailCore + 1;
    int64_t tailCoreNum = totalCoreNum - formerCoreNum;
    int64_t capacity = GetDiv(numOutTokens, tiling.get_numExpert());
    int64_t tokensNum = tiling.get_tokensNum();
    OP_CHECK_IF(capacity > tokensNum,
                OP_LOGE(context->GetNodeName(),
                        "Capacity should be smaller than tokens_num, but now capacity is %ld, tokens_num is %ld.",
                        capacity, tokensNum),
                return ge::GRAPH_FAILED);
    tiling.set_formerCoreNum(formerCoreNum);
    tiling.set_tailCoreNum(tailCoreNum);
    tiling.set_rowIdMapEachCore(rowIdMapEachCore);
    tiling.set_rowIdMapTailCore(rowIdMapTailCore);
    tiling.set_capacity(capacity);

    // 核内切分策略
    int64_t hiddenSize = tiling.get_hiddenSize();
    auto tokensDtype = context->GetInputDesc(INPUT_UNPERMUTEDOUTPUTD_IDX)->GetDataType();
    int64_t inputTypeLength = GetLengthByType(tokensDtype);
    if (inputTypeLength == 0) {
        OP_LOGE(context->GetNodeName(), "The Dtype size of unpermuted_tokens_grad is 0.");
        return ge::GRAPH_FAILED;
    }
    int64_t inputBlockAlignEleNum = BLOCK_SIZE_32 / inputTypeLength;
    int64_t totalUbSize = tiling.get_totalUbSize();
    int64_t hiddenSizeAlign = AlignUp<int64_t>(hiddenSize, inputBlockAlignEleNum);
    int64_t hiddenSizeOneLoopMax = (totalUbSize - BLOCK_SIZE_32 - H_LOOP_ALIGN_BUFFER_LENGTH * SIZE_OF_FLOAT) /
                                   (BUFFER_NUM * (H_BUFFER_NUM + inputTypeLength));
    int64_t hiddenSizeMax = hiddenSizeOneLoopMax * H_LOOP_ALIGN_BUFFER_LENGTH;
    OP_CHECK_IF(
        hiddenSize > hiddenSizeMax,
        OP_LOGE(context->GetNodeName(), "The maximum hidden_size supported is %d, but got %d, which is too large.",
                hiddenSizeMax, hiddenSize),
        return ge::GRAPH_FAILED);
    if (hiddenSizeOneLoopMax < hiddenSizeAlign) { // hiddensize需要切分
        hiddenSizeAlign = AlignDown<int64_t>(hiddenSizeOneLoopMax, inputBlockAlignEleNum);
    }
    int64_t hiddenSizeLoopTimes = CeilDiv(hiddenSize, hiddenSizeAlign);
    int64_t hiddenSizeLoopTimesAlign = AlignUp<int64_t>(hiddenSizeLoopTimes, FP32_BLOCK_NUM);
    int64_t hiddenSizeTail = hiddenSize - (hiddenSizeLoopTimes - 1) * hiddenSizeAlign;
    tiling.set_hiddenSizeAlign(hiddenSizeAlign);
    tiling.set_hiddenSizeLoopTimes(hiddenSizeLoopTimes);
    tiling.set_hiddenSizeLoopTimesAlign(hiddenSizeLoopTimesAlign);
    tiling.set_hiddenSizeTail(hiddenSizeTail);
    tiling.set_inputReserveNum(PROB_NONE_PERLOOP_NUM);
    tiling.set_indicesReserveNum(PROB_NONE_PERLOOP_NUM); // indicesNumPerLoop
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForProbNotNonePadFalse(const gert::TilingContext* context,
                                                    MoeTokenUnpermuteWithRoutingMapGradTilingData& tiling)
{
    int64_t numOutTokens = tiling.get_numOutTokens();
    int64_t tokensNum = tiling.get_tokensNum();
    if (tokensNum == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t topK = GetDiv(numOutTokens, tokensNum);
    OP_CHECK_IF(topK > MAX_TOP_K,
                OP_LOGE(context->GetNodeName(), "The topK_num needs to be smaller than 512, but now is %ld.", topK),
                return ge::GRAPH_FAILED);
    int64_t numExpert = tiling.get_numExpert();
    OP_CHECK_IF(
        topK > numExpert,
        OP_LOGE(context->GetNodeName(),
                "The topK_num needs to be smaller than experts_num, but now topK_num is %ld, experts_num is %ld.", topK,
                numExpert),
        return ge::GRAPH_FAILED);
    int64_t totalCoreNum = static_cast<int64_t>(context->GetBlockDim());
    int64_t tokenNumTailCore = GetDiv(tokensNum, totalCoreNum);
    int64_t formerCoreNum = GetRem(tokensNum, totalCoreNum);
    int64_t tokenNumEachCore = formerCoreNum == 0 ? tokenNumTailCore : tokenNumTailCore + 1;
    int64_t tailCoreNum = totalCoreNum - formerCoreNum;
    int64_t rowIdMapEachCore = tokenNumEachCore * topK;
    int64_t rowIdMapTailCore = tokenNumTailCore * topK;
    tiling.set_topK(topK);
    tiling.set_formerCoreNum(formerCoreNum);
    tiling.set_tailCoreNum(tailCoreNum);
    tiling.set_tokenNumEachCore(tokenNumEachCore);
    tiling.set_tokenNumTailCore(tokenNumTailCore);
    tiling.set_rowIdMapEachCore(rowIdMapEachCore);
    tiling.set_rowIdMapTailCore(rowIdMapTailCore);
    int64_t hiddenSize = tiling.get_hiddenSize();
    int64_t inputTypeLength = GetLengthByType(context->GetInputDesc(INPUT_UNPERMUTEDOUTPUTD_IDX)->GetDataType());
    if (inputTypeLength == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t inputBlockAlignEleNum = BLOCK_SIZE_32 / inputTypeLength;
    int64_t hiddenSizeAlign = AlignUp<int64_t>(hiddenSize, inputBlockAlignEleNum);
    int64_t indicesReserveNumMax = rowIdMapEachCore;
    int64_t indicesReserveNum = topK <= INDICES_RESERVE_MAX_NUM ?
                                    std::min(indicesReserveNumMax, AlignDown<int64_t>(INDICES_RESERVE_MAX_NUM, topK)) :
                                    topK;
    int64_t indicesReserveNumAlign = AlignUp<int64_t>(indicesReserveNum, inputBlockAlignEleNum);
    int64_t numExpertAlign = AlignUp<int64_t>(numExpert, BLOCK_SIZE_32);
    int64_t probTypeLength = GetLengthByType(context->GetInputDesc(INPUT_PROB_IDX)->GetDataType());
    int64_t remainSize = tiling.get_totalUbSize() - inputTypeLength * INDICES_RESERVE_MAX_NUM -
                         INDICES_RESERVE_MAX_NUM * SIZE_OF_FLOAT * INDICES_FP32_BUFFER_NUM;
    int64_t numExpertSize = (1 + probTypeLength) * numExpertAlign;
    int64_t hiddenNum =
        H_BUFFER_NUM_PAD_FALSE * BUFFER_NUM * inputTypeLength + H_FP32_BUFFER_NUM_PAD_FALSE * SIZE_OF_FLOAT;

    int64_t numExpertMax = (remainSize - hiddenNum) / (1 + probTypeLength);
    OP_CHECK_IF(
        numExpert > numExpertMax,
        OP_LOGE(context->GetNodeName(), "The maximum experts_num supported is %d, but got %d, which is too large.",
                numExpertMax, numExpert),
        return ge::GRAPH_FAILED);
    int64_t hiddenSizeTmpMax = (remainSize - numExpertSize) / hiddenNum;
    hiddenSizeAlign = hiddenSizeTmpMax < hiddenSizeAlign ? AlignDown<int64_t>(hiddenSizeTmpMax, inputBlockAlignEleNum) :
                                                           hiddenSizeAlign;
    int64_t hiddenSizeLoopTimes = CeilDiv(hiddenSize, hiddenSizeAlign);
    int64_t hiddenSizeTail = hiddenSize - (hiddenSizeLoopTimes - 1) * hiddenSizeAlign;
    tiling.set_hiddenSizeAlign(hiddenSizeAlign);
    tiling.set_hiddenSizeLoopTimes(hiddenSizeLoopTimes);
    tiling.set_hiddenSizeTail(hiddenSizeTail);
    tiling.set_indicesReserveNum(indicesReserveNum);
    tiling.set_indicesReserveNumAlign(indicesReserveNumAlign);
    tiling.set_numExpertAlign(numExpertAlign);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PreInitForTiling(gert::TilingContext* context,
                                        MoeTokenUnpermuteWithRoutingMapGradTilingData& tiling)
{
    const gert::StorageShape* unpermutedOutputDShape = context->GetInputShape(INPUT_UNPERMUTEDOUTPUTD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, unpermutedOutputDShape);
    auto unpermutedOutputDDimNum = unpermutedOutputDShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        unpermutedOutputDDimNum != UNPERMUTE_GRAD_DIM_NUM,
        OP_LOGE(context->GetNodeName(),
                "Check unpermutedTokensGrad shape failed, the dims of unpermutedTokensGrad should be %u, but get %zu.",
                UNPERMUTE_GRAD_DIM_NUM, unpermutedOutputDDimNum),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* outIndexShape = context->GetInputShape(INPUT_SORTED_TWICE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outIndexShape);
    auto outIndexDimNum = outIndexShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        outIndexDimNum != INDEX_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "Check outIndex shape failed, the dims of outIndex should be %u, but get %zu.",
                INDEX_DIM_NUM, outIndexDimNum),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* permuteTokenIdShape = context->GetInputShape(INPUT_SORTED_TWICE_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, permuteTokenIdShape);
    auto permuteTokenIdDimNum = permuteTokenIdShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(permuteTokenIdDimNum != INDEX_DIM_NUM,
                OP_LOGE(context->GetNodeName(),
                        "Check permuteTokenId shape failed, the dims of permuteTokenId should be %u, but get %zu.",
                        INDEX_DIM_NUM, permuteTokenIdDimNum),
                return ge::GRAPH_FAILED);
    int64_t tokensNum = unpermutedOutputDShape->GetStorageShape().GetDim(DIM_0);
    int64_t hiddenSize = unpermutedOutputDShape->GetStorageShape().GetDim(DIM_1);
    int64_t numOutTokens = outIndexShape->GetStorageShape().GetDim(DIM_0);
    int64_t permuteTokenIdNums = permuteTokenIdShape->GetStorageShape().GetDim(DIM_0);
    OP_CHECK_IF(
        numOutTokens != permuteTokenIdNums,
        OP_LOGE(
            context->GetNodeName(),
            "The dim 0 length of outIndex and permuteTokenId should be equal, but dim 0 length of outIndex is %ld, dim "
            "0 length of permuteTokenId is %ld.",
            numOutTokens, permuteTokenIdNums),
        return ge::GRAPH_FAILED);
    tiling.set_tokensNum(tokensNum);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_numOutTokens(numOutTokens);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputForProbIsNotNone(gert::TilingContext* context,
                                                  MoeTokenUnpermuteWithRoutingMapGradTilingData& tiling)
{
    const gert::StorageShape* routingMapShape = context->GetInputShape(INPUT_ROUTING_MAP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, routingMapShape);
    auto routingMapDimNum = routingMapShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        routingMapDimNum != UNPERMUTE_GRAD_DIM_NUM,
        OP_LOGE(context->GetNodeName(),
                "Check routingMapOptional shape failed, the dims of routingMapOptional should be %u, but get %zu.",
                UNPERMUTE_GRAD_DIM_NUM, routingMapDimNum),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* permutedTokensMapShape = context->GetInputShape(INPUT_PERMUTED_TOKENS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, permutedTokensMapShape);
    auto permutedTokensDimNum = permutedTokensMapShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        permutedTokensDimNum != UNPERMUTE_GRAD_DIM_NUM,
        OP_LOGE(
            context->GetNodeName(),
            "Check permutedTokensOptional shape failed, the dims of permutedTokensOptional should be %u, but get %zu.",
            UNPERMUTE_GRAD_DIM_NUM, permutedTokensDimNum),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* probsShape = context->GetInputShape(INPUT_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, probsShape);
    auto probsDimNum = probsShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(probsDimNum != UNPERMUTE_GRAD_DIM_NUM,
                OP_LOGE(context->GetNodeName(),
                        "Check probsOptional shape failed, the dims of probsOptional should be %u, but get %zu.",
                        UNPERMUTE_GRAD_DIM_NUM, probsDimNum),
                return ge::GRAPH_FAILED);
    int64_t numExpert = probsShape->GetStorageShape().GetDim(DIM_1);
    tiling.set_numExpert(numExpert);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4MoeTokenUnpermuteWithRoutingMapGrad(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "MoeTokenUnpermuteWithRoutingMapGradTiling tiling start");
    auto tokenDtype = context->GetInputDesc(INPUT_UNPERMUTEDOUTPUTD_IDX)->GetDataType();
    auto probTensor = context->GetOptionalInputTensor(INPUT_PROB_IDX);
    auto probDtype = (probTensor == nullptr) ? tokenDtype : context->GetInputDesc(INPUT_PROB_IDX)->GetDataType();
    OP_CHECK_IF(tokenDtype != probDtype && (tokenDtype != ge::DataType::DT_BF16 || probDtype != ge::DataType::DT_FLOAT),
                OP_LOGE(context->GetNodeName(),
                        "The input probs data type only supports being consistent with the input "
                        "unpermuted_tokens_grad, or when unpermuted_tokens_grad is BFLOAT16, probs can be FLOAT."),
                return ge::GRAPH_FAILED);
    MoeTokenUnpermuteWithRoutingMapGradTilingData tiling;
    OP_CHECK_IF(ge::GRAPH_SUCCESS != PreInitForTiling(context, tiling),
                OP_LOGE(context->GetNodeName(), "The input shape is invalid."), return ge::GRAPH_FAILED);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(totalCoreNum);
    uint64_t totalUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);
    int64_t ubSize = static_cast<int64_t>(totalUbSize);
    tiling.set_totalUbSize(ubSize);

    auto attrPtr = context->GetAttrs();
    const bool* paddedModePtr = attrPtr->GetAttrPointer<bool>(ATTR_PADDEDMODE_IDX);
    bool paddedMode = *paddedModePtr;
    OP_LOGD(context->GetNodeName(), ">>> [MoeTokenUnpermuteWithRoutingMapGradTiling] paddedMode: %d", paddedMode);
    if (probTensor == nullptr) {
        OP_CHECK_IF(ge::GRAPH_SUCCESS != TilingForProbIsNone(context, tiling, paddedMode),
                    OP_LOGE(context->GetNodeName(), "Tiling( Prob is None ) failed."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckInputForProbIsNotNone(context, tiling),
                    OP_LOGE(context->GetNodeName(), "The input shape is invalid."), return ge::GRAPH_FAILED);
        if (paddedMode) {
            OP_CHECK_IF(ge::GRAPH_SUCCESS != TilingForProbNotNonePadTrue(context, tiling),
                        OP_LOGE(context->GetNodeName(), "Tiling( Prob Not None and Pad Mode True ) failed."),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(ge::GRAPH_SUCCESS != TilingForProbNotNonePadFalse(context, tiling),
                        OP_LOGE(context->GetNodeName(), "Tiling( Prob Not None and Pad Mode False ) failed."),
                        return ge::GRAPH_FAILED);
        }
    }
    OP_LOGD(context->GetNodeName(),
            "MoeTokenUnpermuteWithRoutingMapGradTiling MoeTokenUnpermuteWithRoutingMapGradCoreSplitInfo finished");
    MoeTokenUnpermuteWithRoutingMapGradPrintParam(context, tiling);

    uint32_t paddedModeKey = paddedMode ? 1 : 0;
    uint32_t probKey = (probTensor == nullptr) ? 0 : 1;
    uint32_t mixKey = 0;
    if (probDtype != tokenDtype) {
        mixKey = 1;
    }
    uint32_t tilingKey = mixKey * 100 + paddedModeKey * 10 + probKey;
    // 00: padded_mode = False, 不存在prob  01: padded_mode = False, 存在prob
    // 10: padded_mode = True, 不存在prob   11: padded_mode = True, 存在prob
    context->SetTilingKey(tilingKey);
    context->SetScheduleMode(1);
    OP_LOGD(context->GetNodeName(), ">>> [MoeTokenUnpermuteWithRoutingMapGradTiling] tilingKey: %d", tilingKey);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    OP_LOGD(context->GetNodeName(), "MoeTokenUnpermuteWithRoutingMapGradTiling tiling end");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4MoeTokenUnpermuteWithRoutingMapGrad(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTokenUnpermuteWithRoutingMapGrad)
    .Tiling(Tiling4MoeTokenUnpermuteWithRoutingMapGrad)
    .TilingParse<MoeTokenUnpermuteWithRoutingMapGradCompileInfo>(TilingPrepare4MoeTokenUnpermuteWithRoutingMapGrad);

} // namespace optiling
