/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_token_permute_with_ep_grad_tiling.cpp
 * \brief
 */

#include "moe_token_permute_with_ep_grad_tiling.h"

namespace optiling {

const static int64_t UNPERMUTE_WITH_EP_INPUT_TOKENS = 0;
const static int64_t UNPERMUTE_WITH_EP_INPUT_IDX = 1;
const static int64_t UNPERMUTE_WITH_EP_INPUT_PROBS = 2;
const static int64_t UNPERMUTE_WITH_EP_OUTPUT_TOKENS = 0;
const static int64_t UNPERMUTE_WITH_EP_ARRT_TOPK = 0;
const static int64_t UNPERMUTE_WITH_EP_ARRT_RANGE = 1;
const static int64_t RANGE_SIZE = 2;


ge::graphStatus TilingMoeTokenPermuteWithEpGrad(gert::TilingContext* context);

ge::graphStatus TilingMoeTokenPermuteWithEpGrad(gert::TilingContext* context)
{
    return PermuteWithEpGradTilingCompute(context, -1, true);
}

static inline int64_t AlignN(const int64_t x, const int64_t N)
{
    return (x + N - 1) & ~(N - 1);
}

static inline int64_t GetLengthByType(const int32_t dtype)
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

static inline int64_t safeMod(const int64_t a, const int64_t b)
{
    return b == 0 ? 0 : a % b;
}

static inline int64_t safeDiv(const int64_t a, const int64_t b)
{
    return b == 0 ? 0 : a / b;
}

static inline bool isFloatDtype(const int64_t inputDtypeSize)
{
    return inputDtypeSize == GetLengthByType(ge::DT_FLOAT);
}

/*
 * 计算hidden_size=1所需要的Btye
 */
static inline int64_t ComputeUnitHSpace(const int64_t inputDtypeSize, const int64_t bufferNum)
{
    int64_t castNum = isFloatDtype(inputDtypeSize) ? 0 : CAST_NUM;
    return inputDtypeSize * (QUE_NUM + bufferNum - 1) + FLOAT_DATA_SIZE * castNum;
}

static inline int64_t ComputeMaxHiddenSize(MoeTokenUnpermuteWithEpParam& param, int64_t bufferNum)
{
    // sorted_indices和probs的预留空间；topK_num为最大值512时，至少需要5120 Btye。
    const int64_t reserveSpace = 5120;
    int64_t maxHiddenSize =
        safeDiv((param.core.maxCoreMemery - reserveSpace), ComputeUnitHSpace(param.input.tokensDtypeSize, bufferNum));

    return AlignN(maxHiddenSize - ALIGN_512, ALIGN_512);
}

static inline ge::graphStatus MoeTokenUnpermuteWithEpInputParamCheck(
    const gert::TilingContext* context, const bool isUnpermute)
{
    const gert::StorageShape* tokensShape = context->GetInputShape(0);
    const gert::StorageShape* indicesShape = context->GetInputShape(1);
    const gert::StorageShape* probsShape = context->GetInputShape(2);
    auto dataTensor0 = context->GetInputTensor(0);
    auto dataTensor1 = context->GetInputTensor(1);
    auto nodeName = context->GetNodeName();
    const int* tmpTopK = context->GetAttrs()->GetAttrPointer<int>(0);
    int inputTopK = static_cast<int64_t>(*tmpTopK);

    OP_CHECK_IF(
        inputTopK < 1, OP_LOGE(nodeName, "input num TopK cannot less than 1."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tokensShape == nullptr || indicesShape == nullptr || dataTensor0 == nullptr || dataTensor1 == nullptr,
        OP_LOGE(nodeName, "permutedTokens or sortedIndices is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tokensShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(nodeName, "permutedTokens's shape is not 2D."), return ge::GRAPH_FAILED);

    if (probsShape != nullptr) {
        int64_t probsDimNum = probsShape->GetStorageShape().GetDimNum();
        if (isUnpermute) {
            OP_CHECK_IF(
                probsDimNum != 2,
                OP_LOGE(nodeName, "probs's shape(%ld) is not 2D.", probsDimNum),
                return ge::GRAPH_FAILED);

            int64_t tokensNum = probsShape->GetStorageShape().GetDim(0);
            int64_t topK = probsShape->GetStorageShape().GetDim(1);
            int64_t totalLength = indicesShape->GetStorageShape().GetDim(0);

            OP_CHECK_IF(
                topK != inputTopK,
                OP_LOGE(nodeName, "input topK(%d) can not match probs' dim(1).", inputTopK),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                totalLength != tokensNum * topK,
                OP_LOGE(
                    nodeName, "permutedTokens's dim(0) is not equal to probs's dim(0) * probs's dim(1)."),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                topK > 512, OP_LOGE(nodeName, "topK can not larger than 512."),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                probsDimNum != 1,
                OP_LOGE(nodeName, "permutedProbsGrad's shape(%ld) is not 1D.", probsDimNum),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

static inline void Init(
    const gert::TilingContext* context, const int64_t topK, MoeTokenUnpermuteWithEpParam& param, const bool isUnpermute)
{
    auto ascendPlaform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    param.core.maxCoreNum = static_cast<int64_t>(ascendPlaform.GetCoreNumAiv());
    uint64_t maxCoreMemery;
    ascendPlaform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxCoreMemery);
    param.core.maxCoreMemery = static_cast<int64_t>(maxCoreMemery);

    const gert::StorageShape* tokensShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_TOKENS);
    const gert::StorageShape* sortedIndicesShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_IDX);
    const gert::StorageShape* probsShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_PROBS);

    param.input.tokensDtypeSize =
        GetLengthByType(context->GetInputTensor(UNPERMUTE_WITH_EP_INPUT_TOKENS)->GetDataType());
    param.input.indicesDtypeSize = GetLengthByType(context->GetInputTensor(UNPERMUTE_WITH_EP_INPUT_IDX)->GetDataType());
    param.input.isUnpermute = isUnpermute;

    auto attrPtr = context->GetAttrs();
    const int* tmpTopK = attrPtr->GetAttrPointer<int>(UNPERMUTE_WITH_EP_ARRT_TOPK);
    int inputTopK = static_cast<int64_t>(*tmpTopK);
    auto rangePtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(UNPERMUTE_WITH_EP_ARRT_RANGE);
    if (rangePtr != nullptr) {
        OP_CHECK_IF(
            rangePtr->GetSize() != RANGE_SIZE,
            OP_LOGE(context->GetNodeName(), "the size of range only support 2"), return);
        const int64_t* rangeList = reinterpret_cast<const int64_t*>(rangePtr->GetData());
        param.input.start = rangeList[0];
        param.input.end = rangeList[1];
    } else {
        param.input.start = 0;
        param.input.end = 0;
    }

    param.input.numOutTokens = tokensShape->GetStorageShape().GetDim(0); // numOutTokens根据tokens第0维获取；
    param.input.hiddenSize = tokensShape->GetStorageShape().GetDim(1);
    param.input.totalLength = sortedIndicesShape->GetStorageShape().GetDim(
        0); // tokens可能存在numOutTokens，因此totalLength从sortedIndices获取。
    if (isUnpermute) {
        if (probsShape != nullptr) {
            auto dataTensor2 = context->GetInputTensor(2);
            param.input.probsDtypeSize = GetLengthByType(dataTensor2->GetDataType());
            param.input.haveProbs = true;
            param.input.tokensNum = probsShape->GetStorageShape().GetDim(0);
            param.input.topK = probsShape->GetStorageShape().GetDim(1);
            param.input.PermuteProbGradSize = 0;
        } else {
            param.input.topK = inputTopK;
            param.input.PermuteProbGradSize = 1;
            param.input.haveProbs = false;
            param.input.tokensNum = safeDiv(param.input.totalLength, param.input.topK);
        }
    } else {
        param.input.topK = (inputTopK == topK) ? topK : 1;
        param.input.PermuteProbGradSize = 1;
        param.input.haveProbs = (probsShape != nullptr) ? true : false;
        param.input.tokensNum = safeDiv(param.input.totalLength, param.input.topK);
    }
}

static void SetCoreNum(MoeTokenUnpermuteWithEpParam& param)
{
    if (param.input.tokensNum < param.core.maxCoreNum) {
        param.core.usedCoreNum = param.input.tokensNum;
    } else {
        param.core.usedCoreNum = param.core.maxCoreNum;
    }
}

static inline void TilingHiddenSize(MoeTokenUnpermuteWithEpParam& param)
{
    int64_t maxHiddenSize = ComputeMaxHiddenSize(param, MIN_BUFFER_NUM);
    if (AlignN(param.input.hiddenSize, ALIGN_512) <= maxHiddenSize) {
        param.hiddenTiling.length = param.input.hiddenSize;
        param.hiddenTiling.remain = 0;
        param.hiddenTiling.num = 1;
    } else {
        param.hiddenTiling.length = maxHiddenSize;
        param.hiddenTiling.remain = safeMod(param.input.hiddenSize, maxHiddenSize);
        param.hiddenTiling.num = safeDiv(param.input.hiddenSize, maxHiddenSize);
    }
}

static inline void SetBufferNum(MoeTokenUnpermuteWithEpParam& param)
{
    const int64_t maxBufferNum = 4;
    int64_t bufferNum = maxBufferNum;
    while (bufferNum > MIN_BUFFER_NUM && param.hiddenTiling.length > ComputeMaxHiddenSize(param, bufferNum)) {
        bufferNum--;
    }
    param.core.bufferNum = bufferNum;
}

static inline void ComputeRemainMemerySpace(MoeTokenUnpermuteWithEpParam& param)
{
    param.core.remainMemerySpace =
        param.core.maxCoreMemery - AlignN(param.hiddenTiling.length, ALIGN_512) *
                                       ComputeUnitHSpace(param.input.tokensDtypeSize, param.core.bufferNum);
    param.core.remainMemerySpace -= ALIGN_256;
}

static inline void TilingToken(MoeTokenUnpermuteWithEpParam& param)
{
    param.tokenPerCore.length = safeDiv(param.input.tokensNum, param.core.usedCoreNum);
    param.tokenPerCore.num = param.core.usedCoreNum;
    param.tokenPerCore.remain = safeMod(param.input.tokensNum, param.core.usedCoreNum);

    int64_t unitTokenSpace = param.input.indicesDtypeSize;
    if (param.input.haveProbs && param.input.isUnpermute) {
        unitTokenSpace += param.input.probsDtypeSize;
        if (!isFloatDtype(param.input.probsDtypeSize)) {
            unitTokenSpace += static_cast<int64_t>(sizeof(float));
        }
    }

    int64_t probIndiceSpace = param.tokenPerCore.length * param.input.topK * unitTokenSpace;

    if (param.core.remainMemerySpace >= probIndiceSpace) {
        param.tokenTiling.length = param.tokenPerCore.length;
        param.tokenTiling.remain = 0;
        param.tokenTiling.num = 1;
    } else {
        int64_t maxTokenSize = safeDiv(param.core.remainMemerySpace, (param.input.topK * unitTokenSpace));
        param.tokenTiling.length = maxTokenSize;
        param.tokenTiling.remain = safeMod(param.tokenPerCore.length, maxTokenSize);
        param.tokenTiling.num = safeDiv(param.tokenPerCore.length, maxTokenSize);
    }
}
/*
  tilingKey计算规则
    0表示probs为None，
    1 2 3 表示 DT_FLOAT DT_FLOAT16 DT_BF16
 */
static inline void SetTilingKey(const gert::TilingContext* context, const bool isUnpermute, MoeTokenUnpermuteWithEpParam& param) {
  if (param.input.haveProbs) {
    // 存在probs
    if (isUnpermute == true) {
      // 在unpermute的条件下
      int64_t dataTensor2Type = context->GetInputTensor(2)->GetDataType();
      if (dataTensor2Type == ge::DT_FLOAT) {
        param.core.tilingKey = TILINGKEY_FLOAT;
      } else if (dataTensor2Type == ge::DT_FLOAT16) {
        param.core.tilingKey = TILINGKEY_FLOAT16;
      } else if (dataTensor2Type == ge::DT_BF16) {
        param.core.tilingKey = TILINGKEY_BF16;
      } else {
        OP_LOGE(context->GetNodeName(), "the data type of probs is incorrect");
      }
    } else {
      // 在permute的条件下
      param.core.tilingKey = 1;
    }   
  } else {
    // 不存在probs
    param.core.tilingKey = 0;
  }
}

static inline void SetTilingData(gert::TilingContext* context, const MoeTokenUnpermuteWithEpParam& param)
{
    MoeTokenPermuteWithEpGradTilingData tilingData;
    tilingData.set_hidden_size(param.input.hiddenSize);
    tilingData.set_top_k(param.input.topK);
    tilingData.set_start(param.input.start);
    tilingData.set_end(param.input.end);
    tilingData.set_permuted_probs_grad_length(param.input.PermuteProbGradSize);
    tilingData.set_num_out_tokens(param.input.numOutTokens);
    tilingData.set_hidden_splited_length(param.hiddenTiling.length);
    tilingData.set_hidden_splited_num(param.hiddenTiling.num);
    tilingData.set_hidden_splited_remain(param.hiddenTiling.remain);
    tilingData.set_tokens_core_length(param.tokenPerCore.length);
    tilingData.set_tokens_core_remain(param.tokenPerCore.remain);
    tilingData.set_tokens_splited_length(param.tokenTiling.length);
    tilingData.set_tokens_splited_remain(param.tokenTiling.remain);
    tilingData.set_tokens_splited_num(param.tokenTiling.num);
    tilingData.set_buffer_num(param.core.bufferNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(param.core.usedCoreNum);
}

static inline void DebugPrint(const gert::TilingContext* context, const MoeTokenUnpermuteWithEpParam& param)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print MoeTokenUnpermuteWithEp tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "coreNum: %ld", param.core.usedCoreNum);
    OP_LOGD(nodeName, "maxCoreMemery: %ld", param.core.maxCoreMemery);
    OP_LOGD(nodeName, "maxCoreNum: %ld", param.core.maxCoreNum);
    OP_LOGD(nodeName, "remainMemerySpace: %ld", param.core.remainMemerySpace);
    OP_LOGD(nodeName, "tilingKey: %ld", param.core.tilingKey);
    OP_LOGD(nodeName, "bufferNum: %ld", param.core.bufferNum);

    OP_LOGD(nodeName, "totalLength: %ld", param.input.totalLength);
    OP_LOGD(nodeName, "tokensNum: %ld", param.input.tokensNum);
    OP_LOGD(nodeName, "topK: %ld", param.input.topK);
    OP_LOGD(nodeName, "hiddenSize: %ld", param.input.hiddenSize);
    OP_LOGD(nodeName, "haveProbs: %d", param.input.haveProbs);
    OP_LOGD(nodeName, "tokensDtypeSize: %ld", param.input.tokensDtypeSize);
    OP_LOGD(nodeName, "indicesDtypeSize: %ld", param.input.indicesDtypeSize);
    OP_LOGD(nodeName, "probsDtypeSize: %ld", param.input.probsDtypeSize);
    OP_LOGD(nodeName, "start: %ld", param.input.start);
    OP_LOGD(nodeName, "end: %ld", param.input.end);
    OP_LOGD(nodeName, "numOutTokens: %ld", param.input.numOutTokens);

    OP_LOGD(nodeName, "hiddenTilingParam.length: %ld", param.hiddenTiling.length);
    OP_LOGD(nodeName, "hiddenTilingParam.num: %ld", param.hiddenTiling.num);
    OP_LOGD(nodeName, "hiddenTilingParam.remain: %ld", param.hiddenTiling.remain);

    OP_LOGD(nodeName, "tokenTilingParam.length: %ld", param.tokenTiling.length);
    OP_LOGD(nodeName, "tokenTilingParam.num: %ld", param.tokenTiling.num);
    OP_LOGD(nodeName, "tokenTilingParam.remain: %ld", param.tokenTiling.remain);

    OP_LOGD(nodeName, "tokenPerCore.length: %ld", param.tokenPerCore.length);
    OP_LOGD(nodeName, "tokenPerCore.num: %ld", param.tokenPerCore.num);
    OP_LOGD(nodeName, "tokenPerCore.remain: %ld", param.tokenPerCore.remain);

    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print MoeTokenUnpermuteWithEp tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus PermuteWithEpGradTilingCompute(gert::TilingContext* context, const int64_t topK, const bool isUnpermute)
{
    MoeTokenUnpermuteWithEpParam param;

    if (MoeTokenUnpermuteWithEpInputParamCheck(context, isUnpermute) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    Init(context, topK, param, isUnpermute);
    SetCoreNum(param);
    TilingHiddenSize(param);
    SetBufferNum(param);
    ComputeRemainMemerySpace(param);
    TilingToken(param);
    SetTilingKey(context, isUnpermute, param);
    SetTilingData(context, param);
    DebugPrint(context, param);

    return context->SetTilingKey(param.core.tilingKey);
}

static ge::graphStatus Tiling4MoeTokenPermuteWithEpGrad(gert::TilingContext* context)
{
    const int* top_k = context->GetAttrs()->GetAttrPointer<int>(0);
    int64_t topk = static_cast<int64_t>(*top_k);
    return PermuteWithEpGradTilingCompute(context, topk, false);
}

static ge::graphStatus TilingPrepareForMoeTokenPermuteWithEpGrad(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct MoeTokenPermuteWithEpGradCompileInfo {
};

IMPL_OP_OPTILING(MoeTokenPermuteWithEpGrad)
    .Tiling(Tiling4MoeTokenPermuteWithEpGrad)
    .TilingParse<MoeTokenPermuteWithEpGradCompileInfo>(TilingPrepareForMoeTokenPermuteWithEpGrad);
} // namespace optiling