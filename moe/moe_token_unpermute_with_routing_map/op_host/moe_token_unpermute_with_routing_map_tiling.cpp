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
 * \file moe_token_unpermute_with_routing_map.cc
 * \brief
 */
#include "moe_token_unpermute_with_routing_map_tiling.h"

namespace optiling {
namespace moe_token_unpermute_with_routing_map {
constexpr int64_t FLOAT_DATA_SIZE = 4;
constexpr int64_t MIN_BUFFER_NUM = 2;
constexpr int64_t ALIGN_512 = 512;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t QUE_NUM = 2;
constexpr int64_t CAST_NUM = 2;
constexpr int64_t TILINGKEY_PROBS = 1;
constexpr static uint64_t DOUBLE_BUFFER = 2;
constexpr static uint64_t IO_QUE = 2;
constexpr static uint64_t MASK_ONE_DATA_SIZE = 7;   // doublebufer 2* 1(int8)+4(int32)+1(int8)
constexpr static uint64_t INDEX_ONE_DATA_SIZE = 12; // doublebufer 2 *4(int32)
const static int64_t ONE_BLOCK_BYTE = 32;
const static int64_t NUM_TWO = 2;
constexpr static uint64_t BLOCK_SIZE = 256;

struct CoreParam {
    int64_t maxCoreMemery = 0;
    int64_t maxCoreNum = 0;
    int64_t usedCoreNum = 0;
    int64_t remainMemerySpace = 0;
    int64_t bufferNum = 0;
    int64_t tilingKey = 0;
    int64_t maxUsedCoreNum = 0;
};

struct InputParam {
    int64_t tokensNum = 0;
    int64_t topK = 0;
    int64_t hiddenSize = 0;
    int64_t totalLength = 0;
    int64_t numOutTokens = 0;
    int64_t tokensDtypeSize = 0;
    int64_t numExperts = 1;
    int64_t indicesDtypeSize = 0;
    int64_t probsDtypeSize = 0;
    bool haveProbs = false;
};

struct TilingParam {
    int64_t length = 0;
    int64_t num = 0;
    int64_t remain = 0;
};

struct MoeTokenUnpermuteWithRoutingMapParam {
    InputParam input;
    TilingParam hiddenTiling;
    TilingParam tokenTiling;
    TilingParam tokenPerCore;
    CoreParam core;
};

constexpr uint64_t INPUT_SORTED_INDICES_INDEX = 1;
constexpr uint64_t INPUT_PROBS_OPTIONAL_INDEX = 3;

constexpr uint64_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint64_t FP32_INT32_DTYPE_SIZE = 4;

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

static inline int64_t safeDiv(const int64_t a, const int64_t b)
{
    return b == 0 ? 0 : a / b;
}

static inline int64_t safeMod(const int64_t a, const int64_t b)
{
    return b == 0 ? 0 : a % b;
}

static inline int64_t AlignN(const int64_t x, const int64_t N)
{
    return (x + N - 1) & ~(N - 1);
}

static inline bool isFloatDtype(const int64_t inputDtypeSize)
{
    return inputDtypeSize == GetLengthByType(ge::DT_FLOAT);
}

int64_t GetDivRem(const int64_t value1, const int64_t value2)
{
    if (value2 == 0) {
        return value1;
    }
    return value1 % value2;
}

int64_t GetCeilInt(const int64_t value1, const int64_t value2)
{
    if (value2 == 0) {
        return value2;
    }
    return (value1 + value2 - 1) / value2;
}

int64_t GetDiv(const int64_t value1, const int64_t value2)
{
    if (value2 == 0) {
        return value2;
    }
    return value1 / value2;
}

int64_t GetAlign(const int64_t value1, const int64_t value2)
{
    if (value2 == 0) {
        return value2;
    }
    return value1 / value2 * value2;
}

int64_t GetCeilAlign(const int64_t value1, const int64_t value2)
{
    if (value2 == 0) {
        return value2;
    }
    return (value1 + value2 - 1) / value2 * value2;
}

static void InitSplitInfoPad(gert::TilingContext *context, TilingPadParams &param, uint64_t dType)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t maxUbSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
    uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();

    uint64_t rowVal = param.num_experts * param.capacity;
    param.front_core = GetDivRem(rowVal, coreNum) != 0 ? GetDivRem(rowVal, coreNum) : coreNum;
    param.tail_core = rowVal <= coreNum ? 0 : coreNum - param.front_core;
    param.core_num_use = param.front_core + param.tail_core;
    param.num_tokens_each_front_core = GetCeilInt(rowVal, coreNum);
    param.num_tokens_each_tail_core = GetDiv(rowVal, coreNum);

    uint64_t dtypeSize = (dType == ge::DT_FLOAT) ? FP32_INT32_DTYPE_SIZE : FP16_BF16_DTYPE_SIZE;
    uint64_t allSize = dtypeSize;
    uint64_t maxNPerLoopForUb = maxUbSize / allSize;

    param.loop_time_each_front_core = GetCeilInt(param.num_tokens_each_front_core, maxNPerLoopForUb);
    param.num_tokens_front_core_each_loop =
        (param.num_tokens_each_front_core < maxNPerLoopForUb) ? param.num_tokens_each_front_core : maxNPerLoopForUb;
    param.num_tokens_front_core_last_loop = GetDivRem(param.num_tokens_each_front_core, maxNPerLoopForUb);

    param.loop_time_each_tail_core = GetCeilInt(param.num_tokens_each_tail_core, maxNPerLoopForUb);
    param.num_tokens_tail_core_each_loop =
        (param.num_tokens_each_tail_core < maxNPerLoopForUb) ? param.num_tokens_each_tail_core : maxNPerLoopForUb;
    param.num_tokens_tail_core_last_loop = GetDivRem(param.num_tokens_each_tail_core, maxNPerLoopForUb);
}

/*
 * 计算hidden_size=1所需要的Btye
 */

static inline int64_t ComputeUnitHSpace(const int64_t inputDtypeSize, const int64_t bufferNum)
{
    int64_t castNum = isFloatDtype(inputDtypeSize) ? 0 : CAST_NUM;
    return inputDtypeSize * (QUE_NUM + bufferNum - 1) + FLOAT_DATA_SIZE * castNum;
}

static inline int64_t ComputeMaxHiddenSize(MoeTokenUnpermuteWithRoutingMapParam &param, int64_t bufferNum)
{
    // sorted_indices和probs的预留空间；topK_num为最大值512时，至少需要5120 Btye。
    const int64_t reserveSpace = 5120;
    int64_t maxHiddenSize =
        safeDiv((param.core.maxCoreMemery - reserveSpace), ComputeUnitHSpace(param.input.tokensDtypeSize, bufferNum));

    return AlignN(maxHiddenSize - ALIGN_512, ALIGN_512);
}

static inline ge::graphStatus InputParamCheck(const gert::TilingContext *context)
{
    const gert::StorageShape *tokensShape = context->GetInputShape(0);
    const gert::StorageShape *indicesShape = context->GetInputShape(1);
    const gert::StorageShape *probsShape = context->GetOptionalInputShape(3);
    auto attrs = context->GetAttrs();
    auto restoreAttr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    auto attrData = reinterpret_cast<const int64_t *>(restoreAttr->GetData());
    int64_t restore_shape = attrData[0];
    auto dataTensor0 = context->GetInputTensor(0);
    auto dataTensor1 = context->GetInputTensor(1);
    auto nodeName = context->GetNodeName();

    OP_CHECK_IF(tokensShape == nullptr || indicesShape == nullptr || dataTensor0 == nullptr || dataTensor1 == nullptr,
                OP_LOGE(nodeName, "[MoeTokenUnpermuteWithRoutingMap] permutedTokens or sortedIndices is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokensShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE(nodeName, "[MoeTokenUnpermuteWithRoutingMap] permutedTokens's shape is not 2D."),
                return ge::GRAPH_FAILED);

    if (probsShape == nullptr) {
        int64_t topK = tokensShape->GetStorageShape().GetDim(0) / restore_shape;
        OP_CHECK_IF(topK > 512, OP_LOGE(nodeName, "[MoeTokenUnpermuteWithRoutingMap] topK can not larger than 512."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(probsShape->GetStorageShape().GetDimNum() != 2,
                    OP_LOGE(nodeName, "[MoeTokenUnpermuteWithRoutingMap] probs's shape is not 2D."),
                    return ge::GRAPH_FAILED);

        int64_t topK = tokensShape->GetStorageShape().GetDim(0) / probsShape->GetStorageShape().GetDim(0);

        OP_CHECK_IF(topK > 512, OP_LOGE(nodeName, "[MoeTokenUnpermuteWithRoutingMap] topK can not larger than 512."),
                    return ge::GRAPH_FAILED);

        if (topK > probsShape->GetStorageShape().GetDim(1)) {
            OP_LOGW(nodeName, "[MoeTokenUnpermuteWithRoutingMap] topK is larger than experts_num, which may cause "
                              "unexpected behavior.");
        }
    }
    return ge::GRAPH_SUCCESS;
}

static inline void Init(gert::TilingContext *context, MoeTokenUnpermuteWithRoutingMapParam &param)
{
    size_t sysWorkspaceSize = 16 * 1024 * 1024 * 2;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    auto ascendPlaform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    param.core.maxCoreNum = static_cast<int64_t>(ascendPlaform.GetCoreNumAiv());
    uint64_t maxCoreMemery;
    ascendPlaform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxCoreMemery);
    param.core.maxCoreMemery = static_cast<int64_t>(maxCoreMemery);

    const gert::StorageShape *tokensShape = context->GetInputShape(0);
    const gert::StorageShape *sortedIndicesShape = context->GetInputShape(1);
    const gert::StorageShape *probsShape = context->GetOptionalInputShape(3);
    auto attrs = context->GetAttrs();
    auto dataTensor0 = context->GetInputTensor(0);
    auto dataTensor1 = context->GetInputTensor(1);

    param.input.tokensDtypeSize = GetLengthByType(dataTensor0->GetDataType());
    param.input.indicesDtypeSize = GetLengthByType(dataTensor1->GetDataType());
    param.input.numOutTokens = tokensShape->GetStorageShape().GetDim(0); // numOutTokens根据tokens第0维获取；
    param.input.hiddenSize = tokensShape->GetStorageShape().GetDim(1);
    param.input.totalLength = sortedIndicesShape->GetStorageShape().GetDim(
        0); // tokens可能存在numOutTokens，因此totalLength从sortedIndices获取。
    if (probsShape != nullptr) {
        auto dataTensor2 = context->GetOptionalInputTensor(3);
        param.input.probsDtypeSize = GetLengthByType(dataTensor2->GetDataType());
        param.input.haveProbs = true;
        param.input.tokensNum = probsShape->GetStorageShape().GetDim(0);
        param.input.numExperts = probsShape->GetStorageShape().GetDim(1);
        param.input.topK = tokensShape->GetStorageShape().GetDim(0) / param.input.tokensNum;
    } else {
        auto restoreAttr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
        auto attrData = reinterpret_cast<const int64_t *>(restoreAttr->GetData());
        int64_t restore_shape = attrData[0];
        param.input.topK = tokensShape->GetStorageShape().GetDim(0) / restore_shape;
        param.input.tokensNum = safeDiv(param.input.totalLength, param.input.topK);
    }
}

static void SetCoreNum(MoeTokenUnpermuteWithRoutingMapParam &param)
{
    if (param.input.tokensNum < param.core.maxCoreNum) {
        param.core.usedCoreNum = param.input.tokensNum;
    } else {
        param.core.usedCoreNum = param.core.maxCoreNum;
    }
}

static inline void TilingHiddenSize(MoeTokenUnpermuteWithRoutingMapParam &param)
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

static inline void SetBufferNum(MoeTokenUnpermuteWithRoutingMapParam &param)
{
    const int64_t maxBufferNum = 4;
    int64_t bufferNum = maxBufferNum;
    while (bufferNum > MIN_BUFFER_NUM && param.hiddenTiling.length > ComputeMaxHiddenSize(param, bufferNum)) {
        bufferNum--;
    }
    param.core.bufferNum = bufferNum;
}

static inline void ComputeRemainMemerySpace(MoeTokenUnpermuteWithRoutingMapParam &param)
{
    param.core.remainMemerySpace =
        param.core.maxCoreMemery - AlignN(param.hiddenTiling.length, ALIGN_512) *
                                       ComputeUnitHSpace(param.input.tokensDtypeSize, param.core.bufferNum);
    param.core.remainMemerySpace -= ALIGN_256;
}

static inline void TilingToken(MoeTokenUnpermuteWithRoutingMapParam &param)
{
    param.tokenPerCore.length = safeDiv(param.input.tokensNum, param.core.usedCoreNum);
    param.tokenPerCore.num = param.core.usedCoreNum;
    param.tokenPerCore.remain = safeMod(param.input.tokensNum, param.core.usedCoreNum);

    int64_t unitTokenSpace = param.input.indicesDtypeSize;
    if (param.input.haveProbs) {
        unitTokenSpace += param.input.probsDtypeSize;
        if (!isFloatDtype(param.input.probsDtypeSize)) {
            unitTokenSpace += sizeof(float);
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

static inline void Tiling4MaskedSelect(MoeTokenUnpermuteWithRoutingMapTilingData &tilingData,
                                       MoeTokenUnpermuteWithRoutingMapParam &param)
{
    auto mstilingData = &tilingData.maskedSelectParamsOp;

    uint64_t aivUseNum = param.core.maxCoreNum; // Vector核数量
    uint64_t ubSize = param.core.maxCoreMemery; // ubSize大小
    uint64_t totalLength = param.input.tokensNum * param.input.numExperts;

    // ub对齐后长度
    uint64_t formerNum = 0;
    uint64_t formerLength = 0;
    uint64_t formerTileNum = 0;
    uint64_t formerTileLength = 0;
    uint64_t formerLastTileLength = 0;

    uint64_t tailNum = 0;
    uint64_t tailLength = 0;
    uint64_t tailTileNum = 0;
    uint64_t tailTileLength = 0;
    uint64_t tailLastTileLength = 0;

    uint64_t numBlocks = 0;

    // 求单个元素大小
    uint64_t sizeOfDataType = 1;
    sizeOfDataType = param.input.probsDtypeSize;

    // 一个block存放的元素
    uint32_t alignNum = BLOCK_SIZE / NUM_TWO; // 256/<8>=32
    // ub对齐后长度
    uint64_t oneDataSize = IO_QUE * DOUBLE_BUFFER * sizeOfDataType + MASK_ONE_DATA_SIZE + INDEX_ONE_DATA_SIZE;
    uint64_t ubLength = ((ubSize - ONE_BLOCK_BYTE * DOUBLE_BUFFER * DOUBLE_BUFFER) / oneDataSize) / alignNum * alignNum;
    ubLength = ubLength > static_cast<uint64_t>(param.input.tokensNum) ? param.input.tokensNum : ubLength;
    // ub能放的元素个数
    // 运行核数
    numBlocks = (static_cast<uint64_t>(param.input.numExperts) > aivUseNum) ? aivUseNum : param.input.numExperts;
    mstilingData->set_needCoreNum(numBlocks);
    // 切分流程
    formerNum = param.input.numExperts % numBlocks;
    if (formerNum == 0) {
        formerNum = numBlocks;
    }
    tailNum = numBlocks - formerNum;
    formerLength = (param.input.numExperts + numBlocks - 1) / numBlocks * param.input.tokensNum;
    formerTileNum = (formerLength + ubLength - 1) / ubLength;
    formerTileLength = ubLength;
    formerLastTileLength = formerLength % ubLength;
    if (formerLastTileLength == 0) {
        formerLastTileLength = ubLength;
    }
    if (tailNum > 0) {
        tailLength = (totalLength - formerLength * formerNum) / tailNum;
        tailTileNum = (tailLength + ubLength - 1) / ubLength;
        tailTileLength = ubLength;
        tailLastTileLength = tailLength % ubLength;
        if (tailLastTileLength == 0) {
            tailLastTileLength = ubLength;
        }
    }
    param.core.maxUsedCoreNum = std::max(param.core.usedCoreNum, static_cast<int64_t>(numBlocks));
    mstilingData->set_formerNum(formerNum);
    mstilingData->set_formerLength(formerLength);
    mstilingData->set_formertileNum(formerTileNum);
    mstilingData->set_formertileLength(formerTileLength);
    mstilingData->set_formerlasttileLength(formerLastTileLength);
    mstilingData->set_tokenNum(param.input.tokensNum);

    mstilingData->set_tailNum(tailNum);
    mstilingData->set_tailLength(tailLength);
    mstilingData->set_tailtileNum(tailTileNum);
    mstilingData->set_tailtileLength(tailTileLength);
    mstilingData->set_taillasttileLength(tailLastTileLength);
}

/*
  tilingKey计算规则
  第0位：
    0表示probs为None，1表示prob非None。
 */

static inline void SetTilingKey(gert::TilingContext *context, MoeTokenUnpermuteWithRoutingMapParam &param)
{
    if (param.input.haveProbs) {
        // 存在probs
        param.core.tilingKey += TILINGKEY_PROBS;
        context->SetScheduleMode(1);
    }
}

static inline void SetTilingData(gert::TilingContext *context, MoeTokenUnpermuteWithRoutingMapParam &param)
{
    MoeTokenUnpermuteWithRoutingMapTilingData tilingData;
    Tiling4MaskedSelect(tilingData, param);
    tilingData.set_hidden_size(param.input.hiddenSize);
    tilingData.set_used_core_num(param.core.usedCoreNum);
    tilingData.set_top_k(param.input.topK);
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
    context->SetBlockDim(param.core.maxUsedCoreNum);
}

static void InitPad(const gert::TilingContext *context, TilingPadParams &param)
{
    const gert::StorageShape *sortedIndicesShape = context->GetInputShape(INPUT_SORTED_INDICES_INDEX);
    const gert::StorageShape *probsOptionalShape = context->GetInputShape(INPUT_PROBS_OPTIONAL_INDEX);
    uint64_t num_experts_capacity = sortedIndicesShape->GetStorageShape().GetDim(0);
    param.num_tokens = probsOptionalShape->GetStorageShape().GetDim(0);
    param.num_experts = probsOptionalShape->GetStorageShape().GetDim(1);
    param.capacity = num_experts_capacity / param.num_experts;
}

static ge::graphStatus SetTilingKeyPad(gert::TilingContext *context, TilingPadParams &params, uint64_t dType)
{
    (void)params;
    (void)dType;
    uint64_t tilingKey = 1000;
    OP_LOGD(context->GetNodeName(), ">>> [MoeTokenUnpermuteWithRoutingMapPadTilingData] tilingKey: %ld", tilingKey);
    return context->SetTilingKey(tilingKey);
}

static inline void SetTilingDataPad(gert::TilingContext *context, TilingPadParams &param)
{
    MoeTokenUnpermuteWithRoutingMapTilingData tilingData;

    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_core_num_use(param.core_num_use);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_num_tokens(param.num_tokens);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_num_experts(param.num_experts);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_capacity(param.capacity);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_front_core(param.front_core);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_loop_time_each_front_core(
        param.loop_time_each_front_core);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_num_tokens_each_front_core(
        param.num_tokens_each_front_core);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_num_tokens_front_core_each_loop(
        param.num_tokens_front_core_each_loop);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_num_tokens_front_core_last_loop(
        param.num_tokens_front_core_last_loop);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_tail_core(param.tail_core);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_loop_time_each_tail_core(
        param.loop_time_each_tail_core);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_num_tokens_each_tail_core(
        param.num_tokens_each_tail_core);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_num_tokens_tail_core_each_loop(
        param.num_tokens_tail_core_each_loop);
    tilingData.moeTokenUnpermuteWithRoutingMapPadTilingData.set_num_tokens_tail_core_last_loop(
        param.num_tokens_tail_core_last_loop);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(param.core_num_use);

    uint64_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
}

static inline void PrintTilingDataPad(const gert::TilingContext *context, const TilingPadParams &param)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print MoeTokenUnpermuteWithRoutingMap tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "core_num_use: %ld", param.core_num_use);
    OP_LOGD(nodeName, "num_tokens: %ld", param.num_tokens);
    OP_LOGD(nodeName, "num_experts: %ld", param.num_experts);
    OP_LOGD(nodeName, "capacity: %ld", param.capacity);
    OP_LOGD(nodeName, "front_core: %ld", param.front_core);
    OP_LOGD(nodeName, "loop_time_each_front_core: %ld", param.loop_time_each_front_core);

    OP_LOGD(nodeName, "num_tokens_each_front_core: %ld", param.num_tokens_each_front_core);
    OP_LOGD(nodeName, "num_tokens_front_core_each_loop: %ld", param.num_tokens_front_core_each_loop);
    OP_LOGD(nodeName, "num_tokens_front_core_last_loop: %ld", param.num_tokens_front_core_last_loop);
    OP_LOGD(nodeName, "tail_core: %ld", param.tail_core);
    OP_LOGD(nodeName, "loop_time_each_tail_core: %ld", param.loop_time_each_tail_core);
    OP_LOGD(nodeName, "num_tokens_each_tail_core: %ld", param.num_tokens_each_tail_core);
    OP_LOGD(nodeName, "num_tokens_tail_core_each_loop: %ld", param.num_tokens_tail_core_each_loop);
    OP_LOGD(nodeName, "num_tokens_tail_core_last_loop: %ld", param.num_tokens_tail_core_last_loop);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print MoeTokenUnpermuteWithRoutingMap tiling data end <<<<<<<<<<<<<<<<");
}

static ge::graphStatus TilingMoeTokenUnpermuteWithRoutingMapPad(gert::TilingContext *context)
{
    TilingPadParams param;
    InitPad(context, param);
    if (param.capacity > param.num_tokens) {
        OP_LOGW(context->GetNodeName(), "[MoeTokenUnpermuteWithRoutingMap] capacity is larger than tokens_num, which "
                                        "may cause unexpected behavior.");
    }
    auto probDtype = context->GetInputDesc(INPUT_PROBS_OPTIONAL_INDEX)->GetDataType();
    InitSplitInfoPad(context, param, probDtype);
    SetTilingKeyPad(context, param, probDtype);
    SetTilingDataPad(context, param);
    PrintTilingDataPad(context, param);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingCompute(gert::TilingContext *context, MoeTokenUnpermuteWithRoutingMapParam &param)
{
    if (InputParamCheck(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    SetCoreNum(param);
    TilingHiddenSize(param);
    SetBufferNum(param);
    ComputeRemainMemerySpace(param);
    TilingToken(param);
    SetTilingKey(context, param);
    SetTilingData(context, param);

    return context->SetTilingKey(param.core.tilingKey);
}

static ge::graphStatus TilingMoeTokenUnpermuteWithRoutingMap(gert::TilingContext *context)
{
    MoeTokenUnpermuteWithRoutingMapParam param;
    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);
    const bool *pPadded_mode = attrPtr->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, pPadded_mode);
    auto padded_mode = *pPadded_mode;
    if (padded_mode == false) {
        Init(context, param);
        return moe_token_unpermute_with_routing_map::TilingCompute(context, param);
    } else {
        return moe_token_unpermute_with_routing_map::TilingMoeTokenUnpermuteWithRoutingMapPad(context);
    }
}

static ge::graphStatus TilingPrepareForMoeTokenUnpermuteWithRoutingMap(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct MoeTokenUnpermuteWithRoutingMapCompileInfo {};

IMPL_OP_OPTILING(MoeTokenUnpermuteWithRoutingMap)
    .Tiling(TilingMoeTokenUnpermuteWithRoutingMap)
    .TilingParse<MoeTokenUnpermuteWithRoutingMapCompileInfo>(TilingPrepareForMoeTokenUnpermuteWithRoutingMap);
} // namespace moe_token_unpermute_with_routing_map
} // namespace optiling
