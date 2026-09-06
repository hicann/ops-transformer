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
 * \file mega_moe_tiling.cpp
 * \brief
 */

#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_exception_dump.h"
#include "../mega_moe_tiling_host.h"
#include "../../../op_kernel/arch35/mega_moe_tiling.h"
#include "../../../op_kernel/arch35/mega_moe_tiling_key.h"
#include "../../../op_kernel/arch35/common/mega_moe_constants.h"
#include "../../../op_kernel/arch35/common/mega_moe_workspace.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
using namespace MegaMoeImpl;

namespace optiling {
namespace {
// SwiGLU 门控把 GMM1 的输出劈成 gate 和 up 两半，所以 weight1 的行数是 weight2 列数的两倍。
const static int64_t SWIGLU_GATE_UP_SPLIT_FACTOR = 2LL;
// MX 量化的 weight scale 每个专家末尾固定带 2 个 base 分量。
const static int64_t WEIGHT_SCALE_MULTI_BASE_DIM_SIZE = 2LL;
// 共享专家最多 4 个。
const static int64_t MAX_SHARED_EXPERT_NUM = 4LL;
const static int64_t UB_BLOCK_SIZE = 32LL;

const static int64_t FOUR_DIMS = 4LL;
const static int64_t THREE_DIMS = 3LL;
const static int64_t TWO_DIMS = 2LL;
const static int64_t ONE_DIM = 1LL;
const static int64_t MIN_TOPK = 1LL;
const static int64_t MAX_TOPK = 32LL;
const static int64_t MIN_EXPERT_PER_RANK = 1LL;
const static int64_t MAX_EXPERT_PER_RANK = 1024LL;
const static int64_t MIN_H = 1024LL;
const static int64_t MAX_H = 8LL * 1024LL; // 8K
const static int64_t H_ALIGN = 32LL;
const static int64_t W4_K_ALIGN = 64LL;
const static int64_t URMA_H_ALIGN = 1024LL;
const static int64_t MAX_HIDDEN_DIM = 8LL * 1024LL; // 8K
// hiddenDim 是 GMM1 含 gate/up 两路的完整输出宽度；MTE 与 URMA 共用支持尾 tile 的激活 epilogue。
// 256 对齐保证 gate/up 半宽按 128 对齐，下限 512 保证半宽至少覆盖一个完整 tile。
const static int64_t MIN_HIDDEN_DIM = 512LL;
const static int64_t HIDDEN_DIM_ALIGN = 256LL;
const static int64_t MIN_EP_WORLD_SIZE = 2LL;
const static int64_t MAX_MTE_EP_WORLD_SIZE = 1024LL;
const static int64_t MAX_URMA_EP_WORLD_SIZE = 1024LL;
const static int64_t MAX_MOE_EXPERT_NUM = 2048LL;
const static int64_t INPUT_WEIGHT_SCALES_CEIL_ALIGN = 64LL;
const static int64_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 50LL;
constexpr float DEFAULT_ACTIVATION_CLAMP = std::numeric_limits<float>::max();
constexpr float DEFAULT_SWIGLU_OAI_ALPHA = 1.702f;
constexpr float DEFAULT_SWIGLU_OAI_BETA = 1.0f;

constexpr uint32_t GMM_TILE_N = 256U;
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE = 4U;
constexpr int64_t GMM_TILE_STATUS_COUNT_ALIGN = 16LL;

uint32_t CalcMGroupsPerWave(const MegaMoeTilingData *tilingData, uint32_t aicNum)
{
    if (tilingData->hiddenDim == 0U || tilingData->h == 0U || aicNum == 0U) {
        return 1U;
    }

    /*
     * hiddenDim 包含 gate/up 两部分。交织模式每核至少调度 4 个独立 N tile；非交织模式
     * 每个物理任务成对处理 gate/up，原先每核 2 个物理任务同样等价于 4 个逻辑 N tile。
     * hiddenDim 已校验为 GMM_TILE_N(256) 的倍数，CeilDiv 对
     * 非交织半宽产生的尾 tile 只影响 wave 粒度估算，不影响正确性，两种编译模式共用本公式。
     */
    uint64_t gmm1LogicalTilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->hiddenDim, GMM_TILE_N);
    uint64_t gmm2TilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->h, GMM_TILE_N);
    uint64_t gmm1RequiredMGroups = ops::CeilDiv<uint64_t>(
        static_cast<uint64_t>(aicNum) * GMM1_MIN_LOGICAL_TILES_PER_CORE, gmm1LogicalTilesPerMGroup);
    uint64_t gmm2RequiredMGroups = ops::CeilDiv<uint64_t>(static_cast<uint64_t>(aicNum), gmm2TilesPerMGroup);
    return static_cast<uint32_t>(std::max(gmm1RequiredMGroups, gmm2RequiredMGroups));
}

const static uint32_t WEIGHT_MATRIX_ROW_DIM_INDEX = 0U;
const static uint32_t WEIGHT_MATRIX_COLUMN_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MATRIX_DIM_INDEX = 0U;
const static uint32_t WEIGHT_SCALE_GROUP_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MULTI_BASE_DIM_INDEX = 2U;

/*
 * 统计指定动态输入 (tensor list) 中的 tensor 数量。
 */
static uint32_t GetDynamicInputTensorCount(const gert::TilingContext *context, uint32_t inputIndex)
{
    uint32_t tensorCount = 0;
    while (context->GetDynamicInputShape(inputIndex, tensorCount) != nullptr) {
        ++tensorCount;
    }
    return tensorCount;
}

/*
 * 按权重布局返回专家数：逐专家布局取 TensorList 长度，堆叠布局取首个 tensor 的 dim0。
 * 可选输入未传入或首维为 0 时返回 0。
 */
static int64_t GetWeightExpertCount(const gert::TilingContext *context, uint32_t inputIndex,
                                    bool isPerExpertWeightTensor)
{
    const auto *firstTensorShape = context->GetDynamicInputShape(inputIndex, 0);
    if (firstTensorShape == nullptr || firstTensorShape->GetStorageShape().GetDimNum() == 0U ||
        firstTensorShape->GetStorageShape().GetDim(0) <= 0) {
        return 0;
    }
    if (isPerExpertWeightTensor) {
        return static_cast<int64_t>(GetDynamicInputTensorCount(context, inputIndex));
    }
    return firstTensorShape->GetStorageShape().GetDim(0);
}

/*
 * 以单专家视图读取指定维度；堆叠布局会自动跳过最外层的专家维。
 */
static int64_t GetSingleExpertTensorDimSize(const gert::StorageShape *tensorShape, uint32_t singleExpertDimIndex,
                                            bool isPerExpertTensor)
{
    uint32_t expertDimOffset = isPerExpertTensor ? 0U : 1U;
    return tensorShape->GetStorageShape().GetDim(expertDimOffset + singleExpertDimIndex);
}
} // namespace

void PrintMegaMoeTilingData(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);

    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");
    OP_LOGD(nodeName,
            "shape: bs=%u, numMaxTokensPerRank=%u, h=%u, hiddenDim=%u, topK=%u, maxOutputSize=%u, "
            "isPerExpertWeightTensor=%d",
            tilingData->bs, tilingData->numMaxTokensPerRank, tilingData->h, tilingData->hiddenDim, tilingData->topK,
            tilingData->maxOutputSize, tilingData->isPerExpertWeightTensor);
    OP_LOGD(nodeName,
            "topology: moeExpertPerRank=%u, sharedExpertNum=%u, epWorldSize=%u, aicNum=%u, blockAivNum=%u, "
            "blockNumPerEP=%u, topoType=%ld, rankNumPerServer=%u",
            tilingData->moeExpertPerRank, tilingData->sharedExpertNum, tilingData->epWorldSize, tilingData->aicNum,
            tilingData->blockAivNum, tilingData->blockNumPerEP, tilingData->topoType, tilingData->rankNumPerServer);
    OP_LOGD(nodeName, "mode: groupedMatmulMode=%u, combineQuantMode=%ld, clampLimit=%f",
            static_cast<uint32_t>(tilingData->groupedMatmulMode), tilingData->combineQuantMode, tilingData->clampLimit);
    OP_LOGD(nodeName, "combineSync: slotCountPerExpert=%lu", tilingData->combineSyncSlotCountPerExpert);

    const auto &dispatchConfig = tilingData->dispatchBufferConfig;
    OP_LOGD(nodeName, "dispatch: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, copyBufferBytes=%u",
            dispatchConfig.routeItemsPerBatch, dispatchConfig.routeBatchCount, dispatchConfig.bufferCount,
            dispatchConfig.copyBufferBytes);

    const auto &sendMaskConfigWithExtraExpert = tilingData->sendMaskConfigForCoreWithExtraExpert;
    const auto &sendMaskConfigWithoutExtraExpert = tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    OP_LOGD(nodeName, "sendMask: coreCountWithExtraExpert=%u", tilingData->sendMaskCoreCountWithExtraExpert);
    OP_LOGD(nodeName,
            "sendMaskWithExtraExpert: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, bufferBytes=%u",
            sendMaskConfigWithExtraExpert.routeItemsPerBatch, sendMaskConfigWithExtraExpert.routeBatchCount,
            sendMaskConfigWithExtraExpert.bufferCount, sendMaskConfigWithExtraExpert.bufferBytes);
    OP_LOGD(nodeName,
            "sendMaskWithoutExtraExpert: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, bufferBytes=%u",
            sendMaskConfigWithoutExtraExpert.routeItemsPerBatch, sendMaskConfigWithoutExtraExpert.routeBatchCount,
            sendMaskConfigWithoutExtraExpert.bufferCount, sendMaskConfigWithoutExtraExpert.bufferBytes);

    const auto &unpermuteFullChunkConfig = tilingData->unpermuteConfigForFullTokenChunk;
    const auto &unpermuteTailChunkConfig = tilingData->unpermuteConfigForTailTokenChunk;
    OP_LOGD(nodeName, "unpermute: fullTokenChunkCoreCount=%u", tilingData->unpermuteFullTokenChunkCoreCount);
    OP_LOGD(nodeName,
            "unpermuteFullChunk: tokensPerBatch=%d, inputBufferCount=%d, bf16SlotElements=%u, fp32SlotElements=%u, "
            "weightBufferBytes=%u, conversionBufferBytes=%u",
            unpermuteFullChunkConfig.tokensPerBatch, unpermuteFullChunkConfig.inputBufferCount,
            unpermuteFullChunkConfig.bf16SlotElementCount, unpermuteFullChunkConfig.fp32SlotElementCount,
            unpermuteFullChunkConfig.topKWeightsBufferBytes, unpermuteFullChunkConfig.topKWeightsConversionBufferBytes);
    OP_LOGD(nodeName,
            "unpermuteTailChunk: tokensPerBatch=%d, inputBufferCount=%d, bf16SlotElements=%u, fp32SlotElements=%u, "
            "weightBufferBytes=%u, conversionBufferBytes=%u",
            unpermuteTailChunkConfig.tokensPerBatch, unpermuteTailChunkConfig.inputBufferCount,
            unpermuteTailChunkConfig.bf16SlotElementCount, unpermuteTailChunkConfig.fp32SlotElementCount,
            unpermuteTailChunkConfig.topKWeightsBufferBytes, unpermuteTailChunkConfig.topKWeightsConversionBufferBytes);
    OP_LOGD(nodeName, "topkWeightsPrefetch is %d", tilingData->topkWeightsPrefetch);
    OP_LOGD(nodeName, "mGroupsPerWave is %u", tilingData->mGroupsPerWave);
}

void PrintWorkspaceLayout(const struct WorkspaceLayout *layout, const char *nodeName)
{
    OP_LOGD(nodeName, "dispatchRevDataOffset:         %ld\n", layout->dispatchRevDataOffset);
    OP_LOGD(nodeName, "dispatchRevScaleOffset:        %ld\n", layout->dispatchRevScaleOffset);
    OP_LOGD(nodeName, "activationQuantDataOffset:     %ld\n", layout->activationQuantDataOffset);
    OP_LOGD(nodeName, "activationQuantScaleOffset:    %ld\n", layout->activationQuantScaleOffset);
    OP_LOGD(nodeName, "expertRevTokenNumsOffset:      %ld\n", layout->expertRevTokenNumsOffset);
    OP_LOGD(nodeName, "metaInfoOffset:                %ld\n", layout->metaInfoOffset);
    OP_LOGD(nodeName, "flagActivationToGmm2Offset:    %ld\n", layout->flagActivationToGmm2Offset);
    OP_LOGD(nodeName, "flagDispatchToGmm1Offset:      %ld\n", layout->flagDispatchToGmm1Offset);
    OP_LOGD(nodeName, "flagSendCntCalToUpdParamsOffset: %ld\n", layout->flagSendCntCalToUpdParamsOffset);
    OP_LOGD(nodeName, "flagGmmToEpilogueOffset:       %ld\n", layout->flagGmmToEpilogueOffset);
    OP_LOGD(nodeName, "gmm2ReadyOffset:               %ld\n", layout->gmm2ReadyOffset);
    OP_LOGD(nodeName, "gmm2CombineSyncCounterOffset:  %ld\n", layout->gmm2CombineSyncCounterOffset);
    OP_LOGD(nodeName, "gmm2MmadResOffset:             %ld\n", layout->gmm2MmadResOffset);
    OP_LOGD(nodeName, "workspaceSize:                 %ld\n", layout->workspaceSize);
}

void PrintPeermemInfo(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_LOGD(nodeName, "========== PeermemInfo ==========");
    int64_t exceptionDumpRegionSize = tilingData->topoType == TOPO_TYPE_MTE ? EXCEPTION_DUMP_REGION_SIZE : 0;
    OP_LOGD(nodeName, "exceptionDumpRegionSize: {%ld}\n", exceptionDumpRegionSize);
    bool isA4Activation = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
                          tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ;
    PeermemSizeParams params{};
    params.numMaxTokensPerRank = static_cast<int64_t>(tilingData->numMaxTokensPerRank);
    params.topK = static_cast<int64_t>(tilingData->topK);
    params.h = static_cast<int64_t>(tilingData->h);
    params.moeExpertPerRank = static_cast<int64_t>(tilingData->moeExpertPerRank);
    params.epWorldSize = static_cast<int64_t>(tilingData->epWorldSize);
    params.yDtypeSize = SIZE_BF_16;
    params.elemsPerByte = isA4Activation ? 2U : 1U;
    params.topkWeightsPrefetch = tilingData->topkWeightsPrefetch == 1;
    params.isQuantCombine = tilingData->combineQuantMode != COMBINE_NO_QUANT;
    params.topoType = tilingData->topoType;
    params.serverNum = 1;
    if (tilingData->topoType == TOPO_TYPE_URMA) {
        params.serverNum = tilingData->epWorldSize / tilingData->rankNumPerServer;
    }
    PeermemLayoutSizes sizes = CalcPeermemLayoutSizes(params);
    OP_LOGD(nodeName, "peermemDataOffset: {%ld}\n", sizes.dataOffset);
    OP_LOGD(nodeName, "maskRecvSize: {%ld}\n", sizes.maskRecvSize);
    OP_LOGD(nodeName, "expertCountRecvSize: {%ld}\n", sizes.expertCountRecvSize);
    OP_LOGD(nodeName, "dispatchRecordAreaSize: {%ld}\n", sizes.dispatchRecordAreaSize);
    OP_LOGD(nodeName, "combineSendSize: {%ld}\n", sizes.combineSendSize);
    OP_LOGD(nodeName, "total PeermemInfo Size: {%ld}\n", exceptionDumpRegionSize + CalcPeermemLeastSize(params));
}

static ge::DataType GetDataTypeByOpQuantMode(const int64_t opQuantMode)
{
    // unsupport UNQUANT, STATIC, DYNAMIC currently
    switch (opQuantMode) {
        case DISPATCH_QUANT_OUT_DTYPE_E5M2:
            return ge::DT_FLOAT8_E5M2;
        case DISPATCH_QUANT_OUT_DTYPE_E4M3FN:
            return ge::DT_FLOAT8_E4M3FN;
        case DISPATCH_QUANT_OUT_DTYPE_E2M1:
            return ge::DT_FLOAT4_E2M1;
        default:
            return ge::DT_UNDEFINED;
    }
    return ge::DT_UNDEFINED;
}

static int64_t GetOpQuantModeByAttrDispatchOutType(const gert::TilingContext *context, MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    int64_t dispatchQuantOutDtype = static_cast<int64_t>(*dispatchQuantOutDtypePtr);

    int64_t opQuantMode;
    if (dispatchQuantOutDtype == static_cast<int64_t>(ge::DT_FLOAT8_E5M2)) {
        opQuantMode = DISPATCH_QUANT_OUT_DTYPE_E5M2;
    } else if (dispatchQuantOutDtype == static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) {
        opQuantMode = DISPATCH_QUANT_OUT_DTYPE_E4M3FN;
    } else {
        opQuantMode = DISPATCH_QUANT_OUT_DTYPE_E2M1;
    }

    return opQuantMode;
}

static int64_t GetCombineQuantModeByAttr(const gert::TilingContext *context, MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    if (combineQuantModePtr == nullptr) {
        return COMBINE_QUANT_OUT_TYPE_NO_QUANT;
    }
    return static_cast<int64_t>(*combineQuantModePtr);
}

static uint64_t CalTilingKey(const gert::TilingContext *context, MegaMoeConfig &config, MegaMoeTilingData *tilingData,
                             const char *nodeName)
{
    auto attrs = context->GetAttrs();

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    int64_t combineQuantMode = GetCombineQuantModeByAttr(context, config);
    int64_t topoType = TILINGKEY_TPL_MTE;
    if (tilingData->topoType == TOPO_TYPE_URMA) {
        topoType = TILINGKEY_TPL_URMA;
    }
    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    return GET_TPL_TILING_KEY(static_cast<int64_t>(*dispatchQuantModePtr), opQuantMode, combineQuantMode, topoType,
                              topkWeightsType);
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);
    auto rankNumPerServerPtr = attrs->GetAttrPointer<int64_t>(config.attrRankNumPerServerIndex);
    auto topoTypePtr = attrs->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    auto topkWeightsTypePtr = attrs->GetAttrPointer<int64_t>(config.attrTopkWeightsTypeIndex);

    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr || *cclBufferSizePtr < 0,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "cclBufferSize"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxRecvTokenNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "maxRecvTokenNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dispatchQuantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantOutDtypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dispatchQuantOutDtype"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(combineQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "combineQuantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commAlg"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numMaxTokensPerRankPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numMaxTokensPerRank"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(activationPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "activation"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(activationParamsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "activationParams"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankNumPerServerPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "rankNumPerServer"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topoTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topoType"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkWeightsType"),
                    return ge::GRAPH_FAILED);
    const size_t paramCount = activationParamsPtr->GetSize();
    const float *activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();
    OP_TILING_CHECK(paramCount != 0U && activationParams == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "activationParamsData"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * swiglu / swiglustep 只认一个可选参数 clamp——激活值的截断上限。
 * 不传就是不截断（默认取 float 最大值，见 SetActivationAttrParams）。
 */
static ge::graphStatus CheckSwiGluActivationParams(const float *activationParams, size_t paramCount,
                                                   const char *nodeName)
{
    OP_TILING_CHECK(paramCount > 1U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize", std::to_string(paramCount).c_str(),
                                              "0 or 1 for swiglu/swiglustep"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 1U) {
        const float activationClamp = activationParams[0];
        OP_TILING_CHECK(activationClamp < 0.0f || std::isnan(activationClamp),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationClamp", std::to_string(activationClamp).c_str(),
                                                  "should be >= 0 and not NAN"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * swigluoai 要 clamp、alpha、beta 三个参数，三个一起给、或者一个都不给。
 * 不接受给一半：缺的那个没有单独的默认值可退，只有一个都不给时才整体走默认。
 */
static ge::graphStatus CheckSwiGluOaiActivationParams(const float *activationParams, size_t paramCount,
                                                      const char *nodeName)
{
    OP_TILING_CHECK(paramCount != 0U && paramCount != 3U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize", std::to_string(paramCount).c_str(),
                                              "0 or 3 for swigluoai [clamp, alpha, beta]"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 3U) {
        const float activationClamp = activationParams[0];
        OP_TILING_CHECK(activationClamp < 0.0f || std::isnan(activationClamp),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationClamp", std::to_string(activationClamp).c_str(),
                                                  ">= 0 and not NAN"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(!std::isfinite(activationParams[1]),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationAlpha",
                                                  std::to_string(activationParams[1]).c_str(), "finite"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(!std::isfinite(activationParams[2]),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationBeta",
                                                  std::to_string(activationParams[2]).c_str(), "finite"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * situglu 的 beta 必填：它要进分母，取 0 或非有限值算出来就是 NaN。
 * linear_beta 选填，给了就切到 LINEAR 子模式。
 * 注意个数校验必须排在取值之前：一个参数都不传时 activationParams 是空指针，
 * 先取 activationParams[0] 会直接段错误，而不是干净地报错返回。
 */
static ge::graphStatus CheckSituGluActivationParams(const float *activationParams, size_t paramCount,
                                                    const char *nodeName)
{
    OP_TILING_CHECK(paramCount < 1U || paramCount > 2U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize", std::to_string(paramCount).c_str(),
                                              "should be 1 or 2 for situglu"),
                    return ge::GRAPH_FAILED);
    const float beta = activationParams[0];
    OP_TILING_CHECK(!std::isfinite(beta) || beta == 0.0f,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "situglu_beta", std::to_string(beta).c_str(),
                                              "should be finite and non-zero"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 2U) {
        const float linearBeta = activationParams[1];
        OP_TILING_CHECK(!std::isfinite(linearBeta) || linearBeta == 0.0f,
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "situglu_linear_beta", std::to_string(linearBeta).c_str(),
                                                  "should be finite and non-zero"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * URMA Layered 已接入 situglu 分派；其余扩展激活仍保持原约束。
 */
static ge::graphStatus CheckActivationTopoConstraint(const gert::TilingContext *context, MegaMoeConfig &config,
                                                     const char *activationPtr, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto topoTypePtr = attrs->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    OP_TILING_CHECK(
        topoTypePtr != nullptr && *topoTypePtr == TOPO_TYPE_URMA && std::strcmp(activationPtr, "swiglu") != 0 &&
            std::strcmp(activationPtr, "situglu") != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr, "'swiglu' or 'situglu' for URMA topology"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckActivationParams(const gert::TilingContext *context, MegaMoeConfig &config,
                                             const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);
    const size_t paramCount = activationParamsPtr->GetSize();
    const float *activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();

    const bool isSwiGlu = std::strcmp(activationPtr, "swiglu") == 0;
    const bool isSwiGluStep = std::strcmp(activationPtr, "swiglustep") == 0;
    const bool isSwiGluOai = std::strcmp(activationPtr, "swigluoai") == 0;
    const bool isSituGlu = std::strcmp(activationPtr, "situglu") == 0;
    OP_TILING_CHECK(!isSwiGlu && !isSwiGluStep && !isSwiGluOai && !isSituGlu,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr,
                                              "one of 'swiglu', 'swiglustep', 'swigluoai' or 'situglu'"),
                    return ge::GRAPH_FAILED);

    // 拓扑约束先判：URMA 支持 swiglu/situglu，其他激活保持拦截。
    ge::graphStatus topoStatus = CheckActivationTopoConstraint(context, config, activationPtr, nodeName);
    if (topoStatus != ge::GRAPH_SUCCESS) {
        return topoStatus;
    }

    // 上面已经把激活类型收敛成四选一，所以这里最后一个分支就是 situglu。
    if (isSwiGlu || isSwiGluStep) {
        return CheckSwiGluActivationParams(activationParams, paramCount, nodeName);
    }
    if (isSwiGluOai) {
        return CheckSwiGluOaiActivationParams(activationParams, paramCount, nodeName);
    }
    return CheckSituGluActivationParams(activationParams, paramCount, nodeName);
}

/*
 * 下面四组校验按属性的业务类别划分：通信域、专家、量化、容量。
 * 组间顺序服从数据依赖：通信域先给出 epWorldSize，专家组用它推出本卡专家数，
 * 量化组先把 combineQuantMode 校验掉（容量公式要用它决定窗口里要不要留量化输出的位置），
 * 最后容量组才用前面这些值去算 peermem 窗口该开多大。
 */

/*
 * CheckAttrParams 一开始就从输入张量上量出来的几个数，四组校验按需取用。
 * 之所以打包成结构体而不是一串 int64_t 形参：这几个数类型全一样，调用处顺序写错编译器不会报，
 * 只会把容量算错，而且错得没有任何征兆。
 */
struct MegaMoeAttrShapeContext {
    int64_t bs;                   // 本卡这次实际要处理多少 token，取自 x 的第 0 维
    int64_t h;                    // hidden size，取自 x 的第 1 维
    int64_t topK;                 // 一个 token 路由到几个专家，取自 topkIds 的第 1 维
    int64_t yDtypeSize;           // 输出元素几个字节，算 peermem 窗口要用
    int64_t weightMoeExpertCount; // weight1 实际给了几个专家的权重
    bool isPerExpertWeightTensor; // weight 是一个专家一个张量(true)，还是堆成一个大张量(false)
};

/*
 * 通信域这一组：EP 域里有多少张卡、用哪种通信算法。
 * epWorldSize 是后面两组都要用的枢纽值（专家组拿它做整除、容量组拿它算窗口），所以经出参带出去。
 */
static ge::graphStatus CheckCommDomainAttrs(const gert::TilingContext *context, MegaMoeConfig &config,
                                            const char *nodeName, int64_t &epWorldSize)
{
    auto attrs = context->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    epWorldSize = static_cast<int64_t>(*epWorldSizePtr);

    int64_t topoType = *attrs->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    OP_TILING_CHECK(topoType != TOPO_TYPE_MTE && topoType != TOPO_TYPE_URMA,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topoType", std::to_string(topoType).c_str(),
                                              "only support MTE(0) or URMA(1)"),
                    return ge::GRAPH_FAILED);
    int64_t maxEpWorldSize = MAX_MTE_EP_WORLD_SIZE;
    if (topoType == TOPO_TYPE_URMA) {
        maxEpWorldSize = MAX_URMA_EP_WORLD_SIZE;
    }
    OP_TILING_CHECK(epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > maxEpWorldSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EP_WORLD_SIZE) + ", " +
                                               std::to_string(maxEpWorldSize) + "] for the selected topology")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    // 目前只实现了默认那一种通信算法，传了别的名字就是配错了，在 host 直接拦下。
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    OP_TILING_CHECK(std::strcmp(commAlgPtr, "") != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "commAlg", commAlgPtr, "not support, need empty string"),
                    return ge::GRAPH_FAILED);

    auto rankNumPerServerPtr = attrs->GetAttrPointer<int64_t>(config.attrRankNumPerServerIndex);
    int64_t rankNumPerServer = *rankNumPerServerPtr;
    OP_TILING_CHECK(topoType == TOPO_TYPE_URMA && (rankNumPerServer <= 0 || rankNumPerServer > epWorldSize ||
                                                   epWorldSize % rankNumPerServer != 0),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "rankNumPerServer", std::to_string(rankNumPerServer).c_str(),
                                              (std::string("should be in [1, epWorldSize] and divide epWorldSize(") +
                                               std::to_string(epWorldSize) + ") for URMA")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 专家这一组：路由专家总数、由它推出的本卡专家数，以及共享专家个数。
 * 专家要能均分到每张卡，所以本卡专家数 = moeExpertNum / epWorldSize，除法必须排在整除校验之后，
 * 否则不整除的配置会先算出一个取整过的错值。算完再跟 weight1 实际给了几个专家对一下，
 * 对不上说明权重和 attr 配的不是同一套。
 * 共享专家不参与上面这条路由切分，它的个数是从 sharedWeight1 张量形状数出来的，这里只卡上限。
 */
static ge::graphStatus CheckExpertNumAttrs(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName, const MegaMoeAttrShapeContext &shape,
                                           int64_t epWorldSize, int64_t &moeExpertPerRank)
{
    auto attrs = context->GetAttrs();

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    OP_TILING_CHECK((moeExpertNum < epWorldSize || moeExpertNum > MAX_MOE_EXPERT_NUM) || (moeExpertNum % epWorldSize),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertNum", std::to_string(moeExpertNum).c_str(),
                                              (std::string("should in [") + std::to_string(epWorldSize) + ", " +
                                               std::to_string(MAX_MOE_EXPERT_NUM) + "] and mod(..., epWorldSize(" +
                                               std::to_string(epWorldSize) + ")) == 0")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    moeExpertPerRank = moeExpertNum / epWorldSize;
    OP_TILING_CHECK(shape.weightMoeExpertCount != moeExpertPerRank,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "weight1 expert count", std::to_string(shape.weightMoeExpertCount).c_str(),
                        (std::string("should equal the local MoE expert count (moeExpertNum / epWorldSize) = ") +
                         std::to_string(moeExpertPerRank))
                            .c_str()),
                    return ge::GRAPH_FAILED);

    int64_t sharedExpertNum = GetWeightExpertCount(context, config.sharedWeight1Index, shape.isPerExpertWeightTensor);
    OP_TILING_CHECK(sharedExpertNum > MAX_SHARED_EXPERT_NUM,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum", std::to_string(sharedExpertNum).c_str(),
                                              "only support 0-4"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 量化这一组：dispatch 用哪种量化、输出什么 dtype，weight1 的 dtype 跟它对不对得上，
 * 以及 combine 回来的时候要不要量化。
 * 排在容量组之前，是因为 combine 量化模式会决定 peermem 窗口里要不要给量化输出留位置。
 */
static ge::graphStatus CheckQuantAttrs(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    auto attrs = context->GetAttrs();

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t dispatchQuantMode = static_cast<int64_t>(*dispatchQuantModePtr);
    OP_TILING_CHECK(dispatchQuantMode != DISPATCH_QUANT_MODE_MXFP,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "dispatchQuantMode", std::to_string(dispatchQuantMode).c_str(),
                        (std::string("only support mxfp(") + std::to_string(DISPATCH_QUANT_MODE_MXFP) + ")").c_str()),
                    return ge::GRAPH_FAILED);

    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    int64_t dispatchQuantOutDtype = static_cast<int64_t>(*dispatchQuantOutDtypePtr);
    OP_TILING_CHECK(
        dispatchQuantOutDtype != (static_cast<int64_t>(ge::DT_FLOAT8_E5M2)) &&
            dispatchQuantOutDtype != (static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) &&
            dispatchQuantOutDtype != (static_cast<int64_t>(ge::DT_FLOAT4_E2M1)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "dispatchQuantOutDtype", std::to_string(dispatchQuantOutDtype).c_str(),
                                  "only support fp8_e5m2, fp8_e4m3fn and fp4_e2m1"),
        return ge::GRAPH_FAILED);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    ge::DataType refWeightDataType = GetDataTypeByOpQuantMode(opQuantMode);
    OP_TILING_CHECK(
        refWeightDataType == ge::DT_UNDEFINED,
        OP_LOGE(nodeName, "unsupported dispatchQuantMode(%ld), leading out data type to being DT_UNDEFINED.",
                dispatchQuantMode),
        return ge::GRAPH_FAILED);
    // weight1 must match dispatch quant dtype; the only allowed mismatch is A8W4 (fp4_e2m1 + fp8_e4m3fn).
    if (refWeightDataType != weightOneDesc->GetDataType()) {
        std::string weightDtypeErrMsg = std::string("The dtype of weightOne (") +
                                        Ops::Base::ToString(weightOneDesc->GetDataType()) +
                                        ") must match dispatch quant dtype (" + Ops::Base::ToString(refWeightDataType) +
                                        "), or be fp4_e2m1 with fp8_e4m3fn dispatch quant.";
        OP_TILING_CHECK(
            weightOneDesc->GetDataType() != ge::DT_FLOAT4_E2M1 || opQuantMode != DISPATCH_QUANT_OUT_DTYPE_E4M3FN,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "dispatchQuantOutDtype/weight1", weightDtypeErrMsg.c_str(),
                                                  "weight1 dtype mismatch."),
            return ge::GRAPH_FAILED);
    }

    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));

    OP_TILING_CHECK(
        *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_NO_QUANT &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E5M2 &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E4M3FN,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "combineQuantMode", std::to_string(*combineQuantModePtr).c_str(),
                                  "only support no_quant(0), fp8_e5m2(3) and fp8_e4m3fn(4)"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 计算通信窗口的最小容量。各分区尺寸复用 common/mega_moe_peermem.h 中的 Host/Device 公共公式，
 * 保证 Host 侧容量校验与 Kernel 侧地址布局一致。
 * 当前校验阶段尚未确定激活数据的实际压缩比例，因此 elemsPerByte 取 1，按最大存储需求预留空间。
 * 逐字段赋值避免同类型字段因聚合初始化顺序变化而静默错位。
 */
static int64_t CalcLeastCclBufferSize(const gert::TilingContext *context, MegaMoeConfig &config,
                                      const MegaMoeAttrShapeContext &shape, int64_t numMaxTokensPerRank,
                                      int64_t epWorldSize, int64_t moeExpertPerRank, bool topkWeightsPrefetch)
{
    auto topoTypePtr = context->GetAttrs()->GetAttrPointer<int64_t>((config.attrTopoTypeIndex));
    PeermemSizeParams peermemSizeParams{};
    peermemSizeParams.numMaxTokensPerRank = numMaxTokensPerRank;
    peermemSizeParams.topK = shape.topK;
    peermemSizeParams.h = shape.h;
    peermemSizeParams.moeExpertPerRank = moeExpertPerRank;
    peermemSizeParams.epWorldSize = epWorldSize;
    peermemSizeParams.yDtypeSize = shape.yDtypeSize;
    peermemSizeParams.elemsPerByte = 1U;
    peermemSizeParams.topkWeightsPrefetch = topkWeightsPrefetch;
    peermemSizeParams.isQuantCombine = GetCombineQuantModeByAttr(context, config) != COMBINE_NO_QUANT;
    peermemSizeParams.topoType = *topoTypePtr;
    peermemSizeParams.serverNum = 1;
    auto rankNumPerServerPtr = context->GetAttrs()->GetAttrPointer<int64_t>(config.attrRankNumPerServerIndex);
    if (peermemSizeParams.topoType == TOPO_TYPE_URMA) {
        peermemSizeParams.serverNum = epWorldSize / *rankNumPerServerPtr;
    }
    int64_t leastCclBufferSize = CalcPeermemLeastSize(peermemSizeParams);
    // MTE 路径会在 peermem 头部单独留一段异常 dump 区，窗口下限要把这段一起算进去。
    if (*topoTypePtr == TOPO_TYPE_MTE) {
        leastCclBufferSize += EXCEPTION_DUMP_REGION_SIZE;
    }
    return leastCclBufferSize;
}

/*
 * 容量这一组：单卡 token 数上界、topk 权重预取开关、peermem 窗口大小、能收多少 token。
 * topkWeightsType 归在这里，是因为开了预取之后 peermem 里每个 token 要多存一份 topk 权重，
 * 它直接决定窗口要开多大，所以先把它校验掉，再拿去算下面的容量。
 * MTE 下 numMaxTokensPerRank 传 0 表示按本卡 bs 计算。URMA 使用跨卡对称窗口，host 无法从
 * 本卡 bs 推导全卡一致的容量上界，因此必须显式传入非 0 值。
 */
static ge::graphStatus CheckCapacityAttrs(const gert::TilingContext *context, MegaMoeConfig &config,
                                          const char *nodeName, const MegaMoeAttrShapeContext &shape,
                                          int64_t epWorldSize, int64_t moeExpertPerRank)
{
    auto attrs = context->GetAttrs();

    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    OP_TILING_CHECK(numMaxTokensPerRankPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numMaxTokensPerRank"),
                    return ge::GRAPH_FAILED);
    int64_t numMaxTokensPerRank = static_cast<int64_t>(*numMaxTokensPerRankPtr);
    const int64_t topoType = *attrs->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    OP_TILING_CHECK(topoType == TOPO_TYPE_URMA && shape.bs <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "bs", std::to_string(shape.bs).c_str(),
                                              "should be greater than 0 for URMA"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topoType == TOPO_TYPE_URMA && numMaxTokensPerRank == 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "numMaxTokensPerRank", "0",
                                              "should be >= bs and identical on all ranks for URMA"),
                    return ge::GRAPH_FAILED);
    // 这个上界最后按 uint32 落库，越界值不拦会被截断成错误值。
    OP_TILING_CHECK(
        numMaxTokensPerRank < 0 || numMaxTokensPerRank > 0xFFFFFFFFLL ||
            (numMaxTokensPerRank != 0 && shape.bs > numMaxTokensPerRank),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "numMaxTokensPerRank", std::to_string(numMaxTokensPerRank).c_str(),
                                  (std::string("0 for MTE, or in [bs, UINT32_MAX] (URMA requires nonzero), bs is ") +
                                   std::to_string(shape.bs))
                                      .c_str()),
        return ge::GRAPH_FAILED);
    if (numMaxTokensPerRank == 0) {
        numMaxTokensPerRank = shape.bs;
    }
    const int64_t routeCapacity = numMaxTokensPerRank * shape.topK;
    OP_TILING_CHECK(
        topoType == TOPO_TYPE_URMA && routeCapacity > std::numeric_limits<int32_t>::max(),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "numMaxTokensPerRank * topK", std::to_string(routeCapacity).c_str(),
                                  "should be <= INT32_MAX for URMA route indexes"),
        return ge::GRAPH_FAILED);

    const int64_t maxOutputCapacity = numMaxTokensPerRank * epWorldSize * std::min(shape.topK, moeExpertPerRank);
    OP_TILING_CHECK(
        topoType == TOPO_TYPE_URMA && maxOutputCapacity > std::numeric_limits<int32_t>::max(),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "maximum receive token capacity", std::to_string(maxOutputCapacity).c_str(),
                                  "should be <= INT32_MAX for URMA prefix sums"),
        return ge::GRAPH_FAILED);

    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    OP_TILING_CHECK(topkWeightsType != 0 && topkWeightsType != 1,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topkWeightsType", std::to_string(topkWeightsType).c_str(),
                                              "only support 0(disabled) or 1(enabled)"),
                    return ge::GRAPH_FAILED);

    int64_t leastCclBufferSize = CalcLeastCclBufferSize(context, config, shape, numMaxTokensPerRank, epWorldSize,
                                                        moeExpertPerRank, topkWeightsType == 1);
    int64_t cclBufferSize = static_cast<int64_t>(*attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex)));
    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "cclBufferSize", std::to_string(cclBufferSize).c_str(),
                                              (std::string("should >= ") + std::to_string(leastCclBufferSize)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    const int64_t maxRecvTokenNum =
        static_cast<int64_t>(*attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex)));
    OP_TILING_CHECK(
        maxRecvTokenNum < 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "maxRecvTokenNum", std::to_string(maxRecvTokenNum).c_str(), ">= 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topoType == TOPO_TYPE_URMA && maxRecvTokenNum > maxOutputCapacity,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "maxRecvTokenNum", std::to_string(maxRecvTokenNum).c_str(),
                        (std::string("should be in [0, ") + std::to_string(maxOutputCapacity) + "] for URMA").c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    auto yDesc = context->GetOutputDesc(config.yIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    bool isPerExpertWeightTensor = weightOneStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS;
    MegaMoeAttrShapeContext shape{xStorageShape->GetStorageShape().GetDim(0),
                                  xStorageShape->GetStorageShape().GetDim(1),
                                  topkIdsStorageShape->GetStorageShape().GetDim(1),
                                  ge::GetSizeByDataType(yDesc->GetDataType()),
                                  GetWeightExpertCount(context, config.weight1Index, isPerExpertWeightTensor),
                                  isPerExpertWeightTensor};

    int64_t epWorldSize = 0;
    ge::graphStatus checkStatus = CheckCommDomainAttrs(context, config, nodeName, epWorldSize);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    int64_t moeExpertPerRank = 0;
    checkStatus = CheckExpertNumAttrs(context, config, nodeName, shape, epWorldSize, moeExpertPerRank);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    checkStatus = CheckQuantAttrs(context, config, nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    return CheckCapacityAttrs(context, config, nodeName, shape, epWorldSize, moeExpertPerRank);
}

/*
 * 把校验过的属性落进 tilingData：通信域大小、单卡 token 上界、本卡专家数、输出上界、
 * 每个 EP 分几个 block、combine 量化模式。
 * maxOutputSize 在 maxRecvTokenNum 没填时兜底 = numMaxTokensPerRank × epWorldSize × min(topK, 本卡专家数)。
 * URMA 的容量乘积已在 CheckCapacityAttrs 中限制到 INT32_MAX；MTE 保留原有的 uint32 取值口径。
 */
static ge::graphStatus SetBasicAttrParams(const gert::TilingContext *context, MegaMoeConfig &config,
                                          MegaMoeTilingData *tilingData, const char *nodeName, const uint32_t aicNum)
{
    auto attrs = context->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto rankNumPerServerPtr = attrs->GetAttrPointer<int64_t>(config.attrRankNumPerServerIndex);
    auto topoTypePtr = attrs->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    auto topkWeightsTypePtr = attrs->GetAttrPointer<int64_t>(config.attrTopkWeightsTypeIndex);
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));

    tilingData->epWorldSize = *epWorldSizePtr;
    tilingData->topoType = *topoTypePtr;
    tilingData->topkWeightsPrefetch = static_cast<int32_t>(*topkWeightsTypePtr);
    // MTE 下 attr 为 0 时上界取本卡 bs；URMA 已在参数校验阶段拒绝 0。
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    OP_TILING_CHECK(numMaxTokensPerRankPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numMaxTokensPerRank"),
                    return ge::GRAPH_FAILED);
    tilingData->numMaxTokensPerRank =
        *numMaxTokensPerRankPtr != 0 ? static_cast<uint32_t>(*numMaxTokensPerRankPtr) : tilingData->bs;
    tilingData->rankNumPerServer = *topoTypePtr == TOPO_TYPE_URMA ? static_cast<uint32_t>(*rankNumPerServerPtr) :
                                                                    static_cast<uint32_t>(*epWorldSizePtr);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    int64_t moeExpertPerRank = moeExpertNum / static_cast<int64_t>(tilingData->epWorldSize);
    tilingData->moeExpertPerRank = static_cast<uint32_t>(moeExpertPerRank);
    // 接收容量要覆盖所有源卡的最大发送量, 用全卡一致的 numMaxTokensPerRank, 不能用本卡 bs
    tilingData->maxOutputSize = *maxRecvTokenNumPtr != 0 ? *maxRecvTokenNumPtr :
                                                           tilingData->numMaxTokensPerRank * tilingData->epWorldSize *
                                                               std::min(tilingData->topK, tilingData->moeExpertPerRank);
    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->epWorldSize);
    tilingData->combineQuantMode = GetCombineQuantModeByAttr(context, config);

    return ge::GRAPH_SUCCESS;
}

/*
 * 设置激活属性。前五行默认值与其后的 if 链是一个不可分割的顺序整体：
 * activation 为 swiglu 时正是靠默认值生效，if 链没有 else 分支，默认值块也不可挪到 if 链之后。
 */
static void SetActivationAttrParams(const gert::TilingContext *context, MegaMoeConfig &config,
                                    MegaMoeTilingData *tilingData)
{
    auto attrs = context->GetAttrs();

    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);

    const size_t paramCount = activationParamsPtr->GetSize();
    const float *activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();
    tilingData->clampLimit = paramCount == 0U ? DEFAULT_ACTIVATION_CLAMP : activationParams[0];
    tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU);
    tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::DEFAULT);
    tilingData->activationAlpha = 1.0f;
    tilingData->activationBeta = 1.0f;

    if (std::strcmp(activationPtr, "situglu") == 0) {
        tilingData->clampLimit = std::numeric_limits<float>::max();
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SITU);
        tilingData->activationBeta = activationParams[0];
        if (activationParamsPtr->GetSize() == 2U) {
            tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::LINEAR);
            tilingData->activationAlpha = activationParams[1];
        } else {
            tilingData->actSubMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActSubMode::DEFAULT);
            tilingData->activationAlpha = 0.0f;
        }
    } else if (std::strcmp(activationPtr, "swiglustep") == 0) {
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU_STEP);
    } else if (std::strcmp(activationPtr, "swigluoai") == 0) {
        tilingData->actMode = static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SWIGLU_OAI);
        tilingData->activationAlpha = paramCount == 0U ? DEFAULT_SWIGLU_OAI_ALPHA : activationParams[1];
        tilingData->activationBeta = paramCount == 0U ? DEFAULT_SWIGLU_OAI_BETA : activationParams[2];
    }
}

/*
 * 按 weight1 的 dtype、格式与 dispatch 量化模式判定 GMM 模式。
 */
static void SetGroupedMatmulMode(const gert::TilingContext *context, MegaMoeConfig &config,
                                 MegaMoeTilingData *tilingData)
{
    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    if (weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 && opQuantMode == DISPATCH_QUANT_OUT_DTYPE_E4M3FN) {
        // A8W4: fp4_e2m1 weight in NZ_C0_32, dispatched via separate template instantiation
        tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
    } else if (opQuantMode == DISPATCH_QUANT_OUT_DTYPE_E4M3FN &&
               weightOneDesc->GetDataType() == GetDataTypeByOpQuantMode(opQuantMode) &&
               static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) ==
                   ge::FORMAT_FRACTAL_NZ) {
        // A8W8_NZ: fp8_e4m3fn activation × fp8_e4m3fn weight in NZ format (E5M2 not supported)
        tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A8W8_NZ;
    } else if (weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 && opQuantMode == DISPATCH_QUANT_OUT_DTYPE_E2M1) {
        // A4W4: weight1 和 activation 都是 fp4，GMM1 走 generic，GMM2 走 A8W4。
        // NZ format: weight1 为 FRACTAL_NZ → A4W4_NZ；否则为 A4W4（ND 格式）。
        if (static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ) {
            tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A4W4_NZ;
        } else {
            tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A4W4;
        }
    } else {
        // Generic: fp8 activation × fp8 weight in ND format
        tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_GENERAL;
    }
}

/*
 * 设置 prefetch 状态位区上限与 wave 粒度。maxTilesPerExpert 先无条件置零、再由 prefetch 分支覆盖，
 * 非 prefetch 路径依赖这次置零，不可改成只在分支内赋值。
 * 本步读 maxOutputSize，必须排在 SetBasicAttrParams 之后。
 */
static void SetPrefetchAndWaveParams(MegaMoeTilingData *tilingData, const uint32_t aicNum)
{
    // GMM1 tile 状态位区每 expert 的 tile 上限（仅 prefetch 软同步路径使用）。
    // 非交织调度只遍历 hiddenDim / 2，交织调度会遍历完整 hiddenDim；统一按完整 hiddenDim
    // 预留，使两个 clencode 共用同一 workspace 布局并避免交织模式覆盖下一 expert 的状态槽。
    tilingData->maxTilesPerExpert = 0;
    if (tilingData->topkWeightsPrefetch == 1) {
        int64_t maxSchedulerN = static_cast<int64_t>(tilingData->hiddenDim);
        int64_t maxTilesM = ops::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(256));
        int64_t maxTilesN = ops::CeilDiv(maxSchedulerN, static_cast<int64_t>(256));
        tilingData->maxTilesPerExpert =
            static_cast<uint32_t>(ops::CeilAlign(maxTilesM * maxTilesN, GMM_TILE_STATUS_COUNT_ALIGN));
    }

    tilingData->mGroupsPerWave = CalcMGroupsPerWave(tilingData, aicNum);
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, MegaMoeConfig &config,
                                     MegaMoeTilingData *tilingData, const char *nodeName, const uint32_t aicNum)
{
    ge::graphStatus setStatus = SetBasicAttrParams(context, config, tilingData, nodeName, aicNum);
    if (setStatus != ge::GRAPH_SUCCESS) {
        return setStatus;
    }

    SetActivationAttrParams(context, config, tilingData);

    tilingData->sharedExpertNum = static_cast<uint32_t>(
        GetWeightExpertCount(context, config.sharedWeight1Index, tilingData->isPerExpertWeightTensor));

    SetGroupedMatmulMode(context, config, tilingData);

    // 读 maxOutputSize，必须排在 SetBasicAttrParams 之后。
    SetPrefetchAndWaveParams(tilingData, aicNum);

    return ge::GRAPH_SUCCESS;
}

/*
 * 计算每个 dispatch ring slot 的量化 token 与 scale 拷贝区字节数（prefetch 模式再追加对齐后的 weight）。
 */
static uint32_t CalcDispatchCopyBufferBytes(const MegaMoeTilingData *tilingData, uint32_t activationElementsPerByte)
{
    // copyBufferBytes 是每个 dispatch slot 中量化 token 和 scale 的拷贝区大小。
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    uint32_t copyBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        uint32_t weightBytes =
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        copyBufferBytes += weightBytes;
    }
    return copyBufferBytes;
}

/*
 * 计算 dispatch 阶段不随 ring 深度与 route batch 变化的固定 UB 占用。
 */
static uint32_t CalcDispatchFixedBufferBytes(const MegaMoeTilingData *tilingData)
{
    // fixedBufferBytes 包含 cumsumInfoTensor_ 和 expertTokenNumsOutTensor_。
    uint32_t fixedBufferBytes =
        static_cast<uint32_t>(ops::CeilAlign(
            static_cast<uint64_t>(tilingData->epWorldSize) * tilingData->moeExpertPerRank * sizeof(int32_t),
            static_cast<uint64_t>(ALIGN_32))) +
        static_cast<uint32_t>(ops::CeilAlign(static_cast<uint64_t>(tilingData->moeExpertPerRank) * sizeof(int32_t),
                                             static_cast<uint64_t>(ALIGN_32)));
    return fixedBufferBytes;
}

/*
 * 分两步定下 dispatch 阶段的 UB 分配：
 *   第一步：先按一个保守的 route batch 大小起步，把剩下的 UB 全拿去开 ring slot，slot 越多流水越深；
 *   第二步：ring 深度定死之后，UB 若还有富余，再反过来把 route batch 撑大，这样总批数更少。
 * 预算减法使用饱和计算，避免 UB 不足时无符号回绕；bufferCount 必须先限制上限，再保证最小 ring 深度。
 */
static void SelectDispatchRingAndRouteBatch(MegaMoeDispatchBufferConfig &bufferConfig, uint64_t sendTotalNum,
                                            uint64_t alignedTotalRouteItems, uint32_t fixedBufferBytes,
                                            uint32_t dispatchSlotBytes, uint32_t availableUbBytes)
{
    // 第一步：用基准 batch 把 ring 深度定下来。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_RECV_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>(ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));

    // MTE 接收侧只保留一个 int32 topK 有效下标 batch，不再分配 mask 和第二个 index tensor。
    // routeIndexBufferBytes 是该有效下标 tensor 的大小。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    // bytesWithoutDispatchSlots 包含 count/prefix 固定区和一份有效下标 tensor。
    uint32_t bytesWithoutDispatchSlots = fixedBufferBytes + routeIndexBufferBytes;
    // dispatchSlotBudgetBytes 是扣除非 ring tensor 后可用于分配 dispatch slot 的 UB。
    uint32_t dispatchSlotBudgetBytes =
        availableUbBytes > bytesWithoutDispatchSlots ? availableUbBytes - bytesWithoutDispatchSlots : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(dispatchSlotBudgetBytes / dispatchSlotBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_DISPATCH_BUFFER_COUNT);
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_DISPATCH_BUFFER_COUNT);

    // 第二步：ring 深度已定，剩余 UB 用来扩 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithDispatchSlots 包含固定 tensor 和已选中的全部 dispatch slot。
        uint32_t fixedBytesWithDispatchSlots =
            fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * dispatchSlotBytes;
        // routeItemBudgetBytes 是有效下标 tensor 可使用的 UB。
        uint32_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithDispatchSlots ? availableUbBytes - fixedBytesWithDispatchSlots : 0U;
        uint32_t expandedRouteItems = routeItemBudgetBytes / static_cast<uint32_t>(sizeof(int32_t));
        expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
        expandedRouteItems =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
        if (expandedRouteItems > static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>(
                ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
        }
    }
}

static MegaMoeDispatchBufferConfig CalcDispatchBufferConfig(const MegaMoeTilingData *tilingData,
                                                            uint32_t activationElementsPerByte,
                                                            uint32_t availableUbBytes)
{
    MegaMoeDispatchBufferConfig bufferConfig{};
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->numMaxTokensPerRank);
    uint64_t alignedTotalRouteItems = ops::CeilAlign(sendTotalNum, static_cast<uint64_t>(ALIGN_256));
    uint32_t copyBufferBytes = CalcDispatchCopyBufferBytes(tilingData, activationElementsPerByte);
    bufferConfig.copyBufferBytes = copyBufferBytes;
    uint32_t fixedBufferBytes = CalcDispatchFixedBufferBytes(tilingData);
    // 一个 dispatch ring slot 包含 token/scale copy buffer 和一条 32B triple。
    uint32_t dispatchSlotBytes = copyBufferBytes + static_cast<uint32_t>(ALIGN_32);

    SelectDispatchRingAndRouteBatch(bufferConfig, sendTotalNum, alignedTotalRouteItems, fixedBufferBytes,
                                    dispatchSlotBytes, availableUbBytes);
    return bufferConfig;
}

static uint64_t CalcTopkValidIndexRingSlotBytes(uint32_t routeItemsPerBatch, uint32_t topK)
{
    // routeItemsPerBatch 按 256 个 item 对齐，但 topK 不一定整除 256，因此后续 batch 可能从某个
    // token 的 topK 段中间开始。同一 token 的 topK 专家不重复，单个专家每个 token 至多匹配一个下标；
    // 一个 batch 最多跨越 CeilDiv(routeItemsPerBatch + topK - 1, topK) 个 token，slot 按此上界预留空间。
    uint64_t maxMatchedRouteItems =
        ops::CeilDiv(static_cast<uint64_t>(routeItemsPerBatch) + topK - 1U, static_cast<uint64_t>(topK));
    uint64_t validIndexBytes = ops::CeilAlign(maxMatchedRouteItems * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32));
    return static_cast<uint64_t>(routeItemsPerBatch) / BITS_PER_BYTE + validIndexBytes;
}

// MTE Wave producer：ring slot 同时保存临时 compare mask 与本专家的 topK 有效下标。
static MegaMoeSendMaskBufferConfig CalcTopkValidIndexBufferConfig(const MegaMoeTilingData *tilingData,
                                                                  uint32_t fixedBufferBytes, uint32_t ownedExpertCount,
                                                                  uint32_t availableUbBytes)
{
    MegaMoeSendMaskBufferConfig bufferConfig{};
    // sendTotalNum 表示所有专家合计最多发送的 topK 有效下标数，不是 token 数。发送批网格按
    // numMaxTokensPerRank * topK 的容量上界划分，确保各 Rank 使用一致批次数；Kernel 再按本 Rank
    // 的实际 bs * topK 对每批有效长度进行裁剪。
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->topK;
    uint64_t alignedTotalRouteItems = ops::CeilAlign(sendTotalNum, static_cast<uint64_t>(ALIGN_256));

    // Stage 1：使用基准 batch 确定 route ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_SEND_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>(ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
    bufferConfig.bufferBytes = static_cast<uint32_t>(
        CalcTopkValidIndexRingSlotBytes(static_cast<uint32_t>(bufferConfig.routeItemsPerBatch), tilingData->topK));

    // topkIdsTensor 和 gather 输出 tensor 各占一份 int32 route batch。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t bytesWithoutRouteBuffers = fixedBufferBytes + 2U * routeIndexBufferBytes;
    // routeBufferBudgetBytes 是扣除固定 tensor 和两份 route tensor 后可用于 ring slot 的 UB。
    uint32_t routeBufferBudgetBytes =
        availableUbBytes > bytesWithoutRouteBuffers ? availableUbBytes - bytesWithoutRouteBuffers : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(routeBufferBudgetBytes / bufferConfig.bufferBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_SEND_MASK_BUFFER_COUNT);

    // ring 深度超过当前核的实际发送次数不会增加流水重叠。
    uint64_t routePushCount = static_cast<uint64_t>(bufferConfig.routeBatchCount) * ownedExpertCount;
    if (routePushCount > 0U && static_cast<uint64_t>(bufferConfig.bufferCount) > routePushCount) {
        bufferConfig.bufferCount = static_cast<int32_t>(routePushCount);
    }
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);

    // Stage 2：固定 ring 深度，用剩余 UB 扩大 route batch。容量公式包含 topK 有效下标的占用。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // 每个 slot 预留有效下标的 32B 对齐余量和 batch 边界余量，其余 UB 用于随 batch 线性增长的数据区。
        uint64_t fixedBytesWithRoutePadding =
            static_cast<uint64_t>(fixedBufferBytes) +
            static_cast<uint64_t>(bufferConfig.bufferCount) * (ALIGN_32 + 2U * sizeof(int32_t));
        uint64_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithRoutePadding ? availableUbBytes - fixedBytesWithRoutePadding : 0U;
        // 两份 route tensor 各占 32bit/item；每个 ring slot 占 1bit mask 和约 32/topK bit 有效下标。
        uint64_t expandedRouteItems =
            routeItemBudgetBytes * BITS_PER_BYTE /
            (2U * sizeof(int32_t) * BITS_PER_BYTE + static_cast<uint64_t>(bufferConfig.bufferCount) +
             ops::CeilDiv(static_cast<uint64_t>(bufferConfig.bufferCount) * sizeof(int32_t) * BITS_PER_BYTE,
                          static_cast<uint64_t>(tilingData->topK)));
        expandedRouteItems = expandedRouteItems / ALIGN_256 * ALIGN_256;
        expandedRouteItems = std::min(expandedRouteItems, alignedTotalRouteItems);
        if (expandedRouteItems > static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>(
                ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
            bufferConfig.bufferBytes = static_cast<uint32_t>(
                CalcTopkValidIndexRingSlotBytes(static_cast<uint32_t>(expandedRouteItems), tilingData->topK));
        }
    }
    return bufferConfig;
}

/*
 * 计算 unpermute 的 slot 与 scale 尺寸：单 token 的 BF16 搬入区 + FP32 计算区，以及 combine quant 的 scale 展开区。
 * dataSlotBytes 与 scaleBytes 经出参带出，供后续两个阶段共用。
 */
static void CalcUnpermuteSlotAndScaleBytes(const MegaMoeTilingData *tilingData,
                                           MegaMoeUnpermuteBufferConfig &bufferConfig, uint32_t &dataSlotBytes,
                                           uint32_t &scaleBytes)
{
    uint32_t bf16SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(uint16_t), static_cast<uint64_t>(ALIGN_32)));
    uint32_t fp32SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    // dataSlotBytes 是同一 token 的 BF16 搬入区和 FP32 计算区之和。
    dataSlotBytes = bf16SlotBytes + fp32SlotBytes;
    bufferConfig.bf16SlotElementCount = bf16SlotBytes / sizeof(uint16_t);
    bufferConfig.fp32SlotElementCount = fp32SlotBytes / sizeof(float);

    // scaleBytes 是 combine quant 使用的 BF16/FP32 scale 展开区大小。
    scaleBytes = 0U;
    if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleElementCount = (tilingData->h + ALIGN_32 - 1U) / ALIGN_32;
        scaleBytes = static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(uint16_t) * DEQUANT_BF16_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32))) +
                     static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32)));
    }
}

/*
 * 按 weight 元素个数刷新 unpermute 的 weight 区大小：FP32 主区必算，转换中转区只有需要 dtype
 * 转换时才占 UB。两个阶段定完 tokensPerBatch 后都要重算一次，所以抽出来复用。
 */
static void SetUnpermuteWeightBufferBytes(MegaMoeUnpermuteBufferConfig &bufferConfig, uint32_t weightElementCount,
                                          uint32_t topKWeightsConversionElementBytes)
{
    bufferConfig.topKWeightsBufferBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    if (topKWeightsConversionElementBytes > 0U) {
        bufferConfig.topKWeightsConversionBufferBytes = static_cast<uint32_t>(
            ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * topKWeightsConversionElementBytes,
                           static_cast<uint64_t>(ALIGN_32)));
    }
}

static MegaMoeUnpermuteBufferConfig CalcUnpermuteBufferConfig(const MegaMoeTilingData *tilingData,
                                                              uint32_t coreTokenCount,
                                                              uint32_t topKWeightsConversionElementBytes,
                                                              uint32_t availableUbBytes)
{
    MegaMoeUnpermuteBufferConfig bufferConfig{};
    if (coreTokenCount == 0U) {
        return bufferConfig;
    }

    uint32_t dataSlotBytes = 0U;
    uint32_t scaleBytes = 0U;
    CalcUnpermuteSlotAndScaleBytes(tilingData, bufferConfig, dataSlotBytes, scaleBytes);

    // 第一步：先按基准 weight batch 定住每批 token 数和 weight 区，再拿剩余 UB 开输入 ring。
    uint32_t baseTokensPerBatch = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / tilingData->topK;
    bufferConfig.tokensPerBatch = static_cast<int32_t>(std::min(baseTokensPerBatch, coreTokenCount));
    uint32_t weightElementCount = static_cast<uint32_t>(bufferConfig.tokensPerBatch) * tilingData->topK;
    SetUnpermuteWeightBufferBytes(bufferConfig, weightElementCount, topKWeightsConversionElementBytes);

    // bytesBeforeInputBuffers 包含 weight、scale 和一个累加/输出 data slot。
    uint32_t bytesBeforeInputBuffers = bufferConfig.topKWeightsBufferBytes +
                                       bufferConfig.topKWeightsConversionBufferBytes + scaleBytes + dataSlotBytes;
    uint32_t inputBufferBudgetBytes =
        availableUbBytes > bytesBeforeInputBuffers ? availableUbBytes - bytesBeforeInputBuffers : 0U;
    bufferConfig.inputBufferCount = static_cast<int32_t>(inputBufferBudgetBytes / dataSlotBytes);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, MAX_UNPERMUTE_INPUT_BUFFER_COUNT);
    int32_t accumulationItemCount =
        bufferConfig.tokensPerBatch * static_cast<int32_t>(tilingData->topK + tilingData->sharedExpertNum);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, accumulationItemCount);
    bufferConfig.inputBufferCount = std::max(bufferConfig.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);

    // 第二步：ring 深度已定，UB 还有富余就把 weight batch 撑大。
    if (baseTokensPerBatch < coreTokenCount) {
        // fixedBytes 包含 scale、一个累加/输出 slot 和已经选中的所有输入 slot。
        uint32_t fixedBytes = scaleBytes + (static_cast<uint32_t>(bufferConfig.inputBufferCount) + 1U) * dataSlotBytes;
        // weightBudgetBytes 是 FP32 weight 及可选转换中转区可使用的 UB。
        uint32_t weightBudgetBytes = availableUbBytes - fixedBytes - UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES;
        uint32_t weightBytesPerToken =
            tilingData->topK * (static_cast<uint32_t>(sizeof(float)) + topKWeightsConversionElementBytes);
        uint32_t expandedTokensPerBatch = std::min(weightBudgetBytes / weightBytesPerToken, coreTokenCount);
        if (expandedTokensPerBatch > static_cast<uint32_t>(bufferConfig.tokensPerBatch)) {
            bufferConfig.tokensPerBatch = static_cast<int32_t>(expandedTokensPerBatch);
            weightElementCount = expandedTokensPerBatch * tilingData->topK;
            SetUnpermuteWeightBufferBytes(bufferConfig, weightElementCount, topKWeightsConversionElementBytes);
        }
    }
    return bufferConfig;
}

static uint64_t CalcCombineSyncSlotCountPerExpert(const MegaMoeTilingData *tilingData)
{
    // MTE 统一使用 per-expert AIC ready 表；group counter 服务 URMA layered Combine，量化与非量化均需要。
    if (tilingData->topoType != TOPO_TYPE_URMA || tilingData->moeExpertPerRank == 0U) {
        return 0U;
    }

    // 上述 guard 保证这里只剩 URMA；layered Combine 仅由 subBlockIdx=1 的半数 AIV 执行。
    uint64_t combineCoreCount = tilingData->blockAivNum / 2U;
    // 同一 token 的 topK expert id 不重复，因此单 expert 从每张卡最多接收 bs 个 token。
    uint64_t maxTokenCountForOneExpert =
        static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->epWorldSize;
    uint64_t maxTokenGroupCountForOneExpert =
        ops::CeilDiv(maxTokenCountForOneExpert, static_cast<uint64_t>(COMBINE_TOKEN_GROUP_SIZE));
    // Workspace 在路由结果产生前分配，因此每个本卡 MoE expert 都按独立最坏情况预留 slot。
    return std::max(maxTokenGroupCountForOneExpert, combineCoreCount);
}

static uint64_t CalcHostFlagElementCount(const MegaMoeTilingData *tilingData)
{
    uint64_t maxWavesPerExpert = ops::CeilDiv<uint64_t>(tilingData->maxOutputSize, L1_TILE_M_256);
    uint64_t waveFlagSlotsPerExpert = maxWavesPerExpert * INT_CACHELINE;
    uint64_t activationFlagSlotsPerExpert =
        tilingData->topoType == TOPO_TYPE_MTE ? waveFlagSlotsPerExpert : INT_CACHELINE;
    uint64_t moeExpertCount = tilingData->moeExpertPerRank;

    uint64_t flagElementCount = moeExpertCount * (activationFlagSlotsPerExpert + waveFlagSlotsPerExpert +
                                                  static_cast<uint64_t>(INT_CACHELINE) * tilingData->aicNum);
    bool isW4Mode = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
                    tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
                    tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ;
    if (isW4Mode || (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode == COMBINE_NO_QUANT)) {
        flagElementCount += static_cast<uint64_t>(tilingData->aicNum) * INT_CACHELINE;
    }
    if (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        flagElementCount += moeExpertCount * tilingData->aicNum * INT_CACHELINE;
    }
    if (tilingData->topoType == TOPO_TYPE_URMA) {
        flagElementCount += tilingData->combineSyncSlotCountPerExpert * moeExpertCount * INT_CACHELINE;
    }
    if (tilingData->sharedExpertNum > 0 && tilingData->topoType == TOPO_TYPE_MTE) {
        uint64_t tokenGroupCount = ops::CeilDiv<uint64_t>(tilingData->bs, L1_TILE_M_256);
        flagElementCount += tokenGroupCount * tilingData->sharedExpertNum * INT_CACHELINE;
    }
    return flagElementCount;
}

/*
 * 设置 topK 有效下标发送的两套 UB 配置：先算固定占用，再按 expert 分核的两类 core 各算一套。
 * 本函数读 tilingData->combineSyncSlotCountPerExpert（经 CalcHostFlagElementCount），
 * 该字段必须在调用前写好。
 */
static void SetTopkValidIndexBufferConfigs(MegaMoeTilingData *tilingData, uint32_t activationElementsPerByte,
                                           uint32_t availableUbBytes)
{
    uint64_t totalFlagElementCount = CalcHostFlagElementCount(tilingData);
    uint32_t resetElementCountPerCore =
        static_cast<uint32_t>(ops::CeilDiv(totalFlagElementCount, static_cast<uint64_t>(tilingData->blockAivNum)));
    uint32_t resetBatchElementCount = std::min(resetElementCountPerCore, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    uint32_t resetTensorBytes =
        ops::CeilAlign(resetBatchElementCount, static_cast<uint32_t>(INT32_PER_256B)) * sizeof(int32_t);
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    // 与 kernel xOutTensorSize 一致：token 和 scale 分别对齐，prefetch 模式再追加对齐后的 weight。
    uint32_t quantOutputBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        quantOutputBufferBytes +=
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    }
    uint32_t quantInputBufferBytes = ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_128)) * sizeof(uint16_t);
    // sendCntAccTensor_ 按本卡 MoE 专家数分配，与 kernel 地址布局一致。
    uint32_t maxExpertCountPerCore =
        ops::CeilDiv(tilingData->epWorldSize * tilingData->moeExpertPerRank, tilingData->blockAivNum);
    uint32_t sendCountAccumulatorBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(maxExpertCountPerCore) * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32)));
    // mxTempTensor_ 占 2KB，xOutTensor_ 和 xInTensor_ 各使用双 buffer。
    uint32_t sendMaskFixedBufferBytes = resetTensorBytes + 2U * 1024U + 2U * quantOutputBufferBytes +
                                        2U * quantInputBufferBytes + sendCountAccumulatorBytes;

    /*
     * 与 kernel topK 有效下标发送的 expert 连续均衡分核一一对应。totalExpertCount 除以 blockAivNum 后，
     * 前 remainder 个 AIV job 各多处理一个 expert，其余 job 处理 quotient 个 expert，因此这里只需
     * 预计算两套配置。Dispatch/Combine 的 wave 内 token 轮转由各自阶段完成，与这里的一次性发送分核无关。
     *
     * 若修改发送阶段的 expert 分核方式、ownedExpertCount 或 routePushCount 计算，必须同步更新
     * 这里的两类 core 划分和 kernel 配置选择条件。
     */
    uint32_t totalExpertCount = tilingData->epWorldSize * tilingData->moeExpertPerRank;
    uint32_t expertCountPerCoreWithoutExtraExpert = totalExpertCount / tilingData->blockAivNum;
    tilingData->sendMaskCoreCountWithExtraExpert = totalExpertCount % tilingData->blockAivNum;
    uint32_t expertCountPerCoreWithExtraExpert = expertCountPerCoreWithoutExtraExpert + 1U;
    tilingData->sendMaskConfigForCoreWithExtraExpert = CalcTopkValidIndexBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithExtraExpert, availableUbBytes);
    tilingData->sendMaskConfigForCoreWithoutExtraExpert = CalcTopkValidIndexBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithoutExtraExpert, availableUbBytes);
}

/*
 * 设置 Unpermute 的完整 chunk 与 tail chunk 两套 UB 配置，并记录完整 chunk 对应的 core 数。
 */
static void SetUnpermuteBufferConfigs(MegaMoeTilingData *tilingData, ge::DataType topKWeightsDataType,
                                      uint32_t availableUbBytes)
{
    /*
     * 与 kernel Unpermute 开头的 TilingByCore(m_, ..., align=1) 一一对应。TilingByCore 使用：
     *   fullTokenChunkSize = ceil(bs / blockAivNum)
     * 为连续 core 分配等长完整 chunk，最后一个活跃 core 可能只处理 tail，后续 core 的 coreLen 为 0
     * 并在读取配置前返回。因此 host 只需预计算“完整 chunk”和“tail chunk”两套配置，并记录完整
     * chunk 对应的 core 数作为 kernel 选择边界。
     *
     * 若修改 TilingByCore、Unpermute 的 align 参数或分核方式，必须同步更新下面的 chunk 推导以及
     * kernel 中 UnpermuteBuffInit 的配置选择条件。
     */
    // FP32 可直接搬入计算 buffer；其他已支持类型按实际元素大小预留转换中转区。
    uint32_t topKWeightsConversionElementBytes =
        topKWeightsDataType == ge::DT_FLOAT ? 0U : static_cast<uint32_t>(ge::GetSizeByDataType(topKWeightsDataType));
    uint32_t fullTokenChunkSize = ops::CeilDiv(tilingData->bs, tilingData->blockAivNum);
    uint32_t activeCoreCount = ops::CeilDiv(tilingData->bs, fullTokenChunkSize);
    uint32_t tailTokenChunkSize = tilingData->bs - (activeCoreCount - 1U) * fullTokenChunkSize;
    bool tailIsFullTokenChunk = tailTokenChunkSize == fullTokenChunkSize;
    tilingData->unpermuteFullTokenChunkCoreCount = tailIsFullTokenChunk ? activeCoreCount : activeCoreCount - 1U;
    tilingData->unpermuteConfigForFullTokenChunk =
        CalcUnpermuteBufferConfig(tilingData, fullTokenChunkSize, topKWeightsConversionElementBytes, availableUbBytes);
    tilingData->unpermuteConfigForTailTokenChunk =
        tailIsFullTokenChunk ? MegaMoeUnpermuteBufferConfig{} :
                               CalcUnpermuteBufferConfig(tilingData, tailTokenChunkSize,
                                                         topKWeightsConversionElementBytes, availableUbBytes);
}

static ge::graphStatus SetAdaptiveBufferConfigs(const gert::TilingContext *context, MegaMoeConfig &config,
                                                MegaMoeTilingData *tilingData, uint32_t availableUbBytes)
{
    const char *nodeName = context->GetNodeName();
    auto topKWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    OP_TILING_CHECK(topKWeightsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkWeights"),
                    return ge::GRAPH_FAILED);

    uint32_t activationElementsPerByte =
        GetOpQuantModeByAttrDispatchOutType(context, config) == DISPATCH_QUANT_OUT_DTYPE_E2M1 ? 2U : 1U;
    // 所有 AIV 核共用同一套 dispatch UB 配置：这块布局和怎么分核无关。
    // kernel 那边如果改成按核拆 tensor，或者动了 copyTmp 的槽位布局，这里要跟着改。
    tilingData->dispatchBufferConfig =
        CalcDispatchBufferConfig(tilingData, activationElementsPerByte, availableUbBytes);

    // 先于 topK 有效下标配置写入：CalcHostFlagElementCount 会读这个字段累加 flag 区大小，
    // 该依赖只经 tilingData 传递，调换顺序会静默改变 flag 区尺寸。
    tilingData->combineSyncSlotCountPerExpert = CalcCombineSyncSlotCountPerExpert(tilingData);

    SetTopkValidIndexBufferConfigs(tilingData, activationElementsPerByte, availableUbBytes);

    ge::DataType topKWeightsDataType = topKWeightsDesc->GetDataType();
    SetUnpermuteBufferConfigs(tilingData, topKWeightsDataType, availableUbBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrAndSetTilingData(const gert::TilingContext *context, MegaMoeConfig &config,
                                                 MegaMoeTilingData *tilingData, const uint32_t aicNum)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckActivationParams(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check activation params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrParams(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetAttrParams(context, config, tilingData, nodeName, aicNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, const WorkspaceLayout &workspaceLayout,
                                    const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspace"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(workspaceLayout.workspaceSize == 0LL,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "workspaceSize",
                                              std::to_string(workspaceLayout.workspaceSize).c_str(), "non-zero"),
                    return ge::GRAPH_FAILED);

    int64_t workspaceSize = sysWorkspaceSize + workspaceLayout.workspaceSize + RESERVED_WORKSPACE_SIZE;
    workspace[0] = workspaceSize;

    OP_LOGD(nodeName, "sysWorkspaceSize: %ld \n", sysWorkspaceSize);
    OP_LOGD(nodeName, "mega_moe_tiling workspaceSize: %ld \n", workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context, MegaMoeConfig &config,
                                             const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsDesc);

    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(xActiveMaskDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "xActiveMask", "not null", "null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scalesDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "scales", "not null", "null"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

struct NamedIndex {
    uint32_t index;
    const char *name;
};

// 组合一组专家的两个权重和对应 scale 输入，供 MoE 专家与共享专家复用同一组内校验。
struct ExpertWeightTensorGroupInputs {
    NamedIndex weightOne;
    NamedIndex weightTwo;
    NamedIndex weightScalesOne;
    NamedIndex weightScalesTwo;
    const char *expertTypeName;
};

/*
 * 校验同一 TensorList 内所有 tensor 均为预期维数，且 shape、dtype 和 format 一致。
 * 以第 0 项为基准，从第 1 项开始逐项比较。
 */
static ge::graphStatus CheckTensorPropertiesWithinList(const gert::TilingContext *context, const NamedIndex &input,
                                                       uint32_t expectedDimNum, const char *nodeName)
{
    uint32_t tensorCount = GetDynamicInputTensorCount(context, input.index);
    OP_TILING_CHECK(tensorCount == 0U, OP_LOGE_WITH_INVALID_INPUT(nodeName, input.name), return ge::GRAPH_FAILED);

    auto referenceShape = context->GetDynamicInputShape(input.index, 0);
    auto referenceDesc = context->GetDynamicInputDesc(input.index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceDesc);
    uint32_t referenceDimNum = referenceShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(referenceDimNum != expectedDimNum,
                    OP_LOGE(nodeName, "%s[0] must be %uD.", input.name, expectedDimNum), return ge::GRAPH_FAILED);

    for (uint32_t tensorIdx = 1; tensorIdx < tensorCount; ++tensorIdx) {
        auto currentShape = context->GetDynamicInputShape(input.index, tensorIdx);
        auto currentDesc = context->GetDynamicInputDesc(input.index, tensorIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentDesc);

        uint32_t currentDimNum = currentShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(currentDimNum != expectedDimNum,
                        OP_LOGE(nodeName, "%s[%u] must be %uD.", input.name, tensorIdx, expectedDimNum),
                        return ge::GRAPH_FAILED);
        bool shapeMismatch = false;
        for (uint32_t dimIdx = 0; dimIdx < referenceDimNum && !shapeMismatch; ++dimIdx) {
            shapeMismatch =
                currentShape->GetStorageShape().GetDim(dimIdx) != referenceShape->GetStorageShape().GetDim(dimIdx);
        }
        OP_TILING_CHECK(
            shapeMismatch,
            OP_LOGE(nodeName, "%s[%u] must have the same shape as %s[0].", input.name, tensorIdx, input.name),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(currentDesc->GetDataType() != referenceDesc->GetDataType(),
                        OP_LOGE(nodeName, "The dtype of %s[%u] must match %s[0].", input.name, tensorIdx, input.name),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(currentDesc->GetStorageFormat() != referenceDesc->GetStorageFormat(),
                        OP_LOGE(nodeName, "The format of %s[%u] must match %s[0].", input.name, tensorIdx, input.name),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 MoE 专家或共享专家的 weight1、weight2、weightScales1 和 weightScales2 四个 TensorList。
 * 先校验每个 TensorList 内部的 tensor 属性一致，再校验四个 TensorList 的长度及堆叠布局下的专家数一致。
 */
static ge::graphStatus CheckExpertWeightInputGroupLayout(const gert::TilingContext *context,
                                                         const ExpertWeightTensorGroupInputs &inputs,
                                                         uint32_t weightDimNum, const char *nodeName)
{
    // scale 比对应的 weight 多一个 multi-base 维度。
    uint32_t scaleDimNum = weightDimNum + 1U;
    OP_TILING_CHECK(
        CheckTensorPropertiesWithinList(context, inputs.weightOne, weightDimNum, nodeName) != ge::GRAPH_SUCCESS ||
            CheckTensorPropertiesWithinList(context, inputs.weightTwo, weightDimNum, nodeName) != ge::GRAPH_SUCCESS ||
            CheckTensorPropertiesWithinList(context, inputs.weightScalesOne, scaleDimNum, nodeName) !=
                ge::GRAPH_SUCCESS ||
            CheckTensorPropertiesWithinList(context, inputs.weightScalesTwo, scaleDimNum, nodeName) !=
                ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "%s weight layout is invalid.", inputs.expertTypeName), return ge::GRAPH_FAILED);

    // 四个 TensorList 以相同下标表示同一专家，因此长度必须一致；堆叠布局只允许一个 tensor。
    uint32_t weightOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightOne.index);
    uint32_t weightTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightTwo.index);
    uint32_t weightScalesOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesOne.index);
    uint32_t weightScalesTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesTwo.index);
    bool tensorCountsMismatch = weightTwoTensorCount != weightOneTensorCount ||
                                weightScalesOneTensorCount != weightOneTensorCount ||
                                weightScalesTwoTensorCount != weightOneTensorCount;
    OP_TILING_CHECK(
        tensorCountsMismatch || (weightDimNum == THREE_DIMS && weightOneTensorCount != 1U),
        OP_LOGE(nodeName,
                "%s, %s, %s and %s must contain the same number of tensors; stacked layout requires "
                "exactly one tensor.",
                inputs.weightOne.name, inputs.weightTwo.name, inputs.weightScalesOne.name, inputs.weightScalesTwo.name),
        return ge::GRAPH_FAILED);

    if (weightDimNum == THREE_DIMS) {
        // 堆叠布局中四个 tensor 的 dim0 都表示专家数，必须一致。
        auto weightOneShape = context->GetDynamicInputShape(inputs.weightOne.index, 0);
        auto weightTwoShape = context->GetDynamicInputShape(inputs.weightTwo.index, 0);
        auto weightScalesOneShape = context->GetDynamicInputShape(inputs.weightScalesOne.index, 0);
        auto weightScalesTwoShape = context->GetDynamicInputShape(inputs.weightScalesTwo.index, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);
        int64_t expertCount = weightOneShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(weightTwoShape->GetStorageShape().GetDim(0) != expertCount ||
                            weightScalesOneShape->GetStorageShape().GetDim(0) != expertCount ||
                            weightScalesTwoShape->GetStorageShape().GetDim(0) != expertCount,
                        OP_LOGE(nodeName, "Dim0 of %s, %s, %s and %s must have matching expert counts for %s inputs.",
                                inputs.weightOne.name, inputs.weightTwo.name, inputs.weightScalesOne.name,
                                inputs.weightScalesTwo.name, inputs.expertTypeName),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 以 MoE weight1[0] 的维数确定公共权重布局，2D 表示逐专家 TensorList，3D 表示单 tensor 堆叠。
 * 校验 MoE 的 weight1、weight2、weightScales1 和 weightScales2。
 * 检查共享专家的对应输入是否同时缺省或同时存在，并在存在时校验其布局与 MoE 一致。
 */
static ge::graphStatus CheckMoeAndSharedWeightInputLayouts(const gert::TilingContext *context,
                                                           const MegaMoeConfig &config, const char *nodeName)
{
    auto weightOneShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    uint32_t weightDimNum = weightOneShape->GetStorageShape().GetDimNum();
    bool isPerExpertWeightTensor = weightDimNum == TWO_DIMS;
    OP_TILING_CHECK(
        weightDimNum != TWO_DIMS && weightDimNum != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "weight1", (std::to_string(weightDimNum) + "D").c_str(),
                                                 "weight1 must be one stacked 3D tensor or a list of 2D tensors."),
        return ge::GRAPH_FAILED);

    const ExpertWeightTensorGroupInputs moeInputs{{config.weight1Index, "weight1"},
                                                  {config.weight2Index, "weight2"},
                                                  {config.weightScales1Index, "weight_scales1"},
                                                  {config.weightScales2Index, "weight_scales2"},
                                                  "MoE expert"};
    const ExpertWeightTensorGroupInputs sharedInputs{{config.sharedWeight1Index, "shared_weight1"},
                                                     {config.sharedWeight2Index, "shared_weight2"},
                                                     {config.sharedWeightScales1Index, "shared_weight_scales1"},
                                                     {config.sharedWeightScales2Index, "shared_weight_scales2"},
                                                     "shared expert"};
    OP_TILING_CHECK(CheckExpertWeightInputGroupLayout(context, moeInputs, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "MoE expert weight layout is invalid."), return ge::GRAPH_FAILED);

    // 检查共享专家的四个 weight/scale 输入是否同时存在或同时缺省。
    bool hasSharedWeightOne = GetWeightExpertCount(context, config.sharedWeight1Index, isPerExpertWeightTensor) > 0;
    bool hasSharedWeightTwo = GetWeightExpertCount(context, config.sharedWeight2Index, isPerExpertWeightTensor) > 0;
    bool hasSharedWeightScalesOne =
        GetWeightExpertCount(context, config.sharedWeightScales1Index, isPerExpertWeightTensor) > 0;
    bool hasSharedWeightScalesTwo =
        GetWeightExpertCount(context, config.sharedWeightScales2Index, isPerExpertWeightTensor) > 0;
    OP_TILING_CHECK(hasSharedWeightOne != hasSharedWeightTwo || hasSharedWeightOne != hasSharedWeightScalesOne ||
                        hasSharedWeightOne != hasSharedWeightScalesTwo,
                    OP_LOGE(nodeName, "Shared expert weights and weight scales must be provided together."),
                    return ge::GRAPH_FAILED);
    if (hasSharedWeightOne) {
        OP_TILING_CHECK(
            CheckExpertWeightInputGroupLayout(context, sharedInputs, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Shared expert weight layout is invalid."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                            const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    auto weightTwoStorageShape = context->GetDynamicInputShape(config.weight2Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);

    uint32_t weightOneDimNum = weightOneStorageShape->GetStorageShape().GetDimNum();
    bool isPerExpertWeightTensor = weightOneDimNum == TWO_DIMS;

    // 去掉可选的专家维后，weight1 和 weight2 均按单专家二维矩阵读取行数和列数。
    const int64_t weightOneRowCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightOneColumnCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightTwoRowCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightTwoColumnCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);

    // 单专家 GMM 形状：weight1=[N, H]，weight2=[H, N/2]，x=[BS, H]。
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xColumnCount = xStorageShape->GetStorageShape().GetDim(1);
    const std::string commonMatrixDimensionsString = "[" + std::to_string(weightOneColumnCount) + ", " +
                                                     std::to_string(weightTwoRowCount) + ", " +
                                                     std::to_string(xColumnCount) + "]";
    OP_TILING_CHECK(weightOneColumnCount != weightTwoRowCount || weightOneColumnCount != xColumnCount,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "weight1, weight2 and x", commonMatrixDimensionsString.c_str(),
                        "The column count of weight1 and the row count of weight2 must equal the column count of x."),
                    return ge::GRAPH_FAILED);

    const std::string weightRowColumnDimensionsString =
        "[" + std::to_string(weightOneRowCount) + ", " + std::to_string(weightTwoColumnCount) + "]";
    OP_TILING_CHECK(weightOneRowCount != weightTwoColumnCount * SWIGLU_GATE_UP_SPLIT_FACTOR,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "weight1 and weight2", weightRowColumnDimensionsString.c_str(),
                        "The row count of weight1 must equal the column count of weight2 multiplied by 2."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验共享专家 TensorList 与对应 MoE TensorList 的 dtype 一致，并按 dimensions 逐项校验单专家 shape。
 * 两个 TensorList 的内部一致性已在前序校验，因此只需校验第 0 项。
 */
static ge::graphStatus CheckSharedTensorMatchesMoeTensor(const gert::TilingContext *context,
                                                         const NamedIndex &sharedInput, const NamedIndex &moeInput,
                                                         bool isPerExpertWeightTensor, const NamedIndex *dimensions,
                                                         uint32_t dimensionCount, bool checkStorageFormat,
                                                         const char *nodeName)
{
    auto sharedDesc = context->GetDynamicInputDesc(sharedInput.index, 0);
    auto sharedShape = context->GetDynamicInputShape(sharedInput.index, 0);
    auto moeDesc = context->GetDynamicInputDesc(moeInput.index, 0);
    auto moeShape = context->GetDynamicInputShape(moeInput.index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sharedDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, sharedShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, moeDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, moeShape);

    OP_TILING_CHECK(sharedDesc->GetDataType() != moeDesc->GetDataType(),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, (std::string(sharedInput.name) + " dtype").c_str(),
                                              std::to_string(sharedDesc->GetDataType()).c_str(),
                                              (std::string("must be same as ") + moeInput.name).c_str()),
                    return ge::GRAPH_FAILED);
    if (checkStorageFormat) {
        OP_TILING_CHECK(
            sharedDesc->GetStorageFormat() != moeDesc->GetStorageFormat(),
            OP_LOGE(nodeName, "%s and %s must use the same storage format.", sharedInput.name, moeInput.name),
            return ge::GRAPH_FAILED);
    }

    for (uint32_t dimensionIdx = 0; dimensionIdx < dimensionCount; ++dimensionIdx) {
        const auto &dimension = dimensions[dimensionIdx];
        const int64_t sharedDimensionSize =
            GetSingleExpertTensorDimSize(sharedShape, dimension.index, isPerExpertWeightTensor);
        const int64_t moeDimensionSize =
            GetSingleExpertTensorDimSize(moeShape, dimension.index, isPerExpertWeightTensor);
        OP_TILING_CHECK(sharedDimensionSize != moeDimensionSize,
                        OP_LOGE_FOR_INVALID_VALUE(
                            nodeName, (std::string(sharedInput.name) + " " + dimension.name).c_str(),
                            std::to_string(sharedDimensionSize).c_str(),
                            (std::string("must be equal to ") + moeInput.name + " " + dimension.name).c_str()),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSharedExpertInputs(const gert::TilingContext *context, MegaMoeConfig &config,
                                               const char *nodeName)
{
    const NamedIndex sharedWeightOneInput{config.sharedWeight1Index, "shared_weight1"};
    const NamedIndex sharedWeightTwoInput{config.sharedWeight2Index, "shared_weight2"};
    const NamedIndex sharedScaleOneInput{config.sharedWeightScales1Index, "shared_weight_scales1"};
    const NamedIndex sharedScaleTwoInput{config.sharedWeightScales2Index, "shared_weight_scales2"};
    const NamedIndex moeWeightOneInput{config.weight1Index, "weight1"};
    const NamedIndex moeWeightTwoInput{config.weight2Index, "weight2"};
    const NamedIndex moeScaleOneInput{config.weightScales1Index, "weight_scales1"};
    const NamedIndex moeScaleTwoInput{config.weightScales2Index, "weight_scales2"};

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);

    bool isPerExpertWeightTensor = weightOneStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS;
    if (GetWeightExpertCount(context, config.sharedWeight1Index, isPerExpertWeightTensor) == 0) {
        return ge::GRAPH_SUCCESS;
    }
    // 组内一致性已校验，跨组关系只需比较第 0 项。
    const NamedIndex weightDimensions[] = {
        {WEIGHT_MATRIX_ROW_DIM_INDEX, "row count"},
        {WEIGHT_MATRIX_COLUMN_DIM_INDEX, "column count"},
    };
    OP_TILING_CHECK(
        CheckSharedTensorMatchesMoeTensor(context, sharedWeightOneInput, moeWeightOneInput, isPerExpertWeightTensor,
                                          weightDimensions, TWO_DIMS, true, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check shared_weight1 failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckSharedTensorMatchesMoeTensor(context, sharedWeightTwoInput, moeWeightTwoInput, isPerExpertWeightTensor,
                                          weightDimensions, TWO_DIMS, true, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check shared_weight2 failed."), return ge::GRAPH_FAILED);

    const NamedIndex scaleDimensions[] = {
        {WEIGHT_SCALE_MATRIX_DIM_INDEX, "matrix dimension"},
        {WEIGHT_SCALE_GROUP_DIM_INDEX, "group dimension"},
        {WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, "multi-base dimension"},
    };
    OP_TILING_CHECK(
        CheckSharedTensorMatchesMoeTensor(context, sharedScaleOneInput, moeScaleOneInput, isPerExpertWeightTensor,
                                          scaleDimensions, THREE_DIMS, false, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check shared_weight_scales1 failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckSharedTensorMatchesMoeTensor(context, sharedScaleTwoInput, moeScaleTwoInput, isPerExpertWeightTensor,
                                          scaleDimensions, THREE_DIMS, false, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check shared_weight_scales2 failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                            const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    auto yStorageShape = context->GetOutputShape(config.yIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorageShape);
    OP_TILING_CHECK(
        yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "y", (std::to_string(yStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);
    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "y dim0 = %ld", yDim0);
    OP_LOGD(nodeName, "y dim1 = %ld", yDim1);

    OP_TILING_CHECK(
        yDim0 != bs || yDim1 != h,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "y", (std::string("[") + std::to_string(yDim0) + ", " + std::to_string(yDim1) + "]").c_str(),
            (std::string("The shape of y must be [bs, h] = [") + std::to_string(bs) + ", " + std::to_string(h) + "].")
                .c_str()),
        return ge::GRAPH_FAILED);

    auto expertTokenNumsStorageShape = context->GetOutputShape(config.expertTokenNumsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsStorageShape);
    OP_TILING_CHECK(
        expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "expert_token_nums",
            (std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsDim0);

    // expertTokenNums 仅报告 MoE 专家的 token 数，不包含共享专家。
    auto attrs = context->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    int64_t moeExpertPerRank = static_cast<int64_t>(*moeExpertNumPtr) / static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(
        expertTokenNumsDim0 != moeExpertPerRank,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "expertTokenNums", (std::string("dim0=") + std::to_string(expertTokenNumsDim0)).c_str(),
            (std::string("The shape [dim0] of expertTokenNums must be equal to moeExpertPerRank(") +
             std::to_string(moeExpertPerRank) + ").")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一组 weight 和它的 scale 形状对不对得上。weight 每 INPUT_WEIGHT_SCALES_CEIL_ALIGN 列共用一个
 * scale，所以 scale 的形状必然是 [weight 行数, ceil(weight 列数 / 该对齐值)]。
 * weight1 和 weight2 两组走的是同一套规则，所以共用这个函数。
 * scale 末尾那个 multi-base 维两组要求一致，在调用方 CheckWeightScalesTensorDim 里一起比。
 */
static ge::graphStatus CheckWeightScaleTensorDim(const gert::StorageShape *weightScaleStorageShape,
                                                 const gert::StorageShape *weightStorageShape,
                                                 bool isPerExpertWeightTensor, const char *weightScaleName,
                                                 const char *weightName, const char *nodeName)
{
    const int64_t weightScaleMatrixDimSize =
        GetSingleExpertTensorDimSize(weightScaleStorageShape, WEIGHT_SCALE_MATRIX_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightScaleGroupDimSize =
        GetSingleExpertTensorDimSize(weightScaleStorageShape, WEIGHT_SCALE_GROUP_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightRowCount =
        GetSingleExpertTensorDimSize(weightStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightColumnCount =
        GetSingleExpertTensorDimSize(weightStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);

    OP_TILING_CHECK(weightScaleMatrixDimSize != weightRowCount,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, weightScaleName,
                        (std::string("matrix dimension=") + std::to_string(weightScaleMatrixDimSize)).c_str(),
                        (std::string("The matrix dimension of ") + weightScaleName + " must equal the row count of " +
                         weightName + "(" + std::to_string(weightRowCount) + ").")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(weightScaleGroupDimSize != ops::CeilDiv(weightColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, weightScaleName,
                        (std::string("group dimension=") + std::to_string(weightScaleGroupDimSize)).c_str(),
                        (std::string("The group dimension of ") + weightScaleName + " must equal CeilDiv(" +
                         weightName + " column count, INPUT_WEIGHT_SCALES_CEIL_ALIGN) = " +
                         std::to_string(ops::CeilDiv(weightColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN)) + ".")
                            .c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightScalesTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                                  const char *nodeName)
{
    auto weightScalesOneStorageShape = context->GetDynamicInputShape(config.weightScales1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneStorageShape);
    auto weightScalesTwoStorageShape = context->GetDynamicInputShape(config.weightScales2Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoStorageShape);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    auto weightTwoStorageShape = context->GetDynamicInputShape(config.weight2Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);
    bool isPerExpertWeightTensor = weightOneStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS;

    ge::graphStatus checkStatus =
        CheckWeightScaleTensorDim(weightScalesOneStorageShape, weightOneStorageShape, isPerExpertWeightTensor,
                                  "weightScales1", "weight1", nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }
    checkStatus = CheckWeightScaleTensorDim(weightScalesTwoStorageShape, weightTwoStorageShape, isPerExpertWeightTensor,
                                            "weightScales2", "weight2", nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    // multi-base 是两组 scale 的共同约束，一次性比对两者。
    const int64_t weightScalesOneMultiBaseDimSize = GetSingleExpertTensorDimSize(
        weightScalesOneStorageShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightScalesTwoMultiBaseDimSize = GetSingleExpertTensorDimSize(
        weightScalesTwoStorageShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, isPerExpertWeightTensor);
    const std::string scaleMultiBaseDimensionsString = "[" + std::to_string(weightScalesOneMultiBaseDimSize) + ", " +
                                                       std::to_string(weightScalesTwoMultiBaseDimSize) + "]";
    OP_TILING_CHECK(weightScalesOneMultiBaseDimSize != WEIGHT_SCALE_MULTI_BASE_DIM_SIZE ||
                        weightScalesTwoMultiBaseDimSize != WEIGHT_SCALE_MULTI_BASE_DIM_SIZE,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "weightScales1, weightScales2", scaleMultiBaseDimensionsString.c_str(),
                        "The per-expert trailing dimension of weightScales1 and weightScales2 must be 2."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验基础输入张量的维度：context 为 1D，x / topkIds / topkWeights 为 2D，
 * 并校验三者之间的 dim0 / dim1 一致性。
 */
static ge::graphStatus CheckBasicInputTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                                const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "context", (std::to_string(contextStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of context must be 1D."),
        return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "x", (std::to_string(xStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of x must be 2D."),
                    return ge::GRAPH_FAILED);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    OP_TILING_CHECK(topkIdsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkIds"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "topkIds", (std::to_string(topkIdsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of topkIds must be 2D."),
        return ge::GRAPH_FAILED);
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkIds dim0 = %ld", topkIdsDim0);
    OP_LOGD(nodeName, "topkIds dim1 = %ld", topkIdsDim1);

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkWeights"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "topkWeights",
                        (std::to_string(topkWeightsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of topkWeights must be 2D."),
                    return ge::GRAPH_FAILED);
    const int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkWeights dim0 = %ld", topkWeightsDim0);
    OP_LOGD(nodeName, "topkWeights dim1 = %ld", topkWeightsDim1);

    OP_TILING_CHECK(xDim0 != topkIdsDim0 || xDim0 != topkWeightsDim0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "x, topkIds, topkWeights",
                        (std::string("[") + std::to_string(xDim0) + ", " + std::to_string(topkIdsDim0) + ", " +
                         std::to_string(topkWeightsDim0) + "]")
                            .c_str(),
                        "The shape [dim0] of x, topkIds, and topkWeights must be equal."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        topkIdsDim1 != topkWeightsDim1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "topkIds, topkWeights",
            (std::string("[") + std::to_string(topkIdsDim1) + ", " + std::to_string(topkWeightsDim1) + "]").c_str(),
            "The shape [dim1] of topkIds and topkWeights must be equal."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验专家权重相关输入与输出的维度：权重布局、weight 形状、weight scale 形状、共享专家输入和输出形状。
 */
static ge::graphStatus CheckExpertWeightAndOutputTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                                           const char *nodeName)
{
    OP_TILING_CHECK(CheckMoeAndSharedWeightInputLayouts(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "expert weight tensor layout is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "weight params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightScalesTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check weight scales tensor dim failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckSharedExpertInputs(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check shared expert inputs failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckOutputTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "output params shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    ge::graphStatus checkStatus = CheckBasicInputTensorDim(context, config, nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    return CheckExpertWeightAndOutputTensorDim(context, config, nodeName);
}

/*
 * 校验基础输入张量的 dtype：context / x / topkIds / topkWeights。
 */
static ge::graphStatus CheckBasicInputDataType(const gert::TilingContext *context, MegaMoeConfig &config,
                                               const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "context",
                                                          Ops::Base::ToString(contextDesc->GetDataType()).c_str(),
                                                          "The dtype of context must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        xDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "The dtype of x must be DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "topkIds",
                                                          Ops::Base::ToString(topkIdsDesc->GetDataType()).c_str(),
                                                          "The dtype of topkIds must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((topkWeightsDesc->GetDataType() != ge::DT_BF16) && (topkWeightsDesc->GetDataType() != ge::DT_FLOAT)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "topkWeights",
                                              Ops::Base::ToString(topkWeightsDesc->GetDataType()).c_str(),
                                              "The dtype of topkWeights must be DT_FLOAT or DT_BF16."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验专家权重与其 scale 的 dtype：两个 weight 各自在量化类型集合内、彼此一致，scale 固定为 E8M0。
 */
static ge::graphStatus CheckWeightAndScaleDataType(const gert::TilingContext *context, MegaMoeConfig &config,
                                                   const char *nodeName)
{
    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);

    OP_TILING_CHECK(
        ((weightOneDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
         (weightOneDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
         (weightOneDesc->GetDataType() != ge::DT_FLOAT4_E2M1)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            nodeName, "weightOne", Ops::Base::ToString(weightOneDesc->GetDataType()).c_str(),
            "The dtype of weightOne must be within the range DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
         (weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
         (weightTwoDesc->GetDataType() != ge::DT_FLOAT4_E2M1)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            nodeName, "weightTwo", Ops::Base::ToString(weightTwoDesc->GetDataType()).c_str(),
            "The dtype of weightTwo must be within the range DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        weightOneDesc->GetDataType() != weightTwoDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, "weightOne, weightTwo",
                                               (std::string("[") + Ops::Base::ToString(weightOneDesc->GetDataType()) +
                                                ", " + Ops::Base::ToString(weightTwoDesc->GetDataType()) + "]")
                                                   .c_str(),
                                               "The dtypes of weightOne and weightTwo must be the same."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesOneDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "weightScalesOne", Ops::Base::ToString(weightScalesOneDesc->GetDataType()).c_str(),
                        "The dtype of weightScalesOne must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesTwoDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "weightScalesTwo", Ops::Base::ToString(weightScalesTwoDesc->GetDataType()).c_str(),
                        "The dtype of weightScalesTwo must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验输出张量的 dtype：y 与 expertTokenNums。
 */
static ge::graphStatus CheckOutputDataType(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(
        yDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "y", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                              "The dtype of y must be DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "expertTokenNums", Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str(),
                        "The dtype of expertTokenNums must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    ge::graphStatus checkStatus = CheckBasicInputDataType(context, config, nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    checkStatus = CheckWeightAndScaleDataType(context, config, nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    return CheckOutputDataType(context, config, nodeName);
}

static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context, MegaMoeConfig &config,
                                         const char *nodeName)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "x format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkIds format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkWeights format is invalid."), return ge::GRAPH_FAILED);

    bool isW4 = weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1;
    int64_t dispatchOutType = GetOpQuantModeByAttrDispatchOutType(context, config);
    ge::Format weightOnePrimaryFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat()));
    ge::Format weightTwoPrimaryFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightTwoDesc->GetStorageFormat()));
    OP_TILING_CHECK(!isW4 && weightOnePrimaryFormat != weightTwoPrimaryFormat,
                    OP_LOGE(nodeName, "weight1 and weight2 must use the same ND/NZ path."), return ge::GRAPH_FAILED);
    if (isW4 && dispatchOutType == DISPATCH_QUANT_OUT_DTYPE_E4M3FN) {
        OP_TILING_CHECK(weightOneDesc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ_C0_32,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "weight1",
                                                   Ops::Base::ToString(weightOneDesc->GetStorageFormat()).c_str(),
                                                   "FORMAT_FRACTAL_NZ_C0_32"),
                        return ge::GRAPH_FAILED);
    }
    if (isW4 && dispatchOutType == DISPATCH_QUANT_OUT_DTYPE_E2M1) {
        OP_TILING_CHECK(weightOneDesc->GetStorageFormat() != ge::FORMAT_ND &&
                            weightOneDesc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "weight1",
                                                   Ops::Base::ToString(weightOneDesc->GetStorageFormat()).c_str(),
                                                   "FORMAT_ND or FORMAT_FRACTAL_NZ"),
                        return ge::GRAPH_FAILED);
    }
    // W4 GMM2 固定使用 ZN/C0=32 prologue；weight2 不能再以逻辑 ND 描述交给 kernel 猜测物理布局。
    if (isW4) {
        OP_TILING_CHECK(weightTwoDesc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ_C0_32,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "weight2",
                                                   Ops::Base::ToString(weightTwoDesc->GetStorageFormat()).c_str(),
                                                   "FORMAT_FRACTAL_NZ_C0_32"),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "weightScalesOne format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesTwoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "weightScalesTwo format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "y format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "expertTokenNums format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMegaMoe(const gert::TilingContext *context, MegaMoeConfig &config,
                                          const char *nodeName)
{
    OP_TILING_CHECK(CheckTensorPtrNullptr(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDataType(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorFormat(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 topK 与 token 隐藏维 H：取值范围，以及按拓扑和权重分形确定的 H 对齐要求。
 */
static ge::graphStatus CheckTopKAndHParam(const gert::TilingContext *context, MegaMoeConfig &config,
                                          const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);

    // topk范围
    OP_TILING_CHECK(
        topkIdsDim1 < MIN_TOPK || topkIdsDim1 > MAX_TOPK,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "topK", std::to_string(topkIdsDim1).c_str(), "only support [1, 32]"),
        return ge::GRAPH_FAILED);

    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    // 检查 H 范围 [1K, 8K]
    OP_TILING_CHECK(
        xDim1 < MIN_H || xDim1 > MAX_H,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "H", std::to_string(xDim1).c_str(),
            (std::string("should in [") + std::to_string(MIN_H) + ", " + std::to_string(MAX_H) + "]").c_str()),
        return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    auto topoTypePtr = attrs->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    bool isW4Nz =
        weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 &&
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ;
    // URMA/Layered 保持 1K 对齐；W4 的 NZ_C0_32 分形要求 GMM K 按 64 对齐；其余 MTE 路径按 32 对齐。
    int64_t requiredHAlignment = *topoTypePtr == TOPO_TYPE_URMA ? URMA_H_ALIGN : (isW4Nz ? W4_K_ALIGN : H_ALIGN);
    OP_TILING_CHECK(
        xDim1 % requiredHAlignment != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "H", std::to_string(xDim1).c_str(),
                                  (std::string("multiple of ") + std::to_string(requiredHAlignment)).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验本卡 MoE 专家数与 GMM1 输出维 hiddenDim：取值范围与统一对齐要求。
 */
static ge::graphStatus CheckExpertAndHiddenDimParam(const gert::TilingContext *context, MegaMoeConfig &config,
                                                    const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    bool isPerExpertWeightTensor = weightOneStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS;
    int64_t moeExpertPerRank = GetWeightExpertCount(context, config.weight1Index, isPerExpertWeightTensor);
    OP_TILING_CHECK(moeExpertPerRank < MIN_EXPERT_PER_RANK || moeExpertPerRank > MAX_EXPERT_PER_RANK,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertPerRank", std::to_string(moeExpertPerRank).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EXPERT_PER_RANK) + ", " +
                                               std::to_string(MAX_EXPERT_PER_RANK) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    int64_t hiddenDim =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    OP_TILING_CHECK(hiddenDim < MIN_HIDDEN_DIM || hiddenDim > MAX_HIDDEN_DIM,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "hiddenDim", std::to_string(hiddenDim).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_HIDDEN_DIM) + ", " +
                                               std::to_string(MAX_HIDDEN_DIM) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hiddenDim % HIDDEN_DIM_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "hiddenDim", std::to_string(hiddenDim).c_str(),
                                              (std::string("multiple of ") + std::to_string(HIDDEN_DIM_ALIGN)).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParam(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    ge::graphStatus checkStatus = CheckTopKAndHParam(context, config, nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    return CheckExpertAndHiddenDimParam(context, config, nodeName);
}

static ge::graphStatus SetInputParam(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                     MegaMoeConfig &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    bool isPerExpertWeightTensor = weightOneStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS;
    int64_t hiddenDim =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);

    tilingData->bs = static_cast<uint32_t>(bs);
    tilingData->h = static_cast<uint32_t>(h);
    tilingData->hiddenDim = static_cast<uint32_t>(hiddenDim);
    tilingData->topK = static_cast<uint32_t>(topK);
    tilingData->isPerExpertWeightTensor = isPerExpertWeightTensor;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetInput(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                        MegaMoeConfig &config, const char *nodeName)
{
    OP_TILING_CHECK(TilingCheckMegaMoe(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check input failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputParam(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check input param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetInputParam(context, tilingData, config) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "set input param failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MegaMoe tiling check func.");

    MegaMoeTilingData *tilingData = context->GetTilingData<MegaMoeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrPtrNullptr(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check attr pointers failed."), return ge::GRAPH_FAILED);

    // Input check & set
    OP_TILING_CHECK(CheckAndSetInput(context, tilingData, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Check and set input failed."), return ge::GRAPH_FAILED);

    // Platform Info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    tilingData->aicNum = aicNum;
    tilingData->blockAivNum = aivNum;
    OP_TILING_CHECK(aivNum <= 0 || aicNum <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "aivNum/aicNum",
                                              (std::to_string(aivNum) + ", " + std::to_string(aicNum)).c_str(),
                                              "should both be > 0"),
                    return ge::GRAPH_FAILED);

    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    context->SetScheduleMode(1); // batch model, all cores start at the same time
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);

    // Attr check & set
    OP_TILING_CHECK(CheckAttrAndSetTilingData(context, config, tilingData, aicNum) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        SetAdaptiveBufferConfigs(context, config, tilingData, static_cast<uint32_t>(ubSize)) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Setting adaptive buffer configs failed."), return ge::GRAPH_FAILED);

    // Cal TilingKey
    uint64_t tilingKey = CalTilingKey(context, config, tilingData, nodeName);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // WorkspaceSize
    WorkspaceLayout workspaceLayout(tilingData);
    OP_TILING_CHECK(SetWorkspace(context, workspaceLayout, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Tiling set workspace Failed"), return ge::GRAPH_FAILED);

    // Print Info
    PrintMegaMoeTilingData(tilingData, nodeName);
    PrintWorkspaceLayout(&workspaceLayout, nodeName);
    PrintPeermemInfo(tilingData, nodeName);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeTilingFunc(gert::TilingContext *context)
{
    MegaMoeConfig config;
    ge::graphStatus ret;

    ret = MegaMoeTilingFuncImplPublic(context, config);

    return ret;
}

struct MegaMoeCompileInfo {};
static ge::graphStatus TilingParseForMegaMoe(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaMoe).Tiling(MegaMoeTilingFunc).TilingParse<MegaMoeCompileInfo>(TilingParseForMegaMoe);

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
inline void MegaMoeExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2Exception::Mc2ExceptionImpl(args, userdata, "MegaMoe");
}

__attribute__((constructor)) void RegisterMegaMoeExceptionFunc()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;

    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MegaMoe", "Get runtime version failed when register exception func.");
        return;
    }
    if (aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MegaMoe", "Get metadef version failed when register exception func.");
        return;
    }

    if (runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MegaMoe",
                "The runtime(%d) or metadata(%d) version is lower than the version(%d) supporting exception func.",
                runtimeVersionNum, metadefVersionNum, EXCEPTION_DUMP_SUPPORT_VERSION);
        return;
    }

    IMPL_OP(MegaMoe).ExceptionDumpParseFunc(MegaMoeExceptionImplWrapper);
}
#endif
} // namespace optiling
