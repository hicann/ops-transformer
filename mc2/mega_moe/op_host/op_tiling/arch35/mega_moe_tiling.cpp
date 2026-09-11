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
#include <unordered_map>

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
// SwiGLU 将 GMM1 输出划分为 gate 和 up 两部分，因此 weight1 行数为 weight2 列数的两倍。
const static int64_t SWIGLU_GATE_UP_SPLIT_FACTOR = 2LL;
// MX 量化的 weight scale 每个专家末尾固定带 2 个 base 分量。
const static int64_t WEIGHT_SCALE_MULTI_BASE_DIM_SIZE = 2LL;
// 当前实现支持的共享专家数量上限。
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
const static uint32_t WEIGHT_MATRIX_ROW_DIM_INDEX = 0U;
const static uint32_t WEIGHT_MATRIX_COLUMN_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MATRIX_DIM_INDEX = 0U;
const static uint32_t WEIGHT_SCALE_GROUP_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MULTI_BASE_DIM_INDEX = 2U;

struct ExpertWeightInputIndices {
    uint32_t weightOne;
    uint32_t weightTwo;
    uint32_t weightScalesOne;
    uint32_t weightScalesTwo;
};

struct ExpertParams {
    ExpertWeightInputIndices inputs;
    const char *expertTypeName;
    int64_t expertCount;
    ge::DataType quantOutDtype;
    uint8_t gmmMode;
};

struct MegaMoeExpertParams {
    ExpertParams moe;
    ExpertParams shared;
    bool isPerExpertWeightTensor;
};

static const std::unordered_map<ge::DataType, int64_t> EXPERT_QUANT_MODE_MAP = {
    {ge::DT_FLOAT8_E5M2, EXPERT_QUANT_OUT_DTYPE_E5M2},
    {ge::DT_FLOAT8_E4M3FN, EXPERT_QUANT_OUT_DTYPE_E4M3FN},
    {ge::DT_FLOAT4_E2M1, EXPERT_QUANT_OUT_DTYPE_E2M1},
};

/*
 * 根据接口索引构造 MoE 与共享专家参数；这里只建立输入角色映射，不读取或校验 tensor。
 */
static MegaMoeExpertParams MakeExpertParams(const MegaMoeConfig &config)
{
    MegaMoeExpertParams params{};
    params.moe.inputs = {config.weight1Index, config.weight2Index, config.weightScales1Index,
                         config.weightScales2Index};
    params.moe.expertTypeName = "MoE expert";
    params.shared.inputs = {config.sharedWeight1Index, config.sharedWeight2Index, config.sharedWeightScales1Index,
                            config.sharedWeightScales2Index};
    params.shared.expertTypeName = "shared expert";
    return params;
}

/*
 * 根据 GMM1/GMM2 的逻辑 tile 数和 AIC 数量，计算单个 wave 覆盖的 M group 数。
 */
uint32_t CalcMGroupsPerWave(const MegaMoeTilingData *tilingData, uint32_t aicNum)
{
    if (tilingData->hiddenDim == 0U || tilingData->h == 0U || aicNum == 0U) {
        return 1U;
    }

    /*
     * hiddenDim 包含 gate/up 两部分。交织模式每核至少调度 4 个独立 N tile；非交织模式
     * 每个物理任务成对处理 gate/up，原先每核 2 个物理任务同样等价于 4 个逻辑 N tile。
     * hiddenDim 已校验为 GMM_TILE_N(256) 的倍数（MTE；URMA 仍为 1024），CeilDiv 对
     * 非交织半宽产生的尾 tile 只影响 wave 粒度估算，不影响正确性，两种编译模式共用本公式。
     */
    uint64_t gmm1LogicalTilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->hiddenDim, GMM_TILE_N);
    uint64_t gmm2TilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->h, GMM_TILE_N);
    uint64_t gmm1RequiredMGroups = ops::CeilDiv<uint64_t>(
        static_cast<uint64_t>(aicNum) * GMM1_MIN_LOGICAL_TILES_PER_CORE, gmm1LogicalTilesPerMGroup);
    uint64_t gmm2RequiredMGroups = ops::CeilDiv<uint64_t>(static_cast<uint64_t>(aicNum), gmm2TilesPerMGroup);
    return static_cast<uint32_t>(std::max(gmm1RequiredMGroups, gmm2RequiredMGroups));
}

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

/*
 * 输出问题规模、执行模式以及各阶段自适应缓冲区配置，供 tiling 诊断使用。
 */
void PrintMegaMoeTilingData(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);
    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");

    // 问题规模、专家拓扑及执行模式。
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
    OP_LOGD(nodeName, "mode: moeGmmMode=%u, sharedGmmMode=%u, combineQuantMode=%ld, clampLimit=%f",
            static_cast<uint32_t>(tilingData->moeGmmMode), static_cast<uint32_t>(tilingData->sharedGmmMode),
            tilingData->combineQuantMode, tilingData->clampLimit);
    OP_LOGD(nodeName, "combineSync: slotCountPerExpert=%lu", tilingData->combineSyncSlotCountPerExpert);
    OP_LOGD(nodeName, "topkWeightsPrefetch is %d", tilingData->topkWeightsPrefetch);
    OP_LOGD(nodeName, "mGroupsPerWave is %u", tilingData->mGroupsPerWave);

    // 各阶段的 UB 分批及缓冲区配置。
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
}

/*
 * 输出 workspace 各分区的偏移和总大小，供 host/device 布局核对使用。
 */
void PrintWorkspaceLayout(const struct WorkspaceLayout *layout, const char *nodeName)
{
    OP_LOGD(nodeName, "dispatchRevDataOffset:         %ld\n", layout->dispatchRevDataOffset);
    OP_LOGD(nodeName, "dispatchRevScaleOffset:        %ld\n", layout->dispatchRevScaleOffset);
    OP_LOGD(nodeName, "activationQuantDataOffset:     %ld\n", layout->activationQuantDataOffset);
    OP_LOGD(nodeName, "activationQuantScaleOffset:    %ld\n", layout->activationQuantScaleOffset);
    OP_LOGD(nodeName, "expertRevTokenNumsOffset:      %ld\n", layout->expertRevTokenNumsOffset);
    OP_LOGD(nodeName, "metaInfoOffset:                %ld\n", layout->metaInfoOffset);
    OP_LOGD(nodeName, "flagActivationToGmm2Offset:    %ld\n", layout->flagActivationToGmm2Offset);
    OP_LOGD(nodeName, "sharedActivationToGmm2Offset:  %ld\n", layout->sharedActivationToGmm2Offset);
    OP_LOGD(nodeName, "flagDispatchToGmm1Offset:      %ld\n", layout->flagDispatchToGmm1Offset);
    OP_LOGD(nodeName, "flagSendCntCalToUpdParamsOffset: %ld\n", layout->flagSendCntCalToUpdParamsOffset);
    OP_LOGD(nodeName, "flagGmmToEpilogueOffset:       %ld\n", layout->flagGmmToEpilogueOffset);
    OP_LOGD(nodeName, "gmm2ReadyOffset:               %ld\n", layout->gmm2ReadyOffset);
    OP_LOGD(nodeName, "gmm2CombineSyncCounterOffset:  %ld\n", layout->gmm2CombineSyncCounterOffset);
    OP_LOGD(nodeName, "gmm2MmadResOffset:             %ld\n", layout->gmm2MmadResOffset);
    OP_LOGD(nodeName, "workspaceSize:                 %ld\n", layout->workspaceSize);
}

/*
 * 按 kernel 共用的 peermem 布局公式计算并输出各分区大小。
 */
void PrintPeermemInfo(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_LOGD(nodeName, "========== PeermemInfo ==========");
    int64_t exceptionDumpRegionSize = tilingData->topoType == TOPO_TYPE_MTE ? EXCEPTION_DUMP_REGION_SIZE : 0;
    OP_LOGD(nodeName, "exceptionDumpRegionSize: {%ld}\n", exceptionDumpRegionSize);
    PeermemSizeParams params{};
    params.numMaxTokensPerRank = static_cast<int64_t>(tilingData->numMaxTokensPerRank);
    params.topK = static_cast<int64_t>(tilingData->topK);
    params.h = static_cast<int64_t>(tilingData->h);
    params.moeExpertPerRank = static_cast<int64_t>(tilingData->moeExpertPerRank);
    params.epWorldSize = static_cast<int64_t>(tilingData->epWorldSize);
    params.yDtypeSize = SIZE_BF_16;
    params.elemsPerByte = IsA4W4GmmMode(tilingData->moeGmmMode) ? 2U : 1U;
    params.topkWeightsPrefetch = tilingData->topkWeightsPrefetch == 1;
    params.isQuantCombine = tilingData->combineQuantMode != COMBINE_NO_QUANT;
    params.topoType = tilingData->topoType;
    params.serverNum =
        tilingData->topoType == TOPO_TYPE_URMA ? tilingData->epWorldSize / tilingData->rankNumPerServer : 1;
    PeermemLayoutSizes sizes = CalcPeermemLayoutSizes(params);
    OP_LOGD(nodeName, "peermemDataOffset: {%ld}\n", sizes.dataOffset);
    OP_LOGD(nodeName, "maskRecvSize: {%ld}\n", sizes.maskRecvSize);
    OP_LOGD(nodeName, "expertCountRecvSize: {%ld}\n", sizes.expertCountRecvSize);
    OP_LOGD(nodeName, "dispatchRecordAreaSize: {%ld}\n", sizes.dispatchRecordAreaSize);
    OP_LOGD(nodeName, "combineSendSize: {%ld}\n", sizes.combineSendSize);
    OP_LOGD(nodeName, "total PeermemInfo Size: {%ld}\n", exceptionDumpRegionSize + CalcPeermemLeastSize(params));
}

/*
 * 将专家权重类型、量化输出类型、通信拓扑和可选能力编码为 kernel tiling key。
 */
static uint64_t CalcTilingKey(const gert::TilingContext *context, const MegaMoeExpertParams &expertParams,
                              const MegaMoeTilingData *tilingData)
{
    auto moeWeightDesc = context->GetDynamicInputDesc(expertParams.moe.inputs.weightOne, 0);
    auto sharedWeightDesc = expertParams.shared.expertCount > 0 ?
                                context->GetDynamicInputDesc(expertParams.shared.inputs.weightOne, 0) :
                                moeWeightDesc;

    int64_t topoType = TILINGKEY_TPL_MTE;
    if (tilingData->topoType == TOPO_TYPE_URMA) {
        topoType = TILINGKEY_TPL_URMA;
    }

    return GET_TPL_TILING_KEY(
        static_cast<int64_t>(moeWeightDesc->GetDataType()), static_cast<int64_t>(sharedWeightDesc->GetDataType()),
        DISPATCH_QUANT_MODE_MXFP, EXPERT_QUANT_MODE_MAP.at(expertParams.moe.quantOutDtype),
        EXPERT_QUANT_MODE_MAP.at(expertParams.shared.quantOutDtype), static_cast<int64_t>(tilingData->combineQuantMode),
        topoType, static_cast<int64_t>(tilingData->topkWeightsPrefetch));
}

/*
 * 集中校验后续阶段使用的必需属性指针，后续属性处理函数不再重复判空。
 * 涉及的直接属性：moe_expert_num、ep_world_size、ccl_buffer_size、max_recv_token_num、
 * dispatch_quant_mode、dispatch_quant_out_dtype、shared_expert_quant_out_dtype、combine_quant_mode、
 * comm_alg、num_max_tokens_per_rank、topo_type、topk_weights_type、activation 和 activation_params。
 */
static ge::graphStatus CheckRequiredAttrPtrNullptr(const gert::TilingContext *context, const MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    auto sharedExpertQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrSharedExpertQuantOutDtypeIndex));
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    auto topoTypePtr = attrs->GetAttrPointer<int64_t>((config.attrTopoTypeIndex));
    auto topkWeightsTypePtr = attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);
    auto rankNumPerServerPtr = attrs->GetAttrPointer<int64_t>(config.attrRankNumPerServerIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, moeExpertNumPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, epWorldSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, cclBufferSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxRecvTokenNumPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dispatchQuantModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dispatchQuantOutDtypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, sharedExpertQuantOutDtypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, combineQuantModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, commAlgPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, numMaxTokensPerRankPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, topoTypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsTypePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, activationPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, activationParamsPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, rankNumPerServerPtr);
    const size_t paramCount = activationParamsPtr->GetSize();
    const float *activationParams = paramCount == 0U ? nullptr : activationParamsPtr->GetData();
    if (paramCount != 0U) {
        OP_CHECK_NULL_WITH_CONTEXT(context, activationParams);
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 swiglu/swiglustep 的可选 clamp 参数；该参数表示激活值截断上限。
 * 未配置时使用 float 最大值，等价于不截断（参见 SetActivationAttrParams）。
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
 * 校验 swigluoai 的 clamp、alpha 和 beta 参数。
 * 三个参数必须同时配置；全部省略时统一使用默认值，不支持部分配置。
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
 * situglu 的 beta 必填：Ascend 950PR/Ascend 950DT 仅支持大于 0 的有限值。
 * linear_beta 选填，给了就切到 LINEAR 子模式，同样仅支持大于 0 的有限值。
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
    OP_TILING_CHECK(!std::isfinite(beta) || beta <= 0.0f,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "situglu_beta", std::to_string(beta).c_str(),
                                              "should be finite and greater than 0"),
                    return ge::GRAPH_FAILED);
    if (paramCount == 2U) {
        const float linearBeta = activationParams[1];
        OP_TILING_CHECK(!std::isfinite(linearBeta) || linearBeta <= 0.0f,
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "situglu_linear_beta", std::to_string(linearBeta).c_str(),
                                                  "should be finite and greater than 0"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验激活类型、参数个数与取值，并检查激活类型和通信拓扑的兼容性。
 */
static ge::graphStatus CheckActivationParams(const gert::TilingContext *context, const MegaMoeConfig &config,
                                             int64_t topoType, const char *nodeName)
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

    // URMA Layered 已接入 swiglu/situglu，其余扩展激活仍保持拦截。
    OP_TILING_CHECK(
        topoType == TOPO_TYPE_URMA && !isSwiGlu && !isSituGlu,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr, "'swiglu' or 'situglu' for URMA topology"),
        return ge::GRAPH_FAILED);

    // 前置类型校验已将其余分支收敛为 situglu。
    if (isSwiGlu || isSwiGluStep) {
        return CheckSwiGluActivationParams(activationParams, paramCount, nodeName);
    }
    if (isSwiGluOai) {
        return CheckSwiGluOaiActivationParams(activationParams, paramCount, nodeName);
    }
    return CheckSituGluActivationParams(activationParams, paramCount, nodeName);
}

/*
 * 校验 EP 通信域规模并写入 tiling data，供专家划分与容量计算使用。
 */
static ge::graphStatus CheckAndSetEpWorldSizeAttr(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                  MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    int64_t epWorldSize = static_cast<int64_t>(*epWorldSizePtr);
    int64_t maxEpWorldSize = tilingData->topoType == TOPO_TYPE_URMA ? MAX_URMA_EP_WORLD_SIZE : MAX_MTE_EP_WORLD_SIZE;
    OP_TILING_CHECK(epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > maxEpWorldSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EP_WORLD_SIZE) + ", " +
                                               std::to_string(maxEpWorldSize) + "] for the selected topology")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    tilingData->epWorldSize = static_cast<uint32_t>(epWorldSize);
    int64_t rankNumPerServer = *attrs->GetAttrPointer<int64_t>(config.attrRankNumPerServerIndex);
    OP_TILING_CHECK(
        tilingData->topoType == TOPO_TYPE_URMA &&
            (rankNumPerServer <= 0 || rankNumPerServer > epWorldSize || epWorldSize % rankNumPerServer != 0),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "rankNumPerServer", std::to_string(rankNumPerServer).c_str(),
                                  "should be in [1, epWorldSize] and divide epWorldSize for URMA"),
        return ge::GRAPH_FAILED);
    tilingData->rankNumPerServer = tilingData->topoType == TOPO_TYPE_URMA ? static_cast<uint32_t>(rankNumPerServer) :
                                                                            static_cast<uint32_t>(epWorldSize);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验通信算法属性；当前仅支持默认的空字符串配置。
 */
static ge::graphStatus CheckCommAlgAttr(const gert::TilingContext *context, const MegaMoeConfig &config,
                                        const char *nodeName)
{
    auto commAlgPtr = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    OP_TILING_CHECK(std::strcmp(commAlgPtr, "") != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "commAlg", commAlgPtr, "not support, need empty string"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验路由专家总数、本 rank 专家数和共享专家数。
 * 路由专家必须在 EP rank 间均分，且派生的本 rank 专家数必须与 weight1 中的专家数一致。
 * 共享专家不参与 EP 路由切分，其数量由 shared weight1 推导，本函数仅校验实现上限。
 */
static ge::graphStatus CheckAndSetExpertCountAttrs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                   const MegaMoeExpertParams &expertParams,
                                                   MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t epWorldSize = static_cast<int64_t>(tilingData->epWorldSize);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    OP_TILING_CHECK((moeExpertNum < epWorldSize || moeExpertNum > MAX_MOE_EXPERT_NUM) || (moeExpertNum % epWorldSize),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertNum", std::to_string(moeExpertNum).c_str(),
                                              (std::string("should in [") + std::to_string(epWorldSize) + ", " +
                                               std::to_string(MAX_MOE_EXPERT_NUM) + "] and mod(..., epWorldSize(" +
                                               std::to_string(epWorldSize) + ")) == 0")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    int64_t moeExpertPerRank = moeExpertNum / epWorldSize;
    OP_TILING_CHECK(moeExpertPerRank < MIN_EXPERT_PER_RANK || moeExpertPerRank > MAX_EXPERT_PER_RANK,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertPerRank", std::to_string(moeExpertPerRank).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EXPERT_PER_RANK) + ", " +
                                               std::to_string(MAX_EXPERT_PER_RANK) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertParams.moe.expertCount != moeExpertPerRank,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "weight1 expert count", std::to_string(expertParams.moe.expertCount).c_str(),
                        (std::string("should equal the local MoE expert count (moeExpertNum / epWorldSize) = ") +
                         std::to_string(moeExpertPerRank))
                            .c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expertParams.shared.expertCount > MAX_SHARED_EXPERT_NUM,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum", std::to_string(expertParams.shared.expertCount).c_str(),
                                  "only support 0-4"),
        return ge::GRAPH_FAILED);

    tilingData->moeExpertPerRank = static_cast<uint32_t>(moeExpertPerRank);
    tilingData->sharedExpertNum = static_cast<uint32_t>(expertParams.shared.expertCount);
    return ge::GRAPH_SUCCESS;
}

/*
 * 判断量化输出类型能否映射到当前 kernel 支持的专家量化模式。
 */
static bool IsSupportedExpertQuantDtype(ge::DataType dtype)
{
    return EXPERT_QUANT_MODE_MAP.count(dtype) > 0U;
}

/*
 * 由单个专家组的量化输出类型、权重类型和两个权重的 format 确定 GMM 实现路径。
 * A4W4 ND 是内部保留能力：weight1 为 ND，weight2 仍为 NZ_C0_32；算子原型不因此扩展支持范围。
 */
static uint8_t ResolveExpertGmmMode(ge::DataType quantOutDtype, ge::DataType weightDtype, ge::Format weightOneFormat,
                                    ge::Format weightTwoFormat)
{
    bool isNdFormat = weightOneFormat == ge::FORMAT_ND && weightTwoFormat == ge::FORMAT_ND;
    bool isNzFormat = weightOneFormat == ge::FORMAT_FRACTAL_NZ && weightTwoFormat == ge::FORMAT_FRACTAL_NZ;
    bool isNzC032Format =
        weightOneFormat == ge::FORMAT_FRACTAL_NZ_C0_32 && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;
    bool isNdAndNzC032Format = weightOneFormat == ge::FORMAT_ND && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;
    bool isNzAndNzC032Format =
        weightOneFormat == ge::FORMAT_FRACTAL_NZ && weightTwoFormat == ge::FORMAT_FRACTAL_NZ_C0_32;

    if (quantOutDtype == ge::DT_FLOAT8_E5M2 && weightDtype == ge::DT_FLOAT8_E5M2 && isNdFormat) {
        return GMM_MODE_A8W8_ND;
    }
    if (quantOutDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT8_E4M3FN && isNdFormat) {
        return GMM_MODE_A8W8_ND;
    }
    if (quantOutDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT8_E4M3FN && isNzFormat) {
        return GMM_MODE_A8W8_NZ;
    }
    if (quantOutDtype == ge::DT_FLOAT8_E4M3FN && weightDtype == ge::DT_FLOAT4_E2M1 && isNzC032Format) {
        return GMM_MODE_A8W4_NZ;
    }
    if (quantOutDtype == ge::DT_FLOAT4_E2M1 && weightDtype == ge::DT_FLOAT4_E2M1 && isNdAndNzC032Format) {
        return GMM_MODE_A4W4_ND;
    }
    if (quantOutDtype == ge::DT_FLOAT4_E2M1 && weightDtype == ge::DT_FLOAT4_E2M1 && isNzAndNzC032Format) {
        return GMM_MODE_A4W4_NZ;
    }
    return std::numeric_limits<uint8_t>::max();
}

/*
 * 校验 dispatch 和 combine 阶段的量化模式，并将 combine 模式写入 tiling data。
 */
static ge::graphStatus CheckAndSetQuantModeAttrs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                 MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    OP_TILING_CHECK(
        *dispatchQuantModePtr != DISPATCH_QUANT_MODE_MXFP,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "dispatch_quant_mode", std::to_string(*dispatchQuantModePtr).c_str(),
                                  std::to_string(DISPATCH_QUANT_MODE_MXFP).c_str()),
        return ge::GRAPH_FAILED);

    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    OP_TILING_CHECK(
        *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_NO_QUANT &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E5M2 &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E4M3FN,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "combineQuantMode", std::to_string(*combineQuantModePtr).c_str(),
                                  "only support no_quant(0), fp8_e5m2(3) and fp8_e4m3fn(4)"),
        return ge::GRAPH_FAILED);
    tilingData->combineQuantMode = static_cast<uint32_t>(*combineQuantModePtr);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验并设置 MoE 与共享专家的量化输出类型，同时解析共享专家类型的继承语义。
 */
static ge::graphStatus CheckAndSetQuantOutDtypes(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                 MegaMoeExpertParams &params, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    ge::DataType moeQuantOutDtype = static_cast<ge::DataType>(*dispatchQuantOutDtypePtr);
    OP_TILING_CHECK(!IsSupportedExpertQuantDtype(moeQuantOutDtype),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "dispatch_quant_out_dtype",
                                              std::to_string(*dispatchQuantOutDtypePtr).c_str(),
                                              "fp8_e5m2, fp8_e4m3fn or fp4_e2m1"),
                    return ge::GRAPH_FAILED);

    auto sharedQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertQuantOutDtypeIndex);
    ge::DataType sharedQuantOutDtype = static_cast<ge::DataType>(*sharedQuantOutDtypePtr);
    bool hasExplicitSharedQuantDtype = sharedQuantOutDtype != ge::DT_UNDEFINED;
    OP_TILING_CHECK(hasExplicitSharedQuantDtype && !IsSupportedExpertQuantDtype(sharedQuantOutDtype),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "shared_expert_quant_out_dtype",
                                              std::to_string(*sharedQuantOutDtypePtr).c_str(),
                                              "DT_UNDEFINED, fp8_e5m2, fp8_e4m3fn or fp4_e2m1"),
                    return ge::GRAPH_FAILED);

    params.moe.quantOutDtype = moeQuantOutDtype;
    params.shared.quantOutDtype =
        params.shared.expertCount <= 0 || !hasExplicitSharedQuantDtype ? moeQuantOutDtype : sharedQuantOutDtype;
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验同一专家组的权重类型一致性，并由量化类型、权重类型及格式解析 GMM 模式。
 */
static ge::graphStatus CheckAndSetExpertGmmMode(const gert::TilingContext *context, ExpertParams &expertParams,
                                                const char *nodeName)
{
    auto weightOneDesc = context->GetDynamicInputDesc(expertParams.inputs.weightOne, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(expertParams.inputs.weightTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);

    ge::DataType weightOneDtype = weightOneDesc->GetDataType();
    ge::DataType weightTwoDtype = weightTwoDesc->GetDataType();
    std::string weightNames = std::string(expertParams.expertTypeName) + " weight1, weight2";
    std::string weightDtypes =
        "[" + Ops::Base::ToString(weightOneDtype) + ", " + Ops::Base::ToString(weightTwoDtype) + "]";
    OP_TILING_CHECK(weightOneDtype != weightTwoDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        nodeName, weightNames.c_str(), weightDtypes.c_str(),
                        "weight1 and weight2 within the same expert group must have the same dtype."),
                    return ge::GRAPH_FAILED);

    ge::Format weightOneFormat = weightOneDesc->GetStorageFormat();
    ge::Format weightTwoFormat = weightTwoDesc->GetStorageFormat();
    expertParams.gmmMode =
        ResolveExpertGmmMode(expertParams.quantOutDtype, weightOneDtype, weightOneFormat, weightTwoFormat);
    std::string actualConfig = "[" + Ops::Base::ToString(expertParams.quantOutDtype) + ", " +
                               Ops::Base::ToString(weightOneDtype) + ", " + Ops::Base::ToString(weightOneFormat) +
                               ", " + Ops::Base::ToString(weightTwoFormat) + "]";
    OP_TILING_CHECK(
        expertParams.gmmMode == std::numeric_limits<uint8_t>::max(),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            nodeName, "quant output dtype, weight dtype, weight1 format, weight2 format", actualConfig.c_str(),
            (std::string(expertParams.expertTypeName) + " configuration does not form a supported GMM mode.").c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 解析 MoE 与共享专家的 GMM mode，并校验 mode 组合与通信拓扑的兼容性。
 */
static ge::graphStatus CheckAndSetExpertGmmModes(const gert::TilingContext *context, MegaMoeExpertParams &params,
                                                 int64_t topoType, MegaMoeTilingData *tilingData, const char *nodeName)
{
    if (CheckAndSetExpertGmmMode(context, params.moe, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    params.shared.gmmMode = params.moe.gmmMode;
    if (params.shared.expertCount > 0 &&
        CheckAndSetExpertGmmMode(context, params.shared, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool hasDifferentExpertGmmConfigs =
        params.shared.gmmMode != params.moe.gmmMode || params.shared.quantOutDtype != params.moe.quantOutDtype;
    OP_TILING_CHECK(
        topoType != TOPO_TYPE_MTE && hasDifferentExpertGmmConfigs,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "topo_type", std::to_string(topoType).c_str(),
                                  "MTE when MoE and shared experts use different quantization configurations"),
        return ge::GRAPH_FAILED);

    tilingData->moeGmmMode = params.moe.gmmMode;
    tilingData->sharedGmmMode = params.shared.gmmMode;
    return ge::GRAPH_SUCCESS;
}

/*
 * 按 host/device 共用的 peermem 布局公式计算通信窗口最小容量
 * （common/mega_moe_peermem.h），保证两侧计算口径一致。
 * 容量校验阶段尚未解析激活存储类型，elementsPerByte 使用 1 作为保守上界。
 * topology、专家数和 combine mode 均来自前序已校验并写入 tilingData 的结果。
 */
static int64_t CalcLeastCclBufferSize(int64_t numMaxTokensPerRank, int64_t yDtypeSize,
                                      const MegaMoeTilingData *tilingData, bool topkWeightsPrefetch)
{
    PeermemSizeParams peermemSizeParams{};
    peermemSizeParams.numMaxTokensPerRank = numMaxTokensPerRank;
    peermemSizeParams.topK = tilingData->topK;
    peermemSizeParams.h = tilingData->h;
    peermemSizeParams.moeExpertPerRank = tilingData->moeExpertPerRank;
    peermemSizeParams.epWorldSize = tilingData->epWorldSize;
    peermemSizeParams.yDtypeSize = yDtypeSize;
    peermemSizeParams.elemsPerByte = 1U;
    peermemSizeParams.topkWeightsPrefetch = topkWeightsPrefetch;
    peermemSizeParams.isQuantCombine = tilingData->combineQuantMode != COMBINE_NO_QUANT;
    peermemSizeParams.topoType = tilingData->topoType;
    peermemSizeParams.serverNum =
        tilingData->topoType == TOPO_TYPE_URMA ? tilingData->epWorldSize / tilingData->rankNumPerServer : 1;
    int64_t leastCclBufferSize = CalcPeermemLeastSize(peermemSizeParams);
    // MTE peermem 头部包含独立异常 dump 区，最小窗口容量需计入该区域。
    if (tilingData->topoType == TOPO_TYPE_MTE) {
        leastCclBufferSize += EXCEPTION_DUMP_REGION_SIZE;
    }
    return leastCclBufferSize;
}

/*
 * 校验并默认化单 rank 最大 token 数，将结果写入 tiling data。
 */
static ge::graphStatus CheckAndSetMaxTokensPerRankAttr(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                       MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    int64_t numMaxTokensPerRank = static_cast<int64_t>(*numMaxTokensPerRankPtr);
    OP_TILING_CHECK(tilingData->topoType == TOPO_TYPE_URMA && tilingData->bs == 0U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "bs", "0", "should be greater than 0 for URMA"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tilingData->topoType == TOPO_TYPE_URMA && numMaxTokensPerRank == 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "numMaxTokensPerRank", "0",
                                              "should be >= bs and identical on all ranks for URMA"),
                    return ge::GRAPH_FAILED);
    // 该值最终存入 uint32 字段，因此同时校验 UINT32_MAX 上界，避免窄化截断。
    OP_TILING_CHECK(numMaxTokensPerRank < 0 || numMaxTokensPerRank > 0xFFFFFFFFLL ||
                        (numMaxTokensPerRank != 0 && tilingData->bs > numMaxTokensPerRank),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "numMaxTokensPerRank", std::to_string(numMaxTokensPerRank).c_str(),
                        (std::string("0 or in [bs, UINT32_MAX], bs is ") + std::to_string(tilingData->bs).c_str())),
                    return ge::GRAPH_FAILED);
    if (numMaxTokensPerRank == 0) {
        numMaxTokensPerRank = tilingData->bs;
    }
    int64_t routeCapacity = numMaxTokensPerRank * static_cast<int64_t>(tilingData->topK);
    int64_t maxOutputCapacity = numMaxTokensPerRank * static_cast<int64_t>(tilingData->epWorldSize) *
                                std::min(tilingData->topK, tilingData->moeExpertPerRank);
    OP_TILING_CHECK(
        tilingData->topoType == TOPO_TYPE_URMA && (routeCapacity > std::numeric_limits<int32_t>::max() ||
                                                   maxOutputCapacity > std::numeric_limits<int32_t>::max()),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "URMA token capacity", std::to_string(maxOutputCapacity).c_str(),
                                  "route indexes and prefix sums should fit in INT32_MAX"),
        return ge::GRAPH_FAILED);
    tilingData->numMaxTokensPerRank = static_cast<uint32_t>(numMaxTokensPerRank);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 topK weight 预取开关并写入 tiling data。
 */
static ge::graphStatus CheckAndSetTopkWeightsTypeAttr(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                      MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    OP_TILING_CHECK(topkWeightsType != 0 && topkWeightsType != 1,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topkWeightsType", std::to_string(topkWeightsType).c_str(),
                                              "only support 0(disabled) or 1(enabled)"),
                    return ge::GRAPH_FAILED);

    tilingData->topkWeightsPrefetch = static_cast<int32_t>(topkWeightsType);
    return ge::GRAPH_SUCCESS;
}

/*
 * 按已解析的 token 上界、tensor 维度及量化配置计算 peermem 最小容量，并校验通信窗口大小。
 */
static ge::graphStatus CheckCclBufferCapacity(const gert::TilingContext *context, const MegaMoeConfig &config,
                                              const MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t yDtypeSize = ge::GetSizeByDataType(context->GetOutputDesc(config.yIndex)->GetDataType());
    int64_t leastCclBufferSize = CalcLeastCclBufferSize(static_cast<int64_t>(tilingData->numMaxTokensPerRank),
                                                        yDtypeSize, tilingData, tilingData->topkWeightsPrefetch == 1);
    int64_t cclBufferSize = static_cast<int64_t>(*attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex)));
    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "cclBufferSize", std::to_string(cclBufferSize).c_str(),
                                              (std::string("should >= ") + std::to_string(leastCclBufferSize)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验接收 token 上限，并据此确定 kernel 的最大输出 token 数。
 */
static ge::graphStatus CheckAndSetMaxRecvTokenNumAttr(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                      MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t maxRecvTokenNum = static_cast<int64_t>(*attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex)));
    OP_TILING_CHECK(maxRecvTokenNum < 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "maxRecvTokenNum", std::to_string(maxRecvTokenNum).c_str(),
                                              "should be greater than or equal to 0"),
                    return ge::GRAPH_FAILED);

    uint64_t defaultMaxOutputSize = static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->epWorldSize *
                                    std::min(tilingData->topK, tilingData->moeExpertPerRank);
    OP_TILING_CHECK(
        tilingData->topoType == TOPO_TYPE_URMA && static_cast<uint64_t>(maxRecvTokenNum) > defaultMaxOutputSize,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "maxRecvTokenNum", std::to_string(maxRecvTokenNum).c_str(),
                                  "should not exceed the URMA maximum receive token capacity"),
        return ge::GRAPH_FAILED);
    tilingData->maxOutputSize = maxRecvTokenNum != 0 ? static_cast<uint32_t>(maxRecvTokenNum) : defaultMaxOutputSize;
    return ge::GRAPH_SUCCESS;
}

/*
 * 将激活属性规范化后写入 tiling data。默认值必须先于条件分支设置：
 * swiglu 直接使用这些默认值，其他激活类型在对应分支中覆盖所需字段。
 */
static void SetActivationAttrParams(const gert::TilingContext *context, const MegaMoeConfig &config,
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
 * 设置 prefetch 状态位区上限与 wave 粒度。maxTilesPerExpert 先无条件置零、再由 prefetch 分支覆盖，
 * 非 prefetch 路径依赖这次置零，不可改成只在分支内赋值。
 * 本步读 maxOutputSize，必须排在 CheckAndSetMaxRecvTokenNumAttr 之后。
 */
static void SetPrefetchAndWaveParams(MegaMoeTilingData *tilingData, const uint32_t aicNum)
{
    // GMM1 tile 状态位区每 expert 的 tile 上限（仅 prefetch 软同步路径使用）。
    // 非交织调度只遍历 hiddenDim / 2，交织调度会遍历完整 hiddenDim；host 无法感知 kernel
    // 编译期开关，因此按完整 hiddenDim 预留，避免交织模式覆盖下一 expert 的状态槽。
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

/*
 * 计算一个 token 的交织量化记录大小。dispatch peermem、SendMask UB 和共享专家量化都必须复用该口径。
 */
static uint32_t CalcQuantTokenAndScaleBytes(const MegaMoeTilingData *tilingData, uint32_t activationElementsPerByte)
{
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    return quantTokenBytes + quantScaleAlignBytes;
}

/*
 * 计算每个 dispatch ring slot 的量化 token 与 scale 拷贝区字节数（prefetch 模式再追加对齐后的 weight）。
 */
static uint32_t CalcDispatchCopyBufferBytes(const MegaMoeTilingData *tilingData, uint32_t activationElementsPerByte)
{
    uint32_t copyBufferBytes = CalcQuantTokenAndScaleBytes(tilingData, activationElementsPerByte);
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
 * 分两阶段确定 dispatch UB 配置：首先以基准 route batch 计算最大 ring 深度，
 * 随后在 ring 深度固定后利用剩余 UB 扩大 route batch，以减少批次数量。
 * 预算减法采用饱和语义以避免无符号回绕；bufferCount 必须依次应用上限和下限约束。
 */
static void SelectDispatchRingAndRouteBatch(MegaMoeDispatchBufferConfig &bufferConfig, uint64_t sendTotalNum,
                                            uint64_t alignedTotalRouteItems, uint32_t fixedBufferBytes,
                                            uint32_t dispatchSlotBytes, uint32_t availableUbBytes)
{
    // 第一阶段：使用基准 batch 确定 ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_RECV_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>(ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));

    // MTE 接收侧只保留一个 int32 topK 有效下标 batch。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t bytesWithoutDispatchSlots = fixedBufferBytes + routeIndexBufferBytes;
    // dispatchSlotBudgetBytes 是扣除非 ring tensor 后可用于分配 ring slot 的 UB。
    uint32_t dispatchSlotBudgetBytes =
        availableUbBytes > bytesWithoutDispatchSlots ? availableUbBytes - bytesWithoutDispatchSlots : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(dispatchSlotBudgetBytes / dispatchSlotBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_DISPATCH_BUFFER_COUNT);
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_DISPATCH_BUFFER_COUNT);

    // 第二阶段：在 ring 深度固定后使用剩余 UB 扩大 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithDispatchSlots 包含固定 tensor 和已选中的 dispatch slot。
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

/*
 * 根据路由规模和可用 UB 计算 dispatch 阶段的分批大小、ring 深度及拷贝缓冲区大小。
 */
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
    // batch 可能从某个 token 的 topK 段中间开始，slot 按该 batch 可跨越的 token 数上界预留。
    uint64_t maxMatchedRouteItems =
        ops::CeilDiv(static_cast<uint64_t>(routeItemsPerBatch) + topK - 1U, static_cast<uint64_t>(topK));
    uint64_t validIndexBytes = ops::CeilAlign(maxMatchedRouteItems * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32));
    return static_cast<uint64_t>(routeItemsPerBatch) / BITS_PER_BYTE + validIndexBytes;
}

/*
 * 针对给定的单核专家数，计算 topK 有效下标发送阶段的路由分批和 ring 配置。
 */
static MegaMoeSendMaskBufferConfig CalcTopkValidIndexBufferConfig(const MegaMoeTilingData *tilingData,
                                                                  uint32_t fixedBufferBytes, uint32_t ownedExpertCount,
                                                                  uint32_t availableUbBytes)
{
    MegaMoeSendMaskBufferConfig bufferConfig{};
    // 发送批网格按 numMaxTokensPerRank * topK 容量上界划分，kernel 再按实际 bs * topK 裁剪。
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->topK;
    uint64_t alignedTotalRouteItems = ops::CeilAlign(sendTotalNum, static_cast<uint64_t>(ALIGN_256));

    // 第一阶段：使用基准 batch 确定 route ring 深度。
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
    uint32_t routeBufferBudgetBytes =
        availableUbBytes > bytesWithoutRouteBuffers ? availableUbBytes - bytesWithoutRouteBuffers : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(routeBufferBudgetBytes / bufferConfig.bufferBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_SEND_MASK_BUFFER_COUNT);

    uint64_t routePushCount = static_cast<uint64_t>(bufferConfig.routeBatchCount) * ownedExpertCount;
    if (routePushCount > 0U && static_cast<uint64_t>(bufferConfig.bufferCount) > routePushCount) {
        bufferConfig.bufferCount = static_cast<int32_t>(routePushCount);
    }
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);

    // 第二阶段：固定 ring 深度，用剩余 UB 扩大 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        uint64_t fixedBytesWithRoutePadding =
            static_cast<uint64_t>(fixedBufferBytes) +
            static_cast<uint64_t>(bufferConfig.bufferCount) * (ALIGN_32 + 2U * sizeof(int32_t));
        uint64_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithRoutePadding ? availableUbBytes - fixedBytesWithRoutePadding : 0U;
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
 * 按 weight 元素数量更新 unpermute weight 区大小：始终分配 FP32 主区，
 * 仅在需要数据类型转换时分配转换暂存区。初始配置和扩批后均复用该计算。
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

/*
 * 根据单核 token 数和 topK weight 转换需求，计算 unpermute 阶段的分批及 UB 布局。
 */
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

    // 第一阶段：按基准 weight batch 确定每批 token 数和 weight 区，再以剩余 UB 分配输入 ring。
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

    // 第二阶段：ring 深度固定后，使用剩余 UB 扩大 weight batch。
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

/*
 * 计算每个本地 MoE 专家所需的 combine 同步槽位数；不使用分组同步时返回 0。
 */
static uint64_t CalcCombineSyncSlotCountPerExpert(const MegaMoeTilingData *tilingData)
{
    // MTE 统一使用 per-expert AIC ready 表；group counter 服务 URMA Layered Combine。
    if (tilingData->topoType != TOPO_TYPE_URMA || tilingData->moeExpertPerRank == 0U) {
        return 0U;
    }

    // Layered Combine 仅由 subBlockIdx=1 的半数 AIV 执行。
    uint64_t combineCoreCount = tilingData->blockAivNum / 2U;
    // 同一 token 的 topK expert id 不重复，因此单 expert 从每张卡最多接收 bs 个 token。
    uint64_t maxTokenCountForOneExpert =
        static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->epWorldSize;
    uint64_t maxTokenGroupCountForOneExpert =
        ops::CeilDiv(maxTokenCountForOneExpert, static_cast<uint64_t>(COMBINE_TOKEN_GROUP_SIZE));
    // Workspace 在路由结果产生前分配，因此每个本卡 MoE expert 都按独立最坏情况预留 slot。
    return std::max(maxTokenGroupCountForOneExpert, combineCoreCount);
}

/*
 * 汇总各执行路径在 workspace 中需要的 host flag 与同步计数器元素数量。
 */
static uint64_t CalcHostFlagElementCount(const MegaMoeTilingData *tilingData)
{
    uint64_t maxWavesPerExpert = ops::CeilDiv<uint64_t>(tilingData->maxOutputSize, L1_TILE_M_256);
    uint64_t waveFlagSlotsPerExpert = maxWavesPerExpert * INT_CACHELINE;
    uint64_t activationFlagSlotsPerExpert =
        tilingData->topoType == TOPO_TYPE_MTE ? waveFlagSlotsPerExpert : INT_CACHELINE;
    uint64_t moeExpertCount = tilingData->moeExpertPerRank;

    uint64_t flagElementCount = moeExpertCount * (activationFlagSlotsPerExpert + waveFlagSlotsPerExpert +
                                                  static_cast<uint64_t>(INT_CACHELINE) * tilingData->aicNum);
    bool hasSharedW4Gmm = tilingData->sharedExpertNum > 0 && IsW4GmmMode(tilingData->sharedGmmMode);
    if (IsW4GmmMode(tilingData->moeGmmMode) || hasSharedW4Gmm ||
        (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode == COMBINE_NO_QUANT)) {
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
        flagElementCount +=
            static_cast<uint64_t>(CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(tilingData->bs))) *
            tilingData->sharedExpertNum;
    }
    return flagElementCount;
}

/*
 * 计算 topK 有效下标发送阶段与路由 ring 深度无关的固定 UB 占用。
 */
static uint32_t CalcTopkValidIndexFixedBufferBytes(const MegaMoeTilingData *tilingData,
                                                   uint32_t moeActivationElementsPerByte,
                                                   uint32_t sharedActivationElementsPerByte,
                                                   bool needsIndependentSharedQuant)
{
    uint64_t totalFlagElementCount = CalcHostFlagElementCount(tilingData);
    uint32_t resetElementCountPerCore =
        static_cast<uint32_t>(ops::CeilDiv(totalFlagElementCount, static_cast<uint64_t>(tilingData->blockAivNum)));
    uint32_t resetBatchElementCount = std::min(resetElementCountPerCore, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    uint32_t resetTensorBytes =
        ops::CeilAlign(resetBatchElementCount, static_cast<uint32_t>(INT32_PER_256B)) * sizeof(int32_t);

    uint32_t moeQuantOutputBufferBytes = CalcDispatchCopyBufferBytes(tilingData, moeActivationElementsPerByte);
    uint32_t quantOutputBufferBytes = moeQuantOutputBufferBytes;
    if (needsIndependentSharedQuant) {
        // 两侧量化分时复用同一组 xOut UB，按较大的单 token 记录预留；共享专家不存储 topK weight。
        uint32_t sharedQuantOutputBufferBytes =
            CalcQuantTokenAndScaleBytes(tilingData, sharedActivationElementsPerByte);
        quantOutputBufferBytes = std::max(moeQuantOutputBufferBytes, sharedQuantOutputBufferBytes);
    }

    uint32_t quantInputBufferBytes = ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_128)) * sizeof(uint16_t);
    uint32_t maxExpertCountPerCore =
        ops::CeilDiv(tilingData->epWorldSize * tilingData->moeExpertPerRank, tilingData->blockAivNum);
    uint32_t sendCountAccumulatorBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(maxExpertCountPerCore) * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32)));
    // mxTempTensor_ 占 2KB，xOutTensor_ 和 xInTensor_ 各使用双 buffer。
    constexpr uint32_t mxTempTensorBytes = 2U * 1024U;
    return resetTensorBytes + mxTempTensorBytes + 2U * quantOutputBufferBytes + 2U * quantInputBufferBytes +
           sendCountAccumulatorBytes;
}

/*
 * 设置 topK 有效下标发送的两套 UB 配置：先算固定占用，再按 expert 分核的两类 core 各算一套。
 * 本函数读 tilingData->combineSyncSlotCountPerExpert（经 CalcHostFlagElementCount），
 * 该字段必须在调用前写好。
 */
static void SetTopkValidIndexBufferConfigs(MegaMoeTilingData *tilingData, uint32_t moeActivationElementsPerByte,
                                           uint32_t sharedActivationElementsPerByte, bool needsIndependentSharedQuant,
                                           uint32_t availableUbBytes)
{
    uint32_t sendMaskFixedBufferBytes = CalcTopkValidIndexFixedBufferBytes(
        tilingData, moeActivationElementsPerByte, sharedActivationElementsPerByte, needsIndependentSharedQuant);

    /*
     * 与 kernel topK 有效下标发送的 expert 连续均衡分核一一对应。前 remainder 个 AIV job
     * 各多处理一个 expert，因此只需预计算两套配置。
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

/*
 * 按依赖顺序生成 dispatch、SendMask 和 unpermute 的自适应 UB 配置。
 */
static void SetAdaptiveBufferConfigs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                     const MegaMoeExpertParams &expertParams, MegaMoeTilingData *tilingData,
                                     uint32_t availableUbBytes)
{
    auto topKWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    uint32_t activationElementsPerByte = expertParams.moe.quantOutDtype == ge::DT_FLOAT4_E2M1 ? 2U : 1U;
    uint32_t sharedActivationElementsPerByte = expertParams.shared.quantOutDtype == ge::DT_FLOAT4_E2M1 ? 2U : 1U;
    bool needsIndependentSharedQuant =
        expertParams.shared.expertCount > 0 && expertParams.shared.quantOutDtype != expertParams.moe.quantOutDtype;
    // 所有 AIV 核共用同一套 dispatch UB 配置，该布局与分核方式无关。
    // 若 kernel 改为按核拆分 tensor 或调整 copyTmp 槽位布局，需同步更新此处计算。
    tilingData->dispatchBufferConfig =
        CalcDispatchBufferConfig(tilingData, activationElementsPerByte, availableUbBytes);

    // 先于 topK 有效下标配置写入：CalcHostFlagElementCount 会读这个字段累加 flag 区大小，
    // 该依赖只经 tilingData 传递，调换顺序会静默改变 flag 区尺寸。
    tilingData->combineSyncSlotCountPerExpert = CalcCombineSyncSlotCountPerExpert(tilingData);

    SetTopkValidIndexBufferConfigs(tilingData, activationElementsPerByte, sharedActivationElementsPerByte,
                                   needsIndependentSharedQuant, availableUbBytes);

    ge::DataType topKWeightsDataType = topKWeightsDesc->GetDataType();
    SetUnpermuteBufferConfigs(tilingData, topKWeightsDataType, availableUbBytes);
}

/*
 * 汇总系统、算子及保留区 workspace 大小，并写入 tiling context。
 */
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

/*
 * 集中校验所有必需输入、输出及动态输入首项的指针，后续 tensor 校验不再重复判空。
 */
static ge::graphStatus CheckRequiredTensorPtrNullptr(const gert::TilingContext *context, const MegaMoeConfig &config)
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
    auto contextShape = context->GetInputShape(config.contextIndex);
    auto xShape = context->GetInputShape(config.xIndex);
    auto topkIdsShape = context->GetInputShape(config.topkIdsIndex);
    auto topkWeightsShape = context->GetInputShape(config.topkWeightsIndex);
    auto weightOneShape = context->GetDynamicInputShape(config.weight1Index, 0);
    auto weightTwoShape = context->GetDynamicInputShape(config.weight2Index, 0);
    auto weightScalesOneShape = context->GetDynamicInputShape(config.weightScales1Index, 0);
    auto weightScalesTwoShape = context->GetDynamicInputShape(config.weightScales2Index, 0);
    auto yShape = context->GetOutputShape(config.yIndex);
    auto expertTokenNumsShape = context->GetOutputShape(config.expertTokenNumsIndex);

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
    OP_CHECK_NULL_WITH_CONTEXT(context, contextShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);

    return ge::GRAPH_SUCCESS;
}

/*
 * arch35 当前不支持 xActiveMask 和 scales，可选输入一旦传入即拒绝。
 */
static ge::graphStatus CheckUnsupportedOptionalInputs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                      const char *nodeName)
{
    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(xActiveMaskDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "xActiveMask", "not null", "null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scalesDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "scales", "not null", "null"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验单个 tensor 的维数是否符合接口约定。
 */
static ge::graphStatus CheckTensorDimNum(const gert::StorageShape *storageShape, uint32_t expectedDimNum,
                                         const char *tensorName, const char *nodeName)
{
    uint32_t actualDimNum = storageShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(actualDimNum != expectedDimNum,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, tensorName, (std::to_string(actualDimNum) + "D").c_str(),
                                                 (std::to_string(expectedDimNum) + "D").c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 TensorList 指定项的维数及各维大小是否与首项一致。
 */
static ge::graphStatus CheckTensorListEntryMatchesReference(const gert::StorageShape *referenceShape,
                                                            const gert::CompileTimeTensorDesc *referenceDesc,
                                                            const gert::StorageShape *currentShape,
                                                            const gert::CompileTimeTensorDesc *currentDesc,
                                                            const std::string &tensorName, uint32_t tensorIndex,
                                                            const char *nodeName)
{
    const std::string entryName = tensorName + "[" + std::to_string(tensorIndex) + "]";
    const uint32_t referenceDimNum = referenceShape->GetStorageShape().GetDimNum();
    const uint32_t currentDimNum = currentShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(currentDimNum != referenceDimNum,
                    OP_LOGE(nodeName,
                            "%s has %u dimensions, but %s[0] has %u dimensions; "
                            "tensors in the same list must have matching dimensions.",
                            entryName.c_str(), currentDimNum, tensorName.c_str(), referenceDimNum),
                    return ge::GRAPH_FAILED);

    bool shapeMismatch = false;
    for (uint32_t dimIndex = 0; dimIndex < referenceDimNum && !shapeMismatch; ++dimIndex) {
        shapeMismatch =
            currentShape->GetStorageShape().GetDim(dimIndex) != referenceShape->GetStorageShape().GetDim(dimIndex);
    }
    OP_TILING_CHECK(shapeMismatch,
                    OP_LOGE_FOR_INVALID_SHAPE(nodeName, entryName.c_str(),
                                              Ops::Base::ToString(currentShape->GetStorageShape()).c_str(),
                                              Ops::Base::ToString(referenceShape->GetStorageShape()).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        currentDesc->GetDataType() != referenceDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, entryName.c_str(), Ops::Base::ToString(currentDesc->GetDataType()).c_str(),
                                  Ops::Base::ToString(referenceDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(currentDesc->GetStorageFormat() != referenceDesc->GetStorageFormat(),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, entryName.c_str(),
                                               Ops::Base::ToString(currentDesc->GetStorageFormat()).c_str(),
                                               Ops::Base::ToString(referenceDesc->GetStorageFormat()).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验同一 TensorList 内所有 tensor 的维数、shape、dtype 和 format 与首项一致。
 * 以第 0 项为基准，从第 1 项开始逐项比较。
 */
static ge::graphStatus CheckTensorListInternalConsistency(const gert::TilingContext *context, uint32_t inputIndex,
                                                          const char *inputRole, const char *expertTypeName,
                                                          const char *nodeName)
{
    std::string inputName = std::string(inputRole) + " of " + expertTypeName;
    uint32_t tensorCount = GetDynamicInputTensorCount(context, inputIndex);
    OP_TILING_CHECK(tensorCount == 0U, OP_LOGE_WITH_INVALID_INPUT(nodeName, inputName.c_str()),
                    return ge::GRAPH_FAILED);

    auto referenceShape = context->GetDynamicInputShape(inputIndex, 0);
    auto referenceDesc = context->GetDynamicInputDesc(inputIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceDesc);
    for (uint32_t tensorIdx = 1; tensorIdx < tensorCount; ++tensorIdx) {
        auto currentShape = context->GetDynamicInputShape(inputIndex, tensorIdx);
        auto currentDesc = context->GetDynamicInputDesc(inputIndex, tensorIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentDesc);

        if (CheckTensorListEntryMatchesReference(referenceShape, referenceDesc, currentShape, currentDesc, inputName,
                                                 tensorIdx, nodeName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightDimRelation(const gert::StorageShape *shape, uint32_t expectedDimNum,
                                              uint32_t referenceDimNum, const char *tensorName,
                                              const char *referenceName, const char *nodeName)
{
    const uint32_t actualDimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        actualDimNum != expectedDimNum,
        OP_LOGE(nodeName, "%s must have %s %s, but got %u and %u dimensions, respectively.", tensorName,
                expectedDimNum == referenceDimNum ? "the same number of dimensions as" : "one more dimension than",
                referenceName, actualDimNum, referenceDimNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧专家的 weight 与 scale tensor 维数符合当前权重组织形式。
 */
static ge::graphStatus CheckWeightAndScaleDimNum(const gert::TilingContext *context, const ExpertParams &expertParams,
                                                 uint32_t weightDimNum, const char *nodeName)
{
    const auto &inputs = expertParams.inputs;
    const uint32_t scaleDimNum = weightDimNum + 1U;
    const std::string nameSuffix = std::string(" of ") + expertParams.expertTypeName;
    const std::string weightOneName = "weight1[0]" + nameSuffix;
    const std::string weightTwoName = "weight2[0]" + nameSuffix;
    const std::string weightScalesOneName = "weight_scales1[0]" + nameSuffix;
    const std::string weightScalesTwoName = "weight_scales2[0]" + nameSuffix;

    auto weightOneShape = context->GetDynamicInputShape(inputs.weightOne, 0);
    auto weightTwoShape = context->GetDynamicInputShape(inputs.weightTwo, 0);
    auto weightScalesOneShape = context->GetDynamicInputShape(inputs.weightScalesOne, 0);
    auto weightScalesTwoShape = context->GetDynamicInputShape(inputs.weightScalesTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);

    bool hasInvalidDimNum =
        CheckWeightDimRelation(weightOneShape, weightDimNum, weightDimNum, weightOneName.c_str(),
                               "weight1[0] of MoE expert", nodeName) != ge::GRAPH_SUCCESS ||
        CheckWeightDimRelation(weightTwoShape, weightDimNum, weightDimNum, weightTwoName.c_str(), weightOneName.c_str(),
                               nodeName) != ge::GRAPH_SUCCESS ||
        CheckWeightDimRelation(weightScalesOneShape, scaleDimNum, weightDimNum, weightScalesOneName.c_str(),
                               weightOneName.c_str(), nodeName) != ge::GRAPH_SUCCESS ||
        CheckWeightDimRelation(weightScalesTwoShape, scaleDimNum, weightDimNum, weightScalesTwoName.c_str(),
                               weightTwoName.c_str(), nodeName) != ge::GRAPH_SUCCESS;
    return hasInvalidDimNum ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧两个 weight scale tensor 的末维符合 multi-base 存储约定。
 * 调用前应先通过 CheckWeightAndScaleDimNum，保证目标维存在。
 */
static ge::graphStatus CheckWeightScaleTrailingDim(const gert::TilingContext *context, const ExpertParams &expertParams,
                                                   bool isPerExpertWeightTensor, const char *nodeName)
{
    auto weightScalesOneShape = context->GetDynamicInputShape(expertParams.inputs.weightScalesOne, 0);
    auto weightScalesTwoShape = context->GetDynamicInputShape(expertParams.inputs.weightScalesTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);

    const int64_t scaleOneMultiBase =
        GetSingleExpertTensorDimSize(weightScalesOneShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t scaleTwoMultiBase =
        GetSingleExpertTensorDimSize(weightScalesTwoShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, isPerExpertWeightTensor);
    const std::string scaleNames = std::string(expertParams.expertTypeName) + " weight_scales1, weight_scales2";
    const std::string trailingDimensions =
        "[" + std::to_string(scaleOneMultiBase) + ", " + std::to_string(scaleTwoMultiBase) + "]";
    OP_TILING_CHECK(
        scaleOneMultiBase != WEIGHT_SCALE_MULTI_BASE_DIM_SIZE || scaleTwoMultiBase != WEIGHT_SCALE_MULTI_BASE_DIM_SIZE,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, scaleNames.c_str(), trailingDimensions.c_str(),
                                               "The trailing dimension of each weight scale tensor must be 2"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧四类 weight/scale 动态输入的 Tensor 数量一致；stacked 布局要求每类仅有一个 Tensor。
 */
static ge::graphStatus CheckWeightAndScaleTensorCounts(const gert::TilingContext *context,
                                                       const ExpertParams &expertParams, bool isPerExpertWeightTensor,
                                                       const char *nodeName)
{
    const auto &inputs = expertParams.inputs;
    uint32_t weightOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightOne);
    uint32_t weightTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightTwo);
    uint32_t weightScalesOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesOne);
    uint32_t weightScalesTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesTwo);
    bool tensorCountsMismatch = weightTwoTensorCount != weightOneTensorCount ||
                                weightScalesOneTensorCount != weightOneTensorCount ||
                                weightScalesTwoTensorCount != weightOneTensorCount;
    OP_TILING_CHECK(
        tensorCountsMismatch || (!isPerExpertWeightTensor && weightOneTensorCount != 1U),
        OP_LOGE(nodeName,
                "weight1, weight2, weight_scales1 and weight_scales2 of %s must contain the same number of tensors; "
                "stacked layout requires exactly one tensor.",
                expertParams.expertTypeName),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 stacked 布局下四类 weight/scale tensor 的专家维大小相互一致。
 */
static ge::graphStatus CheckStackedExpertDimConsistency(const gert::TilingContext *context,
                                                        const ExpertParams &expertParams, const char *nodeName)
{
    const auto &inputs = expertParams.inputs;
    auto weightOneShape = context->GetDynamicInputShape(inputs.weightOne, 0);
    auto weightTwoShape = context->GetDynamicInputShape(inputs.weightTwo, 0);
    auto weightScalesOneShape = context->GetDynamicInputShape(inputs.weightScalesOne, 0);
    auto weightScalesTwoShape = context->GetDynamicInputShape(inputs.weightScalesTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);

    int64_t expertCount = weightOneShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(weightTwoShape->GetStorageShape().GetDim(0) != expertCount ||
                        weightScalesOneShape->GetStorageShape().GetDim(0) != expertCount ||
                        weightScalesTwoShape->GetStorageShape().GetDim(0) != expertCount,
                    OP_LOGE(nodeName,
                            "Dim0 of weight1, weight2, weight_scales1 and weight_scales2 must have matching expert "
                            "counts for %s inputs.",
                            expertParams.expertTypeName),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧专家四类 weight/scale tensor 的列表长度、列表内部属性及堆叠专家维保持一致。
 */
static ge::graphStatus CheckWeightAndScaleConsistency(const gert::TilingContext *context,
                                                      const ExpertParams &expertParams, uint32_t weightDimNum,
                                                      const char *nodeName)
{
    const auto &inputs = expertParams.inputs;
    const bool isPerExpertWeightTensor = weightDimNum == TWO_DIMS;
    if (CheckWeightAndScaleTensorCounts(context, expertParams, isPerExpertWeightTensor, nodeName) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool inconsistentTensorLists =
        CheckTensorListInternalConsistency(context, inputs.weightOne, "weight1", expertParams.expertTypeName,
                                           nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorListInternalConsistency(context, inputs.weightTwo, "weight2", expertParams.expertTypeName,
                                           nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorListInternalConsistency(context, inputs.weightScalesOne, "weight_scales1",
                                           expertParams.expertTypeName, nodeName) != ge::GRAPH_SUCCESS ||
        CheckTensorListInternalConsistency(context, inputs.weightScalesTwo, "weight_scales2",
                                           expertParams.expertTypeName, nodeName) != ge::GRAPH_SUCCESS;
    if (inconsistentTensorLists) {
        return ge::GRAPH_FAILED;
    }
    if (!isPerExpertWeightTensor) {
        return CheckStackedExpertDimConsistency(context, expertParams, nodeName);
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 以 MoE weight1[0] 的维数确定权重输入的组织形式：2D 表示逐专家 TensorList，3D 表示单 tensor 堆叠。
 * 分别校验 MoE 和共享专家的 weight/scale 维数契约与一致性，并记录布局派生信息。
 * 本函数不校验 GMM 矩阵维关系，也不判断 dtype/format 组合是否受支持。
 */
static ge::graphStatus CheckAndSetWeightAndScaleLayout(const gert::TilingContext *context, MegaMoeExpertParams &params,
                                                       MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto weightOneShape = context->GetDynamicInputShape(params.moe.inputs.weightOne, 0);
    uint32_t weightDimNum = weightOneShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        weightDimNum != TWO_DIMS && weightDimNum != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "weight1", (std::to_string(weightDimNum) + "D").c_str(),
                                                 "weight1 must be one stacked 3D tensor or a list of 2D tensors."),
        return ge::GRAPH_FAILED);
    params.isPerExpertWeightTensor = weightDimNum == TWO_DIMS;

    OP_TILING_CHECK(CheckWeightAndScaleDimNum(context, params.moe, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "MoE expert weight or scale tensor dimension is invalid."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckWeightScaleTrailingDim(context, params.moe, params.isPerExpertWeightTensor, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "MoE expert weight scale trailing dimension is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckWeightAndScaleConsistency(context, params.moe, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "MoE expert weight layout is invalid."), return ge::GRAPH_FAILED);
    params.moe.expertCount = GetWeightExpertCount(context, params.moe.inputs.weightOne, params.isPerExpertWeightTensor);

    // 检查共享专家的四个 weight/scale 输入是否同时存在或同时缺省。
    params.shared.expertCount =
        GetWeightExpertCount(context, params.shared.inputs.weightOne, params.isPerExpertWeightTensor);
    bool hasSharedWeightOne = params.shared.expertCount > 0;
    bool hasSharedWeightTwo =
        GetWeightExpertCount(context, params.shared.inputs.weightTwo, params.isPerExpertWeightTensor) > 0;
    bool hasSharedWeightScalesOne =
        GetWeightExpertCount(context, params.shared.inputs.weightScalesOne, params.isPerExpertWeightTensor) > 0;
    bool hasSharedWeightScalesTwo =
        GetWeightExpertCount(context, params.shared.inputs.weightScalesTwo, params.isPerExpertWeightTensor) > 0;
    OP_TILING_CHECK(hasSharedWeightOne != hasSharedWeightTwo || hasSharedWeightOne != hasSharedWeightScalesOne ||
                        hasSharedWeightOne != hasSharedWeightScalesTwo,
                    OP_LOGE(nodeName, "Shared expert weights and weight scales must be provided together."),
                    return ge::GRAPH_FAILED);
    if (hasSharedWeightOne) {
        OP_TILING_CHECK(CheckWeightAndScaleDimNum(context, params.shared, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Shared expert weight or scale tensor dimension is invalid."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(CheckWeightScaleTrailingDim(context, params.shared, params.isPerExpertWeightTensor, nodeName) !=
                            ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Shared expert weight scale trailing dimension is invalid."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            CheckWeightAndScaleConsistency(context, params.shared, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Shared expert weight layout is invalid."), return ge::GRAPH_FAILED);
    }
    tilingData->isPerExpertWeightTensor = params.isPerExpertWeightTensor;
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧 weight1、weight2 与 x 的矩阵维关系。
 */
static ge::graphStatus CheckWeightPairShapeRelations(const gert::TilingContext *context,
                                                     const ExpertParams &expertParams, uint32_t xIndex,
                                                     bool isPerExpertWeightTensor, const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightOne, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    auto weightTwoStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightTwo, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);

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
    const gert::StorageShape *xStorageShape = context->GetInputShape(xIndex);
    const int64_t xColumnCount = xStorageShape->GetStorageShape().GetDim(1);
    const std::string commonMatrixDimensionsString = "[" + std::to_string(weightOneColumnCount) + ", " +
                                                     std::to_string(weightTwoRowCount) + ", " +
                                                     std::to_string(xColumnCount) + "]";
    OP_TILING_CHECK(weightOneColumnCount != weightTwoRowCount || weightOneColumnCount != xColumnCount,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, expertParams.expertTypeName, commonMatrixDimensionsString.c_str(),
                        "weight1 column count and weight2 row count must equal x column count"),
                    return ge::GRAPH_FAILED);

    const std::string weightRowColumnDimensionsString =
        "[" + std::to_string(weightOneRowCount) + ", " + std::to_string(weightTwoColumnCount) + "]";
    OP_TILING_CHECK(weightOneRowCount != weightTwoColumnCount * SWIGLU_GATE_UP_SPLIT_FACTOR,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, expertParams.expertTypeName, weightRowColumnDimensionsString.c_str(),
                        "weight1 row count must equal weight2 column count multiplied by 2"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 分别校验 MoE 与已启用共享专家的 weight1、weight2 矩阵维关系。
 */
static ge::graphStatus CheckWeightShapeRelations(const gert::TilingContext *context,
                                                 const MegaMoeExpertParams &expertParams, uint32_t xIndex,
                                                 const char *nodeName)
{
    if (CheckWeightPairShapeRelations(context, expertParams.moe, xIndex, expertParams.isPerExpertWeightTensor,
                                      nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (expertParams.shared.expertCount <= 0) {
        return ge::GRAPH_SUCCESS;
    }
    return CheckWeightPairShapeRelations(context, expertParams.shared, xIndex, expertParams.isPerExpertWeightTensor,
                                         nodeName);
}

/*
 * 校验输出 y 的维数及 shape 是否与输入 x 一致。
 */
static ge::graphStatus CheckYShape(const gert::TilingContext *context, const MegaMoeConfig &config,
                                   const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    auto yStorageShape = context->GetOutputShape(config.yIndex);
    if (CheckTensorDimNum(yStorageShape, TWO_DIMS, "y", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        yDim0 != bs || yDim1 != h,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "y", (std::string("[") + std::to_string(yDim0) + ", " + std::to_string(yDim1) + "]").c_str(),
            (std::string("The shape of y must be [bs, h] = [") + std::to_string(bs) + ", " + std::to_string(h) + "].")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 expert_token_nums 为一维 tensor，且长度等于本 rank 的 MoE 专家数。
 */
static ge::graphStatus CheckExpertTokenNumsShape(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                 uint32_t moeExpertPerRank, const char *nodeName)
{
    auto expertTokenNumsStorageShape = context->GetOutputShape(config.expertTokenNumsIndex);
    if (CheckTensorDimNum(expertTokenNumsStorageShape, ONE_DIM, "expert_token_nums", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    // expertTokenNums 仅报告 MoE 专家的 token 数，不包含共享专家。
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
 * 校验一组 weight 与对应 scale 的 shape 关系。每 INPUT_WEIGHT_SCALES_CEIL_ALIGN 个 weight 列
 * 共用一个 scale，因此 scale shape 应为 [weight 行数, ceil(weight 列数 / 对齐值)]。
 * weight1 和 weight2 采用相同规则；scale 自身的 shape 约束已在布局解析阶段校验。
 */
static ge::graphStatus CheckWeightScaleShapeRelation(const gert::StorageShape *weightScaleStorageShape,
                                                     const gert::StorageShape *weightStorageShape,
                                                     bool isPerExpertWeightTensor, const char *expertTypeName,
                                                     const char *weightScaleRole, const char *weightRole,
                                                     const char *nodeName)
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
                        nodeName, (std::string(expertTypeName) + " " + weightScaleRole).c_str(),
                        (std::string("matrix dimension=") + std::to_string(weightScaleMatrixDimSize)).c_str(),
                        (std::string("The matrix dimension must equal the row count of ") + expertTypeName + " " +
                         weightRole + "(" + std::to_string(weightRowCount) + ")")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(weightScaleGroupDimSize != ops::CeilDiv(weightColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, (std::string(expertTypeName) + " " + weightScaleRole).c_str(),
                        (std::string("group dimension=") + std::to_string(weightScaleGroupDimSize)).c_str(),
                        (std::string("The group dimension must equal CeilDiv(") + expertTypeName + " " + weightRole +
                         " column count, INPUT_WEIGHT_SCALES_CEIL_ALIGN) = " +
                         std::to_string(ops::CeilDiv(weightColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN)))
                            .c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧两组 weight scale 与对应 weight 的 shape 关系。
 */
static ge::graphStatus CheckWeightScaleShapeRelations(const gert::TilingContext *context,
                                                      const ExpertParams &expertParams, bool isPerExpertWeightTensor,
                                                      const char *nodeName)
{
    // CheckAndSetWeightAndScaleLayout 已保证本侧四个 TensorList 的首项 shape 存在且维数合法。
    auto weightScalesOneStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightScalesOne, 0);
    auto weightScalesTwoStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightScalesTwo, 0);
    auto weightOneStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightOne, 0);
    auto weightTwoStorageShape = context->GetDynamicInputShape(expertParams.inputs.weightTwo, 0);

    ge::graphStatus checkStatus =
        CheckWeightScaleShapeRelation(weightScalesOneStorageShape, weightOneStorageShape, isPerExpertWeightTensor,
                                      expertParams.expertTypeName, "weight_scales1", "weight1", nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }
    checkStatus =
        CheckWeightScaleShapeRelation(weightScalesTwoStorageShape, weightTwoStorageShape, isPerExpertWeightTensor,
                                      expertParams.expertTypeName, "weight_scales2", "weight2", nodeName);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        return checkStatus;
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验非权重输入的维数，并校验 x、topk_ids 和 topk_weights 的 token/topK 维一致性。
 */
static ge::graphStatus CheckNonWeightInputShapes(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                 const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    if (CheckTensorDimNum(contextStorageShape, ONE_DIM, "context", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    if (CheckTensorDimNum(xStorageShape, TWO_DIMS, "x", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    if (CheckTensorDimNum(topkIdsStorageShape, TWO_DIMS, "topkIds", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    if (CheckTensorDimNum(topkWeightsStorageShape, TWO_DIMS, "topkWeights", nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);

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
 * 校验 weight1/weight2/x、weight/scale 以及 MoE/shared hiddenDim 之间的 shape 关系。
 */
static ge::graphStatus CheckWeightAndScaleShapeRelations(const gert::TilingContext *context,
                                                         const MegaMoeConfig &config,
                                                         const MegaMoeExpertParams &expertParams, const char *nodeName)
{
    OP_TILING_CHECK(CheckWeightShapeRelations(context, expertParams, config.xIndex, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "weight shape relation is invalid."), return ge::GRAPH_FAILED);

    if (expertParams.shared.expertCount > 0) {
        // 两侧 W1 column 和 W2 row 已分别校验为 x.H，且各自满足 W1 row = 2 * W2 column。
        // 因此跨侧只需保证 W1 row（GMM1 输出维 hiddenDim）一致。
        auto moeWeightOneShape = context->GetDynamicInputShape(expertParams.moe.inputs.weightOne, 0);
        auto sharedWeightOneShape = context->GetDynamicInputShape(expertParams.shared.inputs.weightOne, 0);
        int64_t moeHiddenDim = GetSingleExpertTensorDimSize(moeWeightOneShape, WEIGHT_MATRIX_ROW_DIM_INDEX,
                                                            expertParams.isPerExpertWeightTensor);
        int64_t sharedHiddenDim = GetSingleExpertTensorDimSize(sharedWeightOneShape, WEIGHT_MATRIX_ROW_DIM_INDEX,
                                                               expertParams.isPerExpertWeightTensor);
        OP_TILING_CHECK(sharedHiddenDim != moeHiddenDim,
                        OP_LOGE_FOR_INVALID_VALUE(
                            nodeName, "shared expert hiddenDim", std::to_string(sharedHiddenDim).c_str(),
                            (std::string("must equal MoE expert hiddenDim ") + std::to_string(moeHiddenDim)).c_str()),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(CheckWeightScaleShapeRelations(context, expertParams.moe, expertParams.isPerExpertWeightTensor,
                                                   nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "MoE weight scale shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        expertParams.shared.expertCount > 0 &&
            CheckWeightScaleShapeRelations(context, expertParams.shared, expertParams.isPerExpertWeightTensor,
                                           nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "shared weight scale shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验非权重输入张量的 dtype：context / x / topkIds / topkWeights。
 */
static ge::graphStatus CheckNonWeightInputDataTypes(const gert::TilingContext *context, const MegaMoeConfig &config,
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
 * 校验输出张量的 dtype：y 与 expertTokenNums。
 */
static ge::graphStatus CheckOutputDataTypes(const gert::TilingContext *context, const MegaMoeConfig &config,
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

/*
 * 判断 tensor 的主格式是否属于 FRACTAL_NZ。
 */
static bool IsNzPrimaryFormat(const gert::CompileTimeTensorDesc *desc)
{
    return static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ;
}

/*
 * 校验非权重输入和输出不使用 FRACTAL_NZ 主格式。
 */
static ge::graphStatus CheckNonWeightTensorFormats(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                   const char *nodeName)
{
    const gert::CompileTimeTensorDesc *descs[] = {
        context->GetInputDesc(config.xIndex), context->GetInputDesc(config.topkIdsIndex),
        context->GetInputDesc(config.topkWeightsIndex), context->GetOutputDesc(config.yIndex),
        context->GetOutputDesc(config.expertTokenNumsIndex)};
    const char *names[] = {"x", "topk_ids", "topk_weights", "y", "expert_token_nums"};
    for (uint32_t index = 0; index < sizeof(descs) / sizeof(descs[0]); ++index) {
        OP_TILING_CHECK(IsNzPrimaryFormat(descs[index]),
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, names[index],
                                                   Ops::Base::ToString(descs[index]->GetStorageFormat()).c_str(),
                                                   "a non-FRACTAL_NZ format"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验一侧 weight scale 的类型和格式；weight 类型与格式组合由后续 GMM 模式解析校验。
 */
static ge::graphStatus CheckExpertScaleDataTypesAndFormats(const gert::TilingContext *context,
                                                           const ExpertParams &expertParams, const char *nodeName)
{
    auto scaleOneDesc = context->GetDynamicInputDesc(expertParams.inputs.weightScalesOne, 0);
    auto scaleTwoDesc = context->GetDynamicInputDesc(expertParams.inputs.weightScalesTwo, 0);
    const std::string scaleNames = std::string(expertParams.expertTypeName) + " weight_scales1, weight_scales2";

    bool invalidDtype =
        scaleOneDesc->GetDataType() != ge::DT_FLOAT8_E8M0 || scaleTwoDesc->GetDataType() != ge::DT_FLOAT8_E8M0;
    const std::string scaleDtypes = "[" + Ops::Base::ToString(scaleOneDesc->GetDataType()) + ", " +
                                    Ops::Base::ToString(scaleTwoDesc->GetDataType()) + "]";
    OP_TILING_CHECK(invalidDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, scaleNames.c_str(), scaleDtypes.c_str(),
                                                           "Both weight scale dtypes must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);

    const std::string scaleFormats = "[" + Ops::Base::ToString(scaleOneDesc->GetStorageFormat()) + ", " +
                                     Ops::Base::ToString(scaleTwoDesc->GetStorageFormat()) + "]";
    OP_TILING_CHECK(
        IsNzPrimaryFormat(scaleOneDesc) || IsNzPrimaryFormat(scaleTwoDesc),
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(nodeName, scaleNames.c_str(), scaleFormats.c_str(),
                                                "Weight scale tensors must use a non-FRACTAL_NZ primary format."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 MoE 和共享专家 weight scale 的数据类型与格式契约。
 */
static ge::graphStatus CheckWeightScaleDataTypesAndFormats(const gert::TilingContext *context,
                                                           const MegaMoeExpertParams &expertParams,
                                                           const char *nodeName)
{
    OP_TILING_CHECK(CheckExpertScaleDataTypesAndFormats(context, expertParams.moe, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "MoE expert weight scale properties are invalid."), return ge::GRAPH_FAILED);

    if (expertParams.shared.expertCount > 0) {
        OP_TILING_CHECK(
            CheckExpertScaleDataTypesAndFormats(context, expertParams.shared, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "shared expert weight scale properties are invalid."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * 建立 weight/scale 布局派生信息，并校验 tensor 的 dtype、format、自身 shape 及跨 tensor shape 关系。
 */
static ge::graphStatus CheckAndSetTensorMetadata(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                 MegaMoeExpertParams &expertParams, MegaMoeTilingData *tilingData,
                                                 const char *nodeName)
{
    OP_TILING_CHECK(CheckAndSetWeightAndScaleLayout(context, expertParams, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expert weight or scale layout is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckNonWeightInputDataTypes(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "non-weight input dtype is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckNonWeightTensorFormats(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "non-weight tensor format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDataTypes(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "output dtype is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckWeightScaleDataTypesAndFormats(context, expertParams, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "weight scale dtypes or formats are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckNonWeightInputShapes(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "non-weight input shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckWeightAndScaleShapeRelations(context, config, expertParams, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "weight or scale shape relation is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckYShape(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "y tensor shape is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 从 x 和 topkIds 提取 bs、H 与 topK，校验取值范围以及按拓扑和权重分形确定的 H 对齐要求。
 */
static ge::graphStatus CheckAndSetTokenDimensions(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                                  const MegaMoeConfig &config, const MegaMoeExpertParams &expertParams,
                                                  const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);

    // 校验 topK 取值范围。
    OP_TILING_CHECK(
        topkIdsDim1 < MIN_TOPK || topkIdsDim1 > MAX_TOPK,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "topK", std::to_string(topkIdsDim1).c_str(), "only support [1, 32]"),
        return ge::GRAPH_FAILED);

    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    // 校验 H 取值范围。
    OP_TILING_CHECK(
        xDim1 < MIN_H || xDim1 > MAX_H,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "H", std::to_string(xDim1).c_str(),
            (std::string("should in [") + std::to_string(MIN_H) + ", " + std::to_string(MAX_H) + "]").c_str()),
        return ge::GRAPH_FAILED);
    bool usesW4C032WeightOne = expertParams.moe.gmmMode == GMM_MODE_A8W4_NZ ||
                               (expertParams.shared.expertCount > 0 && expertParams.shared.gmmMode == GMM_MODE_A8W4_NZ);
    // URMA/Layered 保持 1K 对齐；W4 的 NZ_C0_32 分形要求 GMM K 按 64 对齐；其余 MTE 路径按 32 对齐。
    int64_t requiredHAlignment =
        tilingData->topoType == TOPO_TYPE_URMA ? URMA_H_ALIGN : (usesW4C032WeightOne ? W4_K_ALIGN : H_ALIGN);
    OP_TILING_CHECK(
        xDim1 % requiredHAlignment != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "H", std::to_string(xDim1).c_str(),
                                  (std::string("multiple of ") + std::to_string(requiredHAlignment)).c_str()),
        return ge::GRAPH_FAILED);

    tilingData->bs = static_cast<uint32_t>(bs);
    tilingData->h = static_cast<uint32_t>(xDim1);
    tilingData->topK = static_cast<uint32_t>(topkIdsDim1);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验并设置 GMM1 输出维 hiddenDim：取值范围与统一对齐要求。
 */
static ge::graphStatus CheckAndSetHiddenDim(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                            const MegaMoeExpertParams &expertParams, const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(expertParams.moe.inputs.weightOne, 0);
    int64_t hiddenDim = GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX,
                                                     expertParams.isPerExpertWeightTensor);
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

    tilingData->hiddenDim = static_cast<uint32_t>(hiddenDim);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验通信拓扑枚举并写入 tiling data，供后续模式和资源规划使用。
 */
static ge::graphStatus CheckAndSetTopoType(const gert::TilingContext *context, const MegaMoeConfig &config,
                                           MegaMoeTilingData *tilingData, const char *nodeName)
{
    int64_t topoType = *context->GetAttrs()->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    OP_TILING_CHECK(topoType != TOPO_TYPE_MTE && topoType != TOPO_TYPE_URMA,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topoType", std::to_string(topoType).c_str(),
                                              "only support MTE(0) or URMA(1)"),
                    return ge::GRAPH_FAILED);
    tilingData->topoType = topoType;
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验并记录不依赖 tensor 元数据或问题规模的算子属性。
 * 直接属性：topo_type、ep_world_size、comm_alg、dispatch_quant_mode、combine_quant_mode、
 * activation、activation_params 和 topk_weights_type。
 * 具体属性的拦截项由各子函数保留，本函数只表达该阶段的执行顺序。
 */
static ge::graphStatus CheckAndSetIndependentAttrs(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                   MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(CheckAndSetTopoType(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "topology type is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetEpWorldSizeAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "EP world size is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckCommAlgAttr(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "communication algorithm is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetQuantModeAttrs(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "quantization mode attributes are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckActivationParams(context, config, tilingData->topoType, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "activation parameters are invalid."), return ge::GRAPH_FAILED);
    SetActivationAttrParams(context, config, tilingData);
    OP_TILING_CHECK(CheckAndSetTopkWeightsTypeAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "topK weights type is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 基于已校验的属性和 tensor 元数据，解析专家 GMM 模式、问题维度和专家数量。
 * 直接属性：dispatch_quant_out_dtype、shared_expert_quant_out_dtype 和 moe_expert_num；
 * GMM 模式、token 维度、hiddenDim 及 shared expert 数量由已校验的 tensor 元数据派生。
 */
static ge::graphStatus CheckAndSetExpertExecutionParams(const gert::TilingContext *context, const MegaMoeConfig &config,
                                                        MegaMoeExpertParams &expertParams,
                                                        MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(CheckAndSetQuantOutDtypes(context, config, expertParams, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expert quantization output dtypes are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetExpertGmmModes(context, expertParams, tilingData->topoType, tilingData, nodeName) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expert GMM modes are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAndSetTokenDimensions(context, tilingData, config, expertParams, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "token dimensions are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetHiddenDim(context, tilingData, expertParams, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "hiddenDim is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAndSetExpertCountAttrs(context, config, expertParams, tilingData, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "expert counts are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckExpertTokenNumsShape(context, config, tilingData->moeExpertPerRank, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "expertTokenNums tensor shape is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 读取并校验平台 AIC/AIV 数量与 UB 容量，同时写入 kernel 分核所需字段。
 */
static ge::graphStatus CheckAndSetPlatformParams(gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                                 uint32_t &aicNum, uint64_t &ubSize, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    aicNum = ascendcPlatform.GetCoreNumAic();
    OP_TILING_CHECK(aivNum == 0U || aicNum == 0U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "aivNum/aicNum",
                                              (std::to_string(aivNum) + ", " + std::to_string(aicNum)).c_str(),
                                              "should both be > 0"),
                    return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingData->aicNum = aicNum;
    tilingData->blockAivNum = aivNum;
    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->epWorldSize);
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);
    return ge::GRAPH_SUCCESS;
}

/*
 * 在所有校验和资源规划完成后，统一提交 workspace、block dim、tiling key 及诊断信息。
 */
static ge::graphStatus CommitTilingResult(gert::TilingContext *context, const MegaMoeExpertParams &expertParams,
                                          MegaMoeTilingData *tilingData, const char *nodeName)
{
    WorkspaceLayout workspaceLayout(tilingData);
    OP_TILING_CHECK(SetWorkspace(context, workspaceLayout, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(
        ascendcPlatform.CalcTschBlockDim(tilingData->blockAivNum, tilingData->aicNum, tilingData->blockAivNum));
    context->SetScheduleMode(1); // batch model, all cores start at the same time
    uint64_t tilingKey = CalcTilingKey(context, expertParams, tilingData);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    PrintMegaMoeTilingData(tilingData, nodeName);
    PrintWorkspaceLayout(&workspaceLayout, nodeName);
    PrintPeermemInfo(tilingData, nodeName);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MegaMoe tiling check func.");

    MegaMoeTilingData *tilingData = context->GetTilingData<MegaMoeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    OP_TILING_CHECK(CheckRequiredTensorPtrNullptr(context, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "tensor pointer check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckUnsupportedOptionalInputs(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "unsupported optional input is present."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredAttrPtrNullptr(context, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "attribute pointer check failed."), return ge::GRAPH_FAILED);

    // 独立属性：topo_type、ep_world_size、comm_alg、dispatch_quant_mode、combine_quant_mode、
    // activation、activation_params 和 topk_weights_type。
    OP_TILING_CHECK(CheckAndSetIndependentAttrs(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "independent attributes are invalid."), return ge::GRAPH_FAILED);

    // Tensor 契约：不读取 attr；先确定 weight/scale 组织形式，再校验 dtype、format 和 shape。
    MegaMoeExpertParams expertParams = MakeExpertParams(config);
    OP_TILING_CHECK(CheckAndSetTensorMetadata(context, config, expertParams, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "tensor metadata is invalid."), return ge::GRAPH_FAILED);

    // 专家执行属性：dispatch_quant_out_dtype、shared_expert_quant_out_dtype 和 expert_num。
    OP_TILING_CHECK(
        CheckAndSetExpertExecutionParams(context, config, expertParams, tilingData, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "expert execution parameters are invalid."), return ge::GRAPH_FAILED);

    // 容量属性：num_max_tokens_per_rank、max_recv_token_num 和 ccl_buffer_size。
    OP_TILING_CHECK(CheckAndSetMaxTokensPerRankAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "maximum tokens per rank is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetMaxRecvTokenNumAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "maximum received token count is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckCclBufferCapacity(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CCL buffer capacity is insufficient."), return ge::GRAPH_FAILED);

    // 平台与资源规划。
    uint32_t aicNum = 0U;
    uint64_t ubSize = 0U;
    OP_TILING_CHECK(CheckAndSetPlatformParams(context, tilingData, aicNum, ubSize, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "platform parameters are invalid."), return ge::GRAPH_FAILED);

    SetPrefetchAndWaveParams(tilingData, aicNum);
    SetAdaptiveBufferConfigs(context, config, expertParams, tilingData, static_cast<uint32_t>(ubSize));

    return CommitTilingResult(context, expertParams, tilingData, nodeName);
}

/*
 * 构造默认接口索引配置并进入 arch35 MegaMoe tiling 主流程。
 */
static ge::graphStatus MegaMoeTilingFunc(gert::TilingContext *context)
{
    MegaMoeConfig config;
    return MegaMoeTilingFuncImplPublic(context, config);
}

struct MegaMoeCompileInfo {};
/*
 * MegaMoe tiling parse 回调；当前无需生成额外的编译期信息。
 */
static ge::graphStatus TilingParseForMegaMoe(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaMoe).Tiling(MegaMoeTilingFunc).TilingParse<MegaMoeCompileInfo>(TilingParseForMegaMoe);

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
/*
 * 将 MegaMoe 运行时异常转交 MC2 通用异常处理实现。
 */
inline void MegaMoeExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2Exception::Mc2ExceptionImpl(args, userdata, "MegaMoe");
}

/*
 * 在动态库加载时校验版本并注册 MegaMoe 异常处理回调。
 */
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
