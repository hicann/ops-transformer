/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_kernel/arch35_92/common/mega_moe_gm_layout.h"
#include "../arch35_92/mega_moe_tiling_check.h"
#include "../arch35_92/mega_moe_tiling_print.h"

namespace optiling {
namespace {

/*
 * 建立 weight/scale 布局派生信息，并校验 tensor 的 dtype、format、自身 shape 及跨 tensor shape 关系。
 */
static ge::graphStatus CheckAndSetTensorMetadata(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                 MegaMoeExpertParams& expertParams, MegaMoeTilingData* tilingData,
                                                 const char* nodeName)
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
    OP_TILING_CHECK(CheckDispatchInputShapes(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "dispatch input shapes are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckWeightAndScaleShapeRelations(context, config, expertParams, tilingData->actMode, nodeName) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "weight or scale shape relation is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckYShape(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "y tensor shape is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验并记录不依赖权重元数据或问题规模的算子属性；dispatch_quant_mode 区分内部 MX 量化与预量化直通。
 * 直接属性：topo_type、ep_world_size、comm_alg、dispatch_quant_mode、combine_quant_mode、
 * activation、activation_params 和 topk_weights_type。
 * 具体属性的拦截项由各子函数保留，本函数只表达该阶段的执行顺序。
 */
static ge::graphStatus CheckAndSetIndependentAttrs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                   MegaMoeTilingData* tilingData, const char* nodeName)
{
    OP_TILING_CHECK(CheckAndSetTopoType(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "topology type is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetEpWorldSizeAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "EP world size is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDedupRequestAttrs(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "communication algorithm is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetQuantModeAttrs(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "quantization mode attributes are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckActivationParams(context, config, tilingData->compute.topoType, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "activation parameters are invalid."), return ge::GRAPH_FAILED);
    SetActivationAttrParams(context, config, tilingData);
    OP_TILING_CHECK(CheckAndSetTopkWeightsTypeAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "topK weights type is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 基于已校验的属性和 tensor 元数据，解析专家 GMM 模式、问题维度和专家数量。
 * 直接属性：dispatch_quant_out_dtype、shared_expert_quant_out_dtype 和 moe_expert_num；前两个属性在
 * mode 4 下表示内部量化输出类型，在 mode 0 下表示预量化输入逻辑类型。
 * GMM 模式、token 维度、hiddenDim 及 shared expert 数量由已校验的 tensor 元数据派生。
 */
static ge::graphStatus CheckAndSetExpertExecutionParams(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                        MegaMoeExpertParams& expertParams,
                                                        MegaMoeTilingData* tilingData, const char* nodeName)
{
    OP_TILING_CHECK(CheckAndSetQuantOutDtypes(context, config, expertParams, tilingData->compute.topoType, nodeName) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expert quantization settings are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetExpertGmmModes(context, expertParams, tilingData->compute.topoType, tilingData,
                                              nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "expert GMM modes are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetTokenDimensions(context, tilingData, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "token dimensions are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetHiddenDim(context, tilingData, expertParams, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "hiddenDim is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAndSetExpertCountAttrs(context, config, expertParams, tilingData, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "expert counts are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckExpertTokenNumsShape(context, config, tilingData->compute.moeExpertPerRank, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "expertTokenNums tensor shape is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验并设置 token 容量属性，CCL 容量在完成布局计算后统一检查。
 */
static ge::graphStatus CheckAndSetCapacityAttrs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                MegaMoeTilingData* tilingData, const char* nodeName)
{
    OP_TILING_CHECK(CheckAndSetMaxTokensPerRankAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "maximum tokens per rank is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetMaxRecvTokenNumAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "maximum received token count is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 去重 per-shape 门集合（请求位已由 CheckDedupRequestAttrs 落库）：prepass 可行性护栏 →
 * dispatch 收益门 → combine 容量护栏。combine 位由 combine_comm_mode 显式决定（开即强制走，撞硬约束报错）；
 * dispatch 位保持内部自动裁决。必须在 bs/h/topK/epWorldSize/numMaxTokensPerRank 全部写入
 * tilingData 之后、workspace 与自适应 buffer 规划之前调用（两者按 dedupMode 定布局）。
 */
static ge::graphStatus SetDedupModeParams(const gert::TilingContext* context, const MegaMoeConfig& config,
                                          MegaMoeTilingData* tilingData, const char* nodeName)
{
    (void)context;
    (void)config;
    OP_TILING_CHECK(ApplyDedupPrepassChunkGuard(tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "dedup prepass capacity check failed."), return ge::GRAPH_FAILED);
    ApplyDedupBenefitGate(tilingData, nodeName);
    OP_TILING_CHECK(ApplyDedupCombineCapacityGate(tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "combine dedup capacity check failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckArchitectureCapability(gert::TilingContext* context, const MegaMoeTilingData* tilingData,
                                                   const char* nodeName)
{
    auto supportedGmmMode = [](uint8_t mode) {
        return mode == GMM_MODE_A8W8_ND || mode == GMM_MODE_A8W8_NZ || mode == GMM_MODE_A4W4_ND ||
               mode == GMM_MODE_A4W4_NZ;
    };
    bool moeSupported = supportedGmmMode(tilingData->moeGmmMode);
    bool sharedSupported = tilingData->compute.sharedExpertNum == 0U || supportedGmmMode(tilingData->sharedGmmMode);
    OP_TILING_CHECK(tilingData->compute.topoType != TOPO_TYPE_MTE || !moeSupported || !sharedSupported,
                    OP_LOGE(nodeName, "arch92 only enables A8W8/A4W4 MTE"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 根据路由规模和可用 UB 计算 dispatch 阶段的分批大小、ring 深度及拷贝缓冲区大小。
 */
static MegaMoeDispatchBufferConfig CalcDispatchBufferConfig(const MegaMoeTilingData* tilingData,
                                                            uint32_t activationElementsPerByte,
                                                            uint32_t availableUbBytes)
{
    MegaMoeDispatchBufferConfig bufferConfig{};
    // 调用方传入的预算已扣除 UB 尾部常驻表。
    uint32_t fixedBufferBytes = 0U;
    // 去重模式：批内 desc/扇出 meta/活跃表暂存紧跟 dispatch 固定区尾动态落位，
    // 此处先从自适应预算里扣掉预留，保证暂存区在任意合法规格下不越 UB。
    if (tilingData->compute.dedupMode != 0 && tilingData->compute.topoType == TOPO_TYPE_MTE) {
        availableUbBytes = availableUbBytes > MegaMoeImpl::DEDUP_DISPATCH_UB_RESERVE_BYTES ?
                               availableUbBytes - MegaMoeImpl::DEDUP_DISPATCH_UB_RESERVE_BYTES :
                               0U;
    }
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->compute.numMaxTokensPerRank);
    uint64_t alignedTotalRouteItems = ops::CeilAlign(sendTotalNum, static_cast<uint64_t>(ALIGN_256));
    uint32_t copyBufferBytes = CalcDispatchCopyBufferBytes(tilingData, activationElementsPerByte);
    bufferConfig.copyBufferBytes = copyBufferBytes;

    // 一个 dispatch ring slot 包含 token/scale copy buffer 和一条 32B triple。
    uint32_t dispatchSlotBytes = copyBufferBytes + static_cast<uint32_t>(ALIGN_32);
    uint32_t topkIndexTypeBytes = static_cast<uint32_t>(
        CalcTopkIndexTypeBytes(tilingData->compute.numMaxTokensPerRank, tilingData->compute.topK));

    SelectDispatchRingAndRouteBatch(bufferConfig, sendTotalNum, alignedTotalRouteItems, fixedBufferBytes,
                                    dispatchSlotBytes, topkIndexTypeBytes, availableUbBytes);
    return bufferConfig;
}

static uint64_t CalcCombineSyncSlotCountPerExpert(const MegaMoeTilingData* tilingData)
{
    if (tilingData->compute.topoType != TOPO_TYPE_URMA || tilingData->compute.moeExpertPerRank == 0U) {
        return 0U;
    }

    // Arch92 为 1C1V，每个 Cube 对应的唯一 AIV 都是 Combine 消费核。
    uint64_t combineCoreCount = tilingData->compute.aicNum;
    uint64_t maxTokenCountForOneExpert =
        static_cast<uint64_t>(tilingData->compute.numMaxTokensPerRank) * tilingData->compute.epWorldSize;
    uint64_t maxTokenGroupCountForOneExpert =
        ops::CeilDiv(maxTokenCountForOneExpert, static_cast<uint64_t>(COMBINE_TOKEN_GROUP_SIZE));
    return std::max(maxTokenGroupCountForOneExpert, combineCoreCount);
}

/*
 * 汇总各执行路径在 workspace 中需要的 host flag 与同步计数器元素数量。
 */
static uint64_t CalcHostFlagElementCount(const MegaMoeTilingData* tilingData)
{
    uint64_t maxWavesPerExpert = ops::CeilDiv<uint64_t>(tilingData->compute.maxOutputSize, L1_TILE_M_256);
    uint64_t waveFlagSlotsPerExpert = maxWavesPerExpert * INT_CACHELINE;
    uint64_t activationFlagSlotsPerExpert = waveFlagSlotsPerExpert;
    uint64_t moeExpertCount = tilingData->compute.moeExpertPerRank;

    uint64_t flagElementCount = moeExpertCount * (activationFlagSlotsPerExpert + waveFlagSlotsPerExpert +
                                                  static_cast<uint64_t>(INT_CACHELINE) * tilingData->compute.aicNum);
    // 去重附加计数与 A6 的 Workspace 分配、Kernel 清零范围保持一致。
    if (tilingData->compute.topoType == TOPO_TYPE_MTE &&
        (tilingData->combineQuantMode != COMBINE_NO_QUANT || IsCombineDedupOn(tilingData->compute.dedupMode))) {
        flagElementCount += moeExpertCount * tilingData->compute.aicNum * INT_CACHELINE;
    }
    if (tilingData->compute.topoType == TOPO_TYPE_MTE && tilingData->compute.dedupMode != 0) {
        flagElementCount += static_cast<uint64_t>(INT_CACHELINE) *
                            (1U + static_cast<uint64_t>(tilingData->compute.epWorldSize) * DEDUP_PREPASS_MAX_CHUNKS);
    }
    if (tilingData->compute.topoType == TOPO_TYPE_URMA) {
        flagElementCount += tilingData->compute.combineSyncSlotCountPerExpert * moeExpertCount * INT_CACHELINE;
    }
    if (tilingData->compute.sharedExpertNum > 0 && tilingData->compute.topoType == TOPO_TYPE_MTE) {
        uint64_t tokenGroupCount = ops::CeilDiv<uint64_t>(tilingData->compute.bs, L1_TILE_M_256);
        flagElementCount += tokenGroupCount * tilingData->compute.sharedExpertNum * INT_CACHELINE;
        flagElementCount += static_cast<uint64_t>(CalcSharedActivationFlagElementsPerExpert(
                                static_cast<int64_t>(tilingData->compute.bs))) *
                            tilingData->compute.sharedExpertNum;
    }
    if (tilingData->compute.sharedExpertNum > 0 && tilingData->compute.topoType == TOPO_TYPE_MTE) {
        flagElementCount += static_cast<uint64_t>(tilingData->compute.aicNum) * INT_CACHELINE;
    }
    return flagElementCount;
}

/*
 * 计算 topK 有效下标发送阶段与路由 ring 深度无关的固定 UB 占用。
 */
static uint32_t CalcTopkValidIndexFixedBufferBytes(const MegaMoeTilingData* tilingData,
                                                   uint32_t moeActivationElementsPerByte,
                                                   uint32_t sharedActivationElementsPerByte,
                                                   bool isSharedQuantIndependent, uint32_t topkValidIndexCoreNum)
{
    uint64_t totalFlagElementCount = CalcHostFlagElementCount(tilingData);
    uint32_t resetElementCountPerCore =
        static_cast<uint32_t>(ops::CeilDiv(totalFlagElementCount, static_cast<uint64_t>(tilingData->blockAivNum)));
    uint32_t resetBatchElementCount = std::min(resetElementCountPerCore, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    uint32_t resetTensorBytes =
        ops::CeilAlign(resetBatchElementCount, static_cast<uint32_t>(INT32_PER_256B)) * sizeof(int32_t);

    uint32_t moeQuantOutputBufferBytes = CalcDispatchCopyBufferBytes(tilingData, moeActivationElementsPerByte);
    uint32_t quantOutputBufferBytes = moeQuantOutputBufferBytes;
    if (isSharedQuantIndependent) {
        // 独立共享量化增加一组输出双缓冲，输入和 mxTemp 共用；共享输出不包含 topK weight。
        uint32_t sharedQuantOutputBufferBytes =
            CalcQuantTokenAndScaleBytes(tilingData, sharedActivationElementsPerByte);
        quantOutputBufferBytes += sharedQuantOutputBufferBytes;
    }

    uint32_t quantInputBufferBytes =
        ops::CeilAlign(static_cast<uint32_t>(tilingData->compute.h), static_cast<uint32_t>(ALIGN_128)) *
        sizeof(uint16_t);
    // sendCntAccTensor_ 按每个 topK 有效下标计算发送核负责的最大专家数分配，与 kernel 地址布局一致。
    uint32_t maxExpertCountPerCore = ops::CeilDiv(static_cast<uint32_t>(tilingData->compute.epWorldSize) *
                                                      static_cast<uint32_t>(tilingData->compute.moeExpertPerRank),
                                                  topkValidIndexCoreNum);
    uint32_t sendCountAccumulatorBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(maxExpertCountPerCore) * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32)));
    if (tilingData->preQuantTileTokens != 0U) {
        // 预量化不申请动态量化的 mxTemp/xOut/xIn；两个 stage 按阶段复用。
        return resetTensorBytes + DOUBLE_BUFFER * tilingData->preQuantStageBytes + sendCountAccumulatorBytes;
    }

    // mxTempTensor_ 占 2KB，xOutTensor_ 和 xInTensor_ 各使用双 buffer。
    constexpr uint32_t mxTempTensorBytes = 2U * 1024U;
    return resetTensorBytes + mxTempTensorBytes + 2U * quantOutputBufferBytes + 2U * quantInputBufferBytes +
           sendCountAccumulatorBytes;
}

/*
 * 设置 topK 有效下标发送的两套 UB 配置：先算固定占用，再按 expert 分核的两类 core 各算一套。
 * 本函数读 tilingData->compute.combineSyncSlotCountPerExpert（经 CalcHostFlagElementCount），
 * 该字段必须在调用前写好。
 */
static void SetTopkValidIndexBufferConfigs(MegaMoeTilingData* tilingData, uint32_t moeActivationElementsPerByte,
                                           uint32_t sharedActivationElementsPerByte, bool isSharedQuantIndependent,
                                           uint32_t availableUbBytes)
{
    const bool sharedMte = tilingData->compute.sharedExpertNum > 0U && tilingData->compute.topoType == TOPO_TYPE_MTE;
    const uint32_t topkValidIndexCoreNum = sharedMte ? tilingData->compute.aicNum : tilingData->blockAivNum;
    uint32_t sendMaskFixedBufferBytes =
        CalcTopkValidIndexFixedBufferBytes(tilingData, moeActivationElementsPerByte, sharedActivationElementsPerByte,
                                           isSharedQuantIndependent, topkValidIndexCoreNum);

    /*
     * 按发送核数均衡分配专家，前 remainder 个发送核多处理一个专家。
     * 分别计算这两类发送核的 UB 配置。
     */
    uint32_t totalExpertCount = tilingData->compute.epWorldSize * tilingData->compute.moeExpertPerRank;
    uint32_t expertCountPerCoreWithoutExtraExpert = totalExpertCount / topkValidIndexCoreNum;
    tilingData->sendMaskCoreCountWithExtraExpert = totalExpertCount % topkValidIndexCoreNum;
    uint32_t expertCountPerCoreWithExtraExpert = expertCountPerCoreWithoutExtraExpert + 1U;
    tilingData->sendMaskConfigForCoreWithExtraExpert = CalcTopkValidIndexBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithExtraExpert, availableUbBytes);
    tilingData->sendMaskConfigForCoreWithoutExtraExpert = CalcTopkValidIndexBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithoutExtraExpert, availableUbBytes);
}

static void SetCclLayoutParams(MegaMoeTilingData* tilingData)
{
    auto& compute = tilingData->compute;
    const uint32_t rankNumPerServer =
        tilingData->rankNumPerServer == 0U || tilingData->rankNumPerServer > compute.epWorldSize ?
            static_cast<uint32_t>(compute.epWorldSize) :
            tilingData->rankNumPerServer;
    compute.serverNum = ops::CeilDiv(compute.epWorldSize, static_cast<int64_t>(rankNumPerServer));
    compute.activationOutputDivisor = GetActivationOutputDivisor(tilingData->actMode);
    compute.combineQuant = tilingData->combineQuantMode != COMBINE_NO_QUANT;
    compute.topkWeightsPrefetch = tilingData->isTopkWeightsPrefetch;
    compute.dispatchFp4 = IsA4W4GmmMode(tilingData->moeGmmMode);
    compute.reserveGmm1Output = true;
    compute.reserveCumsum = false;
    compute.sharedInputElementsPerByte = IsA4W4GmmMode(tilingData->sharedGmmMode) ? 2U : 1U;
    // Arch92 keeps its existing full-row workspace and execution policy.
    compute.ring = {};

    auto& comm = tilingData->comm;
    comm.numMaxTokensPerRank = compute.numMaxTokensPerRank;
    comm.topK = compute.topK;
    comm.h = compute.h;
    comm.moeExpertPerRank = compute.moeExpertPerRank;
    comm.epWorldSize = compute.epWorldSize;
    comm.yDtypeSize = SIZE_BF_16;
    comm.elemsPerByte = compute.dispatchFp4 ? 2U : 1U;
    comm.serverCombineReduce = IsServerCombineDedupOn(compute.dedupMode);
    comm.topkWeightsPrefetch =
        compute.topkWeightsPrefetch || IsCombineDedupOn(compute.dedupMode) || comm.serverCombineReduce;
    comm.isQuantCombine = compute.combineQuant;
    comm.topoType = compute.topoType;
    comm.serverNum = compute.serverNum;
}

/*
 * 按依赖顺序生成 dispatch、SendMask 和 unpermute 的自适应 UB 配置。
 */
static ge::graphStatus SetAdaptiveBufferConfigs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                const MegaMoeExpertParams& expertParams, MegaMoeTilingData* tilingData,
                                                uint32_t availableUbBytes, const char* nodeName)
{
    auto topKWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK(availableUbBytes < ARCH92_FIXED_WORK_UB_BYTES + ARCH92_DISPATCH_TABLE_BYTES,
                    OP_LOGE(nodeName,
                            "Arch92 UB is insufficient: available=%u bytes, fixed working area=%u bytes, "
                            "dispatch tables=%u bytes.",
                            availableUbBytes, ARCH92_FIXED_WORK_UB_BYTES, ARCH92_DISPATCH_TABLE_BYTES),
                    return ge::GRAPH_FAILED);
    const uint32_t dispatchUbBytes = (availableUbBytes - ARCH92_DISPATCH_TABLE_BYTES) /
                                     static_cast<uint32_t>(ALIGN_512) * static_cast<uint32_t>(ALIGN_512);
    tilingData->dispatchTableUbBaseAddr = dispatchUbBytes;

    uint32_t activationElementsPerByte = expertParams.moe.quantOutDtype == ge::DT_FLOAT4_E2M1 ? 2U : 1U;
    uint32_t sharedActivationElementsPerByte = expertParams.shared.quantOutDtype == ge::DT_FLOAT4_E2M1 ? 2U : 1U;
    // 所有 AIV 核共用同一套 dispatch UB 配置，该布局与分核方式无关。
    // 若 kernel 改为按核拆分 tensor 或调整 copyTmp 槽位布局，需同步更新此处计算。
    tilingData->dispatchBufferConfig = CalcDispatchBufferConfig(tilingData, activationElementsPerByte, dispatchUbBytes);

    // 先于 topK 有效下标配置写入：CalcHostFlagElementCount 会读这个字段累加 flag 区大小，
    // 该依赖只经 tilingData 传递，调换顺序会静默改变 flag 区尺寸。
    tilingData->compute.combineSyncSlotCountPerExpert = CalcCombineSyncSlotCountPerExpert(tilingData);
    SetCclLayoutParams(tilingData);

    const bool isPreQuantizedX = xDesc != nullptr && IsPreQuantizedXType(xDesc->GetDataType());
    SetPreQuantTileConfig(tilingData, isPreQuantizedX, activationElementsPerByte,
                          topKWeightsDesc == nullptr ? ge::DT_FLOAT : topKWeightsDesc->GetDataType());

    SetTopkValidIndexBufferConfigs(tilingData, activationElementsPerByte, sharedActivationElementsPerByte,
                                   tilingData->compute.sharedInputIndependent == 1U, availableUbBytes);

    ge::DataType topKWeightsDataType = topKWeightsDesc->GetDataType();
    SetUnpermuteBufferConfigs(tilingData, topKWeightsDataType, availableUbBytes,
                              UnpermuteBufferPolicy::SEPARATE_DEQUANTIZATION);
    return ge::GRAPH_SUCCESS;
}

/*
 * A8W4 的 L1 布局由 Host 根据平台报告的 L1 容量统一计算，
 * 供全部 AIC/AIV 消费核使用。
 * M=N=KTile=256：每个 A/W8-B 槽占 64 KiB，每 32 个 K 元素共享一个 scale 字节；
 * A/B scale 覆盖相同的 K 窗口，各自使用双 buffer。
 * 容量按完整 tile 规划，M/N/K 尾块仍复用同一槽位，只缩小搬运范围。
 *
 * 物理布局按地址递增排列：
 * [ A0 ... A(a-1) ][ B physical slots 0 ... b-1 ][ scA0 ][ scB0 ][ scA1 ][ scB1 ]
 * | a * 64 KiB   | b * 64 KiB                  | 2 * (scaleABytes+scaleBBytes) |
 * A 按顺序轮转；B 在两半区间交替，b=4 时物理槽次序为 0、2、1、3。
 *
 * 以 L1 为 512 KiB、max(GMM1 K, GMM2 K)=7168 为例，以下偏移单位均为 KiB：
 * [0] A0 [64] A1 [128] A2 [192] B0 [256] B2 [320] B1 [384] B3
 * [448] scA0 [464] scB0 [480] scA1 [496] scB1 [512]
 * a=3、b=4，每个 scale 槽占 16 KiB，公共 scale K 窗口为 2048。
 *
 * 1. 先预留一组覆盖完整 K 的 A/B scale，估算相等的 A/B 槽数，保留其中的 A 槽数。
 * 2. 改为只预留两组最小 scale（每组覆盖一个 KTile），剩余整槽分配给 B。
 * 3. 将剩余空间分给两组 scale，覆盖范围向下按 KTile 对齐。
 *    第 1 步的完整 K 预算并不表示全部 K scale 都常驻。
 * 4. 在此写出全部偏移，Device 只读取计算好的布局。
 */
static ge::graphStatus CalcMegaMoeL1Layout(uint64_t maxGmmK, uint64_t l1Bytes, MegaMoeL1Layout& layout,
                                           const char* nodeName)
{
    constexpr uint64_t TILE_M = 256;
    constexpr uint64_t TILE_N = 256;
    constexpr uint64_t TILE_K = 256;
    constexpr uint64_t SCALE_BUFFER_NUM = 2;
    constexpr uint64_t K_PER_SCALE_BYTE = 32;
    constexpr uint64_t A_BYTES = TILE_M * TILE_K;
    constexpr uint64_t B_BYTES = TILE_N * TILE_K;
    constexpr uint64_t MIN_SCALE_A_BYTES = TILE_M * (TILE_K / K_PER_SCALE_BYTE);
    constexpr uint64_t MIN_SCALE_B_BYTES = TILE_N * (TILE_K / K_PER_SCALE_BYTE);
    constexpr uint64_t MIN_SCALE_TOTAL_BYTES = SCALE_BUFFER_NUM * (MIN_SCALE_A_BYTES + MIN_SCALE_B_BYTES);
    constexpr uint64_t MIN_L1_BYTES = A_BYTES + B_BYTES + MIN_SCALE_TOTAL_BYTES;

    OP_TILING_CHECK(
        l1Bytes < MIN_L1_BYTES,
        OP_LOGE(nodeName, "platform L1 size %lu is smaller than the minimum required size %lu", l1Bytes, MIN_L1_BYTES),
        return ge::GRAPH_FAILED);

    // 第 1 步：scale 硬件按 K64 成对存储，每行对应两个 scale 字节。
    // M/N 和 tile 字节数已对齐，不需要额外计算 padding。
    const uint64_t fullKScaleBytesPerRow = (maxGmmK + 63) / 64 * 2;
    const uint64_t fullKScalePairBytes = (TILE_M + TILE_N) * fullKScaleBytesPerRow;
    layout = {};
    layout.aBufferNum = fullKScalePairBytes < l1Bytes ? (l1Bytes - fullKScalePairBytes) / (A_BYTES + B_BYTES) : 1;
    layout.aBufferNum = std::max(uint64_t{1}, std::min(layout.aBufferNum, MegaMoeL1Layout::MAX_A_BUFFER_NUM));

    // 第 2 步：固定 A 槽数，先预留两组最小 scale，再分配 B 槽。
    layout.bBufferNum = (l1Bytes - layout.aBufferNum * A_BYTES - MIN_SCALE_TOTAL_BYTES) / B_BYTES;
    layout.bBufferNum = std::min(layout.bBufferNum, MegaMoeL1Layout::MAX_B_BUFFER_NUM);

    // 第 3 步：A/B scale 覆盖相同数量的 K tile，各自使用两个槽位。
    const uint64_t scaleBudgetBytes = l1Bytes - layout.aBufferNum * A_BYTES - layout.bBufferNum * B_BYTES;
    const uint64_t requiredKTiles = std::max(uint64_t{1}, (maxGmmK + TILE_K - 1) / TILE_K);
    const uint64_t scaleKTiles = std::min(scaleBudgetBytes / MIN_SCALE_TOTAL_BYTES, requiredKTiles);
    layout.tileK = TILE_K;
    layout.scaleK = scaleKTiles * TILE_K;
    layout.aSlotBytes = A_BYTES;
    layout.bSlotBytes = B_BYTES;
    layout.scaleASlotBytes = scaleKTiles * MIN_SCALE_A_BYTES;
    layout.scaleBSlotBytes = scaleKTiles * MIN_SCALE_B_BYTES;

    // 第 4 步：顺序排布 A、交错排布 B，再放两组 scale，保证各区域不重叠。
    for (uint64_t slot = 0; slot < layout.aBufferNum; ++slot) {
        layout.aOffsets[slot] = slot * A_BYTES;
    }
    const uint64_t bBase = layout.aBufferNum * A_BYTES;
    for (uint64_t slot = 0; slot < layout.bBufferNum; ++slot) {
        const uint64_t physicalSlot = (slot & 1) * ((layout.bBufferNum + 1) / 2) + slot / 2;
        layout.bOffsets[slot] = bBase + physicalSlot * B_BYTES;
    }
    const uint64_t scaleBase = bBase + layout.bBufferNum * B_BYTES;
    for (uint64_t slot = 0; slot < SCALE_BUFFER_NUM; ++slot) {
        layout.scaleAOffsets[slot] = scaleBase + slot * (layout.scaleASlotBytes + layout.scaleBSlotBytes);
        layout.scaleBOffsets[slot] = layout.scaleAOffsets[slot] + layout.scaleASlotBytes;
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * 在所有校验和资源规划完成后，统一提交 workspace、block dim、tiling key 及诊断信息。
 */
static ge::graphStatus CommitTilingResult(gert::TilingContext* context, const MegaMoeConfig& config,
                                          const MegaMoeExpertParams& expertParams, MegaMoeTilingData* tilingData,
                                          const char* nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t l1Size = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    OP_LOGD(nodeName, "platform L1 size is %lu bytes", l1Size);

    // 共享专家与路由专家使用相同且已校验的 H/hiddenDim 维度。
    OP_TILING_CHECK(CalcMegaMoeL1Layout(std::max(static_cast<uint64_t>(tilingData->compute.h),
                                                 static_cast<uint64_t>(GetActivationOutputWidth(
                                                     tilingData->compute.hiddenDim, tilingData->actMode))),
                                        l1Size, tilingData->a8w4L1Layout, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "calculate A8W4 L1 layout failed"), return ge::GRAPH_FAILED);

    WorkspaceLayout workspaceLayout(tilingData->compute,
                                    WorkspaceLayout::Arch92{tilingData->moeGmmMode, tilingData->sharedGmmMode});
    // Preserve Arch92's existing FP8-sized admission bound for packed FP4 records.
    auto capacityComm = tilingData->comm;
    capacityComm.elemsPerByte = 1U;
    OP_TILING_CHECK(CheckCclBufferCapacity(context, config, tilingData, workspaceLayout, nodeName, &capacityComm) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CCL buffer capacity is insufficient."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    context->SetBlockDim(
        ascendcPlatform.CalcTschBlockDim(tilingData->blockAivNum, tilingData->compute.aicNum, tilingData->blockAivNum));
    context->SetScheduleMode(1);
    uint64_t tilingKey = CalcTilingKey(context, expertParams, tilingData);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    PrintMegaMoeTilingData(tilingData, nodeName);
    PrintWorkspaceLayout(&workspaceLayout, nodeName);
    PrintPeermemInfo(tilingData, nodeName);
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus MegaMoeTilingFuncImplArch92(gert::TilingContext* context, MegaMoeConfig& config)
{
    const char* nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MegaMoe tiling check func.");

    MegaMoeTilingData* tilingData = context->GetTilingData<MegaMoeTilingData>();
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
    // scales 的 descriptor/shape 判空必须先于后续 dtype、format 和 shape 校验。
    OP_TILING_CHECK(CheckScalesInput(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "scales input requirements are not satisfied."), return ge::GRAPH_FAILED);

    // Tensor 契约：不读取 attr；先确定 weight/scale 组织形式，再校验 dtype、format 和 shape。
    MegaMoeExpertParams expertParams = MakeExpertParams(config);
    OP_TILING_CHECK(CheckAndSetTensorMetadata(context, config, expertParams, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "tensor metadata is invalid."), return ge::GRAPH_FAILED);

    // 专家执行属性：dispatch_quant_out_dtype、shared_expert_quant_out_dtype 和 expert_num。
    OP_TILING_CHECK(
        CheckAndSetExpertExecutionParams(context, config, expertParams, tilingData, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "expert execution parameters are invalid."), return ge::GRAPH_FAILED);

    // 容量属性：num_max_tokens_per_rank、max_recv_token_num 和 ccl_buffer_size。
    OP_TILING_CHECK(CheckAndSetCapacityAttrs(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "capacity attributes are invalid."), return ge::GRAPH_FAILED);

    // 通信去重：请求位落库 + per-shape 收益/容量门。prepass 容量护栏要读 numMaxTokensPerRank，
    // 故必须排在容量属性落库之后；CCL 窗口下界在收益门确定最终布局后于 CommitTilingResult 中检查。
    OP_TILING_CHECK(SetDedupModeParams(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "dedup mode params are invalid."), return ge::GRAPH_FAILED);

    // 平台与资源规划。
    uint32_t aicNum = 0U;
    uint64_t ubSize = 0U;
    OP_TILING_CHECK(CheckAndSetPlatformParams(context, tilingData, aicNum, ubSize, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "platform parameters are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckArchitectureCapability(context, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "architecture capability is invalid."), return ge::GRAPH_FAILED);

    SetWaveParams(tilingData, aicNum);
    OP_TILING_CHECK(SetAdaptiveBufferConfigs(context, config, expertParams, tilingData, static_cast<uint32_t>(ubSize),
                                             nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "adaptive UB buffer configuration failed."), return ge::GRAPH_FAILED);

    return CommitTilingResult(context, config, expertParams, tilingData, nodeName);
}

static ge::graphStatus MegaMoeTilingFuncArch92(gert::TilingContext* context)
{
    MegaMoeConfig config;
    return MegaMoeTilingFuncImplArch92(context, config);
}

struct MegaMoeCompileInfoArch92 {};

static ge::graphStatus TilingParseForMegaMoeArch92(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

namespace MegaMoeArch92Tiling {
IMPL_OP_OPTILING(MegaMoe)
    .Tiling(MegaMoeTilingFuncArch92)
    .TilingParse<MegaMoeCompileInfoArch92>(TilingParseForMegaMoeArch92);
} // namespace MegaMoeArch92Tiling

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
__attribute__((constructor)) static void RegisterMegaMoeArch92ExceptionFunc()
{
    RegisterMegaMoeException();
}
#endif

} // namespace optiling
