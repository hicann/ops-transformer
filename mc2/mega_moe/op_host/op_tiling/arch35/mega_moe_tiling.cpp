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

// 校验 URMA 的 Combine 通信模式并设置 dedupMode。
static ge::graphStatus CheckAndSetUrmaDedupMode(int64_t combineCommMode, MegaMoeTilingData* tilingData,
                                                const char* nodeName)
{
    OP_TILING_CHECK(combineCommMode != COMBINE_COMM_MODE_NO_DEDUP && combineCommMode != COMBINE_COMM_MODE_SERVER_DEDUP,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "combine_comm_mode", std::to_string(combineCommMode).c_str(),
                                              "0 or 2 on Layered topology (topo_type=1 or 2)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(combineCommMode == COMBINE_COMM_MODE_SERVER_DEDUP &&
                        tilingData->combineQuantMode != COMBINE_QUANT_OUT_TYPE_NO_QUANT,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "combine_comm_mode", "2",
                        "enabling server combine reduction does not support quantized combine; it requires "
                        "combine_quant_mode=0"),
                    return ge::GRAPH_FAILED);
    tilingData->compute.dedupMode =
        combineCommMode == COMBINE_COMM_MODE_SERVER_DEDUP ? DEDUP_MODE_SERVER_COMBINE : DEDUP_MODE_OFF;
    return ge::GRAPH_SUCCESS;
}

// 校验 MTE 的 Combine 去重配置，并请求后续按形状判断 Dispatch 去重。
static ge::graphStatus CheckAndSetMteDedupMode(const gert::TilingContext* context, const MegaMoeConfig& config,
                                               int64_t combineCommMode, MegaMoeTilingData* tilingData,
                                               const char* nodeName)
{
    OP_TILING_CHECK(combineCommMode != COMBINE_COMM_MODE_NO_DEDUP && combineCommMode != COMBINE_COMM_MODE_RANK_DEDUP,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "combine_comm_mode", std::to_string(combineCommMode).c_str(),
                                              "0 or 1 on MTE topology (topo_type=0)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        combineCommMode == COMBINE_COMM_MODE_RANK_DEDUP &&
            tilingData->combineQuantMode != COMBINE_QUANT_OUT_TYPE_NO_QUANT,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "combine_comm_mode", "1",
                                              "enabling combine dedup does not support quantized combine; it requires "
                                              "combine_quant_mode=0"),
        return ge::GRAPH_FAILED);
    const bool isPreQuantizedX = IsPreQuantizedXType(context->GetInputDesc(config.xIndex)->GetDataType());
    OP_TILING_CHECK(
        combineCommMode == COMBINE_COMM_MODE_RANK_DEDUP && !tilingData->isTopkWeightsPrefetch && isPreQuantizedX,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "combine_comm_mode", "1",
            "enabling combine dedup does not support pre-quantized x together with topk_weights_type=0; "
            "set topk_weights_type=1, or pass a non-pre-quantized x"),
        return ge::GRAPH_FAILED);
    tilingData->compute.dedupMode =
        combineCommMode == COMBINE_COMM_MODE_RANK_DEDUP ? DEDUP_MODE_BOTH : DEDUP_MODE_DISPATCH_ONLY;
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 combine_comm_mode 是否受当前配置支持，并转换为内部 dedupMode。
 * 调用前需完成拓扑、量化模式和权重模式的校验。
 */
static ge::graphStatus CheckAndSetDedupMode(const gert::TilingContext* context, const MegaMoeConfig& config,
                                            MegaMoeTilingData* tilingData, const char* nodeName)
{
    auto attrs = context->GetAttrs();
    auto ccmPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCombineCommModeIndex));
    const int64_t combineCommMode = ccmPtr == nullptr ? COMBINE_COMM_MODE_NO_DEDUP : *ccmPtr;
    OP_TILING_CHECK(
        combineCommMode != COMBINE_COMM_MODE_NO_DEDUP && combineCommMode != COMBINE_COMM_MODE_RANK_DEDUP &&
            combineCommMode != COMBINE_COMM_MODE_SERVER_DEDUP,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "combine_comm_mode", std::to_string(combineCommMode).c_str(), "0, 1 or 2"),
        return ge::GRAPH_FAILED);
    if (IsLayeredTopology(tilingData->compute.topoType)) {
        return CheckAndSetUrmaDedupMode(combineCommMode, tilingData, nodeName);
    }
    if (tilingData->compute.topoType == TOPO_TYPE_MTE) {
        return CheckAndSetMteDedupMode(context, config, combineCommMode, tilingData, nodeName);
    }
    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topo_type", std::to_string(tilingData->compute.topoType).c_str(),
                              "MTE(0), URMA(1) or UBX(2)");
    return ge::GRAPH_FAILED;
}

/*
 * 校验并记录不依赖权重元数据或问题规模的算子属性；dispatch_quant_mode 区分内部 MX 量化与预量化直通。
 * 直接属性：topo_type、ep_world_size、dispatch_quant_mode、combine_quant_mode、activation、
 * activation_params 和 topk_weights_type；基础属性落库后校验 combine_comm_mode 的组合约束。
 * 具体属性的拦截项由各子函数保留，本函数只表达该阶段的执行顺序。
 */
static ge::graphStatus CheckAndSetIndependentAttrs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                   MegaMoeTilingData* tilingData, const char* nodeName)
{
    OP_TILING_CHECK(CheckAndSetTopoType(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "topology type is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetEpWorldSizeAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "EP world size is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetQuantModeAttrs(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "quantization mode attributes are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckActivationParams(context, config, tilingData->compute.topoType, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "activation parameters are invalid."), return ge::GRAPH_FAILED);
    SetActivationAttrParams(context, config, tilingData);
    OP_TILING_CHECK(CheckAndSetTopkWeightsTypeAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "topK weights type is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetDedupMode(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "communication dedup mode request is invalid."), return ge::GRAPH_FAILED);
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
 * 校验每张源卡的输入 token 数上限和本卡的专家路由行数上限，写入 numMaxTokensPerRank 和 maxOutputSize。
 * 这些上限用于后续去重判断和布局规划；CCL 字节容量在 CommitTilingResult 中按最终布局检查。
 */
static ge::graphStatus CheckAndSetTokenCapacityAttrs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                                     MegaMoeTilingData* tilingData, const char* nodeName)
{
    OP_TILING_CHECK(CheckAndSetMaxTokensPerRankAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "maximum tokens per rank is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetMaxRecvTokenNumAttr(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "maximum received token count is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 检查 MTE 去重的容量限制，并在收益不足时关闭 Dispatch 去重。
 * 调用前需完成形状和容量属性校验，后续 workspace 与 buffer 布局使用最终 dedupMode。
 */
static ge::graphStatus CheckMteDedupCapacityAndBenefit(MegaMoeTilingData* tilingData, const char* nodeName)
{
    OP_TILING_CHECK(tilingData->compute.topoType != TOPO_TYPE_MTE || tilingData->compute.dedupMode < 0 ||
                        tilingData->compute.dedupMode > 3,
                    OP_LOGE(nodeName, "MTE dedupMode only supports 0, 1, 2 or 3"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(ApplyDedupPrepassChunkGuard(tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "dedup prepass capacity check failed."), return ge::GRAPH_FAILED);
    ApplyDedupBenefitGate(tilingData, nodeName);
    OP_TILING_CHECK(ApplyDedupCombineCapacityGate(tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "combine dedup capacity check failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/*
 * 计算 dispatch 阶段不随 ring 深度与 route batch 变化的固定 UB 占用。
 */
static uint32_t CalcDispatchFixedBufferBytes(const MegaMoeTilingData* tilingData)
{
    // fixedBufferBytes 包含 cumsumInfoTensor_ 和 expertTokenNumsOutTensor_。
    uint32_t fixedBufferBytes =
        static_cast<uint32_t>(ops::CeilAlign(static_cast<uint64_t>(tilingData->compute.epWorldSize) *
                                                 tilingData->compute.moeExpertPerRank * sizeof(int32_t),
                                             static_cast<uint64_t>(ALIGN_32))) +
        static_cast<uint32_t>(
            ops::CeilAlign(static_cast<uint64_t>(tilingData->compute.moeExpertPerRank) * sizeof(int32_t),
                           static_cast<uint64_t>(ALIGN_32)));
    return fixedBufferBytes;
}

/*
 * 根据路由规模和可用 UB 计算 dispatch 阶段的分批大小、ring 深度及拷贝缓冲区大小。
 */
static MegaMoeDispatchBufferConfig CalcDispatchBufferConfig(const MegaMoeTilingData* tilingData,
                                                            uint32_t activationElementsPerByte,
                                                            uint32_t availableUbBytes)
{
    MegaMoeDispatchBufferConfig bufferConfig{};
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

    uint32_t fixedBufferBytes = CalcDispatchFixedBufferBytes(tilingData);
    // 一个 dispatch ring slot 包含 token/scale copy buffer 和一条 32B triple。
    uint32_t dispatchSlotBytes = copyBufferBytes + static_cast<uint32_t>(ALIGN_32);
    uint32_t topkIndexTypeBytes = static_cast<uint32_t>(
        CalcTopkIndexTypeBytes(tilingData->compute.numMaxTokensPerRank, tilingData->compute.topK));

    SelectDispatchRingAndRouteBatch(bufferConfig, sendTotalNum, alignedTotalRouteItems, fixedBufferBytes,
                                    dispatchSlotBytes, topkIndexTypeBytes, availableUbBytes);
    return bufferConfig;
}

/*
 * 计算每个本地 MoE 专家所需的 combine 同步槽位数；不使用分组同步时返回 0。
 */
static uint64_t CalcCombineSyncSlotCountPerExpert(const MegaMoeTilingData* tilingData)
{
    // MTE 统一使用 per-expert AIC ready 表；group counter 服务 URMA Layered Combine。
    if (!IsLayeredTopology(tilingData->compute.topoType) || tilingData->compute.moeExpertPerRank == 0U) {
        return 0U;
    }

    // Layered Combine 仅由 subBlockIdx=1 的半数 AIV 执行。
    uint64_t combineCoreCount = tilingData->blockAivNum / 2U;
    // 同一 token 的 topK expert id 不重复，因此单 expert 从每张卡最多接收 bs 个 token。
    uint64_t maxTokenCountForOneExpert =
        static_cast<uint64_t>(tilingData->compute.numMaxTokensPerRank) * tilingData->compute.epWorldSize;
    uint64_t maxTokenGroupCountForOneExpert =
        ops::CeilDiv(maxTokenCountForOneExpert, static_cast<uint64_t>(COMBINE_TOKEN_GROUP_SIZE));
    // Workspace 在路由结果产生前分配，因此每个本卡 MoE expert 都按独立最坏情况预留 slot。
    return std::max(maxTokenGroupCountForOneExpert, combineCoreCount);
}

/*
 * 计算 topK 有效下标发送阶段与路由 ring 深度无关的固定 UB 占用。
 */
static uint32_t CalcTopkValidIndexFixedBufferBytes(const MegaMoeTilingData* tilingData,
                                                   uint32_t moeActivationElementsPerByte,
                                                   uint32_t sharedActivationElementsPerByte,
                                                   bool isSharedQuantIndependent, uint32_t topkValidIndexCoreNum)
{
    const WorkspaceLayout workspaceLayout(tilingData->compute);
    uint64_t totalFlagElementCount = static_cast<uint64_t>((workspaceLayout.reset.flags.sizeBytes / SIZE_INT_32));
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
    uint32_t maxExpertCountPerCore =
        ops::CeilDiv(static_cast<uint32_t>(tilingData->compute.epWorldSize * tilingData->compute.moeExpertPerRank),
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
 * 本函数通过 WorkspaceLayout 读取 tilingData->compute.combineSyncSlotCountPerExpert，
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
    compute.reserveGmm1Output = tilingData->moeGmmMode == GMM_MODE_A8W4_NZ || compute.topkWeightsPrefetch;
    compute.reserveCumsum = IsW4GmmMode(tilingData->moeGmmMode);
    compute.sharedInputElementsPerByte = IsA4W4GmmMode(tilingData->sharedGmmMode) ? 2U : 1U;
    compute.ring = CalcCclStageRingConfig(compute);

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
static void SetAdaptiveBufferConfigs(const gert::TilingContext* context, const MegaMoeConfig& config,
                                     const MegaMoeExpertParams& expertParams, MegaMoeTilingData* tilingData,
                                     uint32_t availableUbBytes)
{
    tilingData->dispatchTableUbBaseAddr = 0U;
    auto topKWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);

    uint32_t activationElementsPerByte = expertParams.moe.quantOutDtype == ge::DT_FLOAT4_E2M1 ? 2U : 1U;
    uint32_t sharedActivationElementsPerByte = expertParams.shared.quantOutDtype == ge::DT_FLOAT4_E2M1 ? 2U : 1U;
    // 所有 AIV 核共用同一套 dispatch UB 配置，该布局与分核方式无关。
    // 若 kernel 改为按核拆分 tensor 或调整 copyTmp 槽位布局，需同步更新此处计算。
    tilingData->dispatchBufferConfig =
        CalcDispatchBufferConfig(tilingData, activationElementsPerByte, availableUbBytes);

    // 先于 topK 有效下标配置写入：WorkspaceLayout 会读取本字段计算 flag 区大小，
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
                              UnpermuteBufferPolicy::FUSED_ACCUMULATION);
}

/*
 * A8W4 L1 allocation (512 KiB), computed once on host for all AIC/AIV consumers.
 * M=N=KTile=256: each A/W8-B slot is 64 KiB. Every 32 K elements share one scale
 * byte; A/B scales cover the SAME K window, with two buffers for each matrix.
 * Full tiles determine capacity; M/N/K tails use the same slots and smaller copies.
 *
 * Physical layout, increasing addresses:
 * [ A0 ... A(a-1) ][ B physical slots 0 ... b-1 ][ scA0 ][ scB0 ][ scA1 ][ scB1 ]
 * | a * 64 KiB   | b * 64 KiB                  | 2 * (scaleABytes+scaleBBytes) |
 * A rotates linearly. B alternates between two halves: b=4 -> physical 0,2,1,3.
 *
 * Example max(GMM1 K, GMM2 K)=7168, all offsets below in KiB:
 * [0] A0 [64] A1 [128] A2 [192] B0 [256] B2 [320] B1 [384] B3
 * [448] scA0 [464] scB0 [480] scA1 [496] scB1 [512]
 * a=3, b=4, each scale slot=16 KiB, common scale K window=2048.
 *
 * 1. Reserve one full-K scale pair to estimate equal A/B counts, then retain A count.
 * 2. Reserve only two minimal scale pairs (one KTile each); give remaining slots to B.
 * 3. Allocate remaining space to two scale pairs, rounding coverage DOWN to KTile.
 *    The full-K budget in step 1 does not mean all K scales stay resident.
 * 4. Write every offset here; the device only reads the resulting layout.
 */
static MegaMoeL1Layout CalcMegaMoeL1Layout(uint64_t maxGmmK)
{
    constexpr uint64_t L1_BYTES = 512 * 1024;
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

    // Step 1: scale hardware stores K64 pairs (two scale bytes per row).
    // M/N and tile byte sizes are already aligned; no padding calculations are needed.
    const uint64_t fullKScaleBytesPerRow = (maxGmmK + 63) / 64 * 2;
    const uint64_t fullKScalePairBytes = (TILE_M + TILE_N) * fullKScaleBytesPerRow;
    MegaMoeL1Layout layout{};
    layout.aBufferNum = fullKScalePairBytes < L1_BYTES ? (L1_BYTES - fullKScalePairBytes) / (A_BYTES + B_BYTES) : 1;
    layout.aBufferNum = std::max(uint64_t{1}, std::min(layout.aBufferNum, MegaMoeL1Layout::MAX_A_BUFFER_NUM));

    // Step 2: A count is fixed; reserve two minimal scale pairs before assigning B.
    layout.bBufferNum = (L1_BYTES - layout.aBufferNum * A_BYTES - MIN_SCALE_TOTAL_BYTES) / B_BYTES;
    layout.bBufferNum = std::min(layout.bBufferNum, MegaMoeL1Layout::MAX_B_BUFFER_NUM);

    // Step 3: both scale matrices cover the same number of K tiles; each has two slots.
    const uint64_t scaleBudgetBytes = L1_BYTES - layout.aBufferNum * A_BYTES - layout.bBufferNum * B_BYTES;
    const uint64_t requiredKTiles = std::max(uint64_t{1}, (maxGmmK + TILE_K - 1) / TILE_K);
    const uint64_t scaleKTiles = std::min(scaleBudgetBytes / MIN_SCALE_TOTAL_BYTES, requiredKTiles);
    layout.tileK = TILE_K;
    layout.scaleK = scaleKTiles * TILE_K;
    layout.aSlotBytes = A_BYTES;
    layout.bSlotBytes = B_BYTES;
    layout.scaleASlotBytes = scaleKTiles * MIN_SCALE_A_BYTES;
    layout.scaleBSlotBytes = scaleKTiles * MIN_SCALE_B_BYTES;

    // Step 4: pack A, interleaved B and the two scale pairs without overlap.
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
    return layout;
}

/*
 * 在所有校验和资源规划完成后，统一提交 workspace、block dim、tiling key 及诊断信息。
 */
static ge::graphStatus CommitTilingResult(gert::TilingContext* context, const MegaMoeConfig& config,
                                          const MegaMoeExpertParams& expertParams, MegaMoeTilingData* tilingData,
                                          const char* nodeName)
{
    // 共享专家与路由专家的 H、hiddenDim 已校验一致。
    tilingData->a8w4L1Layout = CalcMegaMoeL1Layout(
        std::max(static_cast<uint64_t>(tilingData->compute.h),
                 static_cast<uint64_t>(GetActivationOutputWidth(tilingData->compute.hiddenDim, tilingData->actMode))));

    WorkspaceLayout workspaceLayout(tilingData->compute);
    OP_TILING_CHECK(CheckCclBufferCapacity(context, config, tilingData, workspaceLayout, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CCL buffer capacity is insufficient."), return ge::GRAPH_FAILED);
    tilingData->cclBufferSize =
        static_cast<uint64_t>(*context->GetAttrs()->GetAttrPointer<int64_t>(config.attrCclBufferSizeIndex));
    OP_TILING_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
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

ge::graphStatus MegaMoeTilingFuncImplArch35(gert::TilingContext* context, MegaMoeConfig& config)
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

    // 平台参数：后续容量检查和资源规划共用已校验的核数与 UB 容量。
    uint32_t aicNum = 0U;
    uint64_t ubSize = 0U;
    OP_TILING_CHECK(CheckAndSetPlatformParams(context, tilingData, aicNum, ubSize, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "platform parameters are invalid."), return ge::GRAPH_FAILED);

    // Token 数量上限：num_max_tokens_per_rank 和 max_recv_token_num。
    OP_TILING_CHECK(CheckAndSetTokenCapacityAttrs(context, config, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "token capacity attributes are invalid."), return ge::GRAPH_FAILED);

    // MTE 去重容量与收益检查依赖 token 数量上限；最终模式用于后续布局和 CCL 容量检查。
    if (tilingData->compute.topoType == TOPO_TYPE_MTE) {
        OP_TILING_CHECK(CheckMteDedupCapacityAndBenefit(tilingData, nodeName) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "MTE dedup capacity or benefit check failed."), return ge::GRAPH_FAILED);
    }

    // 按最终模式规划 Wave 和 buffer。
    SetWaveParams(tilingData, aicNum);
    SetAdaptiveBufferConfigs(context, config, expertParams, tilingData, static_cast<uint32_t>(ubSize));

    return CommitTilingResult(context, config, expertParams, tilingData, nodeName);
}

static ge::graphStatus MegaMoeTilingFuncArch35(gert::TilingContext* context)
{
    MegaMoeConfig config;
    return MegaMoeTilingFuncImplArch35(context, config);
}

struct MegaMoeCompileInfoArch35 {};

static ge::graphStatus TilingParseForMegaMoeArch35(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaMoe)
    .Tiling(MegaMoeTilingFuncArch35)
    .TilingParse<MegaMoeCompileInfoArch35>(TilingParseForMegaMoeArch35);

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
__attribute__((constructor)) static void RegisterMegaMoeArch35ExceptionFunc()
{
    RegisterMegaMoeException();
}
#endif

} // namespace optiling
