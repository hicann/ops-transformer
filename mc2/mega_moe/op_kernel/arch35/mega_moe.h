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
 * \file mega_moe.h
 * \brief
 */

#ifndef MEGA_MOE_H
#define MEGA_MOE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../../common/mc2_kernel_utils.h")
#include "../../common/mc2_kernel_utils.h"
#else
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "common/mega_moe_types.h"
#include "common/mega_moe_workspace.h"
#include "common/mega_moe_utils.h"
#include "blaze/epilogue/block_epilogue_activation_mx_quant.h"
#include "stage/mega_moe_token_quant.h"
#include "stage/mega_moe_send_mask.h"
#include "stage/mega_moe_workspace_reset.h"
#include "stage/mega_moe_shared_expert_input.h"
#include "stage/mega_moe_token_dispatch.h"
#include "stage/mega_moe_gmm1_activation.h"
#include "stage/mega_moe_gmm2_combine.h"
#include "stage/mega_moe_unpermute.h"
#if __has_include("../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaMoeTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    template <int32_t QM>
    struct QuantTraits {
        using OutType = fp8_e4m3fn_t;
    };
    template <>
    struct QuantTraits<E5M2_QUANT> {
        using OutType = fp8_e5m2_t;
    };
    template <>
    struct QuantTraits<E2M1_QUANT> {
        using OutType = fp4x2_e2m1_t;
    };
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType =
        typename std::conditional<Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;
    using UnpermuteBufferConfig = MegaMoeUnpermuteBufferConfig;

    __aicore__ inline void InitInputPrepareConfigs(const MoeStageCommonConfig &commonConfig);
    __aicore__ inline void InitTokenDispatchConfig(const MoeStageCommonConfig &commonConfig,
                                                   int32_t dispatchFlagSlotsPerExpert);
    __aicore__ inline void InitGmm1Config(const MoeStageCommonConfig &commonConfig,
                                          int32_t dispatchFlagSlotsPerExpert);
    __aicore__ inline void InitGmm2CombineConfigs(const MoeStageCommonConfig &commonConfig,
                                                  int32_t dispatchFlagSlotsPerExpert);
    __aicore__ inline void InitTokenUnpermuteConfig(const MoeStageCommonConfig &commonConfig);

protected:
    __aicore__ inline SendMaskBufferConfig SendAndQuantBuffInit();
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline UnpermuteBufferConfig InitTokenUnpermuteBuffers();
    __aicore__ inline void ProcessSharedExpertGmm1(int32_t &gmTileSequence);
    __aicore__ inline void ProcessMoeExpertStages(int32_t &gmTileSequence);
    __aicore__ inline void ProcessSharedExpertGmm2(int32_t &gmTileSequence);
    __aicore__ inline void CrossRankSyncInWorldSize();

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs moeWeightTensorListAddrs_{};
    ExpertWeightTensorListAddrs sharedWeightTensorListAddrs_{};
    DispatchPrepareConfig dispatchPrepareConfig_;
    TokenDispatchConfig tokenDispatchConfig_;
    Gmm1ActivationConfig gmm1Config_;
    SendMaskConfig sendMaskConfig_;
    ResetWorkspaceConfig resetWorkspaceConfig_;
    QuantProcessConfig quantProcessConfig_;
    SharedExpertPrepareConfig sharedExpertPrepareConfig_;
    Gmm2Config gmm2Config_;
    QuantCombineConfig quantCombineConfig_;
    QuantCombineBufferConfig quantCombineBufferConfig_;
    TokenUnpermuteConfig tokenUnpermuteConfig_;
    // A8W4 路径下 RunGmm1A8W4 会覆盖 V1 UB，导致 UB 上跨 expert 的状态
    // 无法保持。tokenDispatchScratch_ 中的 cumsumInfoGlobalTensor 作为 GM 持久备份：
    // ComputeExpertTokenCountAndNotify 中 Load → 计算 → Store；Dispatch/Export 从 GM 恢复。

    uint32_t k_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint16_t gmm2PingPongIdx_ = 0;
    // 主线 shared-expert 特性成员
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;

    static constexpr uint32_t A_ELEMS_PER_BYTE = PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = PackedElementTraits<Weight1Type>::ELEMENTS_PER_BYTE;
    // ENABLE_A8W4: A8W8 路径（fp8 act + fp4 w1），GMM1 使用 A8W4 prologue（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4: A4W4 路径（fp4 act + fp4 weight），GMM2 复用 A8W4 prologue。
    //             a4w4 场景下 GMM1 走 generic a4w4、GMM2 走 a8w4，避免两段都用 a4w4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M =
        TopkWeightsPrefetch ? L1_TILE_M_128 : L1_TILE_M_256;
    QuantProcessScratch<ActivationType> quantScratch_;
    SharedExpertPrepareScratch<ActivationType> sharedExpertPrepareScratch_;
    SendMaskScratch sendMaskScratch_;
    LocalTensor<int32_t> resetTensor_;

    // GMM2 走 A8W4 且 QuantMode 为 a4w4（E2M1）时，ActivationQuant 输出需提升为 fp8_e4m3fn_t。
    // 同时当 Weight2 非 fp4 但 QuantMode==E2M1 时（generic GMM2 路径），也需 promotion，
    // 否则会出现 A=QuantOutType(fp4) vs B=Weight1Type(fp8) 的类型不匹配。
    using ActivationQuantOutType =
        typename std::conditional<(QuantMode == E2M1_QUANT), fp8_e4m3fn_t, QuantOutType>::type;

    // ActivationQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = PackedElementTraits<ActivationQuantOutType>::ELEMENTS_PER_BYTE;

    using BlockEpilogue =
        BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                       EPILOGUE_TILE_M, L1_TILE_N, TopkWeightsPrefetch>;
    using SharedBlockEpilogue =
        BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                       L1_TILE_M_256, L1_TILE_N, false>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
    TokenDispatchScratch<ActivationType> tokenDispatchScratch_;
    Gmm1ActivationScratch gmm1ActivationScratch_;
    Gmm2Scratch gmm2Scratch_;
    TokenUnpermuteScratch tokenUnpermuteScratch_;
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitInputPrepareConfigs(
    const MoeStageCommonConfig &commonConfig)
{
    dispatchPrepareConfig_ = {
        .job = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_},
        .common = commonConfig};

    uint32_t quantScaleNumAlignPerToken = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenAlignBytes =
        Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256)) *
        sizeof(ActivationType);
    uint32_t quantScaleAlignBytes =
        Ops::Base::CeilAlign(quantScaleNumAlignPerToken * static_cast<uint32_t>(sizeof(QuantScaleOutType)),
                             static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenScaleAlignBytes = quantTokenAlignBytes + quantScaleAlignBytes;
    if constexpr (TopkWeightsPrefetch) {
        uint32_t weightAlignBytes = Ops::Base::CeilAlign(
            static_cast<uint32_t>(params_.tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        quantTokenScaleAlignBytes += weightAlignBytes;
    }
    quantProcessConfig_ = {
        .quantTokenAlignBytes = quantTokenAlignBytes,
        .quantScaleAlignBytes = quantScaleAlignBytes,
        .quantTokenScaleAlignBytes = quantTokenScaleAlignBytes,
        .quantScaleNumAlignPerToken = quantScaleNumAlignPerToken};

    int32_t compareCount =
        Ops::Base::CeilAlign(static_cast<int64_t>(params_.tilingData->bs) *
                                 static_cast<int64_t>(params_.tilingData->topK) *
                                 static_cast<int64_t>(sizeof(int32_t)),
                             static_cast<int64_t>(ALIGN_256)) /
        static_cast<int64_t>(sizeof(int32_t));
    uint32_t maskAlignSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(compareCount) / 8, static_cast<int64_t>(ALIGN_32));
    uint64_t maskWinOffset = static_cast<uint64_t>(
        params_.peermemInfo.maskRecvPtr - params_.peermemInfo.rankSyncInWorldPtr);
    const SendMaskBufferConfig &bufferConfig =
        aivCoreIdx_ < params_.tilingData->sendMaskCoreCountWithExtraExpert ?
            params_.tilingData->sendMaskConfigForCoreWithExtraExpert :
            params_.tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    sendMaskConfig_ = {
        .maskAlignSize = maskAlignSize,
        .maskWinOffset = maskWinOffset,
        .bufferConfig = bufferConfig};

    resetWorkspaceConfig_ = {
        .flagElementCount = static_cast<int32_t>(CalcResetFlagElementCount(params_.tilingData)),
        .resetBatchElementCount = 0};

    if (commonConfig.sharedExpertNum > 0U) {
        sharedExpertPrepareConfig_ = {
            .quantTokenAlignBytes = quantProcessConfig_.quantTokenAlignBytes,
            .quantScaleAlignBytes = quantProcessConfig_.quantScaleAlignBytes,
            .quantTokenScaleAlignBytes = quantProcessConfig_.quantTokenScaleAlignBytes};
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenDispatchConfig(
    const MoeStageCommonConfig &commonConfig, int32_t dispatchFlagSlotsPerExpert)
{
    uint64_t quantWinOffset = static_cast<uint64_t>(
        params_.peermemInfo.quantTokenScalePtr - params_.peermemInfo.rankSyncInWorldPtr);
    tokenDispatchConfig_ = {
        .common = commonConfig,
        .blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
        .countWorkspace = {.blockIdx = blockIdx_, .blockNum = params_.tilingData->aicNum},
        .maxOutputSize = params_.tilingData->maxOutputSize,
        .bufferConfig = params_.tilingData->dispatchBufferConfig,
        .maskAlignSize = sendMaskConfig_.maskAlignSize,
        .quantWinOffset = quantWinOffset,
        .quantTokenAlignBytes = quantProcessConfig_.quantTokenAlignBytes,
        .quantScaleAlignBytes = quantProcessConfig_.quantScaleAlignBytes,
        .quantTokenScaleAlignBytes = quantProcessConfig_.quantTokenScaleAlignBytes,
        .dispatchFlagSlotsPerExpert = dispatchFlagSlotsPerExpert};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitGmm1Config(
    const MoeStageCommonConfig &commonConfig, int32_t dispatchFlagSlotsPerExpert)
{
    gmm1Config_ = {
        .common = commonConfig,
        .blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
        .countWorkspace = {.blockIdx = blockIdx_, .blockNum = params_.tilingData->aicNum},
        .dispatchFlagSlotsPerExpert = dispatchFlagSlotsPerExpert,
        .activationFlagSlotsPerExpert = INT_CACHELINE,
        .maxTilesPerExpert = params_.tilingData->maxTilesPerExpert,
        .groupedMatmulMode = params_.tilingData->groupedMatmulMode,
        .isPerExpertWeightTensor = params_.tilingData->isPerExpertWeightTensor};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitGmm2CombineConfigs(
    const MoeStageCommonConfig &commonConfig, int32_t dispatchFlagSlotsPerExpert)
{
    gmm2Config_ = {
        .common = commonConfig,
        .blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
        .countWorkspace = {.blockIdx = blockIdx_, .blockNum = params_.tilingData->aicNum},
        .activationFlagSlotsPerExpert = INT_CACHELINE,
        .dispatchFlagSlotsPerExpert = dispatchFlagSlotsPerExpert,
        .combineSyncSlotCountPerExpert = params_.tilingData->combineSyncSlotCountPerExpert,
        .groupedMatmulMode = params_.tilingData->groupedMatmulMode,
        .isPerExpertWeightTensor = params_.tilingData->isPerExpertWeightTensor};
    constexpr bool pairedAivCombine = ENABLE_A8W4 || ENABLE_A4W4;
    quantCombineConfig_ = {
        .common = commonConfig,
        .job = {.jobIndex = pairedAivCombine ? blockIdx_ : aivCoreIdx_,
                .totalJobs = pairedAivCombine ? blockNum_ : blockAivNum_},
        .participates = !pairedAivCombine || GetSubBlockIdx() == 1U};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenUnpermuteConfig(
    const MoeStageCommonConfig &commonConfig)
{
    tokenUnpermuteConfig_ = {
        .common = commonConfig,
        .job = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_},
        .quantTokenSizeBytes = quantCombineBufferConfig_.quantTokenSizeBytes,
        .fullTokenChunkJobCount = params_.tilingData->unpermuteFullTokenChunkCoreCount,
        .fullTokenChunkConfig = params_.tilingData->unpermuteConfigForFullTokenChunk,
        .tailTokenChunkConfig = params_.tilingData->unpermuteConfigForTailTokenChunk};
}

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    k_ = tilingData->h;
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    gmm2PingPongIdx_ = 0;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    moeWeightTensorListAddrs_ = {
        .weight1 = weight1,
        .weightScales1 = weightScales1,
        .weight2 = weight2,
        .weightScales2 = weightScales2};
    if (sharedExpertNum_ > 0U) {
        sharedWeightTensorListAddrs_ = {
            .weight1 = sharedWeight1,
            .weightScales1 = sharedWeightScales1,
            .weight2 = sharedWeight2,
            .weightScales2 = sharedWeightScales2};
    }
    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    epilogueOp_.Init({.yGmAddr = params_.workspaceInfo.activationQuantDataPtr,
                      .yScaleGmAddr = params_.workspaceInfo.activationQuantScalePtr,
                      .x2ScaleGmAddr = nullptr,
                      .x1ScaleGmAddr = nullptr,
                      .biasGmAddr = nullptr,
                      .clampLimit = tilingData->clampLimit,
                      .actMode = tilingData->actMode,
                      .actSubMode = tilingData->actSubMode,
                      .activationAlpha = tilingData->activationAlpha,
                      .activationBeta = tilingData->activationBeta});
    MoeStageCommonConfig commonConfig = {
        .rankId = rankId_,
        .worldSize = worldSize_,
        .moeExpertPerRank = moeExpertPerRank_,
        .sharedExpertNum = sharedExpertNum_,
        .tokenNum = tilingData->bs,
        .topK = tilingData->topK,
        .tokenHiddenDim = k_,
        .gmm1OutputDim = tilingData->hiddenDim};
    const int64_t maxOutput = static_cast<int64_t>(tilingData->maxOutputSize);
    const int64_t tileM = static_cast<int64_t>(GMM1_TILE_M);
    int32_t dispatchFlagSlotsPerExpert =
        static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, tileM)) * INT_CACHELINE;
    InitInputPrepareConfigs(commonConfig);
    InitTokenDispatchConfig(commonConfig, dispatchFlagSlotsPerExpert);
    InitGmm1Config(commonConfig, dispatchFlagSlotsPerExpert);
    InitGmm2CombineConfigs(commonConfig, dispatchFlagSlotsPerExpert);
    InitCombineBuffers();
    InitTokenUnpermuteConfig(commonConfig);
}

// 普通模板 Token Dispatch 使用的 UB/GM 视图。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchBuffInit()
{
    const TokenDispatchConfig &context = tokenDispatchConfig_;
    TokenDispatchScratch<ActivationType> &scratch = tokenDispatchScratch_;
    scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    scratch.revTokenElemCnt = context.common.tokenHiddenDim / A_ELEMS_PER_BYTE;
    scratch.revScaleElemCnt =
        Ops::Base::CeilDiv(static_cast<int64_t>(context.common.tokenHiddenDim),
                           static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(context.common.worldSize * context.common.moeExpertPerRank * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_32));
    if constexpr (ENABLE_A8W4) {
        scratch.cumsumInfoGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.cumsumInfoPtr +
            static_cast<uint64_t>(cumsumInfoTensorSize) * context.countWorkspace.blockIdx));
    }
    scratch.cumsumRevCntInRank = 0U;

    uint32_t expertTokenCntTensorAddr = 0U;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    scratch.expertTokenCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr,
                             expertTokenCntTensorSize / sizeof(int32_t));
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    scratch.cumsumInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr,
                             cumsumInfoTensorSize / sizeof(int32_t));
    uint32_t sendCntTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t sendCntTensorSize = context.common.worldSize * static_cast<uint32_t>(ALIGN_32);
    scratch.sendCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntTensorAddr,
                             sendCntTensorSize / sizeof(int32_t));
    uint32_t maskBatchTensorAddr = sendCntTensorAddr + sendCntTensorSize;
    uint32_t maskBatchTensorSize = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8);
    scratch.maskBatchTensor =
        LocalTensor<uint8_t>(TPosition::VECCALC, maskBatchTensorAddr, maskBatchTensorSize);
    scratch.maskBatchU32Tensor =
        LocalTensor<uint32_t>(TPosition::VECCALC, maskBatchTensorAddr,
                              maskBatchTensorSize / sizeof(uint32_t));
    uint32_t validTopkIndexTensorAddr = maskBatchTensorAddr + maskBatchTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(bufferConfig.routeItemsPerBatch * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_32));
    scratch.validTopkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
                             validTopkIndexTensorSize / sizeof(int32_t));
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    scratch.topkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr,
                             validTopkIndexTensorSize / sizeof(int32_t));
    scratch.copyTmpBaseAddr = topkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t copyTmpTotalSize =
        static_cast<uint32_t>(bufferConfig.bufferCount) * context.quantTokenScaleAlignBytes;
    uint32_t expertTokenNumsOutTensorAddr = scratch.copyTmpBaseAddr + copyTmpTotalSize;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(context.common.moeExpertPerRank * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_32));
    scratch.expertTokenNumsOutTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
                             expertTokenNumsOutTensorSize / sizeof(int32_t));
    uint32_t metaInfoTensorAddr = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    uint32_t metaInfoTensorSize =
        static_cast<uint32_t>(bufferConfig.bufferCount) * INT32_PER_256B * sizeof(int32_t);
    scratch.metaInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr,
                             metaInfoTensorSize / sizeof(int32_t));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitCombineBuffers()
{
    quantCombineBufferConfig_ = {.combineUbElementCount = 0, .quantTokenSizeBytes = 0U};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        quantCombineBufferConfig_ = CreateQuantCombineBufferConfig(k_);
    }
}

// ======================================================================================
// SendAndQuantBuffInit：单核 mask/reset/quant/shared-prepare 模块使用的 buffer 申请。
//   shared prepare 复用 quant 输出双 buffer；reset 封顶 DISPATCH_RESET_BATCH。
// ======================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::SendMaskBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    SendMaskBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 = static_cast<uint64_t>(resetWorkspaceConfig_.flagElementCount);
    if constexpr (TopkWeightsPrefetch) {
        uint64_t statusElementCount =
            (static_cast<uint64_t>(moeExpertPerRank_) *
                 static_cast<uint64_t>(params_.tilingData->maxTilesPerExpert) +
             1U) *
            INT_CACHELINE;
        totalFlagInt32 = totalFlagInt32 > statusElementCount ? totalFlagInt32 : statusElementCount;
    }
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    int32_t resetBatchElementCount = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                         static_cast<int32_t>(resetElementCountPerCore) :
                                         DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    uint32_t mxTempTensorSize = 2 * 1024;
    uint32_t xOutTensorSize = quantProcessConfig_.quantTokenScaleAlignBytes;
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * moeExpertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // 必须与 host SetAdaptiveBufferConfigs 的 quotient/remainder 分核保持一致。GatherAndSendExpertMasks 按
    // expertId = aivCoreIdx_ + ownedIdx * blockAivNum_ 遍历，因此前 remainder 个 core 多处理一个 expert。
    bufferConfig = sendMaskConfig_.bufferConfig;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，因此两个 int32 tensor 均天然满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetAddrActual = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    resetWorkspaceConfig_.resetBatchElementCount = resetBatchElementCount;

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    quantScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));

    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantScratch_.xOutTensor0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantScratch_.xOutTensor1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));
    if (sharedExpertNum_ > 0U) {
        sharedExpertPrepareScratch_.copyBuffer0 = quantScratch_.xOutTensor0;
        sharedExpertPrepareScratch_.copyBuffer1 = quantScratch_.xOutTensor1;
    }

    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));

    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize;
    uint32_t sendGatherOutSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));

    uint32_t sendMaskTotalBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.sendMaskTensor = LocalTensor<uint8_t>(TPosition::VECCALC, sendMaskAddr, sendMaskTotalBytes);
    uint32_t sendGatherOutAddr = sendMaskAddr + sendMaskTotalBytes;
    sendMaskScratch_.sendGatherOutTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    sendMaskScratch_.sendCntAccTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
    return bufferConfig;
}

// 在普通模板内构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置。
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenUnpermuteBuffers()
{
    const TokenUnpermuteConfig &context = tokenUnpermuteConfig_;
    TokenUnpermuteScratch &scratch = tokenUnpermuteScratch_;
    UnpermuteBufferConfig bufferConfig =
        context.job.jobIndex < context.fullTokenChunkJobCount ? context.fullTokenChunkConfig :
                                                                context.tailTokenChunkConfig;

    uint32_t bf16ScaleBufAlign = 0U;
    uint32_t fp32ScaleBufAlign = 0U;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(context.common.tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(bfloat16_t)) *
                static_cast<uint32_t>(DEQUANT_BF16_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(float)) *
                static_cast<uint32_t>(DEQUANT_FP32_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
    }

    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    uint32_t dataResBufBytes = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufBytes = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
    uint32_t dataResAddr = 0U;
    scratch.dataResTensor = LocalTensor<bfloat16_t>(
        TPosition::VECCALC, dataResAddr, dataResBufBytes / sizeof(bfloat16_t));
    uint32_t dataResFp32Addr = dataResAddr + dataResBufBytes;
    scratch.dataResFp32Tensor = LocalTensor<float>(
        TPosition::VECCALC, dataResFp32Addr, dataResFp32BufBytes / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufBytes;

    scratch.topKWeightsTensor = LocalTensor<float>(
        TPosition::VECCALC, tempAddr, bufferConfig.topKWeightsBufferBytes / sizeof(float));
    tempAddr += bufferConfig.topKWeightsBufferBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        scratch.topKWeightsBf16Tensor = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr,
            bufferConfig.topKWeightsConversionBufferBytes / sizeof(bfloat16_t));
        tempAddr += bufferConfig.topKWeightsConversionBufferBytes;
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        scratch.bf16ScaleTensor = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        scratch.fp32ScaleTensor = LocalTensor<float>(
            TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
    }
    return bufferConfig;
}

// 全卡同步：各 AIV 按物理核号分摊 rank 握手，末尾执行本卡 AIV 同步。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t *syncRank = reinterpret_cast<__gm__ int32_t *>(params_.peermemInfo.rankSyncInWorldPtr);
    __gm__ int32_t *syncCount = reinterpret_cast<__gm__ int32_t *>(
        params_.peermemInfo.rankSyncInWorldPtr + 48U * 1024U + static_cast<uint64_t>(aivCoreIdx_) * 64U);
    int32_t count = ReadGmByPassDCache(syncCount) + 1;
    for (uint32_t rankIdx = aivCoreIdx_; rankIdx < worldSize_; rankIdx += blockAivNum_) {
        __gm__ int32_t *remoteSyncAddr =
            reinterpret_cast<__gm__ int32_t *>(winRankAddr_[rankIdx]) + rankId_ * INT_CACHELINE;
        WriteGmByPassDCache(remoteSyncAddr, count);
        GmSignalWaitBarrier(syncRank + rankIdx * INT_CACHELINE, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm1(int32_t &gmTileSequence)
{
    if (gmm1Config_.blockJob.totalJobs == 0U ||
        gmm1Config_.blockJob.jobIndex >= gmm1Config_.blockJob.totalJobs || sharedExpertNum_ == 0U) {
        return;
    }
    typename SharedBlockEpilogue::Params epilogueParams{
        .yGmAddr = params_.workspaceInfo.sharedExpertActivationDataPtr,
        .yScaleGmAddr = params_.workspaceInfo.sharedExpertActivationScalePtr,
        .x2ScaleGmAddr = nullptr,
        .x1ScaleGmAddr = nullptr,
        .biasGmAddr = nullptr,
        .clampLimit = params_.tilingData->clampLimit,
        .actMode = params_.tilingData->actMode,
        .actSubMode = params_.tilingData->actSubMode,
        .activationAlpha = params_.tilingData->activationAlpha,
        .activationBeta = params_.tilingData->activationBeta};
    sharedEpilogueOp_.Init(epilogueParams);

    SharedExpertGmm1ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = gmm1Config_.common.tokenNum;
    Get<N_VALUE>(problemShape) = gmm1Config_.common.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = gmm1Config_.common.tokenHiddenDim;
    GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0U;
    int32_t vecSetSyncCom = 0;
    uint16_t pingpongIdx = 0U;
    SharedExpertGmm1ActivationState runtimeState{
        startBlockIdx, vecSetSyncCom, gmTileSequence, pingpongIdx};
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType,
                                           QuantScaleOutType, ENABLE_A8W4>(
            gmm1Config_, params_.workspaceInfo, sharedWeightTensorListAddrs_, sharedEpilogueOp_,
            gmmAddrInfo, sharedExpertIdx);
        RunSharedExpertGmm1ActivationStage<QuantOutType, Weight1Type, ActivationQuantOutType,
                                           QuantScaleOutType, ENABLE_A8W4, GMM1_TILE_M>(
            gmm1Config_, params_, sharedEpilogueOp_, gmmAddrInfo, problemShape,
            runtimeState, sharedExpertIdx);
    }
    EndSync(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm2(int32_t &gmTileSequence)
{
    uint32_t startBlockIdx = 0U;
    int32_t vecSetSyncCom = 0;
    Gmm2RuntimeState runtimeState{
        startBlockIdx, vecSetSyncCom, gmTileSequence, gmm2PingPongIdx_};
    Gmm2ExpertLoopState state = CreateGmm2ExpertLoopState(gmm2Config_);
    GMMAddrInfo gmmAddrInfo{};
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        RunSharedExpertGmm2Stage<QuantOutType, ActivationQuantOutType, Weight1Type,
                                 QuantScaleOutType, ENABLE_A8W4, ENABLE_A4W4,
                                 GMM1_TILE_M, false, false, false>(
            gmm2Config_, params_, sharedWeightTensorListAddrs_, state, gmmAddrInfo,
            runtimeState, sharedExpertIdx);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessMoeExpertStages(int32_t &gmTileSequence)
{
    DispatchBuffInit();
    gmm1ActivationScratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
    Gmm1ExpertLoopState gmm1State = CreateGmm1ExpertLoopState(gmm1Config_);
    GMMAddrInfo gmm1AddrInfo{};
    uint32_t gmm1StartBlockIdx = 0U;
    int32_t gmm1VecSetSyncCom = 0;
    uint16_t gmm1PingPongIdx = 0U;
    Gmm1ActivationState gmm1RuntimeState{
        gmm1StartBlockIdx, gmm1VecSetSyncCom, gmTileSequence, gmm1PingPongIdx};
    uint64_t currentSendCnt = 0U;
    uint64_t nextSendCnt = 0U;

    RunMoeExpertDispatchStage<ActivationType, QuantScaleOutType, ENABLE_A8W4, GMM1_TILE_M,
                              TopkWeightsPrefetch>(
        tokenDispatchConfig_, params_, winRankAddr_, tokenDispatchScratch_, 0U, nextSendCnt);
    for (uint32_t expertIdx = 0U; expertIdx < gmm1Config_.common.moeExpertPerRank; ++expertIdx) {
        currentSendCnt = nextSendCnt;
        if (expertIdx + 1U < gmm1Config_.common.moeExpertPerRank) {
            RunMoeExpertDispatchStage<ActivationType, QuantScaleOutType, ENABLE_A8W4, GMM1_TILE_M,
                                      TopkWeightsPrefetch>(
                tokenDispatchConfig_, params_, winRankAddr_, tokenDispatchScratch_, expertIdx + 1U, nextSendCnt);
        }
        RunMoeExpertGmm1ActivationStage<QuantOutType, ActivationType, Weight1Type, ActivationQuantOutType,
                                        QuantScaleOutType, ENABLE_A8W4, GMM1_TILE_M, EPILOGUE_TILE_M,
                                        TopkWeightsPrefetch, false, false>(
            gmm1Config_, params_, moeWeightTensorListAddrs_, epilogueOp_,
            gmm1ActivationScratch_, gmm1State, gmm1AddrInfo,
            gmm1RuntimeState, expertIdx, currentSendCnt);
    }
    FinishMoeGmm1ActivationStage<ENABLE_A8W4, TopkWeightsPrefetch>(
        gmm1Config_, params_.workspaceInfo, gmm1RuntimeState);

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            ExportExpertTokenCounts<ActivationType, ENABLE_A8W4>(
                tokenDispatchConfig_, tokenDispatchScratch_, params_);
        }
    }
    if constexpr (TopkWeightsPrefetch) {
        SyncAll<false>();
    }

    uint32_t gmm2StartBlockIdx = gmm1RuntimeState.startBlockIdx;
    int32_t gmm2VecSetSyncCom = 0;
    Gmm2RuntimeState gmm2RuntimeState{
        gmm2StartBlockIdx, gmm2VecSetSyncCom, gmTileSequence, gmm2PingPongIdx_};
    gmm2Scratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
    Gmm2ExpertLoopState gmm2State = CreateGmm2ExpertLoopState(gmm2Config_);
    GMMAddrInfo gmm2AddrInfo{};
    constexpr bool configureGmm2Output =
        ENABLE_A8W4 || ENABLE_A4W4 || CombineQuantMode != COMBINE_NO_QUANT;
    constexpr bool configureCombineCounter = CombineQuantMode != COMBINE_NO_QUANT;
    for (uint32_t expertIdx = 0U; expertIdx < gmm2Config_.common.moeExpertPerRank; ++expertIdx) {
        if (!PrepareMoeExpertGmm2Stage < ActivationType, ActivationQuantOutType, QuantScaleOutType,
            PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE,
            PackedElementTraits<Weight1Type>::ELEMENTS_PER_BYTE,
            PackedElementTraits<ActivationQuantOutType>::ELEMENTS_PER_BYTE,
            configureGmm2Output, configureCombineCounter,
            ENABLE_A8W4 || ENABLE_A4W4 > (gmm2Config_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmm2Scratch_,
                                          gmm2State, gmm2AddrInfo, expertIdx)) {
            continue;
        }
        RunMoeExpertGmm2Stage<CombineQuantMode, QuantOutType, ActivationQuantOutType,
                              Weight1Type, QuantScaleOutType, ENABLE_A8W4, ENABLE_A4W4,
                              GMM1_TILE_M, TopkWeightsPrefetch, false, false>(
            gmm2Config_, params_, gmm2AddrInfo, gmm2State, gmm2RuntimeState);
        ScheduleQuantizedMoeExpertCombine<CombineQuantMode>(
            quantCombineConfig_, quantCombineBufferConfig_, params_, gmm2AddrInfo,
            static_cast<uint32_t>(Get<M_VALUE>(gmm2State.problemShape)),
            gmm2State.expertBeforeCnt, expertIdx);
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(gmm2RuntimeState.vecSetSyncCom, gmm2RuntimeState.pingpongIdx);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();

    // Phase 1: 量化 + 共享专家输入拆分 + mask/reset (AIV)
    QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
        dispatchPrepareConfig_, params_, quantProcessConfig_, quantScratch_);
    GatherAndSendExpertMasks(dispatchPrepareConfig_, params_, winRankAddr_, sendMaskConfig_, sendMaskScratch_);
    ResetSyncStatus<TopkWeightsPrefetch>(
        dispatchPrepareConfig_, params_, resetWorkspaceConfig_, resetTensor_);
    if (sharedExpertNum_ > 0) {
        PrepareSharedExpertInput<ActivationType, QuantScaleOutType, A_ELEMS_PER_BYTE>(
            dispatchPrepareConfig_, params_, sharedExpertPrepareConfig_, sharedExpertPrepareScratch_);
    }
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>(); // aic需要等待flag位reset清理完成

    int32_t gmTileSequence = 0;
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm1(gmTileSequence);
    }

    CrossRankSyncInWorldSize();

    // 2. MoE 专家 Dispatch -> GMM1/SwiGLU -> GMM2/Combine。
    ProcessMoeExpertStages(gmTileSequence);

    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm2(gmTileSequence);
    }

    // 3. 本卡数据Unpermute
    if constexpr (g_coreType == AIV) {
        CrossRankSyncInWorldSize();
        UnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

} // namespace MegaMoeImpl
#undef TemplateMegaMoeTypeClass
#undef TemplateMegaMoeTypeFunc
#endif
