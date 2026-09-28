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
 * \file mega_moe_arch35.h
 * \brief
 */

#ifndef MEGA_MOE_ARCH35_H
#define MEGA_MOE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "common/mega_moe_types.h"
#include "common/mega_moe_workspace.h"
#include "common/mega_moe_utils.h"
#include "common/mega_moe_exception_dump_policy.h"
#include "blaze/epilogue/block_epilogue_activation_mx_quant.h"
#include "stage/mega_moe_token_quant.h"
#include "stage/mega_moe_send_mask.h"
#include "stage/mega_moe_workspace_reset.h"
#include "stage/mega_moe_token_dispatch_dedup.h"
#include "stage/mega_moe_gmm1_activation.h"
#include "stage/mega_moe_gmm2_combine_dedup.h"
#include "stage/mega_moe_unpermute.h"
#include "../../../common/op_kernel/quantize_functions.h"

namespace MegaMoeImpl {

using namespace AscendC;

#define TemplateMegaMoeTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename MoeWeightType, int32_t MoeQuantMode, \
        typename SharedWeightType, int32_t SharedQuantMode, int32_t MoeWeight1Format, int32_t MoeWeight2Format, \
        int32_t SharedWeight1Format, int32_t SharedWeight2Format, int32_t CombineQuantMode, bool TopkWeightsPrefetch, \
        typename TopkIndexType
#define TemplateMegaMoeTypeFunc \
    XType, OutputType, TopkWeightsType, MoeWeightType, MoeQuantMode, SharedWeightType, SharedQuantMode, \
        MoeWeight1Format, MoeWeight2Format, SharedWeight1Format, SharedWeight2Format, CombineQuantMode, \
        TopkWeightsPrefetch, TopkIndexType

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    using MoeQuantConfig = QuantConfig<MoeWeightType, MoeQuantMode>;
    using SharedQuantConfig = QuantConfig<SharedWeightType, SharedQuantMode>;
    using QuantOutType = typename MoeQuantConfig::QuantOutType;
    using ActivationType = typename MoeQuantConfig::QuantStorageType;
    using QuantScaleOutType = typename MoeQuantConfig::QuantScaleType;
    using SharedActivationType = typename SharedQuantConfig::QuantStorageType;
    using SharedActivationOutType = typename SharedQuantConfig::ActivationQuantOutType;
    using SharedQuantScaleType = typename SharedQuantConfig::QuantScaleType;
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData,
                                GM_ADDR tilingGM);

private:
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;
    using UnpermuteBufferConfig = MegaMoeUnpermuteBufferConfig;

    __aicore__ inline void InitStageConfigs(MegaMoeTilingData *tilingData);
    __aicore__ inline void InitEpilogueAndCommonConfig(MegaMoeTilingData *tilingData);
    __aicore__ inline void InitInputPrepareConfigs();
    __aicore__ inline void InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                    int32_t activationFlagSlotsPerExpert);
    __aicore__ inline void InitGmmConfigs();
    __aicore__ inline void InitTokenUnpermuteConfig();
    __aicore__ inline uint32_t InitQuantScratchTensors(uint32_t mxTempTensorAddr);

protected:
    using A8W4BlockContext =
        typename GmmKernel::Config<true, 0, typename MoeQuantConfig::ActivationQuantOutType, MoeWeightType, bfloat16_t,
                                   QuantScaleOutType, QuantScaleOutType>::BlockContext;
    using SharedA8W4Config = GmmKernel::Config<true, 0, SharedActivationOutType, SharedWeightType, bfloat16_t,
                                               SharedQuantScaleType, SharedQuantScaleType>;
    typename SharedA8W4Config::BlockContext sharedBlockContext_{};
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void EnterSteadyDispatch();
    __aicore__ inline void InitQuantTokenBufferConfig();
    __aicore__ inline UnpermuteBufferConfig InitTokenUnpermuteBuffers();
    __aicore__ inline void SendTopkIdsIndexAndCount(const WorkRange &ownedExpertRange);
    __aicore__ inline void QuantizeInputTokens(const WorkRange &tokenRange);
    __aicore__ inline void ProcessInputPreparationStage();
    __aicore__ inline void WaitForInputPreparation();
    __aicore__ inline void ExportAndResetExpertCounts();
    __aicore__ inline void RunGmm2CombineForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                   uint32_t &startBlockIdx, uint32_t tokenStartIndexInExpert,
                                                   uint32_t sliceTokenCount,
                                                   WaveCombineBufferConfig &combineBufferConfig,
                                                   uint32_t &combineRowSequence, bool isFinalCombine,
                                                   const A8W4BlockContext &pipeline);
    template <bool WaitForTokenCountReady>
    __aicore__ inline void PrepareGmmExpertState(ExpertLoopState &state, uint32_t expertIdx);
    __aicore__ inline void ConfigureSharedGmm1Input(GMMAddrInfo &gmmAddrInfo, GmmExecutionConfig &gmmConfig,
                                                    Gmm1ActivationSync &sharedGmm1ActivationSync) const;
    __aicore__ inline void RunSharedExpertGmm1Activation(const GMMAddrInfo &gmmAddrInfo,
                                                         const ProblemShape &problemShape,
                                                         const GmmExecutionConfig &gmmConfig,
                                                         GmmRuntimeState &runtimeState, uint32_t sharedExpertIdx);
    __aicore__ inline void RunSharedExpertGmm2(const GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape);
    __aicore__ inline void ProcessSharedExpertGmm1Loop(GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape,
                                                       const GmmExecutionConfig &sharedGmmConfig,
                                                       GmmRuntimeState &runtimeState);
    __aicore__ inline void ProcessSharedExpertGmm1(Gmm1ActivationSync &sharedGmm1ActivationSync);
    __aicore__ inline void ProcessSharedExpertGmm2Loop(GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape);
    __aicore__ inline void ProcessSharedExpertGmm2();
    template <typename Derived>
    __aicore__ inline void ProcessWave(Derived &derived);

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs moeWeightTensorListAddrs_{};
    ExpertWeightTensorListAddrs sharedWeightTensorListAddrs_{};
    MoeStageCommonConfig commonConfig_{};
    GmmExecutionConfig gmmExecutionConfig_{};
    BlockWorkspaceContext countWorkspace_{};
    MoeSyncWorkspaceLayout syncWorkspaceLayout_{};
    // 单次 reset 批量元素数（与 syncWorkspaceLayout_ 描述的清零区域配套）。
    int32_t resetBatchElementCount_ = 0;
    // 输入准备各 stage（quant/mask/reset/shared-prepare/unpermute）共用的逐 AIV 任务分工。
    AivJobContext aivJob_{};
    TokenDispatchConfig tokenDispatchConfig_;
    SendMaskConfig sendMaskConfig_;
    QuantProcessConfig quantProcessConfig_;
    QuantProcessConfig sharedQuantProcessConfig_;
    QuantTokenBufferConfig quantTokenBufferConfig_;
    // Wave Combine 的逻辑任务分工（block 粒度，AIV1 门控在函数内）。
    AivJobContext waveCombineJob_{};
    TokenUnpermuteConfig tokenUnpermuteConfig_;

    uint32_t k_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint16_t gmm1PingPongIdx_ = 0;
    // 同一 AIC/AIV 对复用 gmmToEpilogueFlag；序号在单次 launch 的相关 GMM 阶段间保持单调递增。
    int32_t gmmTileSequence_ = 0;
    // MoE GMM1/GMM2 共用分核游标；A8W8 的共享专家 GMM2 继续沿用该游标。
    uint32_t startBlockIdx_ = 0;
    // 主线 shared-expert 特性成员
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;
    uint32_t mGroupsPerWave_ = 1U;

    static constexpr uint32_t A_ELEMS_PER_BYTE = MoeQuantConfig::A_ELEMS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = MoeQuantConfig::B_ELEMS_PER_BYTE;
    static constexpr bool SHARED_INPUT_REUSES_MOE_QUANT =
        Std::IsSame<typename MoeQuantConfig::QuantOutType, typename SharedQuantConfig::QuantOutType>::value;
    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M = TopkWeightsPrefetch ? L1_TILE_M_128 : L1_TILE_M_256;
    QuantProcessScratch<typename MoeQuantConfig::QuantStorageType> quantScratch_;
    QuantProcessScratch<typename SharedQuantConfig::QuantStorageType> sharedQuantScratch_;
    SendMaskScratch<TopkIndexType> sendMaskScratch_;
    LocalTensor<int32_t> resetTensor_;

    using ActivationQuantOutType = typename MoeQuantConfig::ActivationQuantOutType;
    static constexpr uint32_t C_ELEMS_PER_BYTE = MoeQuantConfig::C_ELEMS_PER_BYTE;

    using BlockEpilogue = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, EPILOGUE_TILE_M, L1_TILE_N,
                                                         TopkWeightsPrefetch>;

    using SharedBlockEpilogue = BlockEpilogueActivationMxQuant<typename SharedQuantConfig::ActivationQuantOutType,
                                                               bfloat16_t, L1_TILE_M_256, L1_TILE_N, false>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
    TokenDispatchScratch<ActivationType, TopkIndexType> tokenDispatchScratch_;
    WaveCombineScratch waveCombineScratch_;
    // 非量化 combine 去重的追加 UB 视图与跨行事件武装状态（排空时消费）。
    WaveCombineDedupScratch waveCombineDedupScratch_;
    TokenUnpermuteScratch tokenUnpermuteScratch_;
    MegaMoeImpl::ExceptionDumpEngine exceptionDump_;
    __gm__ MegaMoeImpl::GmmLoopCount *gmmLoopCount_{nullptr};
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitInputPrepareConfigs()
{
    aivJob_ = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_};
    quantProcessConfig_ =
        CreateQuantProcessConfig<typename MoeQuantConfig::QuantStorageType, typename MoeQuantConfig::QuantScaleType,
                                 TopkWeightsPrefetch, MoeQuantConfig::A_ELEMS_PER_BYTE>(k_, params_);
    if constexpr (SHARED_INPUT_REUSES_MOE_QUANT) {
        sharedQuantProcessConfig_ = quantProcessConfig_;
    } else {
        sharedQuantProcessConfig_ = CreateQuantProcessConfig<typename SharedQuantConfig::QuantStorageType,
                                                             typename SharedQuantConfig::QuantScaleType, false,
                                                             SharedQuantConfig::A_ELEMS_PER_BYTE, false>(k_, params_);
    }
    // 共享专家启用时仅 AIV1 计算发送 topK 有效下标，其余情况保留全 AIV 分工。
    const uint32_t topkValidIndexCoreIdx = sharedExpertNum_ > 0U ? blockIdx_ : aivCoreIdx_;
    sendMaskConfig_ = CreateSendMaskConfig<TopkIndexType>(params_, topkValidIndexCoreIdx);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                                                  int32_t activationFlagSlotsPerExpert)
{
    countWorkspace_ = {.blockIdx = blockIdx_, .blockNum = params_.tilingData->aicNum};
    syncWorkspaceLayout_ = {.dispatchFlagSlotCountPerExpert = dispatchFlagSlotsPerExpert,
                            .activationFlagSlotCountPerExpert = activationFlagSlotsPerExpert,
                            .gmm1TileStatusCountPerExpert = params_.tilingData->maxTilesPerExpert};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitGmmConfigs()
{
    gmmExecutionConfig_ = {.blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
                           .isPerExpertWeightTensor = params_.tilingData->isPerExpertWeightTensor};
    waveCombineJob_ = {.jobIndex = blockIdx_, .totalJobs = blockNum_};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenUnpermuteConfig()
{
    tokenUnpermuteConfig_ = {.job = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_},
                             .quantTokenSizeBytes = quantTokenBufferConfig_.quantTokenSizeBytes,
                             .fullTokenChunkJobCount = params_.tilingData->unpermuteFullTokenChunkCoreCount,
                             .fullTokenChunkConfig = params_.tilingData->unpermuteConfigForFullTokenChunk,
                             .tailTokenChunkConfig = params_.tilingData->unpermuteConfigForTailTokenChunk};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitEpilogueAndCommonConfig(MegaMoeTilingData *tilingData)
{
    epilogueOp_.Init({.yGmAddr = params_.workspaceInfo.activationQuantDataPtr,
                      .yScaleGmAddr = params_.workspaceInfo.activationQuantScalePtr,
                      .clampLimit = tilingData->clampLimit,
                      .actMode = tilingData->actMode,
                      .actSubMode = tilingData->actSubMode,
                      .activationAlpha = tilingData->activationAlpha,
                      .activationBeta = tilingData->activationBeta});
    commonConfig_ = {.rankId = rankId_,
                     .worldSize = worldSize_,
                     .moeExpertPerRank = moeExpertPerRank_,
                     .sharedExpertNum = sharedExpertNum_,
                     .tokenNum = tilingData->bs,
                     .topK = tilingData->topK,
                     .tokenHiddenDim = k_,
                     .gmm1OutputDim = tilingData->hiddenDim};
}

// 初始化各阶段配置及同步标志槽数。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitStageConfigs(MegaMoeTilingData *tilingData)
{
    InitEpilogueAndCommonConfig(tilingData);
    const int64_t maxOutput = static_cast<int64_t>(tilingData->maxOutputSize);
    const int64_t tileM = static_cast<int64_t>(GMM1_TILE_M);
    int32_t dispatchFlagSlotsPerExpert = static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, tileM)) * INT_CACHELINE;
    int32_t activationFlagSlotsPerExpert =
        static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, static_cast<int64_t>(L1_TILE_M_256))) * INT_CACHELINE;
    InitInputPrepareConfigs();
    tokenDispatchConfig_ = CreateTokenDispatchConfig<TopkIndexType>(params_, quantProcessConfig_);
    InitSyncWorkspaceConfigs(dispatchFlagSlotsPerExpert, activationFlagSlotsPerExpert);
    InitGmmConfigs();
    InitQuantTokenBufferConfig();
    InitTokenUnpermuteConfig();
}

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData, GM_ADDR tilingGM)
{
    k_ = tilingData->h;
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    mGroupsPerWave_ = tilingData->mGroupsPerWave;
    gmm1PingPongIdx_ = 0;
    gmmTileSequence_ = 0;
    startBlockIdx_ = 0;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    GM_ADDR dumpBase = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[rankId_]);
    for (int i = 0; i < worldSize_; i++) {
        // g_winRankAddr_从win区地址偏移60K开始用，前面60K是异常dump区
        g_winRankAddr_[i] = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[i]) + EXCEPTION_DUMP_REGION_SIZE;
    }
    params_.aGmAddr = x;
    params_.xScaleGmAddr = scales;
    params_.expertIdxGmAddr = topkIds;
    moeWeightTensorListAddrs_ = {
        .weight1 = weight1, .weightScales1 = weightScales1, .weight2 = weight2, .weightScales2 = weightScales2};
    if (sharedExpertNum_ > 0U) {
        sharedWeightTensorListAddrs_ = {.weight1 = sharedWeight1,
                                        .weightScales1 = sharedWeightScales1,
                                        .weight2 = sharedWeight2,
                                        .weightScales2 = sharedWeightScales2};
    }
    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    {
        WorkspaceLayout workspaceLayout(tilingData);
        params_.workspaceInfo.Bind(workspaceGM, workspaceLayout);
    }
    params_.peermemInfo = PeermemInfo(g_winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    InitStageConfigs(tilingData);

    gmmLoopCount_ =
        MegaMoeImpl::RegisterMegaMoeExceptionDump(exceptionDump_, dumpBase, tilingGM, tilingData, params_.peermemInfo,
                                                  reinterpret_cast<GM_ADDR>(&mc2Context_->epRankId));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::EnterSteadyDispatch()
{
    if (GetSubBlockIdx() == 1U) {
        tokenDispatchConfig_.bufferConfig.bufferCount = MIN_DISPATCH_BUFFER_COUNT;
    }
}

// 普通模板 Token Dispatch 使用的 UB/GM 视图。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchBuffInit()
{
    const TokenDispatchConfig &context = tokenDispatchConfig_;
    TokenDispatchScratch<ActivationType, TopkIndexType> &scratch = tokenDispatchScratch_;
    scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRecvTokenCountPtr));
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    scratch.revTokenElemCnt = commonConfig_.tokenHiddenDim / A_ELEMS_PER_BYTE;
    scratch.revScaleElemCnt = Ops::Base::CeilDiv(static_cast<int64_t>(commonConfig_.tokenHiddenDim),
                                                 static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                              MXFP_MULTI_BASE_SIZE;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(commonConfig_.worldSize * commonConfig_.moeExpertPerRank * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_32));
    if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4 || MoeQuantConfig::AXW_MODE == AxWMode::A4W4) {
        scratch.cumsumInfoGlobalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr +
                                               static_cast<uint64_t>(cumsumInfoTensorSize) * countWorkspace_.blockIdx));
    }

    // 按既定顺序落地址。Tensor 保存所有 [expert][source rank] count 的前缀和。
    // Tensor 大小：worldSize_ * moeExpertPerRank_ * sizeof(int32_t)，向上对齐至 32 字节；
    uint32_t cumsumInfoTensorAddr = 0U;
    scratch.cumsumInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    // topK 有效下标 route 已消除 mask 扫描，只保留一份接收 index batch。
    // Tensor 用途：接收 topkIndex 的当前 batch。
    uint32_t validTopkIndexTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(bufferConfig.routeItemsPerBatch) * static_cast<int64_t>(sizeof(TopkIndexType)),
        static_cast<int64_t>(ALIGN_32));
    scratch.validTopkIndexTensor = LocalTensor<TopkIndexType>(TPosition::VECCALC, validTopkIndexTensorAddr,
                                                              validTopkIndexTensorSize / sizeof(TopkIndexType));
    /*
     * 路由批次 Tensor 后依次放置 copyTmp 环形缓冲区和 32B metaInfo 环形缓冲区。
     * Tensor 用途：DispatchExpertTokens 中的动态 dispatch 环形缓冲区，配合
     * EVENT_ID0..EVENT_ID(bufferCount-1) 形成软流水；
     * 只记基址：槽视图在热路径由 GetDispatchCopyBuffer 现场构造，
     * Tensor 大小：bufferConfig.bufferCount 块（启动轮按 Host 自适应分配 2~6 槽；稳态仅使用前 2 槽，不重新布局），
     * 每块 tokenDispatchConfig_.quantTokenScaleAlignBytes；
     * 该值即 Init() 算好的 Align256(token) + Align32(scale) + optional Align32(weight)，与 host
     * CalcDispatchBufferConfig 的 copyBufferBytes 恒相等，故连续 ring 中每个槽位均保持 32B 对齐。
     */
    scratch.copyTmpBaseAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(bufferConfig.bufferCount) * context.quantTokenScaleAlignBytes;
    uint32_t expertTokenNumsOutTensorAddr = scratch.copyTmpBaseAddr + copyTmpTotalSize;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(commonConfig_.moeExpertPerRank * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.expertTokenNumsOutTensor = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
                                                            expertTokenNumsOutTensorSize / sizeof(int32_t));
    // Tensor 用途：CopyTokensAndMetaForDispatch 中的 metaInfo 环形缓冲区，逐 token 即时写入 GM；
    // Tensor 大小：bufferCount * 32B，与 copyTmp 槽位和事件编号一一对应。
    uint32_t metaInfoTensorAddr = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    uint32_t metaInfoTensorSize = static_cast<uint32_t>(bufferConfig.bufferCount) * INT32_PER_256B * sizeof(int32_t);
    scratch.metaInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr, metaInfoTensorSize / sizeof(int32_t));
    scratch.dedupUbBaseAddr = static_cast<uint32_t>(Ops::Base::CeilAlign(
        static_cast<uint64_t>(metaInfoTensorAddr + metaInfoTensorSize), static_cast<uint64_t>(ALIGN_512)));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitQuantTokenBufferConfig()
{
    quantTokenBufferConfig_ = {.quantTokenSizeBytes = 0U};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        quantTokenBufferConfig_ = CreateQuantTokenBufferConfig(k_);
    }
}

// ======================================================================================
// InitQuantScratchTensors：量化 scratch（mxTemp、xOut 双 buffer、xIn 双 buffer）的地址排布；
//   MoE/shared 共用输入和 mxTemp，输出分别分配；返回量化区结束地址。
// ======================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline uint32_t MegaMoe<TemplateMegaMoeTypeFunc>::InitQuantScratchTensors(uint32_t mxTempTensorAddr)
{
    uint32_t mxTempTensorSize = 2 * 1024;
    uint32_t xOutTensorSize = quantProcessConfig_.quantTokenScaleAlignBytes;
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);

    quantScratch_.inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params_.aGmAddr));
    quantScratch_.outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params_.peermemInfo.quantTokenScalePtr));
    quantScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantScratch_.xOutTensor0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantScratch_.xOutTensor1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));
    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));
    uint32_t quantEndAddr = xInAlignAddr2 + xInAlignSize;
    if constexpr (!SHARED_INPUT_REUSES_MOE_QUANT) {
        if (sharedExpertNum_ > 0U) {
            sharedQuantScratch_.outputGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint8_t *>(params_.workspaceInfo.sharedExpertInputPtr));
            sharedQuantScratch_.inputGm = quantScratch_.inputGm;
            sharedQuantScratch_.xInTensor0 = quantScratch_.xInTensor0;
            sharedQuantScratch_.xInTensor1 = quantScratch_.xInTensor1;
            sharedQuantScratch_.mxTempTensor = quantScratch_.mxTempTensor;
            uint32_t sharedOutputBytes = sharedQuantProcessConfig_.quantTokenScaleAlignBytes;
            sharedQuantScratch_.xOutTensor0 = LocalTensor<SharedActivationType>(
                TPosition::VECCALC, quantEndAddr, sharedOutputBytes / sizeof(SharedActivationType));
            quantEndAddr += sharedOutputBytes;
            sharedQuantScratch_.xOutTensor1 = LocalTensor<SharedActivationType>(
                TPosition::VECCALC, quantEndAddr, sharedOutputBytes / sizeof(SharedActivationType));
            quantEndAddr += sharedOutputBytes;
        }
    }

    /*
     * 输入尾部、临时乘法 scale 尾部和输出 scale 补偶槽必须保持为零，保护 H%64==32 的尾块。
     * 两种量化共用 mxTemp，有效 scale 数相同；独立输出避免不同格式覆盖彼此的 padding。
     * 每次 launch 初始化一次，临时 scale 的 padding 在两次量化中均不写入。
     * 预量化路径只搬入有效 data/scale/weight，输出记录的对齐 padding 同样保留为零。
     */
    LocalTensor<int16_t> quantScratchSpan(TPosition::VECCALC, mxTempTensorAddr,
                                          (quantEndAddr - mxTempTensorAddr) / sizeof(int16_t));
    Duplicate<int16_t>(quantScratchSpan, 0, static_cast<int32_t>((quantEndAddr - mxTempTensorAddr) / sizeof(int16_t)));

    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID2>();
    return quantEndAddr;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    sendMaskScratch_.topkIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.expertIdxGmAddr));

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 = static_cast<uint64_t>(params_.workspaceInfo.flagResetElementCount);
    if constexpr (TopkWeightsPrefetch) {
        uint64_t statusElementCount = static_cast<uint64_t>(params_.workspaceInfo.gmm1TileStatusElementCount);
        totalFlagInt32 = totalFlagInt32 > statusElementCount ? totalFlagInt32 : statusElementCount;
    }
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    int32_t resetBatchElementCount = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                         static_cast<int32_t>(resetElementCountPerCore) :
                                         DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    const uint32_t topkValidIndexCoreNum = sharedExpertNum_ > 0U ? blockNum_ : blockAivNum_;
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * moeExpertPerRank_, topkValidIndexCoreNum);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // 必须与 host SetAdaptiveBufferConfigs 的 quotient/remainder 分核保持一致。route 按连续专家段
    // 分核，因此前 remainder 个 core 多处理一个 expert。
    const SendMaskBufferConfig &bufferConfig = sendMaskConfig_.bufferConfig;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，两种 index 编码均满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t topkIdsIndexTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    uint32_t topkIdsIndexTensorSize =
        static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(TopkIndexType));
    sendMaskScratch_.topkIdsIndexTensor = LocalTensor<TopkIndexType>(TPosition::VECCALC, topkIdsIndexTensorAddr,
                                                                     topkIdsIndexTensorSize / sizeof(TopkIndexType));

    uint32_t resetAddrActual = topkIdsIndexTensorAddr + topkIdsIndexTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    resetBatchElementCount_ = resetBatchElementCount;

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    uint32_t routeRingAddr = InitQuantScratchTensors(mxTempTensorAddr);
    uint32_t routeRingBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.routeRingTensor = LocalTensor<uint8_t>(TPosition::VECCALC, routeRingAddr, routeRingBytes);
    uint32_t sendCntAccAddr = routeRingAddr + routeRingBytes;
    // count 发送阶段，量化、reset 和下标发送已完成，可复用累计 count 之前的工作区。
    // 仅双 xIn 就至少占 4096B，足以容纳一张卡最多 1024 个专家的 count。
    sendMaskScratch_.countSendScratchTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, 0, sendCntAccAddr / sizeof(int32_t));
    sendMaskScratch_.sendCntAccTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
}

// 在普通模板内构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置。
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenUnpermuteBuffers()
{
    return CreateTokenUnpermuteBuffers<TopkWeightsType, CombineQuantMode>(
        tokenUnpermuteConfig_, commonConfig_.tokenHiddenDim, tokenUnpermuteScratch_);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ConfigureSharedGmm1Input(
    GMMAddrInfo &gmmAddrInfo, GmmExecutionConfig &gmmConfig, Gmm1ActivationSync &sharedGmm1ActivationSync) const
{
    const bool useInputDirectly =
        !Std::IsSame<XType, bfloat16_t>::value && commonConfig_.tokenHiddenDim % MXFP_DIVISOR_SIZE == 0U;
    gmmConfig.useStridedInput = !useInputDirectly;
    if (useInputDirectly) {
        gmmAddrInfo.aGlobal = params_.aGmAddr;
        gmmAddrInfo.aScaleGlobal = params_.xScaleGmAddr;
    } else {
        GM_ADDR sharedInputBase = SHARED_INPUT_REUSES_MOE_QUANT ? params_.peermemInfo.quantTokenScalePtr :
                                                                  params_.workspaceInfo.sharedExpertInputPtr;
        gmmAddrInfo.aGlobal = sharedInputBase;
        gmmAddrInfo.aScaleGlobal = sharedInputBase + sharedQuantProcessConfig_.quantTokenAlignBytes;
        gmmConfig.inputLayout = {
            sharedQuantProcessConfig_.quantTokenScaleAlignBytes * SharedQuantConfig::A_ELEMS_PER_BYTE,
            sharedQuantProcessConfig_.quantTokenScaleAlignBytes / static_cast<uint32_t>(sizeof(SharedQuantScaleType))};
    }
    if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4) {
        gmmAddrInfo.gmm1ActivationSync = &sharedGmm1ActivationSync;
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::RunSharedExpertGmm1Activation(
    const GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape, const GmmExecutionConfig &gmmConfig,
    GmmRuntimeState &runtimeState, uint32_t sharedExpertIdx)
{
    uint32_t expertBeforeCnt = sharedExpertIdx * commonConfig_.tokenNum;
    auto runGmm = [&](const auto &...layoutArgs) __attribute__((cce_aicore))
    {
        if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4) {
            RunGmm1A8W4<typename SharedQuantConfig::QuantOutType, SharedWeightType, bfloat16_t, SharedQuantScaleType,
                        SharedQuantScaleType, GMM1_TILE_M, L1_TILE_M_256, false, true, true>(
                sharedBlockContext_, sharedEpilogueOp_, params_, problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
                gmmTileSequence_, gmmConfig.blockJob, expertBeforeCnt, sharedExpertIdx, layoutArgs...);
        } else {
            RunGmm1Generic<typename SharedQuantConfig::QuantOutType, SharedActivationOutType,
                           typename SharedQuantConfig::QuantOutType, bfloat16_t, SharedQuantScaleType,
                           SharedQuantScaleType, SharedWeight1Format != FORMAT_ND, GMM1_TILE_M, L1_TILE_M_256, false,
                           true, true>(sharedEpilogueOp_, params_, problemShape, gmmAddrInfo,
                                       runtimeState.startBlockIdx, runtimeState.vecSetSyncCom, gmmConfig.blockJob,
                                       expertBeforeCnt, sharedExpertIdx, runtimeState.pingpongIdx, nullptr,
                                       SharedQuantConfig::AXW_MODE == AxWMode::A8W8, layoutArgs...);
        }
    };
    if (gmmConfig.useStridedInput) {
        runGmm(gmmConfig.inputLayout);
    } else {
        runGmm();
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm1Loop(
    GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape, const GmmExecutionConfig &sharedGmmConfig,
    GmmRuntimeState &runtimeState)
{
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm1GlobalBuffer<SharedWeightType, SharedActivationOutType, SharedQuantScaleType,
                                           SharedQuantConfig::AXW_MODE == AxWMode::A8W4>(
            commonConfig_, sharedGmmConfig, params_.workspaceInfo, sharedWeightTensorListAddrs_, sharedEpilogueOp_,
            gmmAddrInfo, sharedExpertIdx);
        RunSharedExpertGmm1Activation(gmmAddrInfo, problemShape, sharedGmmConfig, runtimeState, sharedExpertIdx);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm1(
    Gmm1ActivationSync &sharedGmm1ActivationSync)
{
    if (gmmExecutionConfig_.blockJob.totalJobs == 0U ||
        gmmExecutionConfig_.blockJob.jobIndex >= gmmExecutionConfig_.blockJob.totalJobs) {
        return;
    }
    typename SharedBlockEpilogue::Params epilogueParams{
        .yGmAddr = params_.workspaceInfo.sharedExpertActivationDataPtr,
        .yScaleGmAddr = params_.workspaceInfo.sharedExpertActivationScalePtr,
        .clampLimit = params_.tilingData->clampLimit,
        .actMode = params_.tilingData->actMode,
        .actSubMode = params_.tilingData->actSubMode,
        .activationAlpha = params_.tilingData->activationAlpha,
        .activationBeta = params_.tilingData->activationBeta};
    sharedEpilogueOp_.Init(epilogueParams);

    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    GMMAddrInfo gmmAddrInfo{};
    GmmExecutionConfig sharedGmmConfig = gmmExecutionConfig_;
    ConfigureSharedGmm1Input(gmmAddrInfo, sharedGmmConfig, sharedGmm1ActivationSync);
    int32_t vecSetSyncCom = 0;
    GmmRuntimeState runtimeState{startBlockIdx_, vecSetSyncCom, gmm1PingPongIdx_};
    if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4) {
        if (g_coreType == AscendC::AIC || GetSubBlockIdx() == 0U) {
            typename SharedA8W4Config::BlockContext::Block block(params_.tilingData->a8w4L1Layout);
            sharedBlockContext_ = {&block};
            ProcessSharedExpertGmm1Loop(gmmAddrInfo, problemShape, sharedGmmConfig, runtimeState);
            sharedBlockContext_ = {};
        } else {
            ProcessSharedExpertGmm1Loop(gmmAddrInfo, problemShape, sharedGmmConfig, runtimeState);
        }
    } else {
        ProcessSharedExpertGmm1Loop(gmmAddrInfo, problemShape, sharedGmmConfig, runtimeState);
    }
    Gmm1UbActivationSync::EndSync(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
    gmm1PingPongIdx_ = 0U;
    if constexpr (g_coreType == AIV) {
        const bool hasSharedMte3Producer = GetSubBlockIdx() == 0U || SharedQuantConfig::AXW_MODE == AxWMode::A8W4;
        if (hasSharedMte3Producer) {
            // 等共享激活或权重转换搬出完成，后续 MoE 阶段才能通过 MTE2 复用 UB。
            SyncFuncStatic<HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::RunSharedExpertGmm2(const GMMAddrInfo &gmmAddrInfo,
                                                                             const ProblemShape &problemShape)
{
    // 共享 GMM1 和 GMM2 的所有量化模式都按 256-token group 交接 activation；
    // TopK weight prefetch 仍关闭，权重 L2 bypass 仍仅用于 A8W8。
    if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4 || SharedQuantConfig::AXW_MODE == AxWMode::A4W4) {
        RunGmm2A8W4<SharedActivationOutType, SharedWeightType, bfloat16_t, SharedQuantScaleType, SharedQuantScaleType,
                    GMM1_TILE_M, false, true, false, true, false>(
            sharedBlockContext_, problemShape, gmmAddrInfo, startBlockIdx_, gmmExecutionConfig_.blockJob,
            static_cast<uint32_t>(Get<M_VALUE>(problemShape)), 0U);
    } else {
        RunGmm2Generic<COMBINE_NO_QUANT, typename SharedQuantConfig::QuantOutType,
                       typename SharedQuantConfig::QuantOutType, bfloat16_t, SharedQuantScaleType, SharedQuantScaleType,
                       SharedWeight2Format != FORMAT_ND, false, GMM1_TILE_M, false, true, true>(
            problemShape, gmmAddrInfo, startBlockIdx_, gmmExecutionConfig_.blockJob, nullptr, true);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm2Loop(GMMAddrInfo &gmmAddrInfo,
                                                                                     const ProblemShape &problemShape)
{
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm2GlobalBuffer<SharedActivationOutType, SharedWeightType, SharedQuantScaleType,
                                           GMM1_TILE_M>(commonConfig_, gmmExecutionConfig_, params_.workspaceInfo,
                                                        sharedWeightTensorListAddrs_, gmmAddrInfo, sharedExpertIdx);
        RunSharedExpertGmm2(gmmAddrInfo, problemShape);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm2()
{
    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    GMMAddrInfo gmmAddrInfo{};
    if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4 || SharedQuantConfig::AXW_MODE == AxWMode::A4W4) {
        if (g_coreType == AscendC::AIC || GetSubBlockIdx() == 0U) {
            typename SharedA8W4Config::BlockContext::Block block(params_.tilingData->a8w4L1Layout);
            sharedBlockContext_ = {&block};
            ProcessSharedExpertGmm2Loop(gmmAddrInfo, problemShape);
            sharedBlockContext_ = {};
        } else {
            ProcessSharedExpertGmm2Loop(gmmAddrInfo, problemShape);
        }
    } else {
        ProcessSharedExpertGmm2Loop(gmmAddrInfo, problemShape);
    }
}

// W4 基类统一执行原 A8W4/A4W4 的专家 GMM2/Combine；派生模板只提供当前专家 slice。
// 非量化路径由 AIV1 在 GMM2 scheduler 内逐 tile 调用统一的 CombineTokenRange。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount, WaveCombineBufferConfig &combineBufferConfig, uint32_t &combineRowSequence,
    bool isFinalCombine, const A8W4BlockContext &pipeline)
{
    uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    uint32_t gmm2NTileCount = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));
    uint32_t problemTileCount = GetMGroupCountForRows(sliceTokenCount, GMM1_TILE_M) * gmm2NTileCount;
    if (!HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx)) {
        UpdateMoeExpertGmm2GlobalBuffer<MoeWeightType, ActivationQuantOutType, QuantScaleOutType>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmmAddrInfo,
            state, tokenStartIndexInExpert);
        ProblemShape sliceProblemShape = state.problemShape;
        Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
        RunGmm2A8W4<ActivationQuantOutType, MoeWeightType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                    GMM1_TILE_M, TopkWeightsPrefetch, false, false, true, CombineQuantMode == COMBINE_NO_QUANT>(
            pipeline, sliceProblemShape, gmmAddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob, expertTokenCount,
            tokenStartIndexInExpert, &params_);
    }

    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if (IsCombineDedupOn(params_.tilingData->dedupMode)) {
            // combine 去重：跨 WAVE 的专家等最后一个 slice 完成后按专家粒度独立消费（与量化路径
            // 同构；AIC 被跳过 problem 的情形也会到达本分支，per-expert 标记不漏发）。
            if (tokenStartIndexInExpert + sliceTokenCount >= expertTokenCount) {
                NotifyWaveGmm2Ready(waveCombineJob_, params_, state.expertIdx);
                if (isFinalCombine) {
                    if constexpr (g_coreType == AIV) {
                        if (GetSubBlockIdx() == 0U) {
                            // W4 prologue 完成后，末轮 AIV0 才能通过 MTE2 复用同一 UB 区域。
                            SyncFuncStatic<HardEvent::MTE3_MTE2, SYNC_EVENT_ID0>();
                        }
                    }
                    combineBufferConfig =
                        InitWaveCombineBuffersDedup<true>(commonConfig_, waveCombineScratch_, waveCombineDedupScratch_);
                    RunWaveCombineStageDedup<true>(commonConfig_, waveCombineJob_, combineBufferConfig,
                                                   waveCombineScratch_, waveCombineDedupScratch_, params_, state,
                                                   state.expertIdx, combineRowSequence);
                } else {
                    RunWaveCombineStageDedup<false>(commonConfig_, waveCombineJob_, combineBufferConfig,
                                                    waveCombineScratch_, waveCombineDedupScratch_, params_, state,
                                                    state.expertIdx, combineRowSequence);
                }
            }
            return;
        }
        // 基线路径：AIV1 已在上述 GMM2 调用中逐 tile 消费当前 slice；无需在外部重建 scheduler。
        return;
    }

    // 量化 Combine 保持原 W4 语义：跨 WAVE 的专家必须等最后一个 slice 完成后再消费。
    if (tokenStartIndexInExpert + sliceTokenCount < expertTokenCount) {
        return;
    }

    NotifyWaveGmm2Ready(waveCombineJob_, params_, state.expertIdx);
    UpdateMoeExpertCombineGlobalBuffer(params_.workspaceInfo, gmmAddrInfo, state);
    if (isFinalCombine) {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 0U) {
                // W4 prologue 完成后，末轮 AIV0 才能通过 MTE2 复用同一 UB 区域。
                SyncFuncStatic<HardEvent::MTE3_MTE2, SYNC_EVENT_ID0>();
            }
        }
        combineBufferConfig =
            PrepareFinalWaveCombineBuffers<CombineQuantMode>(commonConfig_, combineBufferConfig, waveCombineScratch_);
        RunWaveCombineStage<CombineQuantMode, true>(commonConfig_, waveCombineJob_, combineBufferConfig,
                                                    waveCombineScratch_, params_, gmmAddrInfo, state, state.expertIdx,
                                                    combineRowSequence);
        return;
    }
    RunWaveCombineStage<CombineQuantMode>(commonConfig_, waveCombineJob_, combineBufferConfig, waveCombineScratch_,
                                          params_, gmmAddrInfo, state, state.expertIdx, combineRowSequence);
}

// 从 workspace 读取 token 数并准备 GMM 专家状态；GMM1 可按需先等待 token count ready。
template <TemplateMegaMoeTypeClass>
template <bool WaitForTokenCountReady>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::PrepareGmmExpertState(ExpertLoopState &state,
                                                                               uint32_t expertIdx)
{
    if constexpr (WaitForTokenCountReady) {
        if (GetSubBlockIdx() == 0U && !state.expertCountTableReady) {
            WaitForMoeExpertTokenCountReady(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_, 0U);
            state.expertCountTableReady = true;
        }
    }
    uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
        params_.workspaceInfo.expertRecvTokenCountPtr, countWorkspace_, commonConfig_.moeExpertPerRank, expertIdx);
    UpdateExpertLoopState(state, expertIdx, expertTokenCount);
}

// 仅由 AIV 调用：先发送下标，等待本卡全部 AIV 完成 reset 和量化，再发送 count。
// 空专家范围不发送数据，但仍须消费一次量化完成事件。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendTopkIdsIndexAndCount(const WorkRange &ownedExpertRange)
{
    if (ownedExpertRange.count > 0U) {
        SendTopkIdsIndexForExperts(ownedExpertRange, commonConfig_, g_winRankAddr_, sendMaskConfig_, sendMaskScratch_);
    }
    // 通知在共享量化之后发布，覆盖此前的 reset 和全部量化写回；空专家任务也参与。
    CrossCoreWaitFlag<ALL_AICORE_SYNC_MODE, PIPE_MTE3>(MTE_QUANT_READY_FLAG);
    // 将 reset 和量化完成条件传递给 Scalar，在输入准备末尾消费。
    SetFlag<HardEvent::MTE3_S>(SYNC_EVENT_ID3);
    // AIV0 转发全 AIV 完成条件；有共享专家时，不必等 AIV1 发完下标才启动 AIC。
    if (GetSubBlockIdx() == 0U) {
        CrossCoreSetFlag<AIC_SINGLE_AIV_SYNC_MODE, PIPE_MTE3>(INPUT_READY_FLAG);
    }
    if (ownedExpertRange.count > 0U) {
        PipeBarrier<PIPE_MTE3>();
        const int32_t syncCount = GetSyncCount(params_.peermemInfo.rankSyncInWorldPtr, aivCoreIdx_);
        SendTopkIdsCountForExperts(ownedExpertRange, commonConfig_, syncCount, g_winRankAddr_, sendMaskConfig_,
                                   sendMaskScratch_);
        if (sharedExpertNum_ > 0U || GetSubBlockIdx() == 1U) {
            // 有共享时仅 AIV1 发送；后续共享激活或 Dispatch 都先通过 MTE2 复用源 UB。
            SyncFuncStatic<HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
        } else if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4 || TopkWeightsPrefetch) {
            // 无共享 AIV0：权重转换或 GM 激活路径先通过 MTE2 搬入数据。
            SyncFuncStatic<HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
        } else {
            // 无共享 AIV0：直接 UB 激活路径由 Vector 复用工作区。
            SyncFuncStatic<HardEvent::MTE3_V, SYNC_EVENT_ID3>();
        }
    }
    if (GetSubBlockIdx() == 0U) {
        // FIX 的 UB 输出只写入 AIV0，因此仅由 AIV0 通知 AIC，避免覆盖尚在发送的数据。
        CrossCoreSetFlag<AIC_SINGLE_AIV_SYNC_MODE, PIPE_MTE3>(INPUT_UB_FREE_FLAG);
    }
}

// 只有量化类型不同且启用了共享专家，才传入独立共享量化参数。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::QuantizeInputTokens(const WorkRange &tokenRange)
{
    const TokenQuantParams<MoeQuantMode, QuantOutType, ActivationType> moeQuant{quantProcessConfig_, quantScratch_};
    auto quantize = [&](const auto &moeQuant, const auto &...sharedQuant) __attribute__((cce_aicore))
    {
        QuantizeLocalTokens<TopkWeightsType, TopkWeightsPrefetch>(tokenRange, commonConfig_, params_.probsGmAddr,
                                                                  moeQuant, sharedQuant...);
    };

    if (sharedExpertNum_ > 0U && !SHARED_INPUT_REUSES_MOE_QUANT) {
        const TokenQuantParams<SharedQuantMode, typename SharedQuantConfig::QuantOutType, SharedActivationType>
            sharedQuant{sharedQuantProcessConfig_, sharedQuantScratch_};
        quantize(moeQuant, sharedQuant);
    } else {
        quantize(moeQuant);
    }
}

// 仅由全部 AIV 调用：block 0 的 AIV1 输出数量，全部 AIV0 清零原始 count 表。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExportAndResetExpertCounts()
{
    WorkRange resetRange{};
    if (GetSubBlockIdx() == 0U) {
        resetRange = GetBalancedWorkRange(static_cast<uint32_t>(PEERMEM_MTE_COUNT_REGION_SIZE / sizeof(int32_t)),
                                          {.jobIndex = blockIdx_, .totalJobs = blockNum_});
    } else if (blockIdx_ == 0U) {
        ExportExpertTokenCounts(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
    }
    // 输出核先搬出再消费事件；清零核在写 GM 前等待全部读取完成，每个 AIV 每轮 wait 一次。
    CrossCoreWaitFlag<ALL_AICORE_SYNC_MODE, PIPE_MTE3>(COUNT_TABLE_READ_DONE_FLAG);
    if (resetRange.count > 0U) {
        // AIV0 可能参与末尾的量化 Combine，填零前等待旧 MTE3 读完复用的 UB。
        SyncFuncStatic<HardEvent::MTE3_V, SYNC_EVENT_ID2>();
        Duplicate<int32_t>(resetTensor_, 0, resetTensor_.GetSize());
        SyncFuncStatic<HardEvent::V_MTE3, SYNC_EVENT_ID2>();
        ResetWorkspaceRegion(resetRange, params_.peermemInfo.expertCountRecvPtr, resetBatchElementCount_, resetTensor_);
    }
    // 输出和完整 count 区清零均完成后，才进入共享 GMM2 及出口跨卡握手。
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

// 仅由 AIV 调用，AIC 在 ProcessWave 中等待输入就绪。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessInputPreparationStage()
{
    SendAndQuantBuffInit();
    ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, resetBatchElementCount_, resetTensor_);
    const WorkRange tokenRange = GetBalancedWorkRange(commonConfig_.tokenNum, aivJob_);
    if (tokenRange.count > 0U) {
        if constexpr (Std::IsSame<XType, bfloat16_t>::value) {
            QuantizeInputTokens(tokenRange);
        } else {
            PackPreQuantizedLocalTokens<XType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
                tokenRange, commonConfig_, params_, quantProcessConfig_, params_.peermemInfo.quantTokenScalePtr,
                quantScratch_);
            // 预量化共享专家与 MoE 类型一致；64 对齐时直接读取原始输入，
            // 否则复用这里的 MoE 拼接结果，无需独立打包共享输入。
        }
    }
    // 每个物理 AIV 推进入口轮次一次，空 token 核也参与。
    IncrementSyncCount(params_.peermemInfo.rankSyncInWorldPtr, aivCoreIdx_);
    // 同一 MTE3 流水先完成 reset、MoE 量化及可选共享量化写回，再发布完成通知。
    // 空 token 核也必须报告，保证全 AIV 事件配对。
    CrossCoreSetFlag<ALL_AICORE_SYNC_MODE, PIPE_MTE3>(MTE_QUANT_READY_FLAG);

    // 无共享时由全部 AIV 分配专家；有共享时仅 AIV1 分配，AIV0 保持空范围。
    WorkRange ownedExpertRange{};
    const uint32_t totalExperts = commonConfig_.worldSize * commonConfig_.moeExpertPerRank;
    if (sharedExpertNum_ == 0U) {
        ownedExpertRange = GetBalancedWorkRange(totalExperts, aivJob_);
    } else if (GetSubBlockIdx() == 1U) {
        ownedExpertRange = GetBalancedWorkRange(totalExperts, {.jobIndex = blockIdx_, .totalJobs = blockNum_});
    }
    /*
     * AIV 先发送下标；发送 count 前等待全部 AIV 完成 reset 和量化，由 AIV0 向 AIC 通知输入就绪。
     * 有共享专家时 AIV0 不发送下标，AIC 可与 AIV1 的下标发送并行；FIX 输出另等 AIV0 释放 UB。
     * 共享专家使用 A8W8/A4W4：AIC/AIV0 执行共享 GMM1 和激活，AIV1 执行下标/count 发送。
     * 共享专家使用 A8W4：AIC/AIV0 执行共享 GMM1 和权重转换，AIV1 完成下标/count 发送后执行共享激活。
     * count 携带本轮 epoch，接收端逐项确认远端输入就绪；共享计算不等待入口跨卡握手。
     */
    SendTopkIdsIndexAndCount(ownedExpertRange);
    // Scalar 读取 GM ready 状态前确认 reset 已完成；此事件在 count 发送前发布。
    WaitFlag<HardEvent::MTE3_S>(SYNC_EVENT_ID3);
}

// 仅由 AIC 调用，分别承接输入搬运、Scalar 状态读取和 FIX 输出的依赖。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::WaitForInputPreparation()
{
    CrossCoreWaitFlag<AIC_SINGLE_AIV_SYNC_MODE, PIPE_MTE2>(INPUT_READY_FLAG);
    // 除输入搬运外，Scalar 的状态轮询也必须在全 AIV reset 完成之后。
    SyncFuncStatic<HardEvent::MTE2_S, SYNC_EVENT_ID3>();
    // 各路径统一消费 AIV0 的 UB 释放通知，防止 FIX 覆盖尚在发送的数据。
    CrossCoreWaitFlag<AIC_SINGLE_AIV_SYNC_MODE, PIPE_FIX>(INPUT_UB_FREE_FLAG);
}

/*
 * 所有 MTE Wave 路径共用相同的阶段边界，派生类只负责 MoE 专家 Wave 的具体编排。
 */
template <TemplateMegaMoeTypeClass>
template <typename Derived>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessWave(Derived &derived)
{
    // 保存入口时的溢出模式，计算期间关闭溢出检查。
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);

    // 阶段 1：准备动态量化或预量化输入、清零、本卡输入同步及下标/count 发送。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::INPUT_PREPARE);
    if constexpr (g_coreType == AIV) {
        ProcessInputPreparationStage();
    }
    // AIC 进入计算前等待输入就绪及工作区释放。
    if constexpr (g_coreType == AIC) {
        WaitForInputPreparation();
    }
    // 共享 A8W4 写 GM，由 AIV1 消费；独立事件使剩余 ACK 不阻塞 MoE GMM 启动。
    Gmm1ActivationSync sharedGmm1ActivationSync(1U, GmmEventPair::SHARED_GMM1);
    if (sharedExpertNum_ > 0U) {
        // 阶段 2：可选的共享专家 GMM1 及 Activation。
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM1);
        ProcessSharedExpertGmm1(sharedGmm1ActivationSync);
    }

    // 阶段 3：由派生类编排 MoE 专家的 Dispatch、GMM1/Activation 和 GMM2/Combine。
    Gmm1ActivationSync gmm1ActivationSync(MoeQuantConfig::AXW_MODE == AxWMode::A8W4 ? 1U : 0U);
    Gmm2CombineSync gmm2CombineSync;
    derived.ProcessMoeExpertStages(gmm1ActivationSync, gmm2CombineSync);
    if constexpr (g_coreType == AIV) {
        ExportAndResetExpertCounts();
    }
    if (sharedExpertNum_ > 0U) {
        // 阶段 4：可选的共享专家 GMM2，其结果由输出聚合阶段合并。
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM2);
        ProcessSharedExpertGmm2();
    }

    // 阶段 5：所有 rank 完成 Combine 结果发送后，AIV 执行输出聚合。
    if constexpr (g_coreType == AIV) {
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::CROSS_RANK_SYNC_OUTPUT);
        CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::UNPERMUTE);
        MegaMoeUnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteConfig_, commonConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
    }

    // 三组 GM 硬同步事件独立，末尾统一回收 ACK，避免阻塞后续共享计算。
    // 各阶段的 UB 复用等待和逐 tile 额度限制仍在原位。
    sharedGmm1ActivationSync.EndSync();
    gmm1ActivationSync.EndSync();
    gmm2CombineSync.EndSync();

    // 恢复入口时的溢出模式。
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::COMPLETE);
}

} // namespace MegaMoeImpl
#undef TemplateMegaMoeTypeClass
#undef TemplateMegaMoeTypeFunc
#endif
