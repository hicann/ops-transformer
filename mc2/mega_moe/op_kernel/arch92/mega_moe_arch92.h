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
 * \file mega_moe_arch92.h
 * \brief
 */

#ifndef MEGA_MOE_ARCH92_H
#define MEGA_MOE_ARCH92_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../arch35_92/common/mega_moe_types.h"
#include "../arch35_92/common/mega_moe_gm_layout.h"
#include "../arch35_92/common/mega_moe_arch_policy.h"
#include "../arch35_92/common/mega_moe_utils.h"
#include "../arch35_92/common/mega_moe_arch35_92_exception_dump_policy.h"
#include "../arch35_92/blaze/epilogue/mega_moe_block_epilogue_activation_mx_quant.h"
#include "../arch35_92/stage/mega_moe_token_quant.h"
#include "../arch35_92/stage/mega_moe_send_mask.h"
#include "../arch35_92/stage/mega_moe_workspace_reset.h"
#include "../arch35_92/stage/mega_moe_token_dispatch_dedup.h"
#include "stage/mega_moe_gmm1_activation_arch92.h"
#include "stage/mega_moe_gmm2_combine_arch92.h"
#include "../arch35_92/stage/mega_moe_gmm2_combine_dedup.h"
#include "../arch35_92/stage/mega_moe_unpermute.h"
#include "../../../common/op_kernel/quantize_functions.h"

namespace MegaMoeImpl {

using namespace AscendC;

#define TemplateMegaMoeTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename MoeWeightType, int32_t MoeQuantMode, \
        typename SharedWeightType, int32_t SharedQuantMode, int32_t MoeWeight1Format, int32_t MoeWeight2Format, \
        int32_t SharedWeight1Format, int32_t SharedWeight2Format
#define TemplateMegaMoeTypeFunc \
    XType, OutputType, TopkWeightsType, MoeWeightType, MoeQuantMode, SharedWeightType, SharedQuantMode, \
        MoeWeight1Format, MoeWeight2Format, SharedWeight1Format, SharedWeight2Format

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    using MoeQuantConfig = QuantConfig<MoeWeightType, MoeQuantMode>;
    using SharedQuantConfig = QuantConfig<SharedWeightType, SharedQuantMode>;
    static constexpr bool SHARED_GMM1_USES_GM = SharedQuantConfig::AXW_MODE == AxWMode::A8W4;
    using QuantOutType = typename MoeQuantConfig::QuantOutType;
    using ActivationType = typename MoeQuantConfig::QuantStorageType;
    using QuantScaleOutType = typename MoeQuantConfig::QuantScaleType;
    using SharedActivationType = typename SharedQuantConfig::QuantStorageType;
    using SharedActivationOutType = typename SharedQuantConfig::ActivationQuantOutType;
    static constexpr bool SHARED_FP4_SWIGLU_OUTPUT = Std::IsSame<SharedActivationOutType, fp4x2_e2m1_t>::value;
    using SharedQuantScaleType = typename SharedQuantConfig::QuantScaleType;
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, const MegaMoeTilingData* tilingData,
                                GM_ADDR tilingGM);

private:
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;
    using UnpermuteBufferConfig = MegaMoeUnpermuteBufferConfig;

    __aicore__ inline void InitStageConfigs(const MegaMoeTilingData* tilingData);
    __aicore__ inline void InitEpilogueAndCommonConfig(const MegaMoeTilingData* tilingData);
    __aicore__ inline void InitInputPrepareConfigs();
    __aicore__ inline void InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                    int32_t activationFlagSlotsPerExpert);
    __aicore__ inline void InitGmmConfigs();
    __aicore__ inline void InitTokenUnpermuteConfig();
    __aicore__ inline uint32_t InitPreQuantScratchTensors(uint32_t scratchAddr);
    __aicore__ inline uint32_t InitQuantScratchTensors(uint32_t mxTempTensorAddr);

protected:
    using A8W4BlockContext =
        typename GmmKernel::Config<false, typename MoeQuantConfig::ActivationQuantOutType, MoeWeightType, bfloat16_t,
                                   QuantScaleOutType, QuantScaleOutType>::BlockContext;
    using SharedA8W4Config = GmmKernel::Config<false, SharedActivationOutType, SharedWeightType, bfloat16_t,
                                               SharedQuantScaleType, SharedQuantScaleType>;
    typename SharedA8W4Config::BlockContext sharedBlockContext_{};
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void PrepareMoeExpertTokenCountTable();
    __aicore__ inline void ExportExpertTokenCounts();
    __aicore__ inline void EnterSteadyDispatch();
    __aicore__ inline void InitQuantTokenBufferConfig();
    __aicore__ inline UnpermuteBufferConfig InitTokenUnpermuteBuffers();
    __aicore__ inline void QuantizeInputTokens(const WorkRange& tokenRange);
    __aicore__ inline void PrepareLocalInput();
    __aicore__ inline void ProcessInputPreparationStage(SharedGmmStartCondition sharedGmmStartCondition);
    __aicore__ inline void WaitForInputPreparation(SharedGmmStartCondition sharedGmmStartCondition);
    __aicore__ inline SharedGmmStartCondition GetSharedGmmStartCondition() const;
    __aicore__ inline void ProcessTokenUnpermuteStage();
    __aicore__ inline void SyncBeforeSharedExpertGmm1();
    template <typename Derived>
    __aicore__ inline void SyncBeforeMoeExpertStages(Derived& derived, SharedGmmStartCondition sharedGmmStartCondition);
    __aicore__ inline void ExportAndResetExpertCounts();
    __aicore__ inline void ConfigureSharedGmm1Input(GMMAddrInfo& gmmAddrInfo, GmmExecutionConfig& gmmConfig,
                                                    Gmm1ActivationSync& sharedGmm1ActivationSync) const;
    __aicore__ inline void RunSharedExpertGmm1Activation(const GMMAddrInfo& gmmAddrInfo,
                                                         const ProblemShape& problemShape,
                                                         const GmmExecutionConfig& gmmConfig,
                                                         GmmRuntimeState& runtimeState, uint32_t sharedExpertIdx,
                                                         void* blockMmadContext);
    __aicore__ inline void RunSharedExpertGmm2(const GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape,
                                               void* blockMmadContext);
    __aicore__ inline void ProcessSharedExpertGmm1Loop(GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape,
                                                       const GmmExecutionConfig& sharedGmmConfig,
                                                       GmmRuntimeState& runtimeState, void* blockMmadContext);
    __aicore__ inline void ProcessSharedExpertGmm1(Gmm1ActivationSync& sharedGmm1ActivationSync,
                                                   void* blockMmadContext);
    __aicore__ inline void ProcessSharedExpertGmm2Loop(GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape,
                                                       void* blockMmadContext);
    __aicore__ inline void ProcessSharedExpertGmm2(void* blockMmadContext);
    template <typename Derived>
    __aicore__ inline void ProcessWave(Derived& derived, void* blockMmadContext);

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
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
    uint32_t blockAivNum_ = MegaMoeHardwarePolicy::GetAivBlockNum();
    uint32_t blockIdx_ = MegaMoeHardwarePolicy::GetAicBlockIdx();
    uint32_t aivCoreIdx_ = MegaMoeHardwarePolicy::GetAivBlockIdx();
    uint16_t gmm1PingPongIdx_ = 0;
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

    QuantProcessScratch<typename MoeQuantConfig::QuantStorageType> quantScratch_;
    QuantProcessScratch<typename SharedQuantConfig::QuantStorageType> sharedQuantScratch_;
    PreQuantScratch preQuantScratch_;
    SendMaskScratch sendMaskScratch_;
    __ubuf__ int32_t* resetTensor_ = nullptr;

    using ActivationQuantOutType = typename MoeQuantConfig::ActivationQuantOutType;
    static constexpr uint32_t C_ELEMS_PER_BYTE = MoeQuantConfig::C_ELEMS_PER_BYTE;

    using BlockEpilogue = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, L1_TILE_N>;
    using SharedBlockEpilogue = BlockEpilogueActivationMxQuant<SharedActivationOutType, bfloat16_t, L1_TILE_N>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
    TokenDispatchScratch<ActivationType> tokenDispatchScratch_;
    WaveCombineScratch waveCombineScratch_;
    // 非量化 combine 去重的追加 UB 视图与跨行事件武装状态（排空时消费）。
    WaveCombineDedupScratch waveCombineDedupScratch_;
    TokenUnpermuteScratch tokenUnpermuteScratch_;
    MegaMoeImpl::ExceptionDumpEngine exceptionDump_;
    __gm__ MegaMoeImpl::GmmLoopCount* gmmLoopCount_{nullptr};
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitInputPrepareConfigs()
{
    aivJob_ = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_};
    quantProcessConfig_ =
        CreateQuantProcessConfig<typename MoeQuantConfig::QuantStorageType, typename MoeQuantConfig::QuantScaleType,
                                 MoeQuantConfig::A_ELEMS_PER_BYTE>(k_, params_);
    if constexpr (SHARED_INPUT_REUSES_MOE_QUANT) {
        sharedQuantProcessConfig_ = quantProcessConfig_;
    } else {
        sharedQuantProcessConfig_ = CreateQuantProcessConfig<typename SharedQuantConfig::QuantStorageType,
                                                             typename SharedQuantConfig::QuantScaleType,
                                                             SharedQuantConfig::A_ELEMS_PER_BYTE, false>(k_, params_);
    }
    // 共享专家启用时仅 AIV1 计算发送 topK 有效下标，其余情况保留全 AIV 分工。
    const uint32_t topkValidIndexCoreIdx = sharedExpertNum_ > 0U ? blockIdx_ : aivCoreIdx_;
    sendMaskConfig_ = CreateSendMaskConfig(params_, topkValidIndexCoreIdx);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                                                  int32_t activationFlagSlotsPerExpert)
{
    countWorkspace_ = {.blockIdx = blockIdx_, .blockNum = static_cast<uint32_t>(params_.tilingData->compute.aicNum)};
    syncWorkspaceLayout_ = {.dispatchFlagSlotCountPerExpert = dispatchFlagSlotsPerExpert,
                            .activationFlagSlotCountPerExpert = activationFlagSlotsPerExpert};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitGmmConfigs()
{
    gmmExecutionConfig_ = {.ring = {},
                           .blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
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
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitEpilogueAndCommonConfig(
    const MegaMoeTilingData* tilingData)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const uint32_t activationOutputDivisor = GetActivationOutputDivisor(tilingData->actMode);
    epilogueOp_.Init({.yGmAddr = workspace.routed.gmm2Input.data.get_ptr(buffer),
                      .yScaleGmAddr = workspace.routed.gmm2Input.scale.get_ptr(buffer),
                      .clampLimit = tilingData->clampLimit,
                      .actMode = tilingData->actMode,
                      .actSubMode = tilingData->actSubMode,
                      .activationAlpha = tilingData->activationAlpha,
                      .activationBeta = tilingData->activationBeta,
                      .topkWeightsPrefetch = tilingData->isTopkWeightsPrefetch});
    commonConfig_ = {.rankId = rankId_,
                     .worldSize = worldSize_,
                     .moeExpertPerRank = moeExpertPerRank_,
                     .sharedExpertNum = sharedExpertNum_,
                     .tokenNum = static_cast<uint32_t>(tilingData->compute.bs),
                     .topK = static_cast<uint32_t>(tilingData->compute.topK),
                     .tokenHiddenDim = k_,
                     .gmm1OutputDim = static_cast<uint32_t>(tilingData->compute.hiddenDim),
                     .activationOutputDivisor = activationOutputDivisor};
}

// 初始化各阶段配置及同步标志槽数。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitStageConfigs(const MegaMoeTilingData* tilingData)
{
    InitEpilogueAndCommonConfig(tilingData);
    const int64_t maxOutput = static_cast<int64_t>(static_cast<uint32_t>(tilingData->compute.maxOutputSize));
    const int64_t tileM = static_cast<int64_t>(GMM1_TILE_M);
    int32_t dispatchFlagSlotsPerExpert = static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, tileM)) * INT_CACHELINE;
    int32_t activationFlagSlotsPerExpert =
        static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, static_cast<int64_t>(L1_TILE_M_256))) * INT_CACHELINE;
    InitInputPrepareConfigs();
    tokenDispatchConfig_ = CreateTokenDispatchConfig(params_, quantProcessConfig_);
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
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, const MegaMoeTilingData* tilingData, GM_ADDR tilingGM)
{
    if constexpr (g_coreType == AIV) {
        InitCopyPadding();
    }
    k_ = static_cast<uint32_t>(tilingData->compute.h);
    worldSize_ = static_cast<uint32_t>(tilingData->compute.epWorldSize);
    moeExpertPerRank_ = static_cast<uint32_t>(tilingData->compute.moeExpertPerRank);
    sharedExpertNum_ = static_cast<uint32_t>(tilingData->compute.sharedExpertNum);
    mGroupsPerWave_ = static_cast<uint32_t>(tilingData->compute.mGroupsPerWave);
    gmm1PingPongIdx_ = 0;
    startBlockIdx_ = 0;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
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
    (void)workspaceGM;
    params_.communication.base = g_winRankAddr_[rankId_];
    params_.workspace.layout = WorkspaceLayout(
        tilingData->compute, WorkspaceLayout::Arch92{tilingData->moeGmmMode, tilingData->sharedGmmMode});
    params_.communication.layout = PeermemInfo(tilingData->comm, params_.workspace.layout.workspaceSize);
    params_.workspace.base = params_.communication.layout.workspace.get_ptr(params_.communication.base);
    params_.tilingData = tilingData;
    InitStageConfigs(tilingData);

    gmmLoopCount_ = MegaMoeImpl::RegisterMegaMoeExceptionDump(exceptionDump_, dumpBase, tilingGM, tilingData,
                                                              params_.communication.layout, params_.communication.base,
                                                              reinterpret_cast<GM_ADDR>(&mc2Context_->epRankId));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::EnterSteadyDispatch()
{
    tokenDispatchConfig_.bufferConfig.bufferCount = MIN_DISPATCH_BUFFER_COUNT;
}

// 普通模板 Token Dispatch 使用的 UB/GM 视图。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchBuffInit()
{
    const uint32_t indexBytes = params_.tilingData->topkIndexTypeBytes;
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const TokenDispatchConfig& context = tokenDispatchConfig_;
    TokenDispatchScratch<ActivationType>& scratch = tokenDispatchScratch_;
    scratch.expertRecvCountsGm = reinterpret_cast<__gm__ int32_t*>(workspace.routing.recvCounts.get_ptr(buffer));
    if constexpr (g_coreType == AIC) {
        return;
    }

    const MegaMoeDispatchBufferConfig& bufferConfig = context.bufferConfig;
    scratch.revTokenElemCnt = commonConfig_.tokenHiddenDim / A_ELEMS_PER_BYTE;
    scratch.revScaleElemCnt = Ops::Base::CeilDiv(static_cast<int64_t>(commonConfig_.tokenHiddenDim),
                                                 static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                              MXFP_MULTI_BASE_SIZE;
    static_assert(GetVecLen() <= DISPATCH_CUMSUM_VECTOR_BYTES, "persistent count tables must cover a full vector load");
    constexpr uint32_t cumsumInfoTensorSize = ARCH92_DISPATCH_CUMSUM_BYTES;

    // 常驻表固定预留 12 KiB：cumsum 8 KiB，专家计数 4 KiB；基址由 Host 按实际 UB 容量计算。
    const uint32_t cumsumInfoTensorAddr = params_.tilingData->dispatchTableUbBaseAddr;
    scratch.cumsumInfoTensor = reinterpret_cast<__ubuf__ int32_t*>(cumsumInfoTensorAddr);
    // topK 有效下标 route 已消除 mask 扫描，只保留一份接收 index batch。
    // Tensor 用途：接收 topkIndex 的当前 batch。
    uint32_t validTopkIndexTensorAddr = 0U;
    uint32_t validTopkIndexTensorSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(bufferConfig.routeItemsPerBatch) * static_cast<int64_t>(indexBytes),
                             static_cast<int64_t>(ALIGN_32));
    scratch.validTopkIndexTensor = {reinterpret_cast<__ubuf__ uint8_t*>(validTopkIndexTensorAddr), indexBytes};
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
    constexpr uint32_t expertTokenNumsOutTensorSize = ARCH92_DISPATCH_EXPERT_COUNT_BYTES;
    uint32_t metaInfoTensorAddr = expertTokenNumsOutTensorAddr;
    expertTokenNumsOutTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    scratch.expertTokenNumsOutTensor = reinterpret_cast<__ubuf__ int32_t*>(expertTokenNumsOutTensorAddr);
    // Tensor 用途：CopyTokensAndMetaForDispatch 中的 metaInfo 环形缓冲区，逐 token 即时写入 GM；
    // Tensor 大小：bufferCount * 32B，与 copyTmp 槽位和事件编号一一对应。
    uint32_t metaInfoTensorSize = static_cast<uint32_t>(bufferConfig.bufferCount) * INT32_PER_256B * sizeof(int32_t);
    scratch.metaInfoTensor = reinterpret_cast<__ubuf__ int32_t*>(metaInfoTensorAddr);
    scratch.dedupUbBaseAddr = static_cast<uint32_t>(Ops::Base::CeilAlign(
        static_cast<uint64_t>(metaInfoTensorAddr + metaInfoTensorSize), static_cast<uint64_t>(ALIGN_512)));
}

// A6 的前缀表常驻 UB，只发布专家计数和准备完成标志。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::PrepareMoeExpertTokenCountTable()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    if constexpr (g_coreType == AIC) {
        return;
    }
    LoadAndComputeExpertCountTable(commonConfig_, params_, tokenDispatchScratch_);
    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace_, commonConfig_.moeExpertPerRank, 0U, true);
    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
    CopyUbToGm((tokenDispatchScratch_.expertRecvCountsGm + (countOffset)),
               tokenDispatchScratch_.expertTokenNumsOutTensor,
               commonConfig_.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)));
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();

    __gm__ int32_t* countTableReady = reinterpret_cast<__gm__ int32_t*>(workspace.sync.countsReady.get_ptr(buffer)) +
                                      static_cast<uint64_t>(countWorkspace_.blockIdx) * INT_CACHELINE;
    GmWrite(countTableReady, static_cast<int32_t>(1));
}

// A6 直接使用常驻 UB 的专家计数表导出结果。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExportExpertTokenCounts()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (countWorkspace_.blockIdx != 0U) {
        return;
    }

    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();

    __gm__ int32_t* expertTokenNumsOut = nullptr;
    expertTokenNumsOut = reinterpret_cast<__gm__ int32_t*>(params_.expertTokenNumsOutGmAddr);
    CopyUbToGm(expertTokenNumsOut, tokenDispatchScratch_.expertTokenNumsOutTensor,
               commonConfig_.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)));
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitQuantTokenBufferConfig()
{
    const uint8_t combineQuantMode = params_.tilingData->combineQuantMode;
    quantTokenBufferConfig_ = {.quantTokenSizeBytes = 0U};
    if constexpr (g_coreType == AIV) {
        if (combineQuantMode != COMBINE_NO_QUANT) {
            quantTokenBufferConfig_ = CreateQuantTokenBufferConfig(k_);
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline uint32_t MegaMoe<TemplateMegaMoeTypeFunc>::InitPreQuantScratchTensors(uint32_t scratchAddr)
{
    const uint32_t stageBytes = params_.tilingData->preQuantStageBytes;
    preQuantScratch_.preQuantStage0 = reinterpret_cast<__ubuf__ uint8_t*>(scratchAddr);
    preQuantScratch_.preQuantStage1 = reinterpret_cast<__ubuf__ uint8_t*>(scratchAddr + stageBytes);
    return scratchAddr + DOUBLE_BUFFER * stageBytes;
}

// 动态量化 scratch：MoE/shared 共用输入和 mxTemp，输出分别分配。
template <TemplateMegaMoeTypeClass>
__aicore__ inline uint32_t MegaMoe<TemplateMegaMoeTypeFunc>::InitQuantScratchTensors(uint32_t mxTempTensorAddr)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    uint32_t mxTempTensorSize = 2 * 1024;
    uint32_t xOutTensorSize = quantProcessConfig_.quantTokenScaleAlignBytes;
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);

    quantScratch_.inputGm = reinterpret_cast<__gm__ bfloat16_t*>(params_.aGmAddr);
    quantScratch_.outputGm = reinterpret_cast<__gm__ uint8_t*>(peermem.quantTokenScale.get_ptr(commBuffer));
    quantScratch_.mxTempTensor = reinterpret_cast<__ubuf__ uint16_t*>(mxTempTensorAddr);
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantScratch_.xOutTensor0 = reinterpret_cast<__ubuf__ ActivationType*>(xOutTensorAddr1);
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantScratch_.xOutTensor1 = reinterpret_cast<__ubuf__ ActivationType*>(xOutTensorAddr2);
    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantScratch_.xInTensor0 = reinterpret_cast<__ubuf__ bfloat16_t*>(xInAlignAddr1);
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantScratch_.xInTensor1 = reinterpret_cast<__ubuf__ bfloat16_t*>(xInAlignAddr2);
    uint32_t quantEndAddr = xInAlignAddr2 + xInAlignSize;
    if constexpr (!SHARED_INPUT_REUSES_MOE_QUANT) {
        if (sharedExpertNum_ > 0U) {
            sharedQuantScratch_.outputGm =
                reinterpret_cast<__gm__ uint8_t*>(workspace.shared.gmm1Input.data.get_ptr(buffer));
            sharedQuantScratch_.inputGm = quantScratch_.inputGm;
            sharedQuantScratch_.xInTensor0 = quantScratch_.xInTensor0;
            sharedQuantScratch_.xInTensor1 = quantScratch_.xInTensor1;
            sharedQuantScratch_.mxTempTensor = quantScratch_.mxTempTensor;
            uint32_t sharedOutputBytes = sharedQuantProcessConfig_.quantTokenScaleAlignBytes;
            sharedQuantScratch_.xOutTensor0 = reinterpret_cast<__ubuf__ SharedActivationType*>(quantEndAddr);
            quantEndAddr += sharedOutputBytes;
            sharedQuantScratch_.xOutTensor1 = reinterpret_cast<__ubuf__ SharedActivationType*>(quantEndAddr);
            quantEndAddr += sharedOutputBytes;
        }
    }

    /*
     * 输入尾部、临时乘法 scale 尾部和输出 scale 补偶槽必须保持为零，保护 H%64==32 的尾块。
     * 两种量化共用 mxTemp，有效 scale 数相同；独立输出避免不同格式覆盖彼此的 padding。
     * 每个 AIV 在每次 launch 初始化一次，临时 scale 的 padding 在两次量化中均不写入。
     */
    LocalTensor<int16_t> quantScratchSpan(TPosition::VECCALC, mxTempTensorAddr,
                                          (quantEndAddr - mxTempTensorAddr) / sizeof(int16_t));
    Duplicate<int16_t>(quantScratchSpan, 0, static_cast<int32_t>((quantEndAddr - mxTempTensorAddr) / sizeof(int16_t)));
    PipeFence<PIPE_V, PIPE_MTE2, SYNC_EVENT_ID2>();
    return quantEndAddr;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    const uint32_t indexBytes = params_.tilingData->topkIndexTypeBytes;
    const auto& workspace = params_.workspace.layout;
    sendMaskScratch_.topkIdsGm = reinterpret_cast<__gm__ int32_t*>(params_.expertIdxGmAddr);

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 = static_cast<uint64_t>((workspace.reset.flags.sizeBytes / SIZE_INT_32));
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
    const SendMaskBufferConfig& bufferConfig = sendMaskConfig_.bufferConfig;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，两种 index 编码均满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor = reinterpret_cast<__ubuf__ int32_t*>(topkIdsTensorAddr);

    uint32_t topkIdsIndexTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    uint32_t topkIdsIndexTensorSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(indexBytes);
    sendMaskScratch_.topkIdsIndexTensor = {reinterpret_cast<__ubuf__ uint8_t*>(topkIdsIndexTensorAddr), indexBytes};

    uint32_t resetAddrActual = topkIdsIndexTensorAddr + topkIdsIndexTensorSize;
    resetTensor_ = reinterpret_cast<__ubuf__ int32_t*>(resetAddrActual);
    {
        LocalTensor<int32_t> values(TPosition::VECCALC, reinterpret_cast<uint64_t>(resetTensor_),
                                    resetTensorSize / sizeof(int32_t));
        Duplicate<int32_t>(values, 0, resetTensorSize / sizeof(int32_t));
    }
    resetBatchElementCount_ = resetBatchElementCount;

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    uint32_t routeRingAddr;
    if constexpr (Std::IsSame<XType, bfloat16_t>::value) {
        routeRingAddr = InitQuantScratchTensors(mxTempTensorAddr);
    } else {
        routeRingAddr = InitPreQuantScratchTensors(mxTempTensorAddr);
    }
    uint32_t routeRingBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.routeRingTensor = reinterpret_cast<__ubuf__ uint8_t*>(routeRingAddr);
    uint32_t sendCntAccAddr = routeRingAddr + routeRingBytes;
    // count 发送阶段，量化、reset 和下标发送已完成，可复用累计 count 之前的工作区。
    // 仅双 xIn 就至少占 4096B，足以容纳一张卡最多 1024 个专家的 count。
    sendMaskScratch_.countSendScratchTensor = reinterpret_cast<__ubuf__ int32_t*>(0);
    sendMaskScratch_.sendCntAccTensor = reinterpret_cast<__ubuf__ int32_t*>(sendCntAccAddr);
}

// 在普通模板内构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置。
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenUnpermuteBuffers()
{
    const uint8_t combineQuantMode = params_.tilingData->combineQuantMode;
    return CreateTokenUnpermuteBuffers<TopkWeightsType>(combineQuantMode, tokenUnpermuteConfig_,
                                                        commonConfig_.tokenHiddenDim, tokenUnpermuteScratch_);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ConfigureSharedGmm1Input(
    GMMAddrInfo& gmmAddrInfo, GmmExecutionConfig& gmmConfig, Gmm1ActivationSync& sharedGmm1ActivationSync) const
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    const bool useInputDirectly =
        !Std::IsSame<XType, bfloat16_t>::value && commonConfig_.tokenHiddenDim % MXFP_DIVISOR_SIZE == 0U;
    gmmConfig.useStridedInput = !useInputDirectly;
    if (useInputDirectly) {
        gmmAddrInfo.aGlobal = params_.aGmAddr;
        gmmAddrInfo.aScaleGlobal = params_.xScaleGmAddr;
    } else {
        GM_ADDR sharedInputBase = SHARED_INPUT_REUSES_MOE_QUANT ? peermem.quantTokenScale.get_ptr(commBuffer) :
                                                                  workspace.shared.gmm1Input.data.get_ptr(buffer);
        gmmAddrInfo.aGlobal = sharedInputBase;
        gmmAddrInfo.aScaleGlobal = sharedInputBase + sharedQuantProcessConfig_.quantTokenAlignBytes;
        gmmConfig.inputLayout = {
            sharedQuantProcessConfig_.quantTokenScaleAlignBytes * SharedQuantConfig::A_ELEMS_PER_BYTE,
            sharedQuantProcessConfig_.quantTokenScaleAlignBytes / static_cast<uint32_t>(sizeof(SharedQuantScaleType))};
    }
    if constexpr (SHARED_GMM1_USES_GM) {
        gmmAddrInfo.gmm1ActivationSync = &sharedGmm1ActivationSync;
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::RunSharedExpertGmm1Activation(
    const GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape, const GmmExecutionConfig& gmmConfig,
    GmmRuntimeState& runtimeState, uint32_t sharedExpertIdx, void* blockMmadContext)
{
    // 共享专家不使用路由专家的 topk 权重。
    constexpr bool sharedTopkWeightsPrefetch = false;
    uint32_t expertBeforeCnt = sharedExpertIdx * commonConfig_.tokenNum;
    auto runGmm = [&](const auto&... layoutArgs) __attribute__((cce_aicore))
    {
        if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4) {
            RunGmm1A8W4<typename SharedQuantConfig::QuantOutType, SharedWeightType, bfloat16_t, SharedQuantScaleType,
                        SharedQuantScaleType, GMM1_TILE_M, true, true>(
                sharedTopkWeightsPrefetch, sharedBlockContext_, sharedEpilogueOp_, params_, problemShape, gmmAddrInfo,
                runtimeState.startBlockIdx, gmmConfig.blockJob, expertBeforeCnt, sharedExpertIdx,
                commonConfig_.activationOutputDivisor, layoutArgs...);
        } else {
            RunGmm1Generic<typename SharedQuantConfig::QuantOutType, SharedActivationOutType, SharedWeightType,
                           bfloat16_t, SharedQuantScaleType, SharedQuantScaleType, SharedWeight1Format != FORMAT_ND,
                           GMM1_TILE_M, true, true>(
                sharedTopkWeightsPrefetch, sharedEpilogueOp_, params_, problemShape, gmmAddrInfo,
                runtimeState.startBlockIdx, runtimeState.vecSetSyncCom, gmmConfig.blockJob, expertBeforeCnt,
                sharedExpertIdx, runtimeState.pingpongIdx, commonConfig_.activationOutputDivisor, blockMmadContext,
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
    GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape, const GmmExecutionConfig& sharedGmmConfig,
    GmmRuntimeState& runtimeState, void* blockMmadContext)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm1GlobalBuffer<SharedWeightType, SharedActivationOutType, SharedQuantScaleType,
                                           SHARED_GMM1_USES_GM>(commonConfig_, sharedGmmConfig, workspace, buffer,
                                                                sharedWeightTensorListAddrs_, sharedEpilogueOp_,
                                                                gmmAddrInfo, sharedExpertIdx);
        RunSharedExpertGmm1Activation(gmmAddrInfo, problemShape, sharedGmmConfig, runtimeState, sharedExpertIdx,
                                      blockMmadContext);
    }
}

// 等待本核输入准备的 MTE3 搬出完成，再允许共享计算覆盖同一段 UB。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SyncBeforeSharedExpertGmm1()
{
    if constexpr (g_coreType == AIV) {
        if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4) {
            PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID3>();
        } else {
            // 唯一 AIV 在执行共享激活前，等待本核输入和计数的搬出完成。
            PipeFence<PIPE_MTE3, PIPE_V, SYNC_EVENT_ID3>();
        }
    }
}

// 在 MoE 阶段开始前，AIV 承接前序搬出以复用 UB；AIC 承接 reset 完成以读取 GM 状态。
template <TemplateMegaMoeTypeClass>
template <typename Derived>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SyncBeforeMoeExpertStages(
    Derived& derived, SharedGmmStartCondition sharedGmmStartCondition)
{
    if constexpr (g_coreType == AIV) {
        PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID3>();
    } else {
        if (sharedExpertNum_ > 0U) {
            if (sharedGmmStartCondition == SharedGmmStartCondition::InputDirectly) {
                // Scalar 直接等待全部 reset 完成，再读取 MoE 的 GM 状态。
                SyncIntraWait<PIPE_S>(INPUT_PREPARE_DONE_FLAG);
            } else {
                // 共享使用 quant/pack 结果：BF16 输入，或预量化输入的 H 仅满足 32 对齐。
                // 输入准备通知已在共享 GMM 前由 MTE2 消费，此处将完成条件传递给 Scalar。
                // 此同步也会等待此前共享 GMM 的 MTE2 搬入完成。
                PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID3>();
            }
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm1(
    Gmm1ActivationSync& sharedGmm1ActivationSync, void* blockMmadContext)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    if (gmmExecutionConfig_.blockJob.totalJobs == 0U ||
        gmmExecutionConfig_.blockJob.jobIndex >= gmmExecutionConfig_.blockJob.totalJobs) {
        return;
    }
    typename SharedBlockEpilogue::Params epilogueParams{
        .yGmAddr = workspace.shared.gmm2Input.data.get_ptr(buffer),
        .yScaleGmAddr = workspace.shared.gmm2Input.scale.get_ptr(buffer),
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
            ProcessSharedExpertGmm1Loop(gmmAddrInfo, problemShape, sharedGmmConfig, runtimeState, blockMmadContext);
            sharedBlockContext_ = {};
        } else {
            ProcessSharedExpertGmm1Loop(gmmAddrInfo, problemShape, sharedGmmConfig, runtimeState, blockMmadContext);
        }
    } else {
        ProcessSharedExpertGmm1Loop(gmmAddrInfo, problemShape, sharedGmmConfig, runtimeState, blockMmadContext);
    }
    if constexpr (!SHARED_GMM1_USES_GM) {
        Gmm1UbActivationSync::EndSync(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
    }
    gmm1PingPongIdx_ = 0U;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::RunSharedExpertGmm2(const GMMAddrInfo& gmmAddrInfo,
                                                                             const ProblemShape& problemShape,
                                                                             void* blockMmadContext)
{
    constexpr bool sharedTopkWeightsPrefetch = false;
    constexpr bool notifyCombineTileReady = false;
    // 共享 GMM1 和 GMM2 的所有量化模式都按 256-token group 交接 activation；
    // topk 权重在激活量化阶段同步计算仍关闭，权重 L2 bypass 仍仅用于 A8W8。
    if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4 ||
                  (SharedQuantConfig::AXW_MODE == AxWMode::A4W4 && !SHARED_FP4_SWIGLU_OUTPUT)) {
        RunGmm2A8W4<SharedActivationOutType, SharedWeightType, bfloat16_t, SharedQuantScaleType, SharedQuantScaleType,
                    GMM1_TILE_M, true, false, true, notifyCombineTileReady>(
            sharedTopkWeightsPrefetch, sharedBlockContext_, problemShape, gmmAddrInfo, startBlockIdx_,
            gmmExecutionConfig_.blockJob, static_cast<uint32_t>(Get<M_VALUE>(problemShape)), 0U, nullptr,
            commonConfig_.activationOutputDivisor);
    } else {
        RunGmm2Generic<SharedActivationOutType, SharedWeightType, bfloat16_t, SharedQuantScaleType,
                       SharedQuantScaleType, SharedWeight2Format != FORMAT_ND, false, GMM1_TILE_M, true, true,
                       notifyCombineTileReady>(COMBINE_NO_QUANT, sharedTopkWeightsPrefetch, problemShape, gmmAddrInfo,
                                               startBlockIdx_, gmmExecutionConfig_.blockJob, blockMmadContext, true, 0U,
                                               nullptr, commonConfig_.activationOutputDivisor, 0U);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm2Loop(GMMAddrInfo& gmmAddrInfo,
                                                                                     const ProblemShape& problemShape,
                                                                                     void* blockMmadContext)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm2GlobalBuffer<SharedActivationOutType, SharedWeightType, SharedQuantScaleType,
                                           GMM1_TILE_M>(commonConfig_, gmmExecutionConfig_, workspace, buffer,
                                                        sharedWeightTensorListAddrs_, gmmAddrInfo, sharedExpertIdx);
        RunSharedExpertGmm2(gmmAddrInfo, problemShape, blockMmadContext);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm2(void* blockMmadContext)
{
    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    GMMAddrInfo gmmAddrInfo{};
    if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4 ||
                  (SharedQuantConfig::AXW_MODE == AxWMode::A4W4 && !SHARED_FP4_SWIGLU_OUTPUT)) {
        if (g_coreType == AscendC::AIC || GetSubBlockIdx() == 0U) {
            typename SharedA8W4Config::BlockContext::Block block(params_.tilingData->a8w4L1Layout);
            sharedBlockContext_ = {&block};
            ProcessSharedExpertGmm2Loop(gmmAddrInfo, problemShape, blockMmadContext);
            sharedBlockContext_ = {};
        } else {
            ProcessSharedExpertGmm2Loop(gmmAddrInfo, problemShape, blockMmadContext);
        }
    } else {
        ProcessSharedExpertGmm2Loop(gmmAddrInfo, problemShape, blockMmadContext);
    }
}

// 只有量化类型不同且启用了共享专家，才传入独立共享量化参数。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::QuantizeInputTokens(const WorkRange& tokenRange)
{
    const bool topkWeightsPrefetch = params_.tilingData->isTopkWeightsPrefetch;
    const TokenQuantParams<MoeQuantMode, QuantOutType, ActivationType> moeQuant{quantProcessConfig_, quantScratch_};
    auto quantize = [&](const auto& moeQuant, const auto&... sharedQuant) __attribute__((cce_aicore))
    {
        QuantizeLocalTokens<TopkWeightsType>(topkWeightsPrefetch, tokenRange, commonConfig_, params_.probsGmAddr,
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

// 全部 AIV 调用。Arch35 由 block 0 AIV1 导出、AIV0 清零；Arch92 唯一 AIV 兼任两者。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExportAndResetExpertCounts()
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    WorkRange resetRange{};
    resetRange = GetBalancedWorkRange(commonConfig_.worldSize * commonConfig_.moeExpertPerRank,
                                      {.jobIndex = blockIdx_, .totalJobs = blockNum_});
    if (blockIdx_ == 0U) {
        ExportExpertTokenCounts();
    }
    // 输出核先搬出再消费事件；清零核在写 GM 前等待全部读取完成，每个 AIV 每轮 wait 一次。
    SyncInterWait<PIPE_MTE3>(COUNT_TABLE_READ_DONE_FLAG);
    if (resetRange.count > 0U) {
        // AIV0 可能参与末尾的量化 Combine，填零前等待旧 MTE3 读完复用的 UB。
        PipeFence<PIPE_MTE3, PIPE_V, SYNC_EVENT_ID2>();
        const uint32_t resetElementCount =
            Ops::Base::CeilAlign(static_cast<uint32_t>(resetBatchElementCount_), static_cast<uint32_t>(INT32_PER_256B));
        {
            LocalTensor<int32_t> values(TPosition::VECCALC, reinterpret_cast<uint64_t>(resetTensor_),
                                        resetElementCount);
            Duplicate<int32_t>(values, 0, resetElementCount);
        }
        PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
        ResetWorkspaceRegionWithStride(resetRange, peermem.expertCountRecv.get_ptr(commBuffer), MTE_COUNT_SLOT_BYTES,
                                       resetTensor_, resetElementCount);
    }
    // 输出和本轮有效 count 槽清零均完成后，才进入共享 GMM2 及出口跨卡握手。
    PipeBarrier<PIPE_ALL>();
    AllVectorBarrier();
}

// 仅由 AIV 调用：准备本核输入，分别发布 reset 完成和 token 数据就绪通知。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::PrepareLocalInput()
{
    const bool topkWeightsPrefetch = params_.tilingData->isTopkWeightsPrefetch;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    // 全部 AIV 均分 reset 和 quant/pack。
    const WorkRange tokenRange = GetBalancedWorkRange(commonConfig_.tokenNum, aivJob_);
    SendAndQuantBuffInit();
    ResetSyncStatus(aivJob_, params_, resetBatchElementCount_, resetTensor_);
    // 本核 reset 任务完成后发布通知；空任务核同样参与全 AIV 事件。
    SyncInterArrive<PIPE_MTE3>(INPUT_RESET_DONE_FLAG);
    if (tokenRange.count > 0U) {
        if constexpr (Std::IsSame<XType, bfloat16_t>::value) {
            QuantizeInputTokens(tokenRange);
        } else {
            PackPreQuantizedLocalTokens<XType, TopkWeightsType>(
                topkWeightsPrefetch, tokenRange, commonConfig_, params_, quantProcessConfig_,
                peermem.quantTokenScale.get_ptr(commBuffer), preQuantScratch_);
            // 预量化共享专家与 MoE 类型一致；64 对齐时直接读取原始输入，
            // 否则复用这里的 MoE 拼接结果，无需独立打包共享输入。
        }
    }
    // 每个物理 AIV 推进入口轮次一次，空 token 核也参与。
    IncrementSyncCount(peermem.rankSync.get_ptr(commBuffer), aivCoreIdx_);
    // MoE 及可选共享 token 数据和 scale 写回后，发布 token 数据就绪通知。
    // 空 token 核也必须报告，保证全 AIV 事件配对。
    SyncInterArrive<PIPE_MTE3>(TOKEN_DATA_READY_FLAG);
}

// 仅由 AIV 调用：准备输入、发送下标，等待 token 数据就绪后发送 count。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessInputPreparationStage(
    SharedGmmStartCondition sharedGmmStartCondition)
{
    const uint32_t indexBytes = params_.tilingData->topkIndexTypeBytes;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    PrepareLocalInput();
    auto sendIndexAndCount = [&](const WorkRange& ownedExpertRange, bool notifyAicAfterQuantOrPack)
        __attribute__((cce_aicore))
    {
        if (ownedExpertRange.count > 0U) {
            SendTopkIdsIndexForExperts(indexBytes, ownedExpertRange, commonConfig_, g_winRankAddr_, sendMaskConfig_,
                                       sendMaskScratch_);
        }
        SyncInterWait<PIPE_MTE3>(INPUT_RESET_DONE_FLAG);
        PipeNotify<PIPE_MTE3, PIPE_S>(SYNC_EVENT_ID3);
        SyncInterWait<PIPE_MTE3>(TOKEN_DATA_READY_FLAG);
        if (notifyAicAfterQuantOrPack) {
            SyncIntraArrive<PIPE_MTE3>(INPUT_PREPARE_DONE_FLAG);
        }
        if (ownedExpertRange.count > 0U) {
            PipeBarrier<PIPE_MTE3>();
            auto* syncCountAddr = GetSyncCountAddress(peermem.rankSync.get_ptr(commBuffer), aivCoreIdx_);
            const int32_t syncCount = GmRead(syncCountAddr);
            SendTopkIdsCountForExperts(ownedExpertRange, commonConfig_, syncCount, g_winRankAddr_, sendMaskConfig_,
                                       sendMaskScratch_);
        }
    };

    const uint32_t totalExperts = commonConfig_.worldSize * commonConfig_.moeExpertPerRank;
    if (sharedGmmStartCondition == SharedGmmStartCondition::NoSharedExpert) {
        // 全部 AIV 分担发送，并在 count 发送前通知 AIC，reset 和 quant/pack 已完成。
        const WorkRange ownedExpertRange = GetBalancedWorkRange(totalExperts, aivJob_);
        sendIndexAndCount(ownedExpertRange, true);
    } else {
        // 唯一 AIV 分担发送任务，等待 reset 和 quant/pack 完成后发布 count。
        const WorkRange ownedExpertRange =
            GetBalancedWorkRange(totalExperts, {.jobIndex = blockIdx_, .totalJobs = blockNum_});
        sendIndexAndCount(ownedExpertRange, true);
    }

    if constexpr (!SHARED_GMM1_USES_GM) {
        if (sharedExpertNum_ > 0U) {
            // 配对 AIC 向 UB 写入结果前，先释放输入及计数缓冲区。
            SyncIntraArrive<PIPE_MTE3>(INPUT_UB_FREE_FLAG);
        }
    }
    // 后继 Scalar 状态读取承接已记录的 reset 完成条件。
    PipeWait<PIPE_MTE3, PIPE_S>(SYNC_EVENT_ID3);
}

// 仅由 AIC 调用，分别承接输入搬运、Scalar 状态读取和 FIX 输出的依赖。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::WaitForInputPreparation(
    SharedGmmStartCondition sharedGmmStartCondition)
{
    // 共享直接读取原始输入时，MTE2 可提前搬入；输入准备完成通知留到 MoE 前消费。
    if (sharedGmmStartCondition != SharedGmmStartCondition::InputDirectly) {
        SyncIntraWait<PIPE_MTE2>(INPUT_PREPARE_DONE_FLAG);
    }
    if (sharedGmmStartCondition == SharedGmmStartCondition::NoSharedExpert) {
        // 直接进入 MoE，Scalar 状态轮询需要等待全部 reset 完成。
        PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID3>();
    }
    // FIX 写入结果前，等待配对计算侧 AIV 释放输入及计数使用的 UB。
    if constexpr (!SHARED_GMM1_USES_GM) {
        if (sharedExpertNum_ > 0U) {
            SyncIntraWait<PIPE_FIX>(INPUT_UB_FREE_FLAG);
        }
    }
}

// 根据共享专家及其输入来源，确定共享 GMM 的启动条件。
template <TemplateMegaMoeTypeClass>
__aicore__ inline SharedGmmStartCondition MegaMoe<TemplateMegaMoeTypeFunc>::GetSharedGmmStartCondition() const
{
    if (sharedExpertNum_ == 0U) {
        return SharedGmmStartCondition::NoSharedExpert;
    }
    const bool sharedInputDirectly =
        !Std::IsSame<XType, bfloat16_t>::value && commonConfig_.tokenHiddenDim % MXFP_DIVISOR_SIZE == 0U;
    if (sharedInputDirectly) {
        return SharedGmmStartCondition::InputDirectly;
    }
    return SharedGmmStartCondition::QuantOrPackDone;
}

// 仅由 AIV 调用：等待全部 rank 完成输出发送，再将专家结果聚合到 token 输出。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessTokenUnpermuteStage()
{
    const uint8_t combineQuantMode = params_.tilingData->combineQuantMode;
    const bool topkWeightsPrefetch = params_.tilingData->isTopkWeightsPrefetch;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::CROSS_RANK_SYNC_OUTPUT);
    CrossRankSyncInWorldSize(peermem.rankSync.get_ptr(commBuffer), rankId_, worldSize_, aivJob_);
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::UNPERMUTE);
    MegaMoeUnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
    UnpermuteTokens<TopkWeightsType, GMM1_TILE_M>(combineQuantMode, topkWeightsPrefetch, tokenUnpermuteConfig_,
                                                  commonConfig_, params_, tokenUnpermuteScratch_,
                                                  unpermuteBufferConfig);
}

/*
 * 所有 MTE Wave 路径共用相同的阶段边界，派生类只负责 MoE 专家 Wave 的具体编排。
 */
template <TemplateMegaMoeTypeClass>
template <typename Derived>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessWave(Derived& derived, void* blockMmadContext)
{
    // 保存入口时的溢出模式，计算期间关闭溢出检查。
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);

    // 阶段 1：AIV 准备输入，AIC 等待输入可读及 UB 释放。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::INPUT_PREPARE);
    const SharedGmmStartCondition sharedGmmStartCondition = GetSharedGmmStartCondition();
    if constexpr (g_coreType == AIV) {
        ProcessInputPreparationStage(sharedGmmStartCondition);
    } else {
        WaitForInputPreparation(sharedGmmStartCondition);
    }

    // GM 交接由 arch35 AIV1 或 arch92 唯一 AIV 消费；独立事件保留至主流程末尾排空。
    constexpr uint16_t sharedActivationSubBlockIdx = 0U;
    Gmm1ActivationSync sharedGmm1ActivationSync(sharedActivationSubBlockIdx, GmmEventPair::SHARED_GMM1);
    if (sharedExpertNum_ > 0U) {
        // 阶段 2：可选的共享专家 GMM1 及 Activation。
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM1);
        SyncBeforeSharedExpertGmm1();
        ProcessSharedExpertGmm1(sharedGmm1ActivationSync, blockMmadContext);
    }

    // 阶段 3：由派生类编排 MoE 专家的 Dispatch、GMM1/Activation 和 GMM2/Combine。
    Gmm1ActivationSync gmm1ActivationSync(MoeQuantConfig::AXW_MODE == AxWMode::A8W4 ? 1U : 0U);
    Gmm2CombineSync gmm2CombineSync;
    SyncBeforeMoeExpertStages(derived, sharedGmmStartCondition);
    derived.ProcessMoeExpertStages(gmm1ActivationSync, gmm2CombineSync);
    if constexpr (g_coreType == AIV) {
        ExportAndResetExpertCounts();
    }
    if (sharedExpertNum_ > 0U) {
        // 阶段 4：可选的共享专家 GMM2，其结果由输出聚合阶段合并。
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM2);
        ProcessSharedExpertGmm2(blockMmadContext);
    }

    // 阶段 5：所有 rank 完成 Combine 结果发送后，AIV 执行输出聚合。
    if constexpr (g_coreType == AIV) {
        ProcessTokenUnpermuteStage();
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
