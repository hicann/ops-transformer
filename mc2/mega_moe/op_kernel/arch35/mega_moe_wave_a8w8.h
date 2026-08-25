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
 * \file mega_moe_wave_a8w8.h
 * \brief MegaMoe A8W8 wave 流水实现
 */

#ifndef MEGA_MOE_WAVE_A8W8_H
#define MEGA_MOE_WAVE_A8W8_H

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
#include "common/mega_moe_exception_dump_policy.h"
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
#define TemplateMegaMoeA8W8WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved
#define TemplateMegaMoeA8W8WaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch, IsGmm1Interleaved

template <TemplateMegaMoeA8W8WaveTypeClass>
class MegaMoeA8W8Wave {
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
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    using ActivationType = QuantOutType;
    __aicore__ inline MegaMoeA8W8Wave(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData,
                                GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;
    using DispatchBufferConfig = MegaMoeDispatchBufferConfig;
    using CombineBufferConfig = WaveCombineBufferConfig;
    using UnpermuteBufferConfig = MegaMoeUnpermuteBufferConfig;
    __aicore__ inline void InitInputPrepareConfigs();
    __aicore__ inline void InitGmmConfigs(int32_t dispatchFlagSlotsPerExpert);
    __aicore__ inline void InitTokenUnpermuteConfig();
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline CombineBufferConfig InitCombineBuffers();
    __aicore__ inline void InitExpertTokenCountExportBuffers();
    __aicore__ inline void ProcessSharedExpertGmm1();
    __aicore__ inline void ProcessMoeExpertStages();
    __aicore__ inline void ProcessSharedExpertGmm2();
    __aicore__ inline void ProcessGmmPipeline();
    __aicore__ inline void DispatchAllExpertTokens();
    __aicore__ inline bool IsSameExpertTokenPosition(const ExpertTokenPosition &currentPosition,
                                                     const ExpertTokenPosition &targetPosition) const;
    __aicore__ inline ExpertTokenPosition ProcessGmm1Wave(ExpertTokenPosition &gmm1Position,
                                                          ExpertLoopState &gmm1ExpertState, GMMAddrInfo &gmm1AddrInfo,
                                                          GmmRuntimeState &runtimeState);
    __aicore__ inline void ProcessGmm2Wave(ExpertTokenPosition &gmm2Position,
                                           const ExpertTokenPosition &waveEndPosition, ExpertLoopState &gmm2ExpertState,
                                           GMMAddrInfo &gmm2AddrInfo, GmmRuntimeState &runtimeState);
    __aicore__ inline void ProcessCombineExperts(uint32_t expertBegin, uint32_t expertEnd,
                                                 ExpertLoopState &combineState, GMMAddrInfo &combineAddrInfo,
                                                 const CombineBufferConfig &bufferConfig);
    __aicore__ inline UnpermuteBufferConfig InitTokenUnpermuteBuffers();

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
    SendMaskConfig sendMaskConfig_;
    QuantProcessConfig quantProcessConfig_;
    TokenDispatchConfig tokenDispatchConfig_;
    // Wave Combine 的逻辑任务分工（block 粒度，AIV1 门控在函数内）。
    AivJobContext waveCombineJob_{};
    TokenUnpermuteConfig tokenUnpermuteConfig_;

    uint32_t k_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t gmm1TilesPerMGroup_ = 1U;
    uint32_t mGroupsPerWave_ = 1U;
    uint16_t gmm1PingPongIdx_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint16_t gmm2PingPongIdx_ = 0;
    // 共享专家相关成员
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;

    static constexpr bool GMM1_INTERLEAVED = IsGmm1Interleaved;

    /*
     * AIV1 上 Dispatch 与 Combine 分阶段复用 UB，进入 Combine 前 Dispatch 的动态 ring 已经排空：
     *   [0, 64 KiB)       Dispatch 的 cumsum 等全流程常驻状态；MoE 流水结束后复用其中最多 36 KiB，
     *                     从 GM 恢复并压紧最多 1024 个专家的 token count；
     *   [64, 160 KiB)     非量化 Combine 的 6 个 BF16 row buffer（H 最大 8 KiB）；
     *                     量化 Combine 使用 2 个 [BF16 row | FP8 data + scale] 槽及共享量化 scratch；
     *   [160, 184 KiB)    空闲；
     *   [184, 187.5 KiB)  GMM2-ready 的 GM 搬入、ReduceSum scratch 和最终 sum；
     *   [187.5, 200 KiB)  空闲；
     *   [200, 248 KiB)    Combine 共用的 meta-info，共 1536 token * 8 int32；
     *   [248, 256 KiB)    硬件保留，不使用。
     */
    LocalTensor<int32_t> resetTensor_;
    QuantProcessScratch<ActivationType> quantProcessScratch_;
    SharedExpertPrepareScratch<ActivationType> sharedExpertPrepareScratch_;
    SendMaskScratch sendMaskScratch_;
    TokenDispatchScratch<ActivationType> tokenDispatchScratch_;
    ExpertTokenCountExportScratch expertTokenCountExportScratch_;
    WaveCombineScratch waveCombineScratch_;
    TokenUnpermuteScratch tokenUnpermuteScratch_;

    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M = TopkWeightsPrefetch ? L1_TILE_M_128 : L1_TILE_M_256;

    using BlockEpilogue =
        BlockEpilogueActivationMxQuant<ActivationType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                       EPILOGUE_TILE_M, L1_TILE_N, TopkWeightsPrefetch, GMM1_INTERLEAVED>;
    using SharedBlockEpilogue =
        BlockEpilogueActivationMxQuant<ActivationType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                       L1_TILE_M_256, L1_TILE_N, false, GMM1_INTERLEAVED>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
    MegaMoeImpl::ExceptionDumpEngine exceptionDump_;
    __gm__ MegaMoeImpl::GmmLoopCount *gmmLoopCount_{nullptr};
};

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::InitInputPrepareConfigs()
{
    aivJob_ = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_};
    quantProcessConfig_ = CreateQuantProcessConfig<ActivationType, QuantScaleOutType, TopkWeightsPrefetch,
                                                   PackedElementTraits<ActivationType>::ELEMENTS_PER_BYTE>(k_, params_);
    sendMaskConfig_ = CreateSendMaskConfig(params_, aivCoreIdx_);
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::InitGmmConfigs(
    int32_t dispatchFlagSlotsPerExpert)
{
    gmmExecutionConfig_ = {.blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
                           .groupedMatmulMode = params_.tilingData->groupedMatmulMode,
                           .isPerExpertWeightTensor = params_.tilingData->isPerExpertWeightTensor};
    countWorkspace_ = {.blockIdx = blockIdx_, .blockNum = params_.tilingData->aicNum};
    syncWorkspaceLayout_ = {.dispatchFlagSlotCountPerExpert = dispatchFlagSlotsPerExpert,
                            .activationFlagSlotCountPerExpert = dispatchFlagSlotsPerExpert,
                            .gmm1TileStatusCountPerExpert = params_.tilingData->maxTilesPerExpert,
                            .combineSyncSlotCountPerExpert = 0U};
    waveCombineJob_ = {.jobIndex = blockIdx_, .totalJobs = blockNum_};
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::InitTokenUnpermuteConfig()
{
    uint32_t quantTokenSizeBytes = 0U;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        quantTokenSizeBytes = CreateQuantCombineBufferConfig(k_).quantTokenSizeBytes;
    }
    tokenUnpermuteConfig_ = {.job = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_},
                             .quantTokenSizeBytes = quantTokenSizeBytes,
                             .fullTokenChunkJobCount = params_.tilingData->unpermuteFullTokenChunkCoreCount,
                             .fullTokenChunkConfig = params_.tilingData->unpermuteConfigForFullTokenChunk,
                             .tailTokenChunkConfig = params_.tilingData->unpermuteConfigForTailTokenChunk};
}

// ========================
// Init：初始化成员并计算地址偏移
// ========================
template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData, GM_ADDR tilingGM)
{
    k_ = tilingData->h;
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    uint32_t gmm1SchedulerWidth = GMM1_INTERLEAVED ? tilingData->hiddenDim : tilingData->hiddenDim / ACTIVATION_N_HALF;
    gmm1TilesPerMGroup_ = Ops::Base::CeilDiv(gmm1SchedulerWidth, static_cast<uint32_t>(L1_TILE_N));
    mGroupsPerWave_ = tilingData->mGroupsPerWave == 0U ? 1U : tilingData->mGroupsPerWave;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    GM_ADDR dumpBase = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer[rankId_]);
    for (int i = 0; i < worldSize_; i++) {
        // g_winRankAddr_从win区地址偏移60K开始用，前面60K是异常dump区
        g_winRankAddr_[i] = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer[i]) + EXCEPTION_DUMP_REGION_SIZE;
    }
    params_.aGmAddr = x;
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
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(g_winRankAddr_[rankId_], tilingData, 1U);
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
    commonConfig_ = {.rankId = rankId_,
                     .worldSize = worldSize_,
                     .moeExpertPerRank = moeExpertPerRank_,
                     .sharedExpertNum = sharedExpertNum_,
                     .tokenNum = tilingData->bs,
                     .topK = tilingData->topK,
                     .tokenHiddenDim = k_,
                     .gmm1OutputDim = tilingData->hiddenDim};
    const int64_t maxOutput = static_cast<int64_t>(tilingData->maxOutputSize);
    const int64_t tileM = static_cast<int64_t>(GMM1_TILE_M);
    int32_t dispatchFlagSlotsPerExpert = static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, tileM)) * INT_CACHELINE;
    InitInputPrepareConfigs();
    tokenDispatchConfig_ = CreateTokenDispatchConfig(params_, quantProcessConfig_, sendMaskConfig_.maskAlignSize);
    InitGmmConfigs(dispatchFlagSlotsPerExpert);
    InitTokenUnpermuteConfig();

    gmmLoopCount_ = MegaMoeImpl::RegisterMegaMoeExceptionDump(exceptionDump_, dumpBase, tilingGM, tilingData,
                                                              params_.peermemInfo, sendMaskConfig_.maskAlignSize,
                                                              reinterpret_cast<GM_ADDR>(&mc2Context_->epRankId));
}

// =================================================================================================
// DispatchBuffInit：申请公共 SendCount 和 Token Dispatch 函数使用的 buffer。
// =================================================================================================
template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::DispatchBuffInit()
{
    DispatchBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return;
    }

    tokenDispatchScratch_.revTokenElemCnt = k_; // A8W8 输出 token 的元素数
    tokenDispatchScratch_.revScaleElemCnt =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE; // 输出 token-scale 元素数，紧密排列

    // 与 route batch 无关的固定占用
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    // sendCntTensor：每 src rank 一个 burst(32B)，共 worldSize*32B。
    uint32_t sendCntTensorSize = worldSize_ * static_cast<uint32_t>(ALIGN_32);
    // Dispatch 的 UB 布局与 AIV 分核无关；对应 host CalcDispatchBufferConfig 的唯一配置。
    bufferConfig = params_.tilingData->dispatchBufferConfig;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址
    // Tensor 用途：ComputeExpertTokenCountAndNotify 记录本卡各专家收到的 token 总数；
    // Tensor 大小：仅记录 count 值且各专家之间复用，申请 32 字节；
    uint32_t expertTokenCntTensorAddr = 0;
    tokenDispatchScratch_.expertTokenCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr, expertTokenCntTensorSize / sizeof(int32_t));
    // Tensor 用途：ComputeExpertTokenCountAndNotify 记录本卡专家收到 token count 的前缀和；
    // Tensor 大小：worldSize_ * moeExpertPerRank_ * sizeof(int32_t)，向上对齐至 32 字节；
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    tokenDispatchScratch_.cumsumInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    // Tensor 用途：ComputeExpertTokenCountAndNotify 暂存各源 rank 的 count；
    // Tensor 大小：每个源 rank 占一个 32B burst，共 worldSize_ * 32B；
    uint32_t sendCntTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    tokenDispatchScratch_.sendCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntTensorAddr, sendCntTensorSize / sizeof(int32_t));
    // Tensor 用途：DispatchExpertTokens 接收当前 batch 的 mask 切片；
    // Tensor 大小：routeItemsPerBatch / 8 字节，每个 bit 对应一个路由项；
    uint32_t maskBatchAddr = sendCntTensorAddr + sendCntTensorSize;
    uint32_t maskBatchSize = static_cast<uint32_t>(routeItemsPerBatch / 8) * static_cast<uint32_t>(sizeof(uint8_t));
    tokenDispatchScratch_.maskBatchTensor =
        LocalTensor<uint8_t>(TPosition::VECCALC, maskBatchAddr, maskBatchSize / sizeof(uint8_t));
    tokenDispatchScratch_.maskBatchU32Tensor =
        LocalTensor<uint32_t>(TPosition::VECCALC, maskBatchAddr, maskBatchSize / sizeof(uint32_t));
    // Tensor 用途：DispatchExpertTokens 中 GatherMask 的目标 Tensor；
    // Tensor 大小：routeItemsPerBatch * sizeof(int32_t)，向上对齐至 32 字节；
    uint32_t validTopkIndexTensorAddr = maskBatchAddr + maskBatchSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(routeItemsPerBatch * sizeof(int32_t)),
                                                             static_cast<int64_t>(ALIGN_32));
    tokenDispatchScratch_.validTopkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    // Tensor 用途：DispatchExpertTokens 中 GatherMask 的源 Tensor，保存本 batch 的全局索引；
    // Tensor 大小：与 validTopkIndexTensor 一致，为 routeItemsPerBatch * sizeof(int32_t)，
    // 向上对齐至 32 字节；
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = validTopkIndexTensorSize;
    tokenDispatchScratch_.topkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, topkIndexTensorSize / sizeof(int32_t));
    // 路由批次 Tensor 后依次放置 copyTmp 环形缓冲区和 32B metaInfo 环形缓冲区。
    // Tensor 用途：DispatchExpertTokens 中的动态 dispatch 环形缓冲区，配合
    // EVENT_ID0..EVENT_ID(bufferCount-1) 形成软流水；
    // 只记基址：槽视图在热路径由 GetDispatchCopyBuffer 现场构造，
    // Tensor 大小：bufferConfig.bufferCount 块（主线自适应 UB 预算给出的 2~6），
    // 每块 tokenDispatchConfig_.quantTokenScaleAlignBytes；
    // 该值即 Init() 算好的 Align256(token) + Align32(scale) + optional Align32(weight)，与 host
    // CalcDispatchBufferConfig 的 copyBufferBytes 恒相等，故连续 ring 中每个槽位均保持 32B 对齐。
    tokenDispatchScratch_.copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t copyTmpTotalSize =
        static_cast<uint32_t>(bufferConfig.bufferCount) * tokenDispatchConfig_.quantTokenScaleAlignBytes;
    // Tensor 用途：CopyTokensAndMetaForDispatch 中的 metaInfo 环形缓冲区，逐 token 即时写入 GM；
    // Tensor 大小：bufferCount * 32B，与 copyTmp 槽位和事件编号一一对应。
    uint32_t metaInfoTensorAddr = tokenDispatchScratch_.copyTmpBaseAddr + copyTmpTotalSize;
    uint32_t metaInfoReserveSize =
        static_cast<uint32_t>(bufferConfig.bufferCount) * static_cast<uint32_t>(INT32_PER_256B) * sizeof(int32_t);
    tokenDispatchScratch_.metaInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr, metaInfoReserveSize / sizeof(int32_t));
    tokenDispatchScratch_.cumsumRevCntInRank = 0U;
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline typename MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::CombineBufferConfig
MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::InitCombineBuffers()
{
    CombineBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }
    if (GetSubBlockIdx() != 1U) {
        return bufferConfig;
    }
    bufferConfig.rowBytes = commonConfig_.tokenHiddenDim * sizeof(bfloat16_t);
    bufferConfig.rowStrideBytes = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(bufferConfig.rowBytes), static_cast<uint64_t>(ALIGN_32)));
    bufferConfig.slotStrideBytes = bufferConfig.rowStrideBytes;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t nScale = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        uint32_t tokenStorageBytes =
            Ops::Base::CeilAlign(commonConfig_.tokenHiddenDim, static_cast<uint32_t>(ALIGN_256));
        uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
        bufferConfig.quantRowStorageBytes =
            Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
        bufferConfig.quantRowElements = bufferConfig.quantRowStorageBytes;
        bufferConfig.slotStrideBytes += bufferConfig.quantRowStorageBytes;
        bufferConfig.quantTempElements =
            Ops::Base::CeilAlign(storedScaleBytes, static_cast<uint32_t>(ALIGN_32)) + storedScaleBytes / 2U;
    }
    constexpr uint32_t rowBufferCount =
        CombineQuantMode == COMBINE_NO_QUANT ? WAVE_COMBINE_NO_QUANT_ROW_BUFFER_NUM : WAVE_COMBINE_QUANT_ROW_BUFFER_NUM;
    uint32_t rowRingBytes = rowBufferCount * bufferConfig.slotStrideBytes;
    waveCombineScratch_.rowBufferTensor =
        LocalTensor<bfloat16_t>(TPosition::VECIN, WAVE_COMBINE_UB_BASE, rowRingBytes / sizeof(bfloat16_t));
    waveCombineScratch_.metaInfoTensor = LocalTensor<int32_t>(TPosition::VECCALC, META_INFO_TENSOR_ADDR,
                                                              WAVE_COMBINE_META_INFO_TOKEN_CAPACITY * META_INFO_SIZE);
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        waveCombineScratch_.quantTempTensor =
            LocalTensor<float>(TPosition::VECIN, WAVE_COMBINE_UB_BASE + rowRingBytes, bufferConfig.quantTempElements);
    }
    return bufferConfig;
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::InitExpertTokenCountExportBuffers()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || gmmExecutionConfig_.blockJob.jobIndex != 0U) {
        return;
    }
    uint32_t expertPerRank = commonConfig_.moeExpertPerRank;
    uint32_t stridedTensorBytes = expertPerRank * static_cast<uint32_t>(ALIGN_32);
    expertTokenCountExportScratch_.stridedTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, 0U, stridedTensorBytes / sizeof(int32_t));
    uint32_t compactTensorBytes =
        Ops::Base::CeilAlign(static_cast<uint64_t>(expertPerRank * sizeof(int32_t)), static_cast<uint64_t>(ALIGN_32));
    expertTokenCountExportScratch_.compactTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, stridedTensorBytes, compactTensorBytes / sizeof(int32_t));
}

// ======================================================================================
// SendAndQuantBuffInit：申请公共 mask、workspace reset 和 token quant 函数使用的 buffer。
// ======================================================================================
template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::SendAndQuantBuffInit()
{
    SendMaskBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 = static_cast<uint64_t>(params_.workspaceInfo.flagResetElementCount);
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    int32_t resetBatchElementCount = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                         static_cast<int32_t>(resetElementCountPerCore) :
                                         DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    uint32_t mxTempTensorSize = 2 * 1024;
    // 单个 xOutTensor 槽位与 dispatch 的 token-scale-weight 通信记录使用相同布局。
    uint32_t xOutTensorSize = quantProcessConfig_.quantTokenScaleAlignBytes;
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * moeExpertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // 必须与 host SetAdaptiveBufferConfigs 的 quotient/remainder 分核保持一致。
    bufferConfig = aivCoreIdx_ < params_.tilingData->sendMaskCoreCountWithExtraExpert ?
                       params_.tilingData->sendMaskConfigForCoreWithExtraExpert :
                       params_.tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，因此两个 int32 tensor 均天然满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetAddrActual = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    quantProcessScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));

    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantProcessScratch_.xOutTensor0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantProcessScratch_.xOutTensor1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));

    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantProcessScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantProcessScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));

    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize;
    /*
     * h%64==32（scale 组数为奇数）时，量化链路存在三处"计算不覆盖、却进入定长通信记录或参与
     * 计算"的跨 launch UB 残留：xIn 尾部（进 ComputeMaxExp 尾块 mask 内 lane）、xOut 记录的
     * scale 偶数补齐槽（ComputeScale 掩码写不到）、mxTemp 的 halfScale 补偶槽（被
     * ComputeFp8Data 尾块 E2B 广播进乘法，0×NaN 仍为 NaN）。残留呈 NaN/大指数位型时整行
     * GMM 输出被污染为 NaN，最终 combine 输出成块清零（首轮 UB 干净故仅多轮调用时显形）。
     * 此处对 [mxTempTensorAddr, sendMaskAddr) 连续 span（mxTemp/xOut0/xOut1/xIn0/xIn1 五段
     * 量化 scratch）一次性清零：span 边界取 sendMaskAddr、与本函数的地址推进公式同源，
     * 中间插入新 buffer 时范围自动跟随；有效区随后每 token 均被完整覆写，残留位恒为良性 0。
     * h%64==0 时不存在上述缝隙，本清零不改变任何可观测行为。
     */
    LocalTensor<int16_t> quantScratchSpan(TPosition::VECCALC, mxTempTensorAddr,
                                          (sendMaskAddr - mxTempTensorAddr) / sizeof(int16_t));
    Duplicate<int16_t>(quantScratchSpan, 0, static_cast<int32_t>((sendMaskAddr - mxTempTensorAddr) / sizeof(int16_t)));
    PipeBarrier<PIPE_V>();
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID2>();
    uint32_t sendGatherOutSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));

    uint32_t sendMaskTotalBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.sendMaskTensor = LocalTensor<uint8_t>(TPosition::VECCALC, sendMaskAddr, sendMaskTotalBytes);
    uint32_t sendGatherOutAddr = sendMaskAddr + sendMaskTotalBytes;
    sendMaskScratch_.sendGatherOutTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    sendMaskScratch_.sendCntAccTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
    if (sharedExpertNum_ > 0U) {
        sharedExpertPrepareScratch_.copyBuffer0 = quantProcessScratch_.xOutTensor0;
        sharedExpertPrepareScratch_.copyBuffer1 = quantProcessScratch_.xOutTensor1;
    }
    resetBatchElementCount_ = resetBatchElementCount;
}

// ===============================================================
// 可选共享专家流程：依次执行共享专家 GMM1 与 SwiGLU，并在结束后重置 GMM 调度状态。
// ===============================================================
template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessSharedExpertGmm1()
{
    if (gmmExecutionConfig_.blockJob.totalJobs == 0U ||
        gmmExecutionConfig_.blockJob.jobIndex >= gmmExecutionConfig_.blockJob.totalJobs) {
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

    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    GMMAddrInfo gmmAddrInfo{};
    int32_t vecSetSyncCom = 0;
    int32_t gmTileSequence = 0;
    GmmRuntimeState runtimeState{startBlockIdx_, vecSetSyncCom, gmTileSequence, gmm1PingPongIdx_};
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationType, QuantScaleOutType, false>(
            commonConfig_, gmmExecutionConfig_, params_.workspaceInfo, sharedWeightTensorListAddrs_, sharedEpilogueOp_,
            gmmAddrInfo, sharedExpertIdx);
        RunSharedExpertGmm1ActivationStage<QuantOutType, Weight1Type, ActivationType, QuantScaleOutType, false,
                                           GMM1_TILE_M, GMM1_INTERLEAVED, true>(
            commonConfig_, gmmExecutionConfig_, params_, sharedEpilogueOp_, gmmAddrInfo, problemShape, runtimeState,
            sharedExpertIdx, nullptr, true);
    }
    EndSync<GMM1_INTERLEAVED>(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
    gmm1PingPongIdx_ = 0;
    startBlockIdx_ = 0; // 共享专家 GMM1 修改了 startBlockIdx_，重置后供 MoE 专家 GMM1 使用
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessSharedExpertGmm2()
{
    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    int32_t vecSetSyncCom = 0;
    int32_t gmTileSequence = 0;
    GmmRuntimeState runtimeState{startBlockIdx_, vecSetSyncCom, gmTileSequence, gmm2PingPongIdx_};
    GMMAddrInfo gmmAddrInfo{};
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm2GlobalBuffer<ActivationType, Weight1Type, QuantScaleOutType, GMM1_TILE_M, false>(
            commonConfig_, gmmExecutionConfig_, params_.workspaceInfo, sharedWeightTensorListAddrs_, gmmAddrInfo,
            sharedExpertIdx);
        RunGmm2ByMode<COMBINE_NO_QUANT, QuantOutType, ActivationType, Weight1Type, QuantScaleOutType, false, false,
                      GMM1_TILE_M, TopkWeightsPrefetch, true, GMM1_INTERLEAVED, true>(
            gmmExecutionConfig_, params_, gmmAddrInfo, problemShape, runtimeState, nullptr, true);
    }
}

// ===============================================================
// Wave 模板主流程。
// ===============================================================
// 在 wave 模板内构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置。
template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline typename MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::UnpermuteBufferConfig
MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::InitTokenUnpermuteBuffers()
{
    return CreateTokenUnpermuteBuffers<TopkWeightsType, CombineQuantMode>(
        tokenUnpermuteConfig_, commonConfig_.tokenHiddenDim, tokenUnpermuteScratch_);
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::DispatchAllExpertTokens()
{
    /**
     * 在 GMM pipeline 开始前完成所有专家的 Dispatch。每个专家单独使用全部
     * Dispatch core，复用与普通模板一致的单专家 Dispatch 入口。
     */
    for (uint32_t expertIdx = 0U; expertIdx < moeExpertPerRank_; ++expertIdx) {
        uint32_t expertTokenCount = 0U;
        RunMoeExpertDispatchStage<ActivationType, QuantScaleOutType, false, GMM1_TILE_M, TopkWeightsPrefetch>(
            tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, countWorkspace_, syncWorkspaceLayout_,
            params_, g_winRankAddr_, tokenDispatchScratch_, expertIdx, expertTokenCount);
    }
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline bool MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::IsSameExpertTokenPosition(
    const ExpertTokenPosition &currentPosition, const ExpertTokenPosition &targetPosition) const
{
    return currentPosition.expertIdx == targetPosition.expertIdx &&
           currentPosition.tokenIndexInExpert == targetPosition.tokenIndexInExpert;
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline ExpertTokenPosition MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessGmm1Wave(
    ExpertTokenPosition &gmm1Position, ExpertLoopState &gmm1ExpertState, GMMAddrInfo &gmm1AddrInfo,
    GmmRuntimeState &runtimeState)
{
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            return gmm1Position;
        }
    }

    uint32_t processedMGroupCount = 0U;
    while (gmm1Position.expertIdx < moeExpertPerRank_ && processedMGroupCount < mGroupsPerWave_) {
        if (gmm1Position.tokenIndexInExpert == 0U) {
            WaitForMoeExpertTokenCountReady(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_,
                                            gmm1Position.expertIdx);
            uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr,
                                                                         countWorkspace_, gmm1Position.expertIdx);
            UpdateExpertLoopState(gmm1ExpertState, gmm1Position.expertIdx, expertTokenCount);
        }

        uint64_t expertRowCount = Get<M_VALUE>(gmm1ExpertState.problemShape);
        if (expertRowCount == 0U || gmm1Position.tokenIndexInExpert >= expertRowCount) {
            ++gmm1Position.expertIdx;
            gmm1Position.tokenIndexInExpert = 0U;
            continue;
        }

        uint32_t remainingMGroupCount = mGroupsPerWave_ - processedMGroupCount;
        uint32_t waveEndTokenIndexInExpert = GetWaveEndRowOffsetInExpert(
            expertRowCount, gmm1Position.tokenIndexInExpert, remainingMGroupCount, GMM1_TILE_M);
        uint32_t waveRowCount = waveEndTokenIndexInExpert - gmm1Position.tokenIndexInExpert;
        ProblemShape gmm1WaveProblemShape = gmm1ExpertState.problemShape;
        Get<M_VALUE>(gmm1WaveProblemShape) = waveEndTokenIndexInExpert - gmm1Position.tokenIndexInExpert;
        UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationType, QuantScaleOutType,
                                        PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE, false,
                                        TopkWeightsPrefetch>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
            gmm1AddrInfo, gmm1ExpertState, gmm1Position.tokenIndexInExpert, gmm1TilesPerMGroup_);
        /*
         * 只有该 problem 从第 0 行覆盖到专家尾，后续 Wave 才不会再次读取同一份 GMM1 权重。
         * 热点专家的任一 Wave problem 都为 false，即使本次不足 256 行也保留正常 L2 cache。
         */
        bool isWholeExpert =
            gmm1Position.tokenIndexInExpert == 0U && static_cast<uint64_t>(waveEndTokenIndexInExpert) == expertRowCount;
        uint32_t waveTokenStartIndex =
            static_cast<uint32_t>(gmm1ExpertState.globalTokenStartIndex) + gmm1Position.tokenIndexInExpert;
        RunGmm1GenericByWeightFormat<QuantOutType, ActivationType, QuantScaleOutType, GMM1_TILE_M, EPILOGUE_TILE_M,
                                     TopkWeightsPrefetch, GMM1_INTERLEAVED, true>(
            gmmExecutionConfig_, params_, epilogueOp_, gmm1AddrInfo, gmm1WaveProblemShape, waveTokenStartIndex,
            runtimeState, gmm1Position.expertIdx, nullptr, isWholeExpert);

        processedMGroupCount += GetMGroupCountForRows(waveRowCount, GMM1_TILE_M);
        gmm1Position.tokenIndexInExpert = waveEndTokenIndexInExpert;
        if (gmm1Position.tokenIndexInExpert >= expertRowCount) {
            ++gmm1Position.expertIdx;
            gmm1Position.tokenIndexInExpert = 0U;
        }
    }
    return gmm1Position;
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessGmm2Wave(
    ExpertTokenPosition &gmm2Position, const ExpertTokenPosition &waveEndPosition, ExpertLoopState &gmm2ExpertState,
    GMMAddrInfo &gmm2AddrInfo, GmmRuntimeState &runtimeState)
{
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            return;
        }
    }

    while (!IsSameExpertTokenPosition(gmm2Position, waveEndPosition) && gmm2Position.expertIdx < moeExpertPerRank_) {
        if (gmm2Position.tokenIndexInExpert == 0U) {
            uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr,
                                                                         countWorkspace_, gmm2Position.expertIdx);
            UpdateExpertLoopState(gmm2ExpertState, gmm2Position.expertIdx, expertTokenCount);
        }

        uint64_t expertRowCount = Get<M_VALUE>(gmm2ExpertState.problemShape);
        if (expertRowCount == 0U || gmm2Position.tokenIndexInExpert >= expertRowCount) {
            ++gmm2Position.expertIdx;
            gmm2Position.tokenIndexInExpert = 0U;
            continue;
        }

        uint32_t waveEndTokenIndexInExpert = static_cast<uint32_t>(expertRowCount);
        if (gmm2Position.expertIdx == waveEndPosition.expertIdx) {
            waveEndTokenIndexInExpert = waveEndPosition.tokenIndexInExpert;
        }
        ProblemShape gmm2WaveProblemShape = gmm2ExpertState.problemShape;
        Get<M_VALUE>(gmm2WaveProblemShape) = waveEndTokenIndexInExpert - gmm2Position.tokenIndexInExpert;
        UpdateMoeExpertGmm2GlobalBuffer<Weight1Type, ActivationType, QuantScaleOutType, true, false, false>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmm2AddrInfo,
            gmm2ExpertState, gmm2Position.tokenIndexInExpert);
        // GMM2 与 GMM1 使用相同保护：只有完整专家 problem 才允许进一步判断是否绕过 L2。
        bool isWholeExpert =
            gmm2Position.tokenIndexInExpert == 0U && static_cast<uint64_t>(waveEndTokenIndexInExpert) == expertRowCount;
        RunGmm2ByMode<COMBINE_NO_QUANT, QuantOutType, ActivationType, Weight1Type, QuantScaleOutType, false, false,
                      GMM1_TILE_M, TopkWeightsPrefetch, false, GMM1_INTERLEAVED, true>(
            gmmExecutionConfig_, params_, gmm2AddrInfo, gmm2WaveProblemShape, runtimeState, nullptr, isWholeExpert);

        gmm2Position.tokenIndexInExpert = waveEndTokenIndexInExpert;
        if (gmm2Position.tokenIndexInExpert >= expertRowCount) {
            NotifyWaveGmm2Ready(waveCombineJob_, params_, gmm2Position.expertIdx);
            ++gmm2Position.expertIdx;
            gmm2Position.tokenIndexInExpert = 0U;
        }
    }
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessCombineExperts(
    uint32_t expertBegin, uint32_t expertEnd, ExpertLoopState &combineState, GMMAddrInfo &combineAddrInfo,
    const CombineBufferConfig &bufferConfig)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    uint32_t rowSequence = 0U;
    for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
        uint32_t expertTokenCount =
            GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, expertIdx);
        UpdateExpertLoopState(combineState, expertIdx, expertTokenCount);
        if (expertTokenCount == 0U) {
            continue;
        }
        UpdateMoeExpertGmm2GlobalBuffer<Weight1Type, ActivationType, QuantScaleOutType, true, false, false>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_,
            combineAddrInfo, combineState);
        RunWaveExpertCombineStage<CombineQuantMode>(commonConfig_, waveCombineJob_, bufferConfig, waveCombineScratch_,
                                                    params_, combineAddrInfo, combineState, combineState.expertIdx,
                                                    rowSequence);
    }
    DrainWaveCombineRowRing<CombineQuantMode>(rowSequence);
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessMoeExpertStages()
{
    // GMM1/GMM2 交错流水只记录一次阶段入口，各 Wave 完成轮次由独立计数记录。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::MOE_GMM1_ACTIVATION);
    uint64_t gmm1Count = 0U;
    uint64_t gmm2Count = 0U;
    tokenDispatchScratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
    DispatchBuffInit();
    CombineBufferConfig combineBufferConfig = InitCombineBuffers();
    InitExpertTokenCountExportBuffers();

    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};
    GMMAddrInfo combineAddrInfo{};
    ExpertLoopState gmm1ExpertState = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2ExpertState = CreateExpertLoopState(commonConfig_);
    ExpertLoopState combineExpertState = CreateExpertLoopState(commonConfig_);
    int32_t vecSetSyncCom = 0;
    int32_t gmm1TileSequence = 0;
    GmmRuntimeState gmm1RuntimeState{startBlockIdx_, vecSetSyncCom, gmm1TileSequence, gmm1PingPongIdx_};

    ExpertTokenPosition waveBeginPosition{};
    ExpertTokenPosition gmm1Position{};
    ExpertTokenPosition gmm2Position{};
    ExpertTokenPosition dispatchEndPosition{};
    uint32_t combineBeginExpertIndex = 0U;

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            DispatchAllExpertTokens();
            dispatchEndPosition.expertIdx = moeExpertPerRank_;
        }
    }

    while (waveBeginPosition.expertIdx < moeExpertPerRank_) {
        ExpertTokenPosition waveEndPosition = waveBeginPosition;
        const uint32_t waveStartBlockIndex = startBlockIdx_;

        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                waveEndPosition = dispatchEndPosition;
            } else {
                waveEndPosition = ProcessGmm1Wave(gmm1Position, gmm1ExpertState, gmm1AddrInfo, gmm1RuntimeState);
            }
        } else {
            waveEndPosition = ProcessGmm1Wave(gmm1Position, gmm1ExpertState, gmm1AddrInfo, gmm1RuntimeState);
        }
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM1, ++gmm1Count);
        const uint32_t gmm1EndBlockIndex = startBlockIdx_;

        int32_t gmm2TileSequence = 0;
        GmmRuntimeState gmm2RuntimeState{startBlockIdx_, vecSetSyncCom, gmm2TileSequence, gmm2PingPongIdx_};
        ProcessGmm2Wave(gmm2Position, waveEndPosition, gmm2ExpertState, gmm2AddrInfo, gmm2RuntimeState);
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);

        const bool hasNextWave = waveEndPosition.expertIdx < moeExpertPerRank_;
        const bool fixedRoleResonance =
            startBlockIdx_ == waveStartBlockIndex && gmm1EndBlockIndex != waveStartBlockIndex;
        if (hasNextWave && fixedRoleResonance) {
            startBlockIdx_ = gmm1EndBlockIndex;
        }

        uint32_t combineEndExpertIndex = waveEndPosition.expertIdx;
        ProcessCombineExperts(combineBeginExpertIndex, combineEndExpertIndex, combineExpertState, combineAddrInfo,
                              combineBufferConfig);
        combineBeginExpertIndex = combineEndExpertIndex;
        waveBeginPosition = waveEndPosition;
    }

    if constexpr (!TopkWeightsPrefetch) {
        EndSync<GMM1_INTERLEAVED>(vecSetSyncCom, gmm1PingPongIdx_);
    }
    gmm1PingPongIdx_ = 0;
    ExportExpertTokenCounts(commonConfig_, gmmExecutionConfig_.blockJob, countWorkspace_, params_,
                            expertTokenCountExportScratch_);
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::ProcessGmmPipeline()
{
    if (sharedExpertNum_ > 0) {
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM1);
        ProcessSharedExpertGmm1();
    }

    // 等待所有 rank 完成本轮输入准备，再读取远端 dispatch 数据。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::CROSS_RANK_SYNC_INPUT);
    CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);

    // Dispatch 的全部专家先完成，随后按 GMM wave 执行 GMM1、SwiGLU、GMM2 与 Combine。
    ProcessMoeExpertStages();

    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    if (sharedExpertNum_ > 0) {
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM2);
        ProcessSharedExpertGmm2();
    }
}

template <TemplateMegaMoeA8W8WaveTypeClass>
__aicore__ inline void MegaMoeA8W8Wave<TemplateMegaMoeA8W8WaveTypeFunc>::Process()
{
    // 保存入口时的溢出模式，并初始化输入准备阶段使用的 UB。
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();

    // 阶段 1：AIV 完成本卡输入量化、路由 mask 推送和 flag 清零。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::INPUT_PREPARE);
    QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
        aivJob_, commonConfig_, params_, quantProcessConfig_, quantProcessScratch_);
    GatherAndSendExpertMasks(aivJob_, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_, sendMaskScratch_);
    ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, resetBatchElementCount_, resetTensor_);
    if (sharedExpertNum_ > 0) {
        // 可选：为共享专家拆分连续布局的输入数据与 scale。
        PrepareSharedExpertInput<ActivationType, QuantScaleOutType, 1U>(
            aivJob_, commonConfig_, params_, quantProcessConfig_, sharedExpertPrepareScratch_);
    } else {
        if constexpr (g_coreType == AIV) {
            PipeBarrier<PIPE_ALL>();
        }
    }
    SyncAll<false>(); // AIC 等待 AIV 完成输入准备与 flag 清零后再进入计算

    ProcessGmmPipeline();

    // 阶段 3：等待所有 rank 的 Combine 发送完成，再执行本卡 Unpermute。
    if constexpr (g_coreType == AIV) {
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::CROSS_RANK_SYNC_OUTPUT);
        CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::UNPERMUTE);
        UnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteConfig_, commonConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
    }
    // 恢复入口时保存的溢出模式。
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::COMPLETE);
}

#undef TemplateMegaMoeA8W8WaveTypeClass
#undef TemplateMegaMoeA8W8WaveTypeFunc

} // namespace MegaMoeImpl
#endif
