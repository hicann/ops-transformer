/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM2_COMBINE_H
#define MEGA_MOE_GMM2_COMBINE_H

#include "../common/mega_moe_sync.h"
#include <type_traits>

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "adv_api/reduce/reduce.h"
#include "../common/mega_moe_arch_policy.h"
#include "../common/mega_moe_gmm_common.h"
#include "../common/mega_moe_utils.h"
#include "../common/mega_moe_copy.h"
#include "../common/mega_moe_mxfp8_utils.h"
#include "../../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../../common/op_kernel/quantize_functions.h"

namespace MegaMoeImpl {

using namespace AscendC;
constexpr uint32_t META_INFO_TENSOR_ADDR = 200U * 1024U;
constexpr int32_t MAX_AICORE_NUM = 36;

struct QuantTokenBufferConfig {
    uint32_t quantTokenSizeBytes;
};

struct WaveCombineBufferConfig {
    uint32_t rowBytes = 0;
    uint32_t rowStrideBytes = 0;
    uint32_t quantRowElements = 0;
    uint32_t quantRowStorageBytes = 0;
    uint32_t slotStrideBytes = 0;
    uint32_t quantTempElements = 0;
    uint32_t rowBufferCount = 0;
};

struct WaveCombineScratch {
    __ubuf__ int32_t* metaInfoTensor = nullptr;
    __ubuf__ bfloat16_t* rowBufferTensor = nullptr;
    __ubuf__ float* quantTempTensor = nullptr;
};

constexpr uint32_t WAVE_COMBINE_MIN_ROW_BUFFER_COUNT = 2U;
constexpr uint32_t WAVE_COMBINE_STEADY_ROW_BUFFER_COUNT = 1U;
constexpr uint32_t WAVE_COMBINE_MAX_ROW_BUFFER_COUNT = 6U;
constexpr uint32_t WAVE_COMBINE_UB_BASE = 64U * 1024U;
// [64 KiB, 184 KiB) is dedicated to the Combine row ring and quant scratch.
// The ready scan starts at 184 KiB, so adaptive row buffers must stay below it.
constexpr uint32_t WAVE_COMBINE_UB_LIMIT = 184U * 1024U;
constexpr uint32_t WAVE_COMBINE_META_INFO_TOKEN_CAPACITY = 1536U;

namespace CombineImpl {

// Combine 发送路由：dispatch 阶段写入 metaInfo 的 (目标rank, 原token行, topk槽) 三元组。
struct CombineTokenRoute {
    uint32_t dstRankId;
    uint32_t tokenIdx;
    uint32_t topkIdx;
};

__aicore__ inline CombineTokenRoute LoadCombineTokenRoute(__ubuf__ int32_t* metaInfoTensor, uint32_t recordIdx)
{
    uint32_t recordBase = recordIdx * META_INFO_SIZE;
    return {static_cast<uint32_t>(metaInfoTensor[recordBase + RANK_ID]),
            static_cast<uint32_t>(metaInfoTensor[recordBase + TOKEN_ID]),
            static_cast<uint32_t>(metaInfoTensor[recordBase + TOPK_INDEX])};
}

// 目标卡 combine 接收区按 (token, topk) 展开的紧凑行号。
__aicore__ inline uint64_t GetCombineDstRowIndex(const CombineTokenRoute& route, const Params& params)
{
    return static_cast<uint64_t>(route.tokenIdx) * params.tilingData->compute.topK + route.topkIdx;
}

// 为 tile 内每个 token 行发送一段有效的 GMM2 tile 数据。
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(uint32_t nLoc, uint32_t n, __ubuf__ int32_t* metaInfoTensor,
                                     __ubuf__ ElementMMadOut2* l0cOutUbGMM2, BlockShape& actualBlockShape,
                                     uint32_t ubTileN, const Params& params)
{
    const auto& peermem = params.communication.layout;
    // 调用方在进入该数据操作前，保证批量加载的 metadata 对 Scalar 可见。
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    __gm__ ElementMMadOut2* gmRemoteD = nullptr;
    uint64_t gmRemoteBaseOffset = static_cast<uint64_t>(peermem.combineSend.offsetBytes);

    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        CombineTokenRoute route = LoadCombineTokenRoute(metaInfoTensor, static_cast<uint32_t>(tileIdx));
        gmRemoteD =
            reinterpret_cast<__gm__ ElementMMadOut2*>(GetRankWinAddrWithOffset(route.dstRankId, gmRemoteBaseOffset));
        uint64_t gmDstOffset = GetCombineDstRowIndex(route, params) * n + nLoc;
        CopyUbToGm(gmRemoteD + gmDstOffset, (l0cOutUbGMM2 + (tileIdx * ubTileN)),
                   Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2));
    }
}

// 发送一行完整的 Combine 数据，BF16 与 FP8 记录复用相同的定长搬运流程。
template <typename Element>
__aicore__ inline void SendCombineTokenRow(uint32_t rowElements, uint64_t gmRemoteBaseOffset,
                                           __ubuf__ int32_t* metaInfoTensor, __ubuf__ Element* rowTensor,
                                           const Params& params)
{
    CombineTokenRoute route = LoadCombineTokenRoute(metaInfoTensor, 0U);

    __gm__ Element* gmRemoteD =
        reinterpret_cast<__gm__ Element*>(GetRankWinAddrWithOffset(route.dstRankId, gmRemoteBaseOffset));
    uint64_t gmDstRowOffset = GetCombineDstRowIndex(route, params) * rowElements;

    CopyUbToGm(gmRemoteD + gmDstRowOffset, rowTensor, static_cast<uint32_t>(rowElements * sizeof(Element)));
}

// 将一条量化 token 记录发送到 metadata 指定的 rank 和目标行。
template <typename QuantOutType>
__aicore__ inline void CombineQuantizedTokens(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                              uint32_t groupIdx, uint32_t rankId, __ubuf__ int32_t* metaInfoTensor,
                                              __ubuf__ QuantOutType* ubQuant, const Params& params,
                                              uint32_t quantTokenSizeBytes)
{
    const auto& peermem = params.communication.layout;
    CombineTokenRoute route = LoadCombineTokenRoute(metaInfoTensor, batchStart);

    __gm__ QuantOutType* gmRemoteD = nullptr;
    uint64_t gmRemoteOffset = static_cast<uint64_t>(peermem.combineSend.offsetBytes);
    __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(route.dstRankId, gmRemoteOffset);
    gmRemoteD = reinterpret_cast<__gm__ QuantOutType*>(dstPeermemPtr);

    uint64_t dstBaseOffset = GetCombineDstRowIndex(route, params) * quantTokenSizeBytes;

    CopyUbToGm(gmRemoteD + dstBaseOffset, ubQuant, quantTokenSizeBytes);
}

} // namespace CombineImpl

// 调用方决定参与核与阶段；这里只计算 Combine 行环布局并绑定当前核的 UB 视图。
template <uint8_t CombineMode, uint32_t MaxRowBufferCount = WAVE_COMBINE_MAX_ROW_BUFFER_COUNT>
__aicore__ inline WaveCombineBufferConfig InitWaveCombineBuffers(const MoeStageCommonConfig& common,
                                                                 WaveCombineScratch& scratch)
{
    static_assert(MaxRowBufferCount > 0U && MaxRowBufferCount <= WAVE_COMBINE_MAX_ROW_BUFFER_COUNT,
                  "invalid Combine row buffer count");
    WaveCombineBufferConfig bufferConfig{};
    bufferConfig.rowBytes = common.tokenHiddenDim * sizeof(bfloat16_t);
    bufferConfig.rowStrideBytes = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(bufferConfig.rowBytes), static_cast<uint64_t>(ALIGN_32)));
    bufferConfig.slotStrideBytes = bufferConfig.rowStrideBytes;
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        uint32_t nScale = Ops::Base::CeilDiv(common.tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        uint32_t tokenStorageBytes = Ops::Base::CeilAlign(common.tokenHiddenDim, static_cast<uint32_t>(ALIGN_256));
        uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
        bufferConfig.quantRowStorageBytes =
            Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
        bufferConfig.quantRowElements = bufferConfig.quantRowStorageBytes;
        bufferConfig.slotStrideBytes += bufferConfig.quantRowStorageBytes;
        bufferConfig.quantTempElements =
            Ops::Base::CeilAlign(storedScaleBytes, static_cast<uint32_t>(ALIGN_32)) + storedScaleBytes / 2U;
    }
    bufferConfig.rowBufferCount = MaxRowBufferCount;
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        uint32_t quantTempBytes = bufferConfig.quantTempElements * sizeof(float);
        uint32_t rowRingBudgetBytes = WAVE_COMBINE_UB_LIMIT - WAVE_COMBINE_UB_BASE - quantTempBytes;
        constexpr uint32_t minRowBufferCount = MaxRowBufferCount < WAVE_COMBINE_MIN_ROW_BUFFER_COUNT ?
                                                   MaxRowBufferCount :
                                                   WAVE_COMBINE_MIN_ROW_BUFFER_COUNT;
        bufferConfig.rowBufferCount = rowRingBudgetBytes / bufferConfig.slotStrideBytes;
        bufferConfig.rowBufferCount =
            bufferConfig.rowBufferCount < minRowBufferCount ? minRowBufferCount : bufferConfig.rowBufferCount;
        bufferConfig.rowBufferCount =
            bufferConfig.rowBufferCount > MaxRowBufferCount ? MaxRowBufferCount : bufferConfig.rowBufferCount;
    }
    uint32_t rowRingBytes = bufferConfig.rowBufferCount * bufferConfig.slotStrideBytes;
    scratch.rowBufferTensor = reinterpret_cast<__ubuf__ bfloat16_t*>(WAVE_COMBINE_UB_BASE);
    scratch.metaInfoTensor = reinterpret_cast<__ubuf__ int32_t*>(META_INFO_TENSOR_ADDR);
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        scratch.quantTempTensor = reinterpret_cast<__ubuf__ float*>(WAVE_COMBINE_UB_BASE + rowRingBytes);
    }
    return bufferConfig;
}

// 构造量化 Unpermute 输入记录的单 token 存储布局。
__aicore__ inline QuantTokenBufferConfig CreateQuantTokenBufferConfig(uint32_t tokenHiddenDim)
{
    uint32_t nScale = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
    uint32_t tokenStorageBytes = Ops::Base::CeilAlign(tokenHiddenDim, static_cast<uint32_t>(ALIGN_256));
    uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
    uint32_t quantTokenSizeBytes =
        Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
    return {quantTokenSizeBytes};
}

// 按专家累计 token 行数滚动物理 AIV 起点，每个 AIV 在当前专家内仍负责一个连续均衡区间。
// arch35 常规 Wave 使用 AIV1，最终专家可展开两组 AIV；arch92 始终使用唯一 AIV 的原始任务编号。
template <bool IncludeAiv0 = false>
__aicore__ inline WorkRange GetWaveCombineOwnedRange(const AivJobContext& job, uint32_t tokenCount,
                                                     uint64_t expertRowPrefix)
{
    if constexpr (g_coreType == AIC) {
        return {};
    }
#ifdef MEGA_MOE_ARCH92_DEVICE
    return GetRotatedBalancedWorkRange(tokenCount, job.jobIndex, job.totalJobs, expertRowPrefix);
#else
    AivJobContext combineJob = job;
    if constexpr (IncludeAiv0) {
        constexpr uint32_t AIV_CORE_COUNT_PER_AIC = MegaMoeHardwarePolicy::VECTOR_CORE_COUNT_PER_CUBE;
        uint32_t vectorRoleIdx = MegaMoeHardwarePolicy::GetSubBlockIdx();
        combineJob = {job.jobIndex * AIV_CORE_COUNT_PER_AIC + vectorRoleIdx, job.totalJobs * AIV_CORE_COUNT_PER_AIC};
    } else if (MegaMoeHardwarePolicy::GetSubBlockIdx() != 1U) {
        return {};
    }
    if (combineJob.totalJobs == 0U || combineJob.jobIndex >= combineJob.totalJobs) {
        return {};
    }
    return GetRotatedBalancedWorkRange(tokenCount, combineJob.jobIndex, combineJob.totalJobs, expertRowPrefix);
#endif
}

// 等待并消费当前 Wave 的 Combine 发送在行环形 buffer 上产生的全部完成事件，使该环不再有在途 slot。
__aicore__ inline void DrainCombineRowBuffers(uint32_t& issuedRowCount, uint32_t rowBufferCount)
{
    uint32_t activeSlotCount = issuedRowCount < rowBufferCount ? issuedRowCount : rowBufferCount;
    for (uint32_t slot = 0U; slot < activeSlotCount; ++slot) {
        PipeWait<PIPE_MTE3, PIPE_MTE2>(
            static_cast<TEventID>(static_cast<int32_t>(EVENT_ID0) + static_cast<int32_t>(slot)));
    }
    /*
     * Drain 会消费所有在途事件，下一段必须从空闲的 slot 0 重新开始；否则该段会把已消费的事件
     * 误判为 buffer 复用依赖并再次等待。W4 的计数跨 Wave 存活，A8W8 则在每次 Combine 调用时
     * 创建新计数；在这里统一重置，可以保证两种生命周期都遵守相同的行环起始状态。
     */
    issuedRowCount = 0U;
}

// 加载一段连续 wave Combine token 的 metadata。
__aicore__ inline void PreloadWaveCombineMetaInfo(const Params& params, WaveCombineScratch& scratch,
                                                  uint64_t gmTokenOffset, uint32_t tokenCount, uint32_t ubTokenOffset)
{
    const auto& workspace = params.workspace.layout;
    const auto& buffer = params.workspace.base;
    if (tokenCount == 0U) {
        return;
    }
    __ubuf__ int32_t* metaInfoUb = (scratch.metaInfoTensor + (ubTokenOffset * META_INFO_SIZE));
    __gm__ int32_t* metaInfoGm = workspace.routing.receivedTokenMetadata.get_ptr<int32_t>(buffer);
    CopyGmToUb(metaInfoUb, (metaInfoGm + (gmTokenOffset * META_INFO_SIZE)),
               (tokenCount * META_INFO_SIZE) * sizeof(int32_t));
    PipeNotify<PIPE_MTE2, PIPE_S>(EVENT_ID0);
    PipeWait<PIPE_MTE2, PIPE_S>(EVENT_ID0);
}

// 搬运并按需量化一个 wave Combine token 后发送；函数内部不包含 GMM2-ready 依赖。
template <uint8_t CombineMode, bool IsBufferReuse>
__aicore__ inline void SendWaveCombineToken(const MoeStageCommonConfig& common,
                                            const WaveCombineBufferConfig& bufferConfig, WaveCombineScratch& scratch,
                                            const Params& params, __gm__ bfloat16_t* gmm2OutGm,
                                            uint64_t gmRemoteBaseOffset, uint32_t tokenLocal,
                                            __ubuf__ int32_t* tokenMetaInfo, uint32_t slot)
{
    TEventID eventId = static_cast<TEventID>(static_cast<int32_t>(EVENT_ID0) + static_cast<int32_t>(slot));
    if constexpr (IsBufferReuse) {
        PipeWait<PIPE_MTE3, PIPE_MTE2>(eventId);
    }
    uint32_t slotElementOffset = slot * bufferConfig.slotStrideBytes / sizeof(bfloat16_t);
    __ubuf__ bfloat16_t* rowUb = (scratch.rowBufferTensor + (slotElementOffset));

    CopyGmToUb(rowUb, (gmm2OutGm + (static_cast<uint64_t>(tokenLocal) * common.tokenHiddenDim)), bufferConfig.rowBytes);
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        PipeNotify<PIPE_MTE2, PIPE_MTE3>(eventId);
        PipeWait<PIPE_MTE2, PIPE_MTE3>(eventId);
        CombineImpl::SendCombineTokenRow<bfloat16_t>(common.tokenHiddenDim, gmRemoteBaseOffset, tokenMetaInfo, rowUb,
                                                     params);
    } else {
        __ubuf__ bfloat16_t* quantUb =
            (scratch.rowBufferTensor + (slotElementOffset + bufferConfig.rowStrideBytes / sizeof(bfloat16_t)));
        PipeNotify<PIPE_MTE2, PIPE_V>(eventId);
        PipeWait<PIPE_MTE2, PIPE_V>(eventId);
        Mxfp8::QuantMxFp8<CombineMode, bfloat16_t>(quantUb, rowUb, scratch.quantTempTensor, common.tokenHiddenDim);
        PipeNotify<PIPE_V, PIPE_MTE3>(eventId);
        PipeWait<PIPE_V, PIPE_MTE3>(eventId);
        using Fp8Type = typename std::conditional<CombineMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
        __ubuf__ Fp8Type* quantSendUb = reinterpret_cast<__ubuf__ Fp8Type*>(quantUb);
        CombineImpl::SendCombineTokenRow<Fp8Type>(bufferConfig.quantRowElements, gmRemoteBaseOffset, tokenMetaInfo,
                                                  quantSendUb, params);
    }
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(eventId);
}

// 使用行级流水发送当前逻辑 AIV 任务负责的已就绪 token 区间。
template <uint8_t CombineMode>
__aicore__ inline void CombineWaveTokenRange(const MoeStageCommonConfig& common,
                                             const WaveCombineBufferConfig& bufferConfig, WaveCombineScratch& scratch,
                                             const Params& params, GM_ADDR gmm2OutGlobal, uint32_t tokenStart,
                                             uint32_t tokenCount, uint32_t metaInfoUbTokenOffset, uint32_t& rowSequence)
{
    const auto& peermem = params.communication.layout;
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (tokenCount == 0U) {
        return;
    }
    __gm__ bfloat16_t* gmm2OutGm = reinterpret_cast<__gm__ bfloat16_t*>(gmm2OutGlobal);
    uint64_t gmRemoteBaseOffset = static_cast<uint64_t>(peermem.combineSend.offsetBytes);
    uint32_t rowBufferCount = bufferConfig.rowBufferCount;
    uint32_t tokenIdxInSlice = 0U;
    uint32_t firstUseCount = rowSequence < rowBufferCount ? rowBufferCount - rowSequence : 0U;
    firstUseCount = firstUseCount < tokenCount ? firstUseCount : tokenCount;
    for (; tokenIdxInSlice < firstUseCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % rowBufferCount;
        __ubuf__ int32_t* tokenMetaInfo =
            (scratch.metaInfoTensor + ((metaInfoUbTokenOffset + tokenIdxInSlice) * META_INFO_SIZE));
        SendWaveCombineToken<CombineMode, false>(common, bufferConfig, scratch, params, gmm2OutGm, gmRemoteBaseOffset,
                                                 tokenStart + tokenIdxInSlice, tokenMetaInfo, slot);
    }
    for (; tokenIdxInSlice < tokenCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % rowBufferCount;
        __ubuf__ int32_t* tokenMetaInfo =
            (scratch.metaInfoTensor + ((metaInfoUbTokenOffset + tokenIdxInSlice) * META_INFO_SIZE));
        SendWaveCombineToken<CombineMode, true>(common, bufferConfig, scratch, params, gmm2OutGm, gmRemoteBaseOffset,
                                                tokenStart + tokenIdxInSlice, tokenMetaInfo, slot);
    }
}

// 等待当前 GMM2 tile 对应的 Activation 输入就绪。
template <typename Config>
__aicore__ inline void WaitForGmm2InputReady(const GMMAddrInfo& gmmAddrInfo, const Config& config, uint32_t mLoc)
{
    if (gmmAddrInfo.activationToGmm2Flag == nullptr) {
        return;
    }
    const uint32_t sourceNFactor = config.activationOutputDivisor;
    if constexpr (Config::IS_WAVE_FLAG_GRAINED) {
        uint32_t waveIdx = mLoc / L1_TILE_M_256;
        uint32_t waveStart = waveIdx * L1_TILE_M_256;
        uint32_t waveM = waveStart + L1_TILE_M_256 > config.m ? config.m - waveStart : L1_TILE_M_256;
        uint32_t targetLoops =
            Ops::Base::CeilDiv(waveM, config.activationTileM) * Ops::Base::CeilDiv(config.k * sourceNFactor, L1_TILE_N);
        uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
        __gm__ int32_t* flagValueAddr = gmmAddrInfo.activationToGmm2Flag + flagOffset;
        WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetLoops));
    } else {
        GmmKernel::BlockScheduler gmmBlockScheduler(
            {config.m, config.k * sourceNFactor, config.n},
            GmmKernel::BlockScheduler::Params{
                asc::te::make_coord(static_cast<int64_t>(config.activationTileM), static_cast<int64_t>(L1_TILE_N))});
        uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
        WaitUntilGmFlagEquals(gmmAddrInfo.activationToGmm2Flag, static_cast<int32_t>(targetLoops));
    }
}

// Notifies all AIV consumers assigned to the completed GMM2 token group.
__aicore__ inline void NotifyCombineConsumersOfTileCompletion(uint32_t rowTileOffset,
                                                              const GroupSyncSlotLayout& slotLayout,
                                                              __gm__ int32_t* expertCounterBase)
{
    PipeNotify<PIPE_FIX, PIPE_S>(0);
    PipeWait<PIPE_FIX, PIPE_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    uint32_t firstSyncSlot = 0;
    uint32_t syncSlotCount = 0;
    GetGroupSyncSlotRange(tokenGroupIndex, slotLayout, firstSyncSlot, syncSlotCount);
    for (uint32_t syncSlot = firstSyncSlot; syncSlot < firstSyncSlot + syncSlotCount; ++syncSlot) {
        atomicAdd(GetCombineSyncCounterAddress(expertCounterBase, syncSlot), int32_t(1));
    }
}

// Notifies the AIV consumer assigned to the completed shared-expert token group.
__aicore__ inline void NotifySharedExpertTileCompletion(uint32_t rowTileOffset, __gm__ int32_t* sharedExpertCounterBase)
{
    PipeNotify<PIPE_FIX, PIPE_S>(0);
    PipeWait<PIPE_FIX, PIPE_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    atomicAdd(GetCombineSyncCounterAddress(sharedExpertCounterBase, tokenGroupIndex), int32_t(1));
}

namespace GmmKernel {

// 执行通用 GMM2 tile 循环。
template <uint8_t CombineQuantMode, typename BlockMmad, bool IsShared, bool IsLayered = false,
          bool NotifyCombineTileReady = false, typename WorkSet, typename Config>
__aicore__ inline void Gmm2AicMmadGeneric(BlockMmad& blockMmad, WorkSet& workSet, const GMMAddrInfo& gmmAddrInfo,
                                          const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                          uint32_t rowOffsetInExpert = 0U, uint32_t expertTokenCount = 0U)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
        uint32_t logicalCoreCount = gmmAddrInfo.gmm2CombineLogicalCoreCount;
        if (logicalCoreCount == 0U) {
            logicalCoreCount = config.blockNum;
            if constexpr (!std::remove_reference_t<decltype(config)>::IS_WAVE_FLAG_GRAINED) {
                logicalCoreCount *= 2U;
            }
        }
        // 分片计算仍按完整专家行数建立同步槽布局，未显式传入时沿用当前计算形状的行数。
        uint32_t fullExpertTokenCount = expertTokenCount != 0U ? expertTokenCount : config.m;
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(fullExpertTokenCount, logicalCoreCount);
    }

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        // Slice only builds tensor views. Keep it ahead of the synchronization waits so the
        // current tile's address calculation is not inserted into the compute critical path.
        auto gmBlockA = workSet.gmA.slice(asc::te::make_coord(mLoc, kLoc),
                                          asc::te::make_shape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        /* E8M0 scales are padded to an even count because GM->L1 moves 64-K scale pairs as b16. */
        auto gmBlockScaleA = workSet.gmScaleA.slice(asc::te::make_coord(mLoc, 0),
                                                    asc::te::make_shape(Get<M_VALUE>(actualShape), config.scaleK));

        if constexpr (std::remove_reference_t<decltype(config)>::IS_WAVE_FLAG_GRAINED) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                WaitForGmm2InputReady(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            WaitForGmm2InputReady(gmmAddrInfo, config, mLoc);
        }

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                   Get<K_VALUE>(actualShape), 0};

        auto gmBlockB = workSet.gmB.slice(asc::te::make_coord(kLoc, nLoc),
                                          asc::te::make_shape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB = workSet.gmScaleB.slice(asc::te::make_coord(0, nLoc),
                                                    asc::te::make_shape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto gmBlockC = workSet.gmC.slice(asc::te::make_coord(mLoc, nLoc),
                                          asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        if constexpr (NotifyCombineTileReady && !IsShared) {
            // 非去重路径逐 tile 发送；去重路径在完整专家结束后独立消费。
            gmmAddrInfo.gmm2CombineSync->WaitForCombine();
        }
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
        if constexpr (NotifyCombineTileReady && !IsShared) {
            // GMM2 tile 与配对 AIV1 的 Combine tile 一对一通知。
            gmmAddrInfo.gmm2CombineSync->NotifyCombine();
        } else if constexpr (IsLayered && !IsShared) {
            NotifyCombineConsumersOfTileCompletion(rowOffsetInExpert + mLoc, groupSyncSlotLayout,
                                                   gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsShared && !IsLayered) {
            NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
    }
}

// 非量化 Combine 的统一入口。AIV1 直接使用 GMM2 已构造的 scheduler，逐 tile 等待配对 AIC 并消费。
// 当前 GMM2 problem 的起始地址和 M 大小共同限定 token 范围，无需再次遍历专家或重建 scheduler。
// UB 从 64 KiB 起步，避开 AIV1 Dispatch 跨 wave 保留在低区的 cumsum 状态。
template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename Config>
__aicore__ inline void CombineTokenRange(Scheduler& scheduler, TensorC& l0cOutGm, const Params& params,
                                         const GMMAddrInfo& gmmAddrInfo, const Config& config, uint32_t startLoopIdx,
                                         uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        gmmAddrInfo.gmm2CombineSync->WaitForGmm2();
        auto tensorBlockGm = l0cOutGm.slice(asc::te::make_coord(mLoc, nLoc),
                                            asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        constexpr uint32_t tileUbBase = WAVE_COMBINE_UB_BASE;
        auto layoutTileUb = MakeLayoutC{}(config.tileM, L1_TILE_N);
        auto tensorBlockUb =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(tileUbBase), layoutTileUb);
        __ubuf__ ElementC* tileUb = reinterpret_cast<__ubuf__ ElementC*>(tileUbBase);
        auto copyGM2UB = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        asc::te::copy(copyGM2UB, tensorBlockUb, tensorBlockGm);

        int32_t lenTile = Get<M_VALUE>(actualShape);
        __gm__ int32_t* metaInfoGm = nullptr;
        __ubuf__ int32_t* metaInfoTensor = reinterpret_cast<__ubuf__ int32_t*>(META_INFO_TENSOR_ADDR);
        metaInfoGm = reinterpret_cast<__gm__ int32_t*>(gmmAddrInfo.metaInfoGlobal +
                                                       static_cast<uint64_t>(mLoc) * META_INFO_SIZE * sizeof(int32_t));
        CopyGmToUb(metaInfoTensor, metaInfoGm, (lenTile * META_INFO_SIZE) * sizeof(int32_t));
        PipeNotify<PIPE_MTE2, PIPE_S>(0);
        PipeWait<PIPE_MTE2, PIPE_S>(0);
        CombineImpl::CombineTokens<ElementC, decltype(actualShape)>(nLoc, config.n, metaInfoTensor, tileUb, actualShape,
                                                                    L1_TILE_N, params);
        PipeNotify<PIPE_MTE3, PIPE_MTE2>(0);
        PipeWait<PIPE_MTE3, PIPE_MTE2>(0);
        gmmAddrInfo.gmm2CombineSync->NotifyGmm2();
    }
}

// 执行 A8W4 GMM2 tile 循环。
template <bool IsShared, bool IsLayered, bool NotifyCombineTileReady, typename BlockMmad, typename Scheduler,
          typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC, typename Config>
__aicore__ inline void Gmm2AicMmadA8W4(BlockMmad& blockMmad, Scheduler& scheduler, TensorA& gmA, TensorScaleA& gmScaleA,
                                       TensorScaleB& gmScaleB, TensorC& l0cOutGm, const GMMAddrInfo& gmmAddrInfo,
                                       const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                       uint32_t expertTokenCount, uint32_t rowOffsetInExpert)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr (IsLayered && !IsShared) {
        uint32_t logicalCoreCount = gmmAddrInfo.gmm2CombineLogicalCoreCount;
        if (logicalCoreCount == 0U) {
            logicalCoreCount = config.blockNum;
        }
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(expertTokenCount, logicalCoreCount);
    }
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        if constexpr (Config::IS_WAVE_FLAG_GRAINED) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                WaitForGmm2InputReady(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            WaitForGmm2InputReady(gmmAddrInfo, config, mLoc);
        }

        auto gmBlockA =
            gmA.slice(asc::te::make_coord(mLoc, 0), asc::te::make_shape(Get<M_VALUE>(actualShape), config.k));
        auto gmBlockScaleA =
            gmScaleA.slice(asc::te::make_coord(mLoc, 0), asc::te::make_shape(Get<M_VALUE>(actualShape), config.scaleK));

        auto gmBlockScaleB =
            gmScaleB.slice(asc::te::make_coord(0, nLoc), asc::te::make_shape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = l0cOutGm.slice(asc::te::make_coord(mLoc, nLoc),
                                            asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        if constexpr (NotifyCombineTileReady && !IsShared) {
            // W4 路径的握手保留运行时判断：该链路经成员函数 RunGmm2CombineForExpert 下发，
            // 将去重开关做成其嵌套模板参数时（template<类参> template<bool>），编译器生成的
            // AIC 代码在 W4 实例上 scalar 访问 L1 越界（2026-09 实测，源码语义与运行时判断
            // 逐行等价仍崩）。Generic 路径经自由函数模板下发不受影响，保持编译期分化。
            if (gmmAddrInfo.gmm2CombineSync != nullptr) {
                gmmAddrInfo.gmm2CombineSync->WaitForCombine();
            }
        }
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
        if constexpr (NotifyCombineTileReady && !IsShared) {
            // AIV1 与本 AIC 重放同一 scheduler。FIX 完成后通知 Combine 读取 GM；
            // W4 prologue 仍由 AIV0 独立推进，不参与该一对一完成链路。
            if (gmmAddrInfo.gmm2CombineSync != nullptr) {
                gmmAddrInfo.gmm2CombineSync->NotifyCombine();
            }
        } else if constexpr (IsLayered && !IsShared) {
            NotifyCombineConsumersOfTileCompletion(rowOffsetInExpert + mLoc, groupSyncSlotLayout,
                                                   gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsShared && !IsLayered) {
            NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
    }
}

// 为一个 GMM2 逻辑 block 的 AIV0 路径展开一段 A8W4 权重。
template <typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void Gmm2Aiv0PrologueA8W4(BlockPrologue& blockPrologue, Scheduler& scheduler, TensorB& gmB,
                                            const Config& config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto nL1Size = Get<N_VALUE>(actualShape);
        blockPrologue(gmB, config.k, nL1Size, nLoc, config.n, 0U);
    }
}

// 根据 GM 地址建立执行资源，并执行通用 GMM2 阶段。

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM2 阶段。

} // namespace GmmKernel

// 紧凑 token 资源按累计行偏移寻址，专家固定资源按 expertIdx 寻址。
template <typename WeightType, typename ActivationOutType, typename QuantScaleType>
__aicore__ inline void UpdateMoeExpertGmm2GlobalBuffer(const GmmExecutionConfig& gmmConfig,
                                                       const MoeSyncWorkspaceLayout& syncLayout,
                                                       const WorkspaceLayout& workspace, GM_ADDR workspaceBase,
                                                       const ExpertWeightTensorListAddrs& weights,
                                                       GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                       uint32_t activationOutputWidth, uint32_t rowOffsetInExpert = 0U)
{
    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t activationOutputElementsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    uint64_t tokenHiddenDim = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
    uint64_t globalTokenStartIndex = static_cast<uint64_t>(state.globalTokenStartIndex) + rowOffsetInExpert;
    uint64_t physicalTokenStartIndex =
        gmmConfig.ring.activation ?
            CalcMteRingPhysicalRow(state.paddedTokenStartIndex + rowOffsetInExpert, gmmConfig.ring.rows) :
            globalTokenStartIndex;
    uint32_t expertMGroupOffset = rowOffsetInExpert / L1_TILE_M_256;
    uint64_t activationScaleWidth =
        Ops::Base::CeilDiv(static_cast<uint64_t>(activationOutputWidth), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;

    // 激活输入按自身循环容量读取；Combine 去重时结果按紧凑逻辑行全量保存。
    const uint64_t outputTokenStartIndex = gmmConfig.ring.gmm2 ? physicalTokenStartIndex : globalTokenStartIndex;
    gmmAddrInfo.gmm2OutGlobal = workspace.routed.gmm2Output.get_ptr(workspaceBase) +
                                outputTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
    gmmAddrInfo.metaInfoGlobal = workspace.routing.receivedTokenMetadata.get_ptr(workspaceBase) +
                                 globalTokenStartIndex * META_INFO_SIZE * sizeof(int32_t);
    gmmAddrInfo.aGlobal =
        workspace.routed.gmm2Input.data.get_ptr(workspaceBase) +
        physicalTokenStartIndex * activationOutputWidth / activationOutputElementsPerByte * sizeof(ActivationOutType);
    gmmAddrInfo.aScaleGlobal = workspace.routed.gmm2Input.scale.get_ptr(workspaceBase) +
                               physicalTokenStartIndex * activationScaleWidth * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<WeightType>(
        weights.weight2, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
        static_cast<uint64_t>(state.expertIdx) * tokenHiddenDim * activationOutputWidth / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
        static_cast<uint64_t>(state.expertIdx) * tokenHiddenDim * activationScaleWidth);
    gmmAddrInfo.activationToGmm2Flag =
        reinterpret_cast<__gm__ int32_t*>(workspace.sync.activationReady.get_ptr(workspaceBase)) +
        static_cast<uint64_t>(state.expertIdx) * syncLayout.activationFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag =
        reinterpret_cast<__gm__ int32_t*>(workspace.sync.dispatchReady.get_ptr(workspaceBase)) +
        static_cast<uint64_t>(state.expertIdx) * syncLayout.dispatchFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
}

// Combine 只消费当前专家的 GMM2 输出；避免复用完整 GMM2 绑定接口时计算无关的输入、权重和 flag 地址。
__aicore__ inline void UpdateMoeExpertCombineGlobalBuffer(const WorkspaceLayout& workspace, GM_ADDR workspaceBase,
                                                          GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state)
{
    uint64_t tokenHiddenDim = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
    uint64_t globalTokenStartIndex = static_cast<uint64_t>(state.globalTokenStartIndex);
    gmmAddrInfo.gmm2OutGlobal = workspace.routed.gmm2Output.get_ptr(workspaceBase) +
                                globalTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
}

template <typename ActivationType, typename WeightType, typename QuantScaleType, uint32_t Gmm1TileM>
__aicore__ inline void UpdateSharedExpertGmm2GlobalBuffer(const MoeStageCommonConfig& commonConfig,
                                                          const GmmExecutionConfig& gmmConfig,
                                                          const WorkspaceLayout& workspace, GM_ADDR workspaceBase,
                                                          const ExpertWeightTensorListAddrs& weights,
                                                          GMMAddrInfo& gmmAddrInfo, uint32_t sharedExpertIdx)
{
    constexpr uint32_t activationElementsPerByte = PackedElementTraits<ActivationType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    uint64_t tokenNum = commonConfig.tokenNum;
    uint64_t tokenHiddenDim = commonConfig.tokenHiddenDim;
    uint64_t activationOutputWidth = commonConfig.gmm1OutputDim / commonConfig.activationOutputDivisor;
    uint64_t activationScaleWidth =
        Ops::Base::CeilDiv(activationOutputWidth, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t sharedExpertTokenStartIndex = static_cast<uint64_t>(sharedExpertIdx) * tokenNum;

    gmmAddrInfo.gmm2OutGlobal = workspace.shared.gmm2Output.get_ptr(workspaceBase) +
                                sharedExpertTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
    gmmAddrInfo.aGlobal =
        workspace.shared.gmm2Input.data.get_ptr(workspaceBase) +
        sharedExpertTokenStartIndex * activationOutputWidth / activationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = workspace.shared.gmm2Input.scale.get_ptr(workspaceBase) +
                               sharedExpertTokenStartIndex * activationScaleWidth * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<WeightType>(
        weights.weight2, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx,
        static_cast<uint64_t>(sharedExpertIdx) * tokenHiddenDim * activationOutputWidth / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx,
        static_cast<uint64_t>(sharedExpertIdx) * tokenHiddenDim * activationScaleWidth);
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    if (workspace.mte.sync.sharedActivationReady.get_ptr(workspaceBase) != nullptr) {
        uint64_t sharedActivationFlagElementCount =
            static_cast<uint64_t>(CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(tokenNum)));
        gmmAddrInfo.activationToGmm2Flag =
            reinterpret_cast<__gm__ int32_t*>(workspace.mte.sync.sharedActivationReady.get_ptr(workspaceBase)) +
            static_cast<uint64_t>(sharedExpertIdx) * sharedActivationFlagElementCount;
    }
    gmmAddrInfo.sharedExpertGmm2TileCounter = nullptr;
    if (workspace.mte.sync.sharedGmm2Tiles.get_ptr(workspaceBase) != nullptr) {
        uint32_t tokenGroupCount = Ops::Base::CeilDiv(commonConfig.tokenNum, Gmm1TileM);
        gmmAddrInfo.sharedExpertGmm2TileCounter =
            reinterpret_cast<__gm__ int32_t*>(workspace.mte.sync.sharedGmm2Tiles.get_ptr(workspaceBase)) +
            static_cast<uint64_t>(sharedExpertIdx) * tokenGroupCount * INT_CACHELINE;
    }
}

constexpr uint32_t WAVE_GMM2_READY_SCAN_UB_ADDR = WAVE_COMBINE_UB_LIMIT;
constexpr uint32_t WAVE_GMM2_READY_MAX_SCAN_BYTES = MAX_AICORE_NUM * INT_CACHELINE * sizeof(int32_t) > ALIGN_512 ?
                                                        MAX_AICORE_NUM* INT_CACHELINE * sizeof(int32_t) :
                                                        ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR =
    WAVE_GMM2_READY_SCAN_UB_ADDR + ((WAVE_GMM2_READY_MAX_SCAN_BYTES + ALIGN_512 - 1U) / ALIGN_512) * ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_SUM_UB_ADDR = WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR + ALIGN_512;

__aicore__ inline uint64_t GetWaveGmm2ReadySlotStride(const AivJobContext& job)
{
    return static_cast<uint64_t>(job.totalJobs) * INT_CACHELINE;
}

// AIC 等待 Fixpipe 写完专家结果后发布，专家粒度 Combine 按原协议消费。
__aicore__ inline void NotifyWaveGmm2Ready(const AivJobContext& job, const Params& params, uint32_t slotIdx)
{
    const auto& workspace = params.workspace.layout;
    const auto& buffer = params.workspace.base;
    if constexpr (g_coreType == AIC) {
        PipeNotify<PIPE_FIX, PIPE_S>(0);
        PipeWait<PIPE_FIX, PIPE_S>(0);
        __gm__ int32_t* readyBase = workspace.mte.sync.gmm2Ready.get_ptr<int32_t>(buffer);
        uint64_t slotOffset = static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(job);
        GmWrite(readyBase + slotOffset + static_cast<uint64_t>(job.jobIndex) * INT_CACHELINE, int32_t(1));
    }
}

// 一个 AIV Combine 任务等待指定专家的全部 GMM2 结果落地。
#if defined(MEGA_MOE_ARCH92_DEVICE)
__aicore__ inline void WaitWaveGmm2Ready(const AivJobContext& job, const Params& params, uint32_t slotIdx)
{
    const auto& workspace = params.workspace.layout;
    const auto& buffer = params.workspace.base;
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (job.totalJobs == 0U) {
        return;
    }
    uint32_t readyElements = job.totalJobs * static_cast<uint32_t>(INT_CACHELINE);
    __gm__ int32_t* readyBase = workspace.mte.sync.gmm2Ready.get_ptr<int32_t>(buffer);
    uint64_t readySlotOffset = static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(job);
    __gm__ int32_t* readyGm = readyBase;
    __ubuf__ int32_t* scanUb = reinterpret_cast<__ubuf__ int32_t*>(WAVE_GMM2_READY_SCAN_UB_ADDR);
    __ubuf__ int32_t* sumUb = reinterpret_cast<__ubuf__ int32_t*>(WAVE_GMM2_READY_SUM_UB_ADDR);
    LocalTensor<int32_t> scan(TPosition::VECCALC, WAVE_GMM2_READY_SCAN_UB_ADDR, readyElements);
    LocalTensor<uint8_t> reduceTmp(TPosition::VECCALC, WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR, ALIGN_512);
    LocalTensor<int32_t> sum(TPosition::VECCALC, WAVE_GMM2_READY_SUM_UB_ADDR, INT_CACHELINE);
    const uint32_t readyShape[] = {1U, readyElements};
    while (true) {
        CopyGmToUb(scanUb, readyGm + readySlotOffset, (readyElements) * sizeof(int32_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(EVENT_ID0);
        PipeWait<PIPE_MTE2, PIPE_V>(EVENT_ID0);
        ReduceSum<int32_t, AscendC::Pattern::Reduce::AR, true>(sum, scan, reduceTmp, readyShape, true);
        PipeNotify<PIPE_V, PIPE_S>(EVENT_ID0);
        PipeWait<PIPE_V, PIPE_S>(EVENT_ID0);
        if (sumUb[0] >= static_cast<int32_t>(job.totalJobs)) {
            return;
        }
        int64_t waitStart = GetDeviceCycle();
        while (GetDeviceCycle() - waitStart < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
}
#else
// Compact搬入各槽首项后，只对实际任务取最小值；原scan预留区覆盖完整向量加载。
__simd_vf__ inline void CheckWaveGmm2ReadyVF(__ubuf__ int32_t* doneUb, __ubuf__ int32_t* scanUb, uint32_t jobCount)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = static_cast<uint32_t>(asc_get_vf_len()) / sizeof(int32_t);
    uint32_t remaining = jobCount;
    vector_bool active = asc_update_mask_b32(remaining);
    vector_int32_t inputReg;
    vector_int32_t resultReg;
    asc_loadalign(inputReg, scanUb);
    asc_reduce_min(resultReg, inputReg, active);
    // 常见任务数不超过一个向量；保留分块处理，避免未来任务数增加时被掩码截断。
    for (uint32_t offset = ELEMENTS_PER_VECTOR; remaining > 0U; offset += ELEMENTS_PER_VECTOR) {
        active = asc_update_mask_b32(remaining);
        asc_loadalign(inputReg, scanUb + offset);
        vector_int32_t blockMinReg;
        asc_reduce_min(blockMinReg, inputReg, active);
        // 仅保留各块lane0的最小值，非空块的active始终包含lane0。
        asc_min(resultReg, resultReg, blockMinReg, active);
    }
    asc_storealign_1st(doneUb, resultReg);
}

__aicore__ inline void WaitWaveGmm2Ready(const AivJobContext& job, const Params& params, uint32_t slotIdx)
{
    const auto& workspace = params.workspace.layout;
    const auto& buffer = params.workspace.base;
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (job.totalJobs == 0U) {
        return;
    }
    __gm__ int32_t* readyBase = workspace.mte.sync.gmm2Ready.get_ptr<int32_t>(buffer);
    uint64_t readySlotOffset = static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(job);
    __gm__ int32_t* readyGm = readyBase;
    __ubuf__ int32_t* scanUb = reinterpret_cast<__ubuf__ int32_t*>(WAVE_GMM2_READY_SCAN_UB_ADDR);
    __ubuf__ int32_t* sumUb = reinterpret_cast<__ubuf__ int32_t*>(WAVE_GMM2_READY_SUM_UB_ADDR);
    while (true) {
        CopyGmToUb(scanUb, readyGm + readySlotOffset,
                   CopyRows{static_cast<uint16_t>(job.totalJobs), sizeof(int32_t), INT_CACHELINE * sizeof(int32_t),
                            sizeof(int32_t)},
                   GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
        PipeNotify<PIPE_MTE2, PIPE_V>(EVENT_ID0);
        PipeWait<PIPE_MTE2, PIPE_V>(EVENT_ID0);
        if constexpr (g_coreType == AIV) {
            CheckWaveGmm2ReadyVF(sumUb, scanUb, job.totalJobs);
        }
        PipeNotify<PIPE_V, PIPE_S>(EVENT_ID0);
        PipeWait<PIPE_V, PIPE_S>(EVENT_ID0);
        if (sumUb[0] != 0) {
            return;
        }
        int64_t waitStart = GetDeviceCycle();
        while (GetDeviceCycle() - waitStart < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
}
#endif

template <uint8_t CombineMode, bool IncludeAiv0 = false>
__aicore__ inline void RunWaveCombineStage(const MoeStageCommonConfig& common, const AivJobContext& job,
                                           const WaveCombineBufferConfig& bufferConfig, WaveCombineScratch& scratch,
                                           const Params& params, const GMMAddrInfo& gmmAddrInfo,
                                           const ExpertLoopState& state, uint32_t expertIdx, uint32_t& rowSequence)
{
    uint32_t currentExpertTokenNum = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    WorkRange currentCoreTokenRange = GetWaveCombineOwnedRange<IncludeAiv0>(
        job, currentExpertTokenNum, static_cast<uint64_t>(state.globalTokenStartIndex));
    if (currentCoreTokenRange.count != 0U) {
        WaitWaveGmm2Ready(job, params, expertIdx);
    }
    for (uint32_t processedTokenNum = 0U; processedTokenNum < currentCoreTokenRange.count;) {
        uint32_t remainingTokenNum = currentCoreTokenRange.count - processedTokenNum;
        uint32_t chunkTokenNum = remainingTokenNum < WAVE_COMBINE_META_INFO_TOKEN_CAPACITY ?
                                     remainingTokenNum :
                                     WAVE_COMBINE_META_INFO_TOKEN_CAPACITY;
        PreloadWaveCombineMetaInfo(
            params, scratch,
            static_cast<uint64_t>(state.globalTokenStartIndex) + currentCoreTokenRange.start + processedTokenNum,
            chunkTokenNum, 0U);
        CombineWaveTokenRange<CombineMode>(common, bufferConfig, scratch, params, gmmAddrInfo.gmm2OutGlobal,
                                           currentCoreTokenRange.start + processedTokenNum, chunkTokenNum, 0U,
                                           rowSequence);
        processedTokenNum += chunkTokenNum;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_H
