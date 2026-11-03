/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM1_ACTIVATION_H
#define MEGA_MOE_GMM1_ACTIVATION_H

#include "../common/mega_moe_sync.h"
#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "../blaze/epilogue/mega_moe_block_epilogue_activation_mx_quant.h"
#include "../common/mega_moe_arch_policy.h"
#include "../common/mega_moe_gmm_common.h"

namespace MegaMoeImpl {

using namespace AscendC;

constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = 256U * 256U;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = 128U * 256U;

// 等待当前 GMM1 tile 对应的 Dispatch 输入就绪。
template <bool IsShared, typename Config>
__aicore__ inline void WaitForGmm1InputReady(const GMMAddrInfo& gmmAddrInfo, const Config& config, uint32_t mLoc)
{
    if constexpr (IsShared) {
        return;
    }
    uint32_t waveIdx = mLoc / config.tileM;
    uint32_t targetValue = (mLoc + config.tileM > config.m) ? (config.m - mLoc) : config.tileM;
    uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
    __gm__ int32_t* flagValueAddr = gmmAddrInfo.dispatchToGmm1Flag + flagOffset;
    WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetValue));
}

namespace GmmKernel {

template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorC, typename TensorMetaInfo>
struct Gmm1WorkSet {
    Scheduler& scheduler;
    TensorA& gmA;
    TensorB& gmB;
    TensorScaleA& gmScaleA;
    TensorScaleB& gmScaleB;
    TensorC& gmC;
    TensorMetaInfo& metaInfoGm;
};

template <bool IsWaveFlagGrained>
__aicore__ inline void NotifyGmm2InputReady(__gm__ int32_t* flagBase, uint32_t flagSlotIdx = 0U)
{
    if (flagBase == nullptr) {
        return;
    }
    if constexpr (IsWaveFlagGrained) {
        flagBase += static_cast<uint64_t>(flagSlotIdx) * INT_CACHELINE;
    }
    atomicAdd(flagBase, 1);
}

// 计算一个 Prefetch GMM1 tile 并直接写入 GM。
template <typename BlockMmad, typename Config, typename TensorB, typename TensorScaleB, typename ActualShape,
          typename TensorA, typename TensorScaleA, typename TensorC>
__aicore__ inline void Gmm1AicMmadTileToGmGeneric(BlockMmad& blockMmad, const Config& config, TensorB& gmB,
                                                  TensorScaleB& gmScaleB, TensorC& gmC, const ActualShape& actualShape,
                                                  uint32_t mLoc, uint32_t nLoc, uint32_t kLoc, TensorA& gmBlockA,
                                                  TensorScaleA& gmBlockScaleA)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    Get<N_VALUE>(singleShape) /= config.activationOutputDivisor;
    auto gmBlockB = gmB.slice(asc::te::make_coord(kLoc, nLoc / config.activationOutputDivisor),
                              asc::te::make_shape(Get<K_VALUE>(actualShape), Get<N_VALUE>(singleShape)));
    auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(0, nLoc / config.activationOutputDivisor),
                                        asc::te::make_shape(config.scaleK, Get<N_VALUE>(singleShape)));
    auto tensorBlockGm = gmC.slice(asc::te::make_coord(mLoc, nLoc),
                                   asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    Gmm1OutputView<decltype(tensorBlockGm)> outputView{tensorBlockGm};
    if (config.activationOutputDivisor == 1U) {
        using KernelConfig = typename Config::KernelConfig;
        using BiasType = typename KernelConfig::BiasType;
        // 普通 MMAD 入口要求传入 bias 张量；Init 已关闭 bias，因此不会读取该空地址。
        auto gmBias =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BiasType*>(0UL)),
                                 typename KernelConfig::MakeLayoutBias{}(1U, config.n));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, outputView, singleShape);
    } else {
        auto weightView = MakeGmm1WeightView(gmBlockB);
        blockMmad(gmBlockA, weightView, gmBlockScaleA, gmBlockScaleB, outputView, singleShape);
    }
}

// 计算一个非 Prefetch GMM1 tile 并写入固定 tile layout 的 UB。
template <typename BlockMmad, typename Config, typename TensorB, typename TensorScaleB, typename ActualShape,
          typename TensorA, typename TensorScaleA, typename TensorC>
__aicore__ inline void Gmm1AicMmadTileToUbGeneric(BlockMmad& blockMmad, const Config& config, TensorB& gmB,
                                                  TensorScaleB& gmScaleB, const ActualShape& actualShape, uint32_t nLoc,
                                                  uint32_t kLoc, TensorA& gmBlockA, TensorScaleA& gmBlockScaleA,
                                                  TensorC& l0cOutUbFirst, TensorC& l0cOutUbSecond, uint16_t pingpongIdx)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    Get<N_VALUE>(singleShape) /= config.activationOutputDivisor;
    auto gmBlockB = gmB.slice(asc::te::make_coord(kLoc, nLoc / config.activationOutputDivisor),
                              asc::te::make_shape(Get<K_VALUE>(actualShape), Get<N_VALUE>(singleShape)));
    auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(0, nLoc / config.activationOutputDivisor),
                                        asc::te::make_shape(config.scaleK, Get<N_VALUE>(singleShape)));
    auto tensorUb = pingpongIdx == 0U ? l0cOutUbFirst : l0cOutUbSecond;
    auto tensorBlockUb = tensorUb.slice(asc::te::make_coord(0, 0),
                                        asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    Gmm1OutputView<decltype(tensorBlockUb)> outputView{tensorBlockUb};
    if (config.activationOutputDivisor == 1U) {
        using KernelConfig = typename Config::KernelConfig;
        using BiasType = typename KernelConfig::BiasType;
        // 普通 MMAD 入口要求传入 bias 张量；Init 已关闭 bias，因此不会读取该空地址。
        auto gmBias =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BiasType*>(0UL)),
                                 typename KernelConfig::MakeLayoutBias{}(1U, config.n));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, outputView, singleShape);
    } else {
        auto weightView = MakeGmm1WeightView(gmBlockB);
        blockMmad(gmBlockA, weightView, gmBlockScaleA, gmBlockScaleB, outputView, singleShape);
    }
}

// AIC 执行一个 A8W4 GMM1 MMAD tile；不管理 Dispatch/GMM1 或 GMM1/Activation 同步，
// AIV0 与 AIC 的 Blaze 内部握手仍由 GMM 实现负责。
template <typename BlockMmad, typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC,
          typename Config, typename ActualShape>
__aicore__ inline void Gmm1AicMmadTileA8W4(BlockMmad& blockMmad, TensorA& gmA, TensorScaleA& gmScaleA,
                                           TensorScaleB& gmScaleB, TensorC& l0cOutGm, const Config& config,
                                           const ActualShape& actualShape, uint32_t mLoc, uint32_t nLoc)
{
    auto gmBlockA = gmA.slice(asc::te::make_coord(mLoc, 0), asc::te::make_shape(Get<M_VALUE>(actualShape), config.k));
    auto gmBlockScaleA =
        gmScaleA.slice(asc::te::make_coord(mLoc, 0), asc::te::make_shape(Get<M_VALUE>(actualShape), config.scaleK));

    uint32_t nOffset = nLoc / config.activationOutputDivisor;
    auto gmBlockScaleB =
        gmScaleB.slice(asc::te::make_coord(0, nOffset), asc::te::make_shape(config.scaleK, Get<N_VALUE>(actualShape)));
    auto tensorBlockGm = l0cOutGm.slice(asc::te::make_coord(mLoc, nLoc),
                                        asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
}

// GM 交接路径的 AIC 将 GMM1 tile 写入 GM，再通过硬同步通知 AIV 后处理。
template <typename MmadContext, bool IsShared, bool TopkWeightsPrefetch, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicMmadToGmGeneric(WorkSet& workSet, const Params& params, const GMMAddrInfo& gmmAddrInfo,
                                              const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                              uint32_t expertIdx, MmadContext* blockMmadContext)
{
    auto& sync = *gmmAddrInfo.gmm1ActivationSync;
    using BlockMmad = decltype(blockMmadContext->blockMmad);
    blockMmadContext->template Init<GmmStage::GMM1>(config);
    auto& blockMmad = blockMmadContext->blockMmad;

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);
        auto gmBlockA = workSet.gmA.slice(asc::te::make_coord(mLoc, kLoc),
                                          asc::te::make_shape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        /*
         * E8M0 scales are stored in 64-K pairs. Keep the padded even scale span in the tensor view;
         * GM->L1 搬运将每一对 scale 作为一个 16 位元素读取。
         */
        auto gmBlockScaleA = workSet.gmScaleA.slice(asc::te::make_coord(mLoc, 0),
                                                    asc::te::make_shape(Get<M_VALUE>(actualShape), config.scaleK));
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            WaitForGmm1InputReady<IsShared>(gmmAddrInfo, config, mLoc);
            lastWaveWaited = waveIdx;
        }
        sync.WaitForActivation();
        Gmm1AicMmadTileToGmGeneric<BlockMmad>(blockMmad, config, workSet.gmB, workSet.gmScaleB, workSet.gmC,
                                              actualShape, mLoc, nLoc, kLoc, gmBlockA, gmBlockScaleA);
        sync.NotifyActivation();
    }
}

// 非 Prefetch 路径的 AIC 将 GMM1 tile 写入 UB，并维持与 AIV 的 ping-pong 同步。
template <typename MmadContext, bool IsShared, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicMmadGeneric(WorkSet& workSet, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                          uint32_t startLoopIdx, uint32_t tileNum, int32_t& vecSetSyncCom,
                                          uint16_t& pingpongIdx, MmadContext* blockMmadContext)
{
    Gmm1UbActivationSync sync(vecSetSyncCom, pingpongIdx);
    using BlockMmad = decltype(blockMmadContext->blockMmad);
    blockMmadContext->template Init<GmmStage::GMM1>(config);
    auto& blockMmad = blockMmadContext->blockMmad;

    using KernelConfig = typename Config::KernelConfig;
    using ElementC = typename KernelConfig::ElementCType;
    using MakeLayoutC = typename KernelConfig::MakeLayoutC;
    uint32_t ubBufSize = config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    auto ubLayout = MakeLayoutC{}(config.tileM, L1_TILE_N);
    auto l0cOutUbFirst =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(ubOffsetFirst), ubLayout);
    auto l0cOutUbSecond =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(ubOffsetSecond), ubLayout);

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);
        auto gmBlockA = workSet.gmA.slice(asc::te::make_coord(mLoc, kLoc),
                                          asc::te::make_shape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        auto gmBlockScaleA = workSet.gmScaleA.slice(asc::te::make_coord(mLoc, 0),
                                                    asc::te::make_shape(Get<M_VALUE>(actualShape), config.scaleK));
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            WaitForGmm1InputReady<IsShared>(gmmAddrInfo, config, mLoc);
            lastWaveWaited = waveIdx;
        }
        sync.WaitForActivation();

        Gmm1AicMmadTileToUbGeneric<BlockMmad>(blockMmad, config, workSet.gmB, workSet.gmScaleB, actualShape, nLoc, kLoc,
                                              gmBlockA, gmBlockScaleA, l0cOutUbFirst, l0cOutUbSecond, pingpongIdx);

        sync.NotifyActivation();
    }
}

// 执行 A8W4 GMM1 AIC tile 循环，并维持 Dispatch/GMM1 和 GMM1/Activation 的同步关系。
template <typename BlockMmad, bool IsShared, bool TopkWeightsPrefetch, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicMmadA8W4(BlockMmad& blockMmad, WorkSet& workSet, const Params& params,
                                       const GMMAddrInfo& gmmAddrInfo, const Config& config, uint32_t startLoopIdx,
                                       uint32_t tileNum, uint32_t expertIdx)
{
    auto& sync = *gmmAddrInfo.gmm1ActivationSync;
    typename BlockMmad::BlockShape l0TileShape{config.blockMmadTiling.tileM, config.blockMmadTiling.tileN, L0_TILE_K,
                                               0};
    typename BlockMmad::ProblemShape matmulShape{config.m, config.schedulerN, config.k, 0};
    // GELU 连续搬运 W1 和 ScaleB，门控激活沿用两个物理半区的交织搬运。
    const bool interleave = config.activationOutputDivisor != 1U;
    blockMmad.template Init<false>(matmulShape, l0TileShape, interleave);

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            WaitForGmm1InputReady<IsShared>(gmmAddrInfo, config, mLoc);
            lastWaveWaited = waveIdx;
        }

        sync.WaitForActivation();
        Gmm1AicMmadTileA8W4(blockMmad, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC, config,
                            actualShape, mLoc, nLoc);

        sync.NotifyActivation();
    }
}

// Prefetch 交错路径从 GM 取回一个子 tile，执行 Activation/量化并通知 GMM2。
template <typename MakeLayoutC, typename ElementC, bool IsWaveFlagGrained, typename Config, typename TensorC,
          typename TensorMetaInfo, typename ActivationQuantOp, typename ActualShape>
__aicore__ inline void Gmm1Aiv0PrefetchEpilogueTileGeneric(const Config& config, TensorC& gmC,
                                                           TensorMetaInfo& metaInfoGm,
                                                           ActivationQuantOp& activationQuantOp,
                                                           const GMMAddrInfo& gmmAddrInfo, uint32_t expertBeforeCnt,
                                                           const ActualShape& actualShape, uint32_t mLoc, uint32_t nLoc)
{
    uint32_t mLen = Get<M_VALUE>(actualShape);
    if (mLen == 0U) {
        return;
    }
    constexpr uint32_t subTileM = L1_TILE_M_128;
    auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
    auto tensorBlockUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(0), layoutL0cUb);
    auto copyGm2Ub = asc::te::make_copy(asc::te::copy_gm_to_ub{});
    auto topkWeightTensor = activationQuantOp.GetTopkWeightTensor();

    PipeNotify<PIPE_V, PIPE_MTE2>(0);
    for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
        uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
        uint32_t subMLoc = mLoc + subOffset;
        uint64_t metaInfoRow = expertBeforeCnt + subMLoc;
        PipeWait<PIPE_V, PIPE_MTE2>(0);
        CopyGmToUb(topkWeightTensor, (metaInfoGm + (metaInfoRow * INT32_PER_256B)),
                   (subM * INT32_PER_256B) * sizeof(int32_t));
        auto tensorBlockGm =
            gmC.slice(asc::te::make_coord(subMLoc, nLoc), asc::te::make_shape(subM, Get<N_VALUE>(actualShape)));
        asc::te::copy(copyGm2Ub, tensorBlockUb, tensorBlockGm);
        PipeNotify<PIPE_MTE2, PIPE_V>(0);
        PipeWait<PIPE_MTE2, PIPE_V>(0);

        uint32_t epilogueN = Get<N_VALUE>(actualShape) / config.activationOutputDivisor;
        uint32_t epilogueNLoc = nLoc / config.activationOutputDivisor;
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, epilogueN, 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            subMLoc * config.outputN + epilogueNLoc,
            subMLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(epilogueNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            static_cast<int64_t>(subMLoc / L1_TILE_M_256),
            0,
            static_cast<int64_t>(metaInfoRow),
            0};
        MegaMoeImpl::SetSaturationStrategy(SaturationStrategy::USE_API);
        activationQuantOp(epilogueShape, epilogueOffset, 0U);
        NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo.activationToGmm2Flag, subMLoc / L1_TILE_M_256);
        PipeNotify<PIPE_V, PIPE_MTE2>(0);
    }
    PipeWait<PIPE_V, PIPE_MTE2>(0);
}

// 非 Prefetch 交错路径直接消费 AIC 的 UB ping-pong tile。
template <bool IsWaveFlagGrained, typename Config, typename ActivationQuantOp, typename ActualShape>
__aicore__ inline void Gmm1Aiv0EpilogueTileGeneric(const Config& config, ActivationQuantOp& activationQuantOp,
                                                   const GMMAddrInfo& gmmAddrInfo, const ActualShape& actualShape,
                                                   uint32_t mLoc, uint32_t nLoc, uint16_t pingpongIdx)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
    uint32_t epilogueN = Get<N_VALUE>(actualShape) / config.activationOutputDivisor;
    uint32_t epilogueNLoc = nLoc / config.activationOutputDivisor;
    Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape), epilogueN, 0, 0};
    Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
        mLoc * config.outputN + epilogueNLoc,
        mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
            CeilDiv(epilogueNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
        static_cast<int64_t>(mLoc / L1_TILE_M_256),
        0,
        0,
        0};
    MegaMoeImpl::SetSaturationStrategy(SaturationStrategy::USE_API);
    activationQuantOp(epilogueShape, epilogueOffset, pingpongIdx);
    NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo.activationToGmm2Flag, mLoc / L1_TILE_M_256);
}

// Prefetch 路径的 AIV1 从 GM 取回 A8W4 GMM1 子 tile，执行 Activation/量化并发布完成状态。
template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename TensorC, typename TensorMetaInfo,
          typename ActivationQuantOp, typename Config, typename ActualShape>
__aicore__ inline void Gmm1Aiv1PrefetchEpilogueTileA8W4(ActivationQuantOp& activationQuantOp, TensorC& l0cOutGm,
                                                        TensorMetaInfo& metaInfoGm, const GMMAddrInfo& gmmAddrInfo,
                                                        const Config& config, const ActualShape& actualShape,
                                                        uint32_t mLoc, uint32_t nLoc, uint32_t expertBeforeCnt)
{
    Gmm1Aiv0PrefetchEpilogueTileGeneric<MakeLayoutC, ElementC, IsWaveFlagGrained>(
        config, l0cOutGm, metaInfoGm, activationQuantOp, gmmAddrInfo, expertBeforeCnt, actualShape, mLoc, nLoc);
}

// 非 Prefetch 路径从 GM 取回完整 GMM1 tile 并执行 Activation/量化。
template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename TensorC, typename ActivationQuantOp,
          typename Config, typename ActualShape>
__aicore__ inline void Gmm1AivReadGmEpilogueTileGeneric(ActivationQuantOp& activationQuantOp, TensorC& l0cOutGm,
                                                        const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                                        const ActualShape& actualShape, uint32_t mLoc, uint32_t nLoc)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
    PipeNotify<PIPE_S, PIPE_MTE2>(0);
    PipeWait<PIPE_S, PIPE_MTE2>(0);
    auto tensorBlockGm = l0cOutGm.slice(asc::te::make_coord(mLoc, nLoc),
                                        asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    auto layoutL0cUb = MakeLayoutC{}(config.tileM, L1_TILE_N);
    auto tensorBlockUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(0), layoutL0cUb);
    auto copyGm2Ub = asc::te::make_copy(asc::te::copy_gm_to_ub{});
    asc::te::copy(copyGm2Ub, tensorBlockUb, tensorBlockGm);
    PipeNotify<PIPE_MTE2, PIPE_V>(0);
    PipeWait<PIPE_MTE2, PIPE_V>(0);
    Gmm1Aiv0EpilogueTileGeneric<IsWaveFlagGrained>(config, activationQuantOp, gmmAddrInfo, actualShape, mLoc, nLoc, 0U);
    PipeNotify<PIPE_S, PIPE_MTE2>(0);
    PipeWait<PIPE_S, PIPE_MTE2>(0);
}

// AIV0 在 UB 拼接两段 W1 权重并展开 A8W4，每个配对 tile 只提交一次 prologue。
template <typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void Gmm1Aiv0PrologueA8W4(BlockPrologue& blockPrologue, Scheduler& scheduler, TensorB& gmB,
                                            const Config& config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto nL1Size = Get<N_VALUE>(actualShape);

        const bool isGelu = config.activationOutputDivisor == 1U;
        auto nOffset = nLoc / config.activationOutputDivisor;
        const uint64_t secondHalfColumnOffset = isGelu ? 0U : config.n / ACTIVATION_N_HALF;
        // W1 两段物理拼接各占一半，与激活输出列宽无关。
        blockPrologue(gmB, config.k, nL1Size, nOffset, config.n, secondHalfColumnOffset);
    }
}

// Prefetch 路径的 AIV0 通过硬同步等待 tile 就绪，再执行 Activation/量化。
template <typename MakeLayoutC, typename ElementC, bool IsWaveFlagGrained, typename WorkSet, typename Config,
          typename ActivationQuantOp>
__aicore__ inline void Gmm1Aiv0PrefetchEpilogueGeneric(WorkSet& workSet, const Params& params,
                                                       const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                                       uint32_t startLoopIdx, uint32_t tileNum,
                                                       ActivationQuantOp& activationQuantOp, uint32_t expertBeforeCnt,
                                                       uint32_t expertIdx)
{
    auto& sync = *gmmAddrInfo.gmm1ActivationSync;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        sync.WaitForGmm1();
        Gmm1Aiv0PrefetchEpilogueTileGeneric<MakeLayoutC, ElementC, IsWaveFlagGrained>(
            config, workSet.gmC, workSet.metaInfoGm, activationQuantOp, gmmAddrInfo, expertBeforeCnt, actualShape, mLoc,
            nLoc);
        sync.NotifyGmm1();
    }
}

// 非 Prefetch 路径的 AIV0 直接消费 AIC 的 UB tile，并维持 ping-pong 同步。
template <bool IsWaveFlagGrained, typename WorkSet, typename Config, typename ActivationQuantOp>
__aicore__ inline void Gmm1Aiv0EpilogueGeneric(WorkSet& workSet, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                               uint32_t startLoopIdx, uint32_t tileNum,
                                               ActivationQuantOp& activationQuantOp, uint16_t& pingpongIdx)
{
    Gmm1UbActivationSync sync(pingpongIdx);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        sync.WaitForGmm1();
        Gmm1Aiv0EpilogueTileGeneric<IsWaveFlagGrained>(config, activationQuantOp, gmmAddrInfo, actualShape, mLoc, nLoc,
                                                       pingpongIdx);
        sync.NotifyGmm1();
    }
}

// 根据 GM 地址建立执行资源，并执行通用 GMM1/Activation 阶段。
template <typename BlockMmad, typename ElementC, typename MakeLayoutC, bool TopkWeightsPrefetch, bool IsShared = false,
          bool IsWaveFlagGrained = false, typename Scheduler, typename Config, typename ActivationQuantOp,
          typename LayoutBundle>
__aicore__ inline void Gmm1ExecGeneric(Scheduler& scheduler, const Params& params, const GMMAddrInfo& gmmAddrInfo,
                                       const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                       ActivationQuantOp& activationQuantOp, int32_t& vecSetSyncCom,
                                       uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t& pingpongIdx,
                                       BlockMmadContext<BlockMmad>* blockMmadContext, bool allowWeightL2Bypass,
                                       const LayoutBundle& layouts)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    auto gmA = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)),
        layouts.a);
    auto gmB = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)),
        layouts.b);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
                                         layouts.scaleA);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
                                         layouts.scaleB);
    if constexpr (IsWaveFlagGrained && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    GM_ADDR cGlobal = 0UL;
    if constexpr (TopkWeightsPrefetch) {
        cGlobal = gmmAddrInfo.gmm1OutGlobal;
    }
    auto gmC = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementC*>(cGlobal)), layouts.c);
    __gm__ float* metaInfoGm = nullptr;
    if constexpr (TopkWeightsPrefetch && g_coreType == AscendC::AIV) {
        metaInfoGm = reinterpret_cast<__gm__ float*>(gmmAddrInfo.metaInfoGlobal);
    }

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmC), decltype(metaInfoGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmC, metaInfoGm};
    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            if constexpr (TopkWeightsPrefetch) {
                Gmm1AicMmadToGmGeneric<BlockMmadContext<BlockMmad>, IsShared, TopkWeightsPrefetch>(
                    workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, expertIdx, blockMmadContext);
            } else {
                Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsShared>(
                    workSet, gmmAddrInfo, config, startLoopIdx, tileNum, vecSetSyncCom, pingpongIdx, blockMmadContext);
            }
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            if constexpr (TopkWeightsPrefetch) {
                Gmm1AicMmadToGmGeneric<BlockMmadContext<BlockMmad>, IsShared, TopkWeightsPrefetch>(
                    workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, expertIdx, &localBlockMmadContext);
            } else {
                Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsShared>(workSet, gmmAddrInfo, config, startLoopIdx,
                                                                          tileNum, vecSetSyncCom, pingpongIdx,
                                                                          &localBlockMmadContext);
            }
        }
    } else {
        if constexpr (TopkWeightsPrefetch) {
            if (IsSubBlockIdx<0U>()) {
                Gmm1Aiv0PrefetchEpilogueGeneric<MakeLayoutC, ElementC, IsWaveFlagGrained>(
                    workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, activationQuantOp, expertBeforeCnt,
                    expertIdx);
            }
        } else if (IsSubBlockIdx<0U>()) {
            Gmm1Aiv0EpilogueGeneric<IsWaveFlagGrained>(workSet, gmmAddrInfo, config, startLoopIdx, tileNum,
                                                       activationQuantOp, pingpongIdx);
        }
    }
}

// Prefetch 路径的 AIV1 通过硬同步等待 tile 就绪，再执行 A8W4 Activation/量化。
template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename WorkSet, typename Config,
          typename ActivationQuantOp>
__aicore__ inline void Gmm1Aiv1PrefetchEpilogueA8W4(WorkSet& workSet, const Params& params,
                                                    const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                                    uint32_t startLoopIdx, uint32_t tileNum,
                                                    ActivationQuantOp& activationQuantOp, uint32_t expertBeforeCnt,
                                                    uint32_t expertIdx)
{
    auto& sync = *gmmAddrInfo.gmm1ActivationSync;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        sync.WaitForGmm1();
        Gmm1Aiv1PrefetchEpilogueTileA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(
            activationQuantOp, workSet.gmC, workSet.metaInfoGm, gmmAddrInfo, config, actualShape, mLoc, nLoc,
            expertBeforeCnt);
        sync.NotifyGmm1();
    }
}

// 非 Prefetch 路径的 AIV1 按 AIC 通知顺序消费 A8W4 GMM1 tile。
template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename WorkSet, typename Config,
          typename ActivationQuantOp>
__aicore__ inline void Gmm1Aiv1EpilogueA8W4(WorkSet& workSet, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                            uint32_t startLoopIdx, uint32_t tileNum,
                                            ActivationQuantOp& activationQuantOp)
{
    auto& sync = *gmmAddrInfo.gmm1ActivationSync;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        sync.WaitForGmm1();
        Gmm1AivReadGmEpilogueTileGeneric<ElementC, MakeLayoutC, IsWaveFlagGrained>(
            activationQuantOp, workSet.gmC, gmmAddrInfo, config, actualShape, mLoc, nLoc);
        sync.NotifyGmm1();
    }
}

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM1/Activation 阶段。
template <typename BlockMmad, typename BlockPrologue, typename ElementC, typename MakeLayoutC, bool IsShared,
          typename Scheduler, typename Config, bool TopkWeightsPrefetch, bool IsWaveFlagGrained,
          typename ActivationQuantOp, typename LayoutBundle>
__aicore__ inline void Gmm1ExecA8W4(Scheduler& scheduler, const Params& params, const GMMAddrInfo& gmmAddrInfo,
                                    const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                    ActivationQuantOp& activationQuantOp, uint32_t expertBeforeCnt, uint32_t expertIdx,
                                    const LayoutBundle& layouts,
                                    const A8W4BlockContext<BlockMmad, BlockPrologue>& pipeline)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    auto gmC = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm1OutGlobal)),
        layouts.c);
    auto gmA = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)),
        layouts.a);
    auto gmB = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)),
        layouts.b);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
                                         layouts.scaleA);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
                                         layouts.scaleB);
    __gm__ float* metaInfoGm = nullptr;
    if constexpr (TopkWeightsPrefetch && g_coreType == AscendC::AIV) {
        metaInfoGm = reinterpret_cast<__gm__ float*>(gmmAddrInfo.metaInfoGlobal);
    }

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmC), decltype(metaInfoGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmC, metaInfoGm};
    if constexpr (g_coreType == AscendC::AIC) {
        Gmm1AicMmadA8W4<BlockMmad, IsShared, TopkWeightsPrefetch>(*pipeline.block, workSet, params, gmmAddrInfo, config,
                                                                  startLoopIdx, tileNum, expertIdx);
    } else if (IsSubBlockIdx<0U>()) {
        Gmm1Aiv0PrologueA8W4<BlockPrologue>(*pipeline.block, workSet.scheduler, workSet.gmB, config, startLoopIdx,
                                            tileNum);
    } else {
        if constexpr (TopkWeightsPrefetch) {
            Gmm1Aiv1PrefetchEpilogueA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(
                workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, activationQuantOp, expertBeforeCnt,
                expertIdx);
        } else {
            Gmm1Aiv1EpilogueA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(workSet, gmmAddrInfo, config, startLoopIdx,
                                                                           tileNum, activationQuantOp);
        }
    }
}

} // namespace GmmKernel

template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsWaveFlagGrained = false, typename BlockEpilogue, typename... LayoutArgs>
__aicore__ inline void RunGmm1Generic(BlockEpilogue& epilogueOp, const Params& params,
                                      const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                      const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom,
                                      const BlockJobContext& blockJob, uint32_t expertBeforeCnt, uint32_t expertIdx,
                                      uint16_t& pingpongIdx, uint32_t activationOutputDivisor = ACTIVATION_N_HALF,
                                      void* blockMmadContext = nullptr, bool allowWeightL2Bypass = false,
                                      const LayoutArgs&... layoutArgs)
{
    using GmmConfig =
        GmmKernel::Config<false, COMBINE_NO_QUANT, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                          IsWeightNZ, TopkWeightsPrefetch, IsShared, false, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM, activationOutputDivisor);

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{asc::te::make_coord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if (!IsSubBlockIdx<0U>()) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }
    if constexpr (g_coreType == AIV) {
        epilogueOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = GmmKernel::BlockMmadContext<BlockMmad>;
    auto* typedBlockMmadContext = reinterpret_cast<MmadContext*>(blockMmadContext);
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;
    auto layouts = GmmConfig::BuildLayouts(config, layoutArgs...);
    GmmKernel::Gmm1ExecGeneric<BlockMmad, ElementC, MakeLayoutC, TopkWeightsPrefetch, IsShared, IsWaveFlagGrained>(
        scheduler, params, gmmAddrInfo, config, startLoopIdx, tileNum, epilogueOp, vecSetSyncCom, expertBeforeCnt,
        expertIdx, pingpongIdx, typedBlockMmadContext, allowWeightL2Bypass, layouts);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsWaveFlagGrained = false, typename BlockEpilogue>
__aicore__ inline void RunGmm1Generic(BlockEpilogue& epilogueOp, const Params& params,
                                      const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                      const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom,
                                      uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t& pingpongIdx,
                                      uint32_t activationOutputDivisor = ACTIVATION_N_HALF,
                                      void* blockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    BlockJobContext blockJob{MegaMoeHardwarePolicy::GetAicBlockIdx(),
                             static_cast<uint32_t>(MegaMoeHardwarePolicy::GetBlockNum())};
    RunGmm1Generic<ElementA, EpilogueElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ,
                   Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch, IsShared, IsWaveFlagGrained>(
        epilogueOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom, blockJob, expertBeforeCnt,
        expertIdx, pingpongIdx, activationOutputDivisor, blockMmadContext, allowWeightL2Bypass);
}

// RunGmm1A8W4：执行 A8W4 prologue（W4→W8）、GMM1、Activation 和量化。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsWaveFlagGrained = false, typename BlockContext, typename BlockEpilogue,
          typename... LayoutArgs>
__aicore__ inline void RunGmm1A8W4(const BlockContext& pipeline, BlockEpilogue& activationQuantOp, const Params& params,
                                   const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                   const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                   const BlockJobContext& blockJob, uint32_t expertBeforeCnt, uint32_t expertIdx,
                                   uint32_t activationOutputDivisor = ACTIVATION_N_HALF,
                                   const LayoutArgs&... layoutArgs)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = GmmKernel::Config<true, 0, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                                        TopkWeightsPrefetch, IsShared, false, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM, activationOutputDivisor);

    if constexpr (g_coreType == AIV) {
        activationQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    using BlockMmad = typename GmmConfig::BlockMmad;
    using BlockPrologue = typename GmmConfig::BlockPrologue;
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{asc::te::make_coord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    auto layouts = GmmConfig::BuildLayouts(config, layoutArgs...);
    GmmKernel::Gmm1ExecA8W4<BlockMmad, BlockPrologue, ElementC, MakeLayoutC, IsShared, GmmKernel::BlockScheduler,
                            decltype(config), TopkWeightsPrefetch, IsWaveFlagGrained>(
        scheduler, params, gmmAddrInfo, config, startLoopIdx, tileNum, activationQuantOp, expertBeforeCnt, expertIdx,
        layouts, pipeline);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

// 使用当前核的任务配置执行 A8W4 GMM1。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsWaveFlagGrained = false, typename BlockContext, typename BlockEpilogue>
__aicore__ inline void RunGmm1A8W4(const BlockContext& pipeline, BlockEpilogue& activationQuantOp, const Params& params,
                                   const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                   const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, uint32_t expertBeforeCnt,
                                   uint32_t expertIdx = 0, uint32_t activationOutputDivisor = ACTIVATION_N_HALF)
{
    BlockJobContext blockJob{MegaMoeHardwarePolicy::GetAicBlockIdx(),
                             static_cast<uint32_t>(MegaMoeHardwarePolicy::GetBlockNum())};
    RunGmm1A8W4<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM, EpilogueTileM,
                TopkWeightsPrefetch, IsShared, IsWaveFlagGrained>(pipeline, activationQuantOp, params, problemShape,
                                                                  gmmAddrInfo, startBlockIdx, blockJob, expertBeforeCnt,
                                                                  expertIdx, activationOutputDivisor);
}

template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          uint32_t ActivationElementsPerByte, bool UseA8W4Gmm1, bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void UpdateMoeExpertGmm1GlobalBuffer(const GmmExecutionConfig& gmmConfig,
                                                       const MoeSyncWorkspaceLayout& syncLayout,
                                                       const WorkspaceLayout& workspace, GM_ADDR workspaceBase,
                                                       const ExpertWeightTensorListAddrs& weights,
                                                       BlockEpilogue& epilogueOp, GMMAddrInfo& gmmAddrInfo,
                                                       const ExpertLoopState& state, uint32_t activationOutputWidth,
                                                       uint32_t rowOffsetInExpert = 0U)
{
    if constexpr (g_coreType == AIV && !UseA8W4Gmm1) {
        if (!IsSubBlockIdx<0U>()) {
            return;
        }
    }

    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElementsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    int64_t n = Get<N_VALUE>(state.problemShape);
    int64_t k = Get<K_VALUE>(state.problemShape);
    int64_t expertOffset = state.expertIdx;
    int64_t globalTokenStartIndex = state.globalTokenStartIndex + rowOffsetInExpert;
    uint64_t physicalTokenStartIndex =
        gmmConfig.ring.gmm1 ?
            CalcMteRingPhysicalRow(state.paddedTokenStartIndex + rowOffsetInExpert, gmmConfig.ring.rows) :
            static_cast<uint64_t>(globalTokenStartIndex);
    // Dispatch 输入按自身循环开关选择行号；GMM1 输出和激活仍按各自的循环物理行寻址。
    const uint64_t dispatchTokenStartIndex =
        gmmConfig.ring.dispatch ? physicalTokenStartIndex : static_cast<uint64_t>(globalTokenStartIndex);
    uint32_t expertMGroupOffset = rowOffsetInExpert / L1_TILE_M_256;
    int64_t scaleK = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    if constexpr (UseA8W4Gmm1 || TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1OutGlobal =
            workspace.routed.gmm1Output.get_ptr(workspaceBase) + physicalTokenStartIndex * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.metaInfoGlobal = workspace.routing.receivedTokenMetadata.get_ptr(workspaceBase);
    gmmAddrInfo.activationToGmm2Flag =
        reinterpret_cast<__gm__ int32_t*>(workspace.sync.activationReady.get_ptr(workspaceBase)) +
        expertOffset * syncLayout.activationFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.aGlobal = workspace.routed.gmm1Input.data.get_ptr(workspaceBase) +
                          dispatchTokenStartIndex * k / ActivationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = workspace.routed.gmm1Input.scale.get_ptr(workspaceBase) +
                               dispatchTokenStartIndex * scaleK * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<WeightType>(weights.weight1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                        static_cast<uint64_t>(state.expertIdx) * n * k / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal =
        GetExpertWeightAddr<QuantScaleType>(weights.weightScales1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                            static_cast<uint64_t>(state.expertIdx) * n * scaleK);
    if constexpr (g_coreType == AIV) {
        bool runsActivation = true;
        if constexpr (UseA8W4Gmm1) {
            runsActivation = IsSubBlockIdx<1U>();
        }
        if (runsActivation) {
            int64_t scaleN = Ops::Base::CeilDiv(static_cast<int64_t>(activationOutputWidth),
                                                static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                             MXFP_MULTI_BASE_SIZE;
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                static_cast<int64_t>(physicalTokenStartIndex) * activationOutputWidth / outputElementsPerByte,
                static_cast<int64_t>(physicalTokenStartIndex) * scaleN,
                expertOffset * syncLayout.activationFlagSlotCountPerExpert / INT_CACHELINE + expertMGroupOffset,
                0L,
                0L,
                0L};
            epilogueOp.UpdateGlobalAddr(vecBaseOffset);
        }
    }
    if constexpr (g_coreType == AIC) {
        gmmAddrInfo.dispatchToGmm1Flag =
            reinterpret_cast<__gm__ int32_t*>(workspace.sync.dispatchReady.get_ptr(workspaceBase)) +
            expertOffset * syncLayout.dispatchFlagSlotCountPerExpert +
            static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    }
}

// 构造一个共享专家的 GMM1 输入、权重、输出与 Activation 量化输出 GM 视图。
template <typename WeightType, typename ActivationOutType, typename QuantScaleType, bool UseGmOutput,
          typename BlockEpilogue>
__aicore__ inline void UpdateSharedExpertGmm1GlobalBuffer(const MoeStageCommonConfig& commonConfig,
                                                          const GmmExecutionConfig& gmmConfig,
                                                          const WorkspaceLayout& workspace, GM_ADDR workspaceBase,
                                                          const ExpertWeightTensorListAddrs& weights,
                                                          BlockEpilogue& epilogueOp, GMMAddrInfo& gmmAddrInfo,
                                                          uint32_t sharedExpertIdx)
{
    constexpr uint32_t weightElemsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElemsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    uint64_t m = commonConfig.tokenNum;
    uint64_t n = commonConfig.gmm1OutputDim;
    uint64_t k = commonConfig.tokenHiddenDim;
    uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t activationN = n / commonConfig.activationOutputDivisor;
    uint64_t scaleN = Ops::Base::CeilDiv(activationN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t expertIdx = sharedExpertIdx;

    if constexpr (UseGmOutput) {
        gmmAddrInfo.gmm1OutGlobal =
            workspace.shared.gmm1Output.get_ptr(workspaceBase) + expertIdx * m * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<WeightType>(weights.weight1, gmmConfig.isPerExpertWeightTensor,
                                                          sharedExpertIdx, expertIdx * n * k / weightElemsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales1, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx, expertIdx * n * scaleK);
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    if (workspace.mte.sync.sharedActivationReady.get_ptr(workspaceBase) != nullptr) {
        uint64_t sharedActivationFlagElementCount =
            static_cast<uint64_t>(CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(m)));
        gmmAddrInfo.activationToGmm2Flag =
            reinterpret_cast<__gm__ int32_t*>(workspace.mte.sync.sharedActivationReady.get_ptr(workspaceBase)) +
            expertIdx * sharedActivationFlagElementCount;
    }
    gmmAddrInfo.metaInfoGlobal = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;

    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            static_cast<int64_t>(expertIdx * m * activationN / outputElemsPerByte),
            static_cast<int64_t>(expertIdx * m * scaleN),
            0L,
            0L,
            0L,
            0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM1_ACTIVATION_H
