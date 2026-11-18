/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEGA_MOE_GMM_BF16_H
#define MEGA_MOE_GMM_BF16_H

#include "../../arch35_92/common/mega_moe_arch_policy.h"
#ifdef MEGA_MOE_ARCH35_DEVICE
#include "../../arch35_92/stage/mega_moe_gmm1_activation.h"
#include "../../arch35_92/stage/mega_moe_gmm2_combine.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"

namespace MegaMoeImpl {

struct Bf16GmmBlockContext {
    template <typename Block>
    __aicore__ inline void Init(Block& block, uint32_t tileN = MegaMoeBf16GmmConfig::TILE_N) const
    {
        using C = MegaMoeBf16GmmConfig;
        typename Block::Params params{};
        params.mL1 = C::TILE_M;
        params.nL1 = tileN;
        params.kL1 = C::L1_TILE_K;
        params.mL0 = C::L0_TILE_M;
        params.nL0 = tileN;
        params.kL0 = C::L0_TILE_K;
        params.l1Stages = C::BUFFER_COUNT;
        params.l0cStages = 1U;
        block.Init(params);
    }
};

namespace Bf16Gmm {
using MakeLayout = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, AscendC::Std::Int<16>>;

template <bool WeightNZ>
struct Kernel {
    using LayoutB = typename AscendC::Std::conditional_t<WeightNZ, asc::te::zn_layout_ptn, asc::te::dn_ext_layout_ptn>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<16>>;
    using Policy = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelMmadMultiBlockBasic, 0>;
    using Block =
        Blaze::Gemm::Block::BlockMmad<Policy, bfloat16_t, asc::te::nd_ext_layout_ptn, bfloat16_t, LayoutB, bfloat16_t,
                                      asc::te::nd_ext_layout_ptn, float, asc::te::nd_ext_layout_ptn>;
};

struct ProblemConfig {
    static constexpr bool IS_WAVE_FLAG_GRAINED = true;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t tileM;
    uint32_t blockNum;
    uint32_t blockIdx;
    uint32_t activationOutputDivisor;
    uint32_t activationTileM = MegaMoeBf16GmmConfig::TILE_M;
};

// GMM2 按 128 行 L0 分条计算，结果以 BF16 写回 GM。
template <typename Block, typename TensorA, typename TensorB, typename TensorC>
__aicore__ inline void ComputeTile(Block& block, TensorA& a, TensorB& b, TensorC& c, uint32_t row, uint32_t column,
                                   uint32_t rows, uint32_t columns, uint32_t k)
{
    auto bias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(static_cast<__gm__ float*>(nullptr)),
                                     asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn>{}(1U, columns));
    for (uint32_t offset = 0U; offset < rows; offset += MegaMoeBf16GmmConfig::L0_TILE_M) {
        const uint32_t count =
            rows - offset < MegaMoeBf16GmmConfig::L0_TILE_M ? rows - offset : MegaMoeBf16GmmConfig::L0_TILE_M;
        auto tileA = a.slice(asc::te::make_coord(row + offset, 0U), asc::te::make_shape(count, k));
        auto tileB = b.slice(asc::te::make_coord(0U, column), asc::te::make_shape(k, columns));
        auto tileC = c.slice(asc::te::make_coord(row + offset, column), asc::te::make_shape(count, columns));
        typename Block::TupleShape shape{count, columns, k, 0U};
        block(tileA, tileB, bias, tileC, shape);
    }
}

// UB 每行按 gate/up 交织；权重使用全局列号，输出使用当前槽的局部行列号。
template <typename Block, typename TensorA, typename TensorB, typename TensorC, typename Coord, typename Extent>
__aicore__ inline void ComputeGmm1Tile(Block& block, TensorA& a, TensorB& b, TensorC& ub, const Coord& coord,
                                       const Extent& extent, const ProblemConfig& config)
{
    using C = MegaMoeBf16GmmConfig;
    const uint32_t column = Get<N_VALUE>(coord) / config.activationOutputDivisor;
    const uint32_t columns = Get<N_VALUE>(extent) / config.activationOutputDivisor;
    const uint32_t branchWidth = config.n / config.activationOutputDivisor;
    const uint32_t ubBranchWidth = C::TILE_N / config.activationOutputDivisor;
    auto bias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(static_cast<__gm__ float*>(nullptr)),
                                     MakeLayout{}(1U, columns));
    for (uint32_t offset = 0U; offset < Get<M_VALUE>(extent); offset += C::L0_TILE_M) {
        const uint32_t remaining = Get<M_VALUE>(extent) - offset;
        const uint32_t rows = remaining < C::L0_TILE_M ? remaining : C::L0_TILE_M;
        auto tileA =
            a.slice(asc::te::make_coord(Get<M_VALUE>(coord) + offset, 0U), asc::te::make_shape(rows, config.k));
        for (uint32_t branch = 0U; branch < config.activationOutputDivisor; ++branch) {
            auto tileB =
                b.slice(asc::te::make_coord(0U, column + branch * branchWidth), asc::te::make_shape(config.k, columns));
            auto tileC =
                ub.slice(asc::te::make_coord(offset, branch * ubBranchWidth), asc::te::make_shape(rows, columns));
            Gmm1OutputView<decltype(tileC)> output{tileC};
            typename Block::TupleShape shape{rows, columns, config.k, 0U};
            block(tileA, tileB, bias, output, shape);
        }
    }
}

template <bool WeightNZ, bool IsShared>
__aicore__ inline void RunGmm1Aic(const Bf16GmmBlockContext& context, GmmKernel::BlockScheduler& scheduler,
                                  const GMMAddrInfo& addresses, const ProblemConfig& config, uint32_t first,
                                  uint32_t tiles, GmmRuntimeState& runtimeState)
{
    using C = MegaMoeBf16GmmConfig;
    using K = Kernel<WeightNZ>;
    typename K::Block block;
    context.Init(block, C::TILE_N / config.activationOutputDivisor);
    auto a = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ bfloat16_t*>(addresses.aGlobal)),
        MakeLayout{}(config.m, config.k));
    auto b = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ bfloat16_t*>(addresses.bGlobal)),
        typename K::MakeLayoutB{}(config.k, config.n));
    auto ub = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(0U),
                                   MakeLayout{}(C::TILE_M, C::TILE_N));
    Gmm1UbActivationSyncBase<C::GMM1_UB_BUFFER_COUNT> sync(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t tile = first; tile < tiles; tile += config.blockNum) {
        const auto coord = scheduler.GetBlockCoord(tile);
        const auto extent = scheduler.GetBlockShape(coord);
        const uint32_t wave = Get<M_VALUE>(coord) / config.tileM;
        if (wave != lastWaveWaited) {
            WaitForGmm1InputReady<IsShared>(addresses, config, Get<M_VALUE>(coord));
            lastWaveWaited = wave;
        }
        sync.WaitForActivation();
        ComputeGmm1Tile(block, a, b, ub, coord, extent, config);
        sync.NotifyActivation();
    }
}

template <typename Epilogue>
__aicore__ inline void RunGmm1Aiv(GmmKernel::BlockScheduler& scheduler, Epilogue& epilogue,
                                  const GMMAddrInfo& addresses, const ProblemConfig& config, uint32_t first,
                                  uint32_t tiles, GmmRuntimeState& runtimeState)
{
    Gmm1UbActivationSyncBase<MegaMoeBf16GmmConfig::GMM1_UB_BUFFER_COUNT> sync(runtimeState.pingpongIdx);
    for (uint32_t tile = first; tile < tiles; tile += config.blockNum) {
        const auto coord = scheduler.GetBlockCoord(tile);
        const auto extent = scheduler.GetBlockShape(coord);
        sync.WaitForGmm1();
        epilogue.RunTile(addresses.metaInfoGlobal, config.n / config.activationOutputDivisor, Get<M_VALUE>(coord),
                         Get<N_VALUE>(coord) / config.activationOutputDivisor, Get<M_VALUE>(extent),
                         Get<N_VALUE>(extent) / config.activationOutputDivisor);
        GmmKernel::NotifyGmm2InputReady<true>(addresses.activationToGmm2Flag, Get<M_VALUE>(coord) / config.tileM);
        sync.NotifyGmm1();
    }
}

template <bool WeightNZ, bool IsShared, typename Epilogue>
__aicore__ inline void RunGmm1(const Bf16GmmBlockContext& context, Epilogue& epilogue, const ProblemShape& shape,
                               const GMMAddrInfo& addresses, const BlockJobContext& job, GmmRuntimeState& runtimeState,
                               uint32_t divisor)
{
    using C = MegaMoeBf16GmmConfig;
    const uint32_t m = static_cast<uint32_t>(Get<M_VALUE>(shape));
    const uint32_t n = static_cast<uint32_t>(Get<N_VALUE>(shape));
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(shape));
    const ProblemConfig config{m, n, k, C::TILE_M, job.totalJobs, job.jobIndex, divisor};
    GmmKernel::BlockScheduler scheduler(
        {m, n, k}, GmmKernel::BlockScheduler::Params{asc::te::make_coord(int64_t{C::TILE_M}, int64_t{C::TILE_N})});
    const uint32_t tiles = static_cast<uint32_t>(scheduler.GetTileNum());
    const uint32_t first = (job.jobIndex + job.totalJobs - runtimeState.startBlockIdx) % job.totalJobs;
    if constexpr (g_coreType == AIC) {
        RunGmm1Aic<WeightNZ, IsShared>(context, scheduler, addresses, config, first, tiles, runtimeState);
    } else if (MegaMoeHardwarePolicy::GetSubBlockIdx() == 0U) {
        RunGmm1Aiv(scheduler, epilogue, addresses, config, first, tiles, runtimeState);
    }
    runtimeState.startBlockIdx = (runtimeState.startBlockIdx + tiles) % job.totalJobs;
}

template <bool WeightNZ, bool IsShared, bool NotifyCombine>
__aicore__ inline void RunGmm2(const Bf16GmmBlockContext& context, const ProblemShape& shape,
                               const GMMAddrInfo& addresses, const Params& params, const BlockJobContext& job,
                               uint32_t& startBlockIdx, uint32_t divisor)
{
    using C = MegaMoeBf16GmmConfig;
    using K = Kernel<WeightNZ>;
    const uint32_t m = static_cast<uint32_t>(Get<M_VALUE>(shape));
    const uint32_t n = static_cast<uint32_t>(Get<K_VALUE>(shape));
    const uint32_t k = static_cast<uint32_t>(Get<N_VALUE>(shape)) / divisor;
    const ProblemConfig config{m, n, k, C::TILE_M, job.totalJobs, job.jobIndex, divisor};
    GmmKernel::BlockScheduler scheduler(
        {m, n, k}, GmmKernel::BlockScheduler::Params{asc::te::make_coord(int64_t{C::TILE_M}, int64_t{C::TILE_N})});
    const uint32_t tiles = static_cast<uint32_t>(scheduler.GetTileNum());
    const uint32_t first = (job.jobIndex + job.totalJobs - startBlockIdx) % job.totalJobs;
    auto c = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ bfloat16_t*>(addresses.gmm2OutGlobal)),
        MakeLayout{}(m, n));
    if constexpr (g_coreType == AIC) {
        typename K::Block block;
        context.Init(block);
        auto a = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ bfloat16_t*>(addresses.aGlobal)),
            MakeLayout{}(m, k));
        auto b = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ bfloat16_t*>(addresses.bGlobal)),
            typename K::MakeLayoutB{}(k, n));
        for (uint32_t tile = first; tile < tiles; tile += job.totalJobs) {
            const auto coord = scheduler.GetBlockCoord(tile);
            const auto extent = scheduler.GetBlockShape(coord);
            const uint32_t row = Get<M_VALUE>(coord);
            WaitForGmm2InputReady(addresses, config, row);
            if constexpr (NotifyCombine) {
                addresses.gmm2CombineSync->WaitForCombine();
            }
            ComputeTile(block, a, b, c, row, Get<N_VALUE>(coord), Get<M_VALUE>(extent), Get<N_VALUE>(extent), k);
            if constexpr (NotifyCombine) {
                addresses.gmm2CombineSync->NotifyCombine();
            } else if constexpr (IsShared) {
                NotifySharedExpertTileCompletion(row, addresses.sharedExpertGmm2TileCounter);
            }
        }
    } else if constexpr (NotifyCombine) {
        if (MegaMoeHardwarePolicy::GetSubBlockIdx() == 1U) {
            GmmKernel::CombineTokenRange<bfloat16_t, MakeLayout>(scheduler, c, params, addresses, config, first, tiles);
        }
    }
    startBlockIdx = (startBlockIdx + tiles) % job.totalJobs;
}

} // namespace Bf16Gmm
} // namespace MegaMoeImpl
#endif
#endif
