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

#include "../arch35_92/common/mega_moe_arch_policy.h"
#ifdef MEGA_MOE_ARCH35_DEVICE
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "../arch35_92/stage/mega_moe_gmm1_activation.h"
#include "../arch35_92/stage/mega_moe_gmm2_combine.h"
#include "mega_moe_epilogue_bf16.h"

namespace MegaMoeImpl {

struct Bf16GmmBlockContext {
    template <typename Block>
    __aicore__ inline void Init(Block& block) const
    {
        using C = MegaMoeBf16GmmConfig;
        typename Block::Params params{};
        params.mL1 = C::TILE_M;
        params.nL1 = C::TILE_N;
        params.kL1 = C::L1_TILE_K;
        params.mL0 = C::L0_TILE_M;
        params.nL0 = C::TILE_N;
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
    using Block = Blaze::Gemm::Block::BlockMmad<Policy, bfloat16_t, asc::te::nd_ext_layout_ptn,
        bfloat16_t, LayoutB, bfloat16_t, asc::te::nd_ext_layout_ptn, float, asc::te::nd_ext_layout_ptn>;
};

struct ProblemConfig {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t tileM;
    uint32_t blockNum;
    uint32_t blockIdx;
};

// 外层任务保持 256 行；普通 BlockMmad 按 128 行 L0 分条，结果统一以 BF16 写回 GM。
template <typename Block, typename TensorA, typename TensorB, typename TensorC>
__aicore__ inline void ComputeTile(Block& block, TensorA& a, TensorB& b, TensorC& c,
                                   uint32_t row, uint32_t column, uint32_t rows, uint32_t columns, uint32_t k)
{
    auto bias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
        static_cast<__gm__ float*>(nullptr)), asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn>{}(1U, columns));
    for (uint32_t offset = 0U; offset < rows; offset += MegaMoeBf16GmmConfig::L0_TILE_M) {
        const uint32_t count = rows - offset < MegaMoeBf16GmmConfig::L0_TILE_M ?
            rows - offset : MegaMoeBf16GmmConfig::L0_TILE_M;
        auto tileA = a.slice(asc::te::make_coord(row + offset, 0U), asc::te::make_shape(count, k));
        auto tileB = b.slice(asc::te::make_coord(0U, column), asc::te::make_shape(k, columns));
        auto tileC = c.slice(asc::te::make_coord(row + offset, column), asc::te::make_shape(count, columns));
        typename Block::TupleShape shape{count, columns, k, 0U};
        block(tileA, tileB, bias, tileC, shape);
    }
}

template <bool WeightNZ, bool IsShared, typename Epilogue>
__aicore__ inline void RunGmm1(const Bf16GmmBlockContext& context, Epilogue& epilogue,
                               const ProblemShape& shape, const GMMAddrInfo& addresses,
                               const BlockJobContext& job, uint32_t& startBlockIdx, uint32_t divisor)
{
    using C = MegaMoeBf16GmmConfig;
    using K = Kernel<WeightNZ>;
    const uint32_t m = static_cast<uint32_t>(Get<M_VALUE>(shape));
    const uint32_t fullN = static_cast<uint32_t>(Get<N_VALUE>(shape));
    const uint32_t k = static_cast<uint32_t>(Get<K_VALUE>(shape));
    const uint32_t n = fullN / divisor;
    const ProblemConfig config{m, n, k, C::TILE_M, job.totalJobs, job.jobIndex};
    GmmKernel::BlockScheduler scheduler({m, n, k},
        GmmKernel::BlockScheduler::Params{asc::te::make_coord(int64_t{C::TILE_M}, int64_t{C::TILE_N})});
    const uint32_t tiles = static_cast<uint32_t>(scheduler.GetTileNum());
    const uint32_t first = (job.jobIndex + job.totalJobs - startBlockIdx) % job.totalJobs;
    if constexpr (g_coreType == AIC) {
        typename K::Block block;
        context.Init(block);
        auto a = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
            reinterpret_cast<__gm__ bfloat16_t*>(addresses.aGlobal)), MakeLayout{}(m, k));
        auto b = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
            reinterpret_cast<__gm__ bfloat16_t*>(addresses.bGlobal)), typename K::MakeLayoutB{}(k, fullN));
        auto c = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
            reinterpret_cast<__gm__ bfloat16_t*>(addresses.gmm1OutGlobal)), MakeLayout{}(m, fullN));
        for (uint32_t tile = first; tile < tiles; tile += job.totalJobs) {
            const auto coord = scheduler.GetBlockCoord(tile);
            const auto extent = scheduler.GetBlockShape(coord);
            const uint32_t row = Get<M_VALUE>(coord), column = Get<N_VALUE>(coord);
            WaitForGmm1InputReady<IsShared>(addresses, config, row);
            addresses.gmm1ActivationSync->WaitForActivation();
            ComputeTile(block, a, b, c, row, column, Get<M_VALUE>(extent), Get<N_VALUE>(extent), k);
            if (divisor == ACTIVATION_N_HALF) {
                ComputeTile(block, a, b, c, row, column + n, Get<M_VALUE>(extent), Get<N_VALUE>(extent), k);
            }
            addresses.gmm1ActivationSync->NotifyActivation();
        }
    } else if (MegaMoeHardwarePolicy::GetSubBlockIdx() == 0U) {
        for (uint32_t tile = first; tile < tiles; tile += job.totalJobs) {
            const auto coord = scheduler.GetBlockCoord(tile);
            const auto extent = scheduler.GetBlockShape(coord);
            addresses.gmm1ActivationSync->WaitForGmm1();
            epilogue.RunTile(addresses.gmm1OutGlobal, addresses.metaInfoGlobal, fullN,
                Get<M_VALUE>(coord), Get<N_VALUE>(coord), Get<M_VALUE>(extent), Get<N_VALUE>(extent));
            addresses.gmm1ActivationSync->NotifyGmm1();
            PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID5>();
            GmmKernel::NotifyGmm2InputReady<true>(addresses.activationToGmm2Flag, Get<M_VALUE>(coord) / C::TILE_M);
        }
    }
    startBlockIdx = (startBlockIdx + tiles) % job.totalJobs;
}

template <bool WeightNZ, bool IsShared, bool NotifyCombine>
__aicore__ inline void RunGmm2(const Bf16GmmBlockContext& context, const ProblemShape& shape,
                               const GMMAddrInfo& addresses, const Params& params,
                               const BlockJobContext& job, uint32_t& startBlockIdx, uint32_t divisor)
{
    using C = MegaMoeBf16GmmConfig;
    using K = Kernel<WeightNZ>;
    const uint32_t m = static_cast<uint32_t>(Get<M_VALUE>(shape));
    const uint32_t n = static_cast<uint32_t>(Get<K_VALUE>(shape));
    const uint32_t k = static_cast<uint32_t>(Get<N_VALUE>(shape)) / divisor;
    const ProblemConfig config{m, n, k, C::TILE_M, job.totalJobs, job.jobIndex};
    GmmKernel::BlockScheduler scheduler({m, n, k},
        GmmKernel::BlockScheduler::Params{asc::te::make_coord(int64_t{C::TILE_M}, int64_t{C::TILE_N})});
    const uint32_t tiles = static_cast<uint32_t>(scheduler.GetTileNum());
    const uint32_t first = (job.jobIndex + job.totalJobs - startBlockIdx) % job.totalJobs;
    auto c = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
        reinterpret_cast<__gm__ bfloat16_t*>(addresses.gmm2OutGlobal)), MakeLayout{}(m, n));
    if constexpr (g_coreType == AIC) {
        typename K::Block block;
        context.Init(block);
        auto a = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
            reinterpret_cast<__gm__ bfloat16_t*>(addresses.aGlobal)), MakeLayout{}(m, k));
        auto b = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
            reinterpret_cast<__gm__ bfloat16_t*>(addresses.bGlobal)), typename K::MakeLayoutB{}(k, n));
        for (uint32_t tile = first; tile < tiles; tile += job.totalJobs) {
            const auto coord = scheduler.GetBlockCoord(tile);
            const auto extent = scheduler.GetBlockShape(coord);
            const uint32_t row = Get<M_VALUE>(coord);
            WaitUntilGmFlagEquals(addresses.activationToGmm2Flag +
                static_cast<uint64_t>(row / C::TILE_M) * INT_CACHELINE,
                static_cast<int32_t>(Ops::Base::CeilDiv(k, C::TILE_N)));
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
