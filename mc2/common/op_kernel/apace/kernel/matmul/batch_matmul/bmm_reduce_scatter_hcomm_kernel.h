/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

// 只负责一组计算资源和单片计算，通信片循环及跨核同步由 fusion 持有。
template <class ProblemShape, class BlockMmad, class BlockScheduler>
class MatmulReduceScatterV2Kernel {
private:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;

public:
    struct BlockMmadParams {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    struct MatmulTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
    };

    struct SliceParams {
        uint32_t rankDim;
        uint64_t perRankM;
        uint64_t sliceStart;
        uint64_t sliceM;
    };

    using BlockSchedulerParams = typename BlockScheduler::Params;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmParams;
        L1Params l1Params;
        BlockSchedulerParams schedulerParams;
        MatmulTiling matmulTiling;
        SliceParams sliceParams;
    };

    __aicore__ inline void InitGroup(const Params& params)
    {
        typename BlockMmad::Params mmParams;
        mmParams.biasGmAddr = params.mmParams.biasGmAddr;
        mmParams.mL1 = params.matmulTiling.baseM;
        mmParams.nL1 = params.matmulTiling.baseN;
        mmParams.kL1 = params.l1Params.kL1;
        mmParams.mL0 = params.matmulTiling.baseM;
        mmParams.nL0 = params.matmulTiling.baseN;
        mmParams.kL0 = params.matmulTiling.baseK;
        mmParams.l1Stages = params.l1Params.l1BufNum;
        mmParams.l0cStages = params.matmulTiling.dbL0C;
        mmOp_.Init(mmParams);
    }

    __aicore__ inline void ProcessSlice(const Params& params)
    {
        BlockScheduler scheduler(params.problemShape, params.schedulerParams);
        ProcessBasicSlice(params, scheduler);
    }

private:
    __aicore__ inline void ProcessBasicSlice(const Params& params, BlockScheduler& scheduler)
    {
        using MakeLayoutA = asc::te::frame_layout_format<LayoutA>;
        using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
        using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
        using MakeLayoutBias = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;
        const auto& sp = params.sliceParams;
        const int64_t m = sp.perRankM * sp.rankDim;
        const int64_t n = asc::te::get<MNK_N>(params.problemShape);
        const int64_t k = asc::te::get<MNK_K>(params.problemShape);
        const int64_t logicalM = sp.sliceM * sp.rankDim;
        auto gmA = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType*>(params.mmParams.aGmAddr)),
            MakeLayoutA{}(m, k));
        auto gmB = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BType*>(params.mmParams.bGmAddr)),
            MakeLayoutB{}(k, n));
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                               reinterpret_cast<__gm__ BiasType*>(params.mmParams.biasGmAddr)),
                                           MakeLayoutBias{}(1L, n));
        auto gmC = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ CType*>(params.mmParams.cGmAddr)),
            MakeLayoutC{}(logicalM, n));

        // 尾轮按全部启动核重新分配，步长与 Host 校验的核数一致。
        const int64_t coreNum = AscendC::GetBlockNum();
        for (int64_t blockIdx = AscendC::GetBlockIdx(); blockIdx < scheduler.GetBlockNums(); blockIdx += coreNum) {
            auto blockShape = scheduler.template GetBlockShape<transB, BType>(blockIdx);
            const int64_t blockM = asc::te::get<MNK_M>(blockShape);
            const int64_t blockN = asc::te::get<MNK_N>(blockShape);
            if (blockM <= 0 || blockN <= 0) {
                continue;
            }
            auto blockCoord = scheduler.GetBlockCoord(blockIdx);
            const int64_t nPos = asc::te::get<MNK_N>(blockCoord);
            int64_t mPos = asc::te::get<MNK_M>(blockCoord);
            auto blockB = gmB.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(k, blockN));
            auto blockBias = gmBias.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, blockN));

            // 输入按完整 rank 行段存放，跨 rank 的块按本通信片内的行数拆分。
            int64_t remainingM = blockM;
            while (remainingM > 0) {
                const uint64_t rankIdx = static_cast<uint64_t>(mPos) / sp.sliceM;
                const uint64_t rankOffset = static_cast<uint64_t>(mPos) % sp.sliceM;
                const int64_t rankRemainingM = sp.sliceM - rankOffset;
                const int64_t subM = remainingM < rankRemainingM ? remainingM : rankRemainingM;
                const int64_t inputM = rankIdx * sp.perRankM + sp.sliceStart + rankOffset;
                auto blockA = gmA.slice(asc::te::make_coord(inputM, 0L), asc::te::make_shape(subM, k));
                auto blockC = gmC.slice(asc::te::make_coord(mPos, nPos), asc::te::make_shape(subM, blockN));
                ProblemShape subShape{subM, blockN, k, 1L};
                // 本地 Matmul 累加 bias，输出转换后由 fusion 执行 ReduceScatter。
                mmOp_(blockA, blockB, blockBias, blockC, subShape);
                mPos += subM;
                remainingM -= subM;
            }
        }
    }

    BlockMmad mmOp_;
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
