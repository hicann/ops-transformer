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
 * \file kernel_matmul_al1_full_load.h
 * \brief
 * For elementwise ADD or MUL fusion, AIC writes MMAD results to UB in MIX mode. The paired AIV applies the epilogue
 * with x3 and writes to GM. When FUSED_OP_TYPE is 0, AIC writes directly to GM in AIC_ONLY mode.
 */

#pragma once

#include "kernel_basic_intf.h"

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_al1_full_load.h"
#include "kernel_universal_decl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class HcGradNnGemm<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                   AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                       KernelMmadMultiBlockAFullLoad, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline HcGradNnGemm()
    {
        if constexpr (HAS_ELEMWISE) {
            if ASCEND_IS_AIV {
                Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
                Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + 1);
            }
        }
    }
    __aicore__ inline ~HcGradNnGemm()
    {
        if constexpr (HAS_ELEMWISE) {
            if ASCEND_IS_AIC {
                Sync::WaitForVector<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG, true);
                Sync::WaitForVector<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1, true);
            }
        }
    }

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    // mmad
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        uint32_t fusionLda = 0;
        Params() = default;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        Init(params);

        // Initialize the block scheduler.
        BlockScheduler bs(params.problemShape, params.schParams);
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t realCoreNums = bs.GetCoreNums();
        if constexpr (HAS_ELEMWISE) {
            if ASCEND_IS_AIV {
                if (!params.mmadParams.splitM && AscendC::GetSubBlockIdx() > 0) {
                    return;
                }
                curBlockIdx /= AscendC::GetTaskRation();
            }
        } else {
            if ASCEND_IS_AIV {
                return;
            }
        }
        if (curBlockIdx >= realCoreNums) {
            return;
        }
        Blaze::Gemm::SetHF32(params.schParams.isHf32);
        BlockMmad blockMmad;
        if ASCEND_IS_AIC {
            blockMmad.Init(params.mmadParams);
        }
        BlockEpilogue epilogueOp;
        if constexpr (HAS_ELEMWISE) {
            // Init accepts only parameters. The non-fused A full-load path uses Empty and skips Init.
            epilogueOp.Init(params.epilogueParams);
        }
        MatmulProcess(params, blockMmad, epilogueOp, bs, curBlockIdx, params.schParams.usedCoreNum, bs.GetBlockNums());
        Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
    }

private:
    __aicore__ inline auto MakeLayoutA2D(Params const& params)
    {
        uint64_t lda = params.fusionLda;
        if constexpr (BlockMmad::TRANS_A) {
            return asc::te::make_pattern_layout<LayoutA, asc::te::layout_trait_default<>>(
                asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, m_), asc::te::make_shape(asc::te::_1{}, k_)),
                asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, asc::te::_1{}),
                                     asc::te::make_stride(asc::te::_0{}, lda)));
        } else {
            return asc::te::make_pattern_layout<LayoutA, asc::te::layout_trait_default<>>(
                asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, m_), asc::te::make_shape(asc::te::_1{}, k_)),
                asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, lda),
                                     asc::te::make_stride(asc::te::_0{}, asc::te::_1{})));
        }
    }

    // For elementwise ADD or MUL, AIC writes MMAD results to UB in MIX mode. The paired AIV applies the epilogue with
    // x3 and writes to GM. In AIC_ONLY mode, AIC writes directly to GM, as in the main implementation.
    __aicore__ inline void MatmulProcess(Params const& params, BlockMmad& blockMmad, BlockEpilogue& epilogueOp,
                                         BlockScheduler& bs, int64_t curBlockIdx, int64_t coreNums,
                                         int64_t totalBlockNums)
    {
        auto layoutA = MakeLayoutA2D(params);
        auto layoutB = MakeLayoutB{}(batch_, k_, n_); // ND layout for B
        auto layoutBias = MakeLayoutBias{}(1L, n_);   // ND layout for Bias
        // A,B,Bias Gm Tensor
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        // Enable dual page tables.
        SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);

        // Process tiles in ping-pong mode
        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto blockShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx);
            auto blockCoord = bs.GetBlockCoord(blockIdx);
            auto shapeN = asc::te::get<MNK_N>(blockShape);
            auto shapeK = asc::te::get<MNK_K>(blockShape);
            auto coordN = asc::te::get<MNK_N>(blockCoord);
            curBatchIdx_ = static_cast<uint64_t>(asc::te::get<MNK_B>(blockCoord));
            if constexpr (HAS_ELEMWISE) {
                // In MIX mode, clamp the tail tile to valid N. AIC writes to UB and AIV runs the epilogue.
                const int64_t shapeM = asc::te::get<MNK_M>(blockShape);
                const int64_t validShapeN = AscendC::Std::min(shapeN, static_cast<int64_t>(n_) - coordN);
                if (shapeM <= 0 || validShapeN <= 0) {
                    continue;
                }
                BlockShape validBlockShape{shapeM, validShapeN, shapeK, 1};

                const uint64_t batchOffsetB = curBatchIdx_ * k_ * n_;
                const int64_t offsetC =
                    (static_cast<int64_t>(curBatchIdx_) * static_cast<int64_t>(m_)) * static_cast<int64_t>(n_) + coordN;
                auto gmBBatch = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_ + batchOffsetB), layoutB);
                // Match the slice coordinates and shape to the nested layoutB, as in the AIC_ONLY subTensorB.
                // The 2D is_flat_batch shortcut only applies to flat layouts. MakeLayoutB produces a nested layout.
                auto subTensorB = gmBBatch.slice(asc::te::make_coord(0L, asc::te::make_coord(0L, coordN)),
                                                 asc::te::make_shape(1L, asc::te::make_shape(shapeK, validShapeN)));
                auto gmBlockB = asc::te::squeeze<0>(subTensorB);
                auto gmBlockBiasValid =
                    gmBias.slice(asc::te::make_coord(0L, coordN), asc::te::make_shape(1L, validShapeN));
                auto layoutUb = MakeLayoutC{}(shapeM, validShapeN);
                auto ubTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, CType>(0), layoutUb);
                if ASCEND_IS_AIC {
                    blockMmad(gmA, gmBlockB, gmBlockBiasValid, ubTensor, validBlockShape);
                }
                if ASCEND_IS_AIV {
                    epilogueOp(validBlockShape, offsetC, params.mmadParams.splitM, params.schParams.baseM,
                               params.schParams.baseN, params.mmadParams.ubDB);
                }
            } else {
                // In AIC_ONLY mode, slice B and C by batch and write MMAD results directly to GM.
                auto layoutC = MakeLayoutC{}(batch_, m_, n_); // ND layout for C
                auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
                // Block offset
                auto subTensorB = gmB.slice(asc::te::make_coord(curBatchIdx_, asc::te::make_coord(0L, coordN)),
                                            asc::te::make_shape(1L, asc::te::make_shape(shapeK, shapeN)));
                auto gmBlockB = asc::te::squeeze<0>(subTensorB);
                auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, coordN), asc::te::make_shape(1L, shapeN));
                auto subTensorC = gmC.slice(asc::te::make_coord(curBatchIdx_, asc::te::make_coord(0L, coordN)),
                                            asc::te::make_shape(1L, asc::te::make_shape(m_, shapeN)));
                auto gmBlockC = asc::te::squeeze<0>(subTensorC);
                blockMmad(gmA, gmBlockB, gmBlockBias, gmBlockC, blockShape);
            }
        }
    }

    __aicore__ inline void Init(Params const& params)
    {
        auto blockMmadParams = params.mmadParams;
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        batch_ = static_cast<uint64_t>(AscendC::Std::max(asc::te::get<MNK_B>(params.problemShape), 1L));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
    }

    template <typename TensorA, typename TensorB>
    __aicore__ inline void SetL2Cache(TensorA& gmA, TensorB& gmB, uint32_t l2CacheMode)
    {
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == B_L2_CACHE_DISABLE) {
            gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == A_L2_CACHE_DISABLE) {
            gmA.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
    }

private:
    static constexpr bool TRANS_A = BlockMmad::TRANS_A;
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    static constexpr bool HAS_ELEMWISE = BlockMmad::DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_ADD ||
                                         BlockMmad::DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_MUL;
    static constexpr uint64_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // Initialize the optional bias input here.

    uint64_t curBatchIdx_ = {0};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t batch_{1};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
