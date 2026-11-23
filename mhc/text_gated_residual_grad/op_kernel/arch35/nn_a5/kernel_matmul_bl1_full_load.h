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
 * \file kernel_matmul_bl1_full_load.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/epilogue/block/block_epilogue_fixpipe.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_bl1_full_load.h"
#include "kernel_universal_decl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class HcGradNnGemm<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                   AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                       KernelMmadMultiBlockBFullLoad, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline HcGradNnGemm()
    {
        if constexpr (BlockMmad::DispatchPolicy::L0C2OUT_MODEL != ON_THE_FLY || HAS_ELEMWISE) {
            if ASCEND_IS_AIV {
                Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);     // ping
                Sync::NotifyCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + 1); // pong
            }
        }
    }
    __aicore__ inline ~HcGradNnGemm()
    {
        if constexpr (BlockMmad::DispatchPolicy::L0C2OUT_MODEL != ON_THE_FLY || HAS_ELEMWISE) {
            if ASCEND_IS_AIC {
                Sync::WaitForVector<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG, true);     // ping
                Sync::WaitForVector<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1, true); // pong
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
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;

    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    // Elementwise add/mul fusion (FUSED_OP_TYPE != 0) runs the MIX data flow (AIC L0C->UB +
    // paired AIV epilogue) even for the ON_THE_FLY L0C2OUT model, mirroring the legacy CMCT
    // MatMulMixWithoutQueActKernel B-full-load OTF template. Without fusion the OTF model keeps
    // the original AIC-only direct L0C->GM write.
    static constexpr bool HAS_ELEMWISE = BlockMmad::DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_ADD ||
                                         BlockMmad::DispatchPolicy::FUSED_OP_TYPE == Blaze::Gemm::OP_TYPE_MUL;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<BType>>>;

    static_assert(
        AscendC::Std::is_one_of_v<
            AscendC::Std::tuple<AType, BType, CType, BiasType>, AscendC::Std::tuple<half, half, half, half>,
            AscendC::Std::tuple<half, half, half, float>, AscendC::Std::tuple<half, half, float, half>,
            AscendC::Std::tuple<half, half, float, float>,
            AscendC::Std::tuple<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>,
            AscendC::Std::tuple<bfloat16_t, bfloat16_t, bfloat16_t, float>,
            AscendC::Std::tuple<bfloat16_t, bfloat16_t, float, bfloat16_t>,
            AscendC::Std::tuple<bfloat16_t, bfloat16_t, float, float>, AscendC::Std::tuple<float, float, float, float>>,
        "Unsupported (AType, BType, CType, BiasType) combination");
    static_assert(!AscendC::Std::is_one_of_v<LayoutA, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "LayoutA and LayoutC cannot be nz_layout_ptn or zn_layout_ptn");

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
        BlockEpilogue epilogueOp;
        BlockMmad blockMmad;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        // OTF without fusion stays AIC-only (direct L0C->GM); 1V2 and fused OTF run the MIX flow.
        if constexpr (BlockMmad::DispatchPolicy::L0C2OUT_MODEL == ON_THE_FLY && !HAS_ELEMWISE) {
            if ASCEND_IS_AIV {
                return;
            }
        } else {
            if ASCEND_IS_AIV {
                if (!params.mmadParams.splitM && AscendC::GetSubBlockIdx() > 0) {
                    return;
                }
                curBlockIdx /= AscendC::GetTaskRation();
            }
        }
        Init(params);

        // Initialize the block scheduler.
        BlockScheduler bs(params.problemShape, params.schParams);
        int64_t realCoreNums = bs.GetCoreNums();
        if (curBlockIdx >= realCoreNums) {
            return;
        }
        Blaze::Gemm::SetHF32(params.schParams.isHf32);

        if constexpr (BlockMmad::DispatchPolicy::L0C2OUT_MODEL == ON_THE_FLY && !HAS_ELEMWISE) {
            blockMmad.Init(params.mmadParams);
            MatmulProcessOnTheFly(params, blockMmad, bs, curBlockIdx, params.schParams.usedCoreNum, bs.GetBlockNums());
        } else {
            if constexpr (HAS_ELEMWISE) {
                epilogueOp.Init(params.epilogueParams); // elemwise: params-only Init
            } else {
                epilogueOp.Init(params.epilogueParams, problemShape_); // fixpipe: mainline two-param
            }
            if ASCEND_IS_AIC {
                blockMmad.Init(params.mmadParams);
            }
            MatmulProcessFixpipe(params, epilogueOp, blockMmad, bs, curBlockIdx, params.schParams.usedCoreNum,
                                 bs.GetBlockNums());
        }
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

    __aicore__ inline void MatmulProcessOnTheFly(Params const& params, BlockMmad& blockMmad, BlockScheduler& bs,
                                                 int64_t curBlockIdx, int64_t coreNums, int64_t totalBlockNums)
    {
        auto layoutA = MakeLayoutA2D(params);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutC = MakeLayoutC{}(m_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);

        uint64_t preBatchIdx = 0;
        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto tileShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx);
            auto tileCoord = bs.GetBlockCoord(blockIdx);
            auto coordM = asc::te::get<MNK_M>(tileCoord);
            auto shapeM = asc::te::get<MNK_M>(tileShape);
            auto shapeK = asc::te::get<MNK_K>(tileShape);
            curBatchIdx_ = static_cast<uint64_t>(asc::te::get<MNK_B>(tileCoord));
            if (preBatchIdx != curBatchIdx_) {
                UpdateBatchOffset(params);
                gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
                gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
                preBatchIdx = curBatchIdx_;
                SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);
            }
            auto gmBlockA = gmA.slice(asc::te::make_coord(coordM, 0L), asc::te::make_shape(shapeM, shapeK));
            auto gmBlockC = gmC.slice(asc::te::make_coord(coordM, 0L), asc::te::make_shape(shapeM, n_));
            blockMmad(gmBlockA, gmB, gmBias, gmBlockC, tileShape);
        }
    }

    __aicore__ inline void MatmulProcessFixpipe(Params const& params, BlockEpilogue& epilogueOp, BlockMmad& blockMmad,
                                                BlockScheduler& bs, int64_t curBlockIdx, int64_t coreNums,
                                                int64_t totalBlockNums)
    {
        auto layoutA = MakeLayoutA2D(params);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutC = MakeLayoutC{}(m_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);

        uint64_t preBatchIdx = 0;
        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto tileShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx);
            auto tileCoord = bs.GetBlockCoord(blockIdx);
            auto coordM = asc::te::get<MNK_M>(tileCoord);
            auto coordN = asc::te::get<MNK_N>(tileCoord);
            auto shapeM = asc::te::get<MNK_M>(tileShape);
            auto shapeN = asc::te::get<MNK_N>(tileShape);
            auto shapeK = asc::te::get<MNK_K>(tileShape);
            curBatchIdx_ = static_cast<uint64_t>(asc::te::get<MNK_B>(tileCoord));
            if (preBatchIdx != curBatchIdx_) {
                UpdateBatchOffset(params);
                gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
                gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
                preBatchIdx = curBatchIdx_;
                SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);
            }
            int64_t offsetC = coordM * n_ + coordN;
            shapeN = AscendC::Std::min(shapeN, static_cast<int64_t>(n_) - coordN);
            TupleShape validBlockShape{shapeM, shapeN, shapeK, asc::te::get<MNK_B>(tileShape)};
            auto layoutUB = MakeLayoutC{}(shapeM, shapeN);
            auto ubLocal = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, CType>(0), layoutUB);
            auto gmBlockA = gmA.slice(asc::te::make_coord(coordM, 0L), asc::te::make_shape(shapeM, shapeK));
            if ASCEND_IS_AIC {
                blockMmad(gmBlockA, gmB, gmBias, ubLocal, validBlockShape);
            }
            if ASCEND_IS_AIV {
                epilogueOp(validBlockShape, offsetC, params.mmadParams.splitM, params.schParams.baseM,
                           params.schParams.baseN, params.mmadParams.ubDB);
            }
        }
    }

    __aicore__ inline void Init(Params const& params)
    {
        if constexpr (BlockMmad::DispatchPolicy::L0C2OUT_MODEL != ON_THE_FLY) {
            problemShape_ = params.problemShape;
        }
        auto blockMmadParams = params.mmadParams;
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        kAlign_ = Blaze::Gemm::CeilAlign(k_, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        isBias_ = blockMmadParams.biasGmAddr != nullptr;
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
    }

    __aicore__ inline void UpdateBatchOffset(Params const& params)
    {
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr) + curBatchIdx_ * m_ * k_;
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr) + curBatchIdx_ * m_ * n_;
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
    static constexpr bool IS_FP32 = (AscendC::Std::is_same_v<BType, float>);
    static constexpr int64_t C0_SIZE = IS_FP32 ? C0_SIZE_fp32 : C0_SIZE_fp16;
    static constexpr bool TRANS_A = BlockMmad::TRANS_A;
    static constexpr bool WEIGHTNZ_FORMAT = BlockMmad::WEIGHTNZ_FORMAT;

    static constexpr uint64_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;
    static constexpr uint16_t FLAG_ID_MAX = 16;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr;
    TupleShape problemShape_{};

    uint64_t curBatchIdx_ = {0};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t kAlign_{1};
    bool isBias_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
