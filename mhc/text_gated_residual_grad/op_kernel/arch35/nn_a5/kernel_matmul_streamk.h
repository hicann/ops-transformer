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
 * \file kernel_matmul_streamk.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"
#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_matmul_streamk_split_k.h"
#include "kernel_universal_decl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class HcGradNnGemm<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_same_v<KernelMultiBlockStreamK, typename BlockMmad_::DispatchPolicy::ScheduleType> &&
        AscendC::Std::is_same_v<KernelMultiBlockStreamK, typename BlockEpilogue_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline HcGradNnGemm() {}
    __aicore__ inline ~HcGradNnGemm() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

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
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, asc::te::layout_trait_default<AType>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, asc::te::layout_trait_default<BType>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, asc::te::layout_trait_default<CType>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, asc::te::layout_trait_default<BiasType>>;

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

        if (params.schParams.usedCoreNum <= 0) {
            return;
        }

        BlockScheduler bs(params.problemShape, params.schParams);
        // L1 & L0 & singlecore, per core use L1 once in stream k
        mL1_ = params.schParams.baseM;
        nL1_ = params.schParams.baseN;
        // Compute tile counts directly along M, N, and K.
        mBlockNums_ = Blaze::Gemm::CeilDiv(m_, mL1_);
        nBlockNums_ = Blaze::Gemm::CeilDiv(n_, nL1_);
        skBlockNums_ = Blaze::Gemm::CeilDiv(k_, params.schParams.singleCoreK);

        if ASCEND_IS_AIC {
            ProcessOnAic(params, bs);
        }

        if ASCEND_IS_AIV {
            ProcessOnAiv(params, bs);
        }
    }

private:
    __aicore__ inline auto MakeLayoutAGm(Params const& params)
    {
        uint64_t lda = params.fusionLda;
        if constexpr (BlockMmad::TRANS_A) {
            return asc::te::make_pattern_layout<LayoutA, asc::te::layout_trait_default<>>(
                asc::te::make_shape(uint64_t(batch_),
                                    asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, uint64_t(m_)),
                                                        asc::te::make_shape(asc::te::_1{}, uint64_t(k_)))),
                asc::te::make_stride(uint64_t(k_) * lda,
                                     asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, asc::te::_1{}),
                                                          asc::te::make_stride(asc::te::_0{}, lda))));
        } else {
            return asc::te::make_pattern_layout<LayoutA, asc::te::layout_trait_default<>>(
                asc::te::make_shape(uint64_t(batch_),
                                    asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, uint64_t(m_)),
                                                        asc::te::make_shape(asc::te::_1{}, uint64_t(k_)))),
                asc::te::make_stride(uint64_t(m_) * lda,
                                     asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, lda),
                                                          asc::te::make_stride(asc::te::_0{}, asc::te::_1{}))));
        }
    }
    __aicore__ inline void ProcessOnAic(Params const& params, BlockScheduler& bs)
    {
        BlockMmad blockMmad;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        if (curBlockIdx >= bs.GetCoreNums()) {
            Sync::NotifyVector<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG, true);
            return;
        }

        Blaze::Gemm::SetHF32(params.schParams.isHf32);

        blockMmad.Init(params.mmadParams);

        int64_t totalBlockNums = bs.GetBlockNums();
        int64_t usedCoreNum = params.schParams.usedCoreNum;
        int64_t tailSKTotalBlockNums =
            static_cast<int64_t>(((mBlockNums_ * nBlockNums_ * batch_) % usedCoreNum) * skBlockNums_);

        auto layoutA = MakeLayoutAGm(params);
        auto layoutB = MakeLayoutB{}(batch_, k_, n_);
        auto layoutC = MakeLayoutC{}(batch_, m_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += usedCoreNum) {
            int64_t tmpBlockIdx = blockIdx;
            if (!bs.CheckIsSkScene(0)) { // SK Preload in DP+SK
                if (blockIdx % usedCoreNum < tailSKTotalBlockNums &&
                    (Blaze::Gemm::CeilDiv(blockIdx + 1, usedCoreNum) ==
                     (Blaze::Gemm::CeilDiv(totalBlockNums, usedCoreNum) - 1))) {
                    tmpBlockIdx = blockIdx + usedCoreNum;
                } else if (blockIdx % usedCoreNum < tailSKTotalBlockNums &&
                           (Blaze::Gemm::CeilDiv(blockIdx + 1, usedCoreNum) ==
                            Blaze::Gemm::CeilDiv(totalBlockNums, usedCoreNum))) {
                    tmpBlockIdx = blockIdx - usedCoreNum;
                }
            }
            BlockShape singleCoreShape = bs.GetBlockShape(tmpBlockIdx);
            BlockShape singleCoreCoord = bs.GetBlockCoord(tmpBlockIdx);
            auto coordM = asc::te::get<MNK_M>(singleCoreCoord);
            auto coordN = asc::te::get<MNK_N>(singleCoreCoord);
            auto coordK = asc::te::get<MNK_K>(singleCoreCoord);
            auto shapeM = asc::te::get<MNK_M>(singleCoreShape);
            auto shapeN = asc::te::get<MNK_N>(singleCoreShape);
            auto shapeK = asc::te::get<MNK_K>(singleCoreShape);
            curBatchIdx_ = asc::te::get<MNK_B>(singleCoreCoord);

            // Use blockSchedulerParams.singleCoreK for split-K.
            int64_t kSingleCore = bs.CheckIsSkScene(tmpBlockIdx) ? params.schParams.singleCoreK : k_;
            int64_t alignedBaseN =
                Blaze::Gemm::CeilAlign(params.schParams.baseN, static_cast<int64_t>(BLOCK_BYTE_SIZE));
            int64_t offsetWorkspace = (((tmpBlockIdx % usedCoreNum) / skBlockNums_) * skBlockNums_ + coordK) *
                                      params.schParams.baseM * alignedBaseN;
            // when fixpipe 1v2, dstStride should align to 32
            auto workspaceStrideColumn0 = BlockMmad::DispatchPolicy::FIXP_OPTI == MatMulL0C2Out::ND_FIXPIPE_1_2 ?
                                              Blaze::Gemm::CeilAlign(shapeN, static_cast<int64_t>(BLOCK_BYTE_SIZE)) :
                                              shapeN;
            auto workspaceShape = asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, shapeM),
                                                      asc::te::make_shape(asc::te::_1{}, shapeN));
            auto workspaceStride = asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, workspaceStrideColumn0),
                                                        asc::te::make_stride(asc::te::_0{}, asc::te::_1{}));
            auto layoutWorkspace =
                asc::te::make_pattern_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<float>>(
                    workspaceShape, workspaceStride);
            // workspace use 1 dim expression, make tensor each calculate
            auto gmWorkSpace = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::gm>(workspaceGmAddr_ + offsetWorkspace), layoutWorkspace);

            // split tensor from gm which needed by current calculate
            auto subTensorA =
                gmA.slice(AscendC::MakeCoord(curBatchIdx_, asc::te::make_coord(coordM * mL1_, coordK * kSingleCore)),
                          AscendC::MakeShape(1L, asc::te::make_shape(shapeM, shapeK)));
            auto gmBlockA = asc::te::squeeze<0>(subTensorA);
            auto subTensorB =
                gmB.slice(AscendC::MakeCoord(curBatchIdx_, asc::te::make_coord(coordK * kSingleCore, coordN * nL1_)),
                          AscendC::MakeShape(1L, asc::te::make_shape(shapeK, shapeN)));
            auto gmBlockB = asc::te::squeeze<0>(subTensorB);
            auto subTensorC =
                gmC.slice(AscendC::MakeCoord(curBatchIdx_, asc::te::make_coord(coordM * mL1_, coordN * nL1_)),
                          AscendC::MakeShape(1L, asc::te::make_shape(shapeM, shapeN)));
            auto gmBlockC = asc::te::squeeze<0>(subTensorC);
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, coordN * nL1_), asc::te::make_shape(1L, shapeN));
            blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, gmWorkSpace, singleCoreShape, coordK,
                      bs.CheckIsSkScene(tmpBlockIdx));

            if (tmpBlockIdx + usedCoreNum >= totalBlockNums) {
                Sync::NotifyVector<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG, true);
            }
        }

        Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
    }

    __aicore__ inline void ProcessOnAiv(Params const& params, BlockScheduler& bs)
    {
        int64_t usedCoreNum = params.schParams.usedCoreNum;
        uint64_t lastLoopTotalCnt = (mBlockNums_ * nBlockNums_ * batch_ % usedCoreNum) * skBlockNums_;
        uint64_t curBlockIdxInAiv = AscendC::GetBlockIdx();
        // Epilogue reads include peer Cube partials. All AIVs report the
        // paired Fixpipe completion before any AIV may read the workspace.
        constexpr uint16_t AIV_WORKSPACE_READY = 6U;
        Sync::WaitForCube<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(AIV_WORKSPACE_READY);
        AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>(AIV_WORKSPACE_READY);
        if (curBlockIdxInAiv >= lastLoopTotalCnt * AscendC::GetTaskRation()) {
            return;
        }

        BlockEpilogue epilogueOp;
        // size of m in L1 & L0 & singlecore, per core use L1 once in stream k
        BlockShape l1Block = {params.schParams.baseM, params.schParams.baseN, params.schParams.kL1, 1};
        BlockShape l0Block = {params.schParams.baseM, params.schParams.baseN, params.schParams.baseK, 1};
        auto epParams = params.epilogueParams;
        epParams.windowLen = params.schParams.windowLen;
        epilogueOp.Init(epParams, params.problemShape, l1Block, {mBlockNums_, nBlockNums_, skBlockNums_, batch_},
                        usedCoreNum, bs.CheckIsSkScene(0), l0Block);
        epilogueOp();
    }

private:
    __aicore__ inline void Init(Params const& params)
    {
        auto blockMmadParams = params.mmadParams;
        m_ = asc::te::get<MNK_M>(params.problemShape);
        n_ = asc::te::get<MNK_N>(params.problemShape);
        k_ = asc::te::get<MNK_K>(params.problemShape);
        batch_ = asc::te::get<MNK_B>(params.problemShape);
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        workspaceGmAddr_ = reinterpret_cast<__gm__ float*>(blockMmadParams.workspaceGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
    }

private:
    static constexpr uint16_t NUM_TWO = 2;
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 0;
    static constexpr uint16_t FLAG_ID_MAX = 16;

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr;
    __gm__ float* workspaceGmAddr_;

    int64_t curBatchIdx_ = {0};
    int64_t m_{1};
    int64_t n_{1};
    int64_t k_{1};
    int64_t batch_{1};
    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t skBlockNums_{0};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
