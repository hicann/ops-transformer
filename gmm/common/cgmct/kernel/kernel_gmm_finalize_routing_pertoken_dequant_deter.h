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
 * \file kernel_gmm_finalize_routing_pertoken_dequant_deter.h
 * \brief Deterministic variant with sliding window flush for A5.
 */

#ifndef KERNEL_GMM_FINALIZE_ROUTING_PERTOKEN_DEQUANT_DETER_H
#define KERNEL_GMM_FINALIZE_ROUTING_PERTOKEN_DEQUANT_DETER_H

#include "kernel_gmm_finalize_routing_pertoken_dequant.h"
#include "../epilogue/block_epilogue_dequant_finalize_routing_deter.h"

namespace Cgmct {
namespace Gemm {
namespace Kernel {

struct DeterSyncConfig {
    uint64_t curM = 0UL;
    uint64_t curGroupM = 0UL;
    uint64_t lowBoundM = 0UL;
    uint64_t windowSize = 0UL;
    uint64_t windowStartM = 0UL;
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockPrologue_,
          class BlockEpilogueDequantFinalizeRoutingDeter_, class BlockScheduler_, typename Enable_ = void>
class KernelGmmFinalizeRoutingPertokenDequantDeter {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
                  "KernelGmmFinalizeRoutingPertokenDequantDeter is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockPrologue_,
          class BlockEpilogueDequantFinalizeRoutingDeter_, class BlockScheduler_>
class KernelGmmFinalizeRoutingPertokenDequantDeter<
    ProblemShape_, BlockMmadBuilder_, BlockPrologue_, BlockEpilogueDequantFinalizeRoutingDeter_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtWithTailSplitScheduler>>>
    : public KernelGmmFinalizeRoutingPertokenDequant<ProblemShape_, BlockMmadBuilder_, BlockPrologue_,
                                                     BlockEpilogueDequantFinalizeRoutingDeter_, BlockScheduler_> {
public:
    using Base = KernelGmmFinalizeRoutingPertokenDequant<ProblemShape_, BlockMmadBuilder_, BlockPrologue_,
                                                         BlockEpilogueDequantFinalizeRoutingDeter_, BlockScheduler_>;

    using Base::aGlobal_;
    using Base::baseOffset_;
    using Base::bGlobal_;
    using Base::blockOffset_;
    using Base::End;
    using Base::epilogueDequantOp_;
    using Base::formatB;
    using Base::GetSplitValueFromGroupList;
    using Base::groupListGm_;
    using Base::isVecSetSyncCom_;
    using Base::l0cOutUb_;
    using Base::mmadOp_;
    using Base::NotifyCube;
    using Base::NotifyVector;
    using Base::problemShape_;
    using Base::prologueOp_;
    using Base::transA;
    using Base::transB;
    using Base::UpdateOffset;
    using Base::WaitForCube;
    using Base::WaitForVector;
    using typename Base::AType;
    using typename Base::BlockCoord;
    using typename Base::BlockEpilogueParams;
    using typename Base::BlockMmadArguments;
    using typename Base::BlockMmadParams;
    using typename Base::BlockPrologueArguments;
    using typename Base::BlockPrologueParams;
    using typename Base::BlockSchedulerOp;
    using typename Base::BlockShape;
    using typename Base::BType;
    using typename Base::CoordClass;
    using typename Base::CType;

    DeterSyncConfig deterSync_;
    uint64_t cumulativeGroupM_ = 0;

    struct GMMTiling : Base::GMMTiling {
        uint32_t deterWorkspaceSize;
        uint32_t coreNum;
        __aicore__ GMMTiling()
            : Base::GMMTiling()
        {}
        __aicore__ GMMTiling(uint32_t groupNum_, uint8_t groupListType_, int32_t baseM_, int32_t baseN_, int32_t baseK_,
                             uint8_t hasBias_, uint32_t deterWsSize_, uint32_t coreNum_)
            : Base::GMMTiling(groupNum_, groupListType_, baseM_, baseN_, baseK_, hasBias_),
              deterWorkspaceSize(deterWsSize_),
              coreNum(coreNum_)
        {}
    };

    struct Arguments {
        typename Base::ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockPrologueArguments prologueArgs;
        BlockEpilogueParams epilogueArgs;
        GMMTiling gmmArgs;
        Arguments() = default;
    };

    struct Params {
        typename Base::ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockPrologueParams prologueParams;
        BlockEpilogueParams epilogueParams;
        GMMTiling gmmParams;
        Params() = default;
    };

    __aicore__ inline void InitParamsAndTensor(const Params &params)
    {
        Get<MNK_N>(problemShape_) = params.gmmParams.matmulTiling->N;
        Get<MNK_K>(problemShape_) = params.gmmParams.matmulTiling->Ka;
        groupListGm_.SetGlobalBuffer((__gm__ int64_t *)params.mmadParams.groupListGmAddr);
    }

    __aicore__ inline void UpdateGlobalBuffer(const Params &params)
    {
        if ASCEND_IS_AIC {
            aGlobal_.SetGlobalBuffer((__gm__ AType *)params.mmadParams.aGmAddr + Get<IDX_A_OFFSETS>(baseOffset_));
            bGlobal_.SetGlobalBuffer((__gm__ BType *)params.mmadParams.bGmAddr + Get<IDX_B_OFFSETS>(baseOffset_));
        }
        if ASCEND_IS_AIV {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                0L,
                Get<IDX_BIAS_OFFSETS>(baseOffset_),
                Get<IDX_X2SCALE_OFFSETS>(baseOffset_),
                Get<IDX_X1SCALE_OFFSETS>(baseOffset_),
                Get<IDX_LOGIT_OFFSETS>(baseOffset_),
                Get<IDX_LOGIT_OFFSETS>(baseOffset_)};
            epilogueDequantOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    }

    __aicore__ inline bool UpdateGroupParams(const Params &params, uint32_t groupIdx)
    {
        UpdateOffset(groupIdx);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.gmmParams.groupListType);
        Get<MNK_M>(problemShape_) = splitValue;
        if (Get<MNK_M>(problemShape_) == 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline void FRDeterministic(const Params &params)
    {
        uint64_t totalM = deterSync_.curGroupM - deterSync_.windowStartM;
        if (totalM == 0) {
            return;
        }
        if (totalM > deterSync_.windowSize) {
            totalM = deterSync_.windowSize;
            deterSync_.curGroupM = deterSync_.windowStartM + deterSync_.windowSize;
        }
        uint64_t coreNumVec = params.gmmParams.coreNum * GetTaskRation();
        uint64_t n = params.gmmParams.matmulTiling->N;
        for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
            auto outRow = epilogueDequantOp_.GetRowIndex(deterSync_.windowStartM + mOffset);
            if (outRow % coreNumVec != GetBlockIdx()) {
                continue;
            }
            epilogueDequantOp_.DeterministicFlushRow(mOffset, outRow, n);
            PipeBarrier<PIPE_MTE3>();
        }
    }

    __aicore__ inline void ProcessSingleGroup(const Params &params, BlockSchedulerOp &bs, uint32_t groupIdx)
    {
        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        bs.UpdateNextProblem(problemShape_);
        epilogueDequantOp_.UpdateNextProblem(problemShape_);
        UpdateGlobalBuffer(params);
        CoordClass coord(m, n, k, params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        BlockCoord tileIdx;
        while (bs.GetTileIdxRowMajor(tileIdx)) {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            blockOffset_ = coord.template GetQuantOffset<GroupedMatmul::QuantMode::PERTOKEN_MODE>(
                Get<IDX_M_TILEIDXS>(tileIdx), Get<IDX_N_TILEIDXS>(tileIdx), Get<IDX_M_TAIL_SPLIT_TILEIDXS>(singleShape),
                Get<IDX_N_TAIL_SPLIT_TILEIDXS>(singleShape));
            int64_t y = Get<IDX_C_OFFSETS>(blockOffset_);
            int64_t tileMOffset = y / n;
            int64_t singleM = Get<MNK_M>(singleShape);
            int64_t wsAccum = static_cast<int64_t>(cumulativeGroupM_ - deterSync_.windowStartM);
            int64_t tileWsEnd = wsAccum + tileMOffset + singleM;
            if (tileWsEnd > static_cast<int64_t>(deterSync_.windowSize)) {
                deterSync_.curGroupM = cumulativeGroupM_ + static_cast<uint64_t>(tileMOffset);
                SyncAll<false>();
                if ASCEND_IS_AIV {
                    FRDeterministic(params);
                }
                deterSync_.windowStartM = deterSync_.curGroupM;
                deterSync_.lowBoundM = deterSync_.curGroupM + deterSync_.windowSize;
                SyncAll<false>();
            }
            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) {
                    WaitForVector();
                }
                AscendC::Std::tuple<int32_t, int32_t, int32_t> mmSingleShape{singleM, Get<MNK_N>(singleShape), k};
                mmadOp_(aGlobal_[Get<IDX_A_OFFSETS>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSETS>(blockOffset_)],
                        l0cOutUb_, mmSingleShape, transA, transB);
                NotifyVector();
            }
            isVecSetSyncCom_ = true;
            if ASCEND_IS_AIV {
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{singleM, Get<MNK_N>(singleShape),
                                                                                      0, 0};
                int64_t nOffset = y - tileMOffset * n;
                epilogueDequantOp_.SetWorkspaceGroupOffset(cumulativeGroupM_ - deterSync_.windowStartM);
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    nOffset,
                    Get<IDX_BIAS_OFFSETS>(blockOffset_),
                    Get<IDX_X2SCALE_OFFSETS>(blockOffset_),
                    Get<IDX_X1SCALE_OFFSETS>(blockOffset_),
                    tileMOffset,
                    tileMOffset};
                WaitForCube();
                epilogueDequantOp_(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        }
    }

    __aicore__ inline void operator()(const Params &params)
    {
        if ASCEND_IS_AIV {
            prologueOp_.Init(params.prologueParams);
            prologueOp_();
        }
        if ASCEND_IS_AIC {
            mmadOp_.Init(const_cast<TCubeTiling *__restrict>(params.gmmParams.matmulTiling), GetTPipePtr());
            l0cOutUb_ = epilogueDequantOp_.GetL0c2UbTensor();
        }
        InitParamsAndTensor(params);
        BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);

        deterSync_.windowSize =
            params.gmmParams.deterWorkspaceSize / (params.gmmParams.matmulTiling->N * sizeof(CType));
        deterSync_.windowSize = (deterSync_.windowSize / params.gmmParams.baseM) * params.gmmParams.baseM;
        deterSync_.lowBoundM = deterSync_.windowSize;

        SyncAll<false>();
        if ASCEND_IS_AIV {
            epilogueDequantOp_.Init(params.epilogueParams, GetTPipePtr());
        }
        uint32_t groupNum = params.gmmParams.groupNum;
        if constexpr (formatB == CubeFormat::NZ) {
            if constexpr (transB) {
                bs.SetTailAlign(1, MATMUL_MNK_ALIGN);
            } else {
                bs.SetTailAlign(1, MATMUL_MNK_ALIGN_INT8);
            }
        }
        for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }
            if ((cumulativeGroupM_ + (uint64_t)Get<MNK_M>(problemShape_)) > deterSync_.lowBoundM) {
                deterSync_.curGroupM = cumulativeGroupM_;
                SyncAll<false>();
                if ASCEND_IS_AIV {
                    FRDeterministic(params);
                }
                deterSync_.windowStartM = deterSync_.curGroupM;
                deterSync_.lowBoundM = deterSync_.curGroupM + deterSync_.windowSize;
                SyncAll<false>();
            }
            ProcessSingleGroup(params, bs, groupIdx);
            cumulativeGroupM_ += Get<MNK_M>(problemShape_);
            deterSync_.curGroupM = cumulativeGroupM_;
        }

        SyncAll<false>();
        if ASCEND_IS_AIV {
            FRDeterministic(params);
        }
        SyncAll<false>();

        End();
    }
};
} // namespace Kernel
} // namespace Gemm
} // namespace Cgmct
#endif
