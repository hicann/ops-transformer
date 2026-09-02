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
 * \file kernel_qgmm_inplace_add_fixpipe_dequant.h
 * \brief
 */

#ifndef KERNEL_QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_H
#define KERNEL_QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_H

#include "../block/block_mmad_fixpipe_dequant.h"
#include "../block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "../block/block_scheduler_utils.h"
#include "../epilogue/cgmct_block_epilogue_empty.h"
#include "../utils/cgmct_common_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/fill_utils.h"
#include "../utils/grouped_matmul_constant.h"
#include "../utils/cgmct_layout_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/tuple_utils.h"
#include "./semaphore.h"
#include "kernel_basic_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

namespace Cgmct {
namespace Gemm {
namespace Kernel {
#define QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_FUN_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::GroupedMatmul;

namespace {
// fixpipe 只能取 32位 dequant scale 的高19位，低13位清零
constexpr uint32_t DEQSCALE_HIGH19_MASK = 0xFFFFE000U;
} // namespace

QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_TEM_PARAMS
class KernelQGmmInplaceAddFixpipeDequant {
public:
    __aicore__ inline KernelQGmmInplaceAddFixpipeDequant() {}
    __aicore__ inline ~KernelQGmmInplaceAddFixpipeDequant() {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    // schedulerOp
    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmad::L1TileShape,
                                                                    typename BlockMmad::L0TileShape, BlockScheduler,
                                                                    transA, transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutB = typename BlockMmad::LayoutB;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // x, w, x1Scale, x2Scale, bias, y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, CubeFormat::ND, CubeFormat::ND>;

    struct GMMTiling {
        uint32_t groupNum;
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1;
        uint32_t kBL1;
        uint8_t dbL0C;
        uint8_t groupListType;
        __aicore__ GMMTiling() {}
        __aicore__ GMMTiling(uint32_t groupNum_, uint32_t m_, uint32_t n_, uint32_t k_, uint32_t baseM_,
                             uint32_t baseN_, uint32_t baseK_, uint32_t kAL1_, uint32_t kBL1_, uint8_t dbL0C_,
                             uint8_t groupListType_)
            : groupNum(groupNum_),
              m(m_),
              n(n_),
              k(k_),
              baseM(baseM_),
              baseN(baseN_),
              baseK(baseK_),
              kAL1(kAL1_),
              kBL1(kBL1_),
              dbL0C(dbL0C_),
              groupListType(groupListType_)
        {}
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        GMMTiling gmmParams;
        Params() = default;
    };

public:
    __aicore__ inline void Init(const Params &params);
    __aicore__ inline void Run(const Params &params);
    __aicore__ inline void operator()(const Params &params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        Run(params);
    }

private:
    __aicore__ inline void ProcessSingleGroup(const Params &params, BlockSchedulerOp &bs, uint32_t groupIdx);
    __aicore__ inline void UpdateOffset(uint32_t groupIdx);
    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx);
    __aicore__ inline void UpdateMMGlobalAddr();
    __aicore__ inline void Iterate(int64_t singleCoreM, int64_t singleCoreN);
    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockSchedulerOp &bs, uint32_t groupIdx);

private:
    BlockMmad mmadOp_;
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<float> x1ScaleGlobal_;
    AscendC::GlobalTensor<float> x2ScaleGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGlobal_;
    AscendC::GlobalTensor<CType> yGlobal_;

    GM_ADDR groupListPtr_;
    GM_ADDR xTensorPtr_;
    GM_ADDR wTensorPtr_;
    GM_ADDR x1ScaleTensorPtr_;
    GM_ADDR x2ScaleTensorPtr_;
    GM_ADDR yTensorPtr_;

    int32_t preOffset_ = 0;
    uint32_t groupNum_;
    uint8_t groupListType_;

    uint64_t scaleScalar_ = 0;
};

QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_TEM_PARAMS
__aicore__ inline void KernelQGmmInplaceAddFixpipeDequant<QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_FUN_PARAMS>::Run(
    const Params &params)
{
    Init(params);
    BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
    AscendC::SetAtomicAdd<float>();
    for (uint32_t loopIdx = 0; loopIdx < groupNum_; ++loopIdx) {
        UpdateOffset(loopIdx);
        // Update the group-specific M/N/K values.
        int32_t splitValue = GetSplitValueFromGroupList(loopIdx);
        Get<MNK_K>(problemShape_) = splitValue;
        if (Get<MNK_M>(problemShape_) <= 0 || Get<MNK_K>(problemShape_) <= 0) {
            continue;
        }
        if ASCEND_IS_AIC {
            mmadOp_.UpdateParamsForNextProblem(problemShape_);
        }

        bs.UpdateNextProblem(problemShape_);
        // Further split the tail tiles of the last group to use more cores when possible.
        if (IsLastGroupAndNeedSplit(bs, loopIdx)) {
            bs.UpdateTailTile();
        }
        UpdateMMGlobalAddr();
        ProcessSingleGroup(params, bs, loopIdx);
    }
    AscendC::SetAtomicNone();
}

QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_TEM_PARAMS
__aicore__ inline void KernelQGmmInplaceAddFixpipeDequant<QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_FUN_PARAMS>::Init(
    const Params &params)
{
    xTensorPtr_ = params.mmadParams.aGmAddr;
    wTensorPtr_ = params.mmadParams.bGmAddr;
    x1ScaleTensorPtr_ = params.mmadParams.x1ScaleGmAddr;
    x2ScaleTensorPtr_ = params.mmadParams.x2ScaleGmAddr;
    groupListPtr_ = params.mmadParams.groupListGmAddr;
    yTensorPtr_ = params.mmadParams.cGmAddr;

    groupNum_ = params.gmmParams.groupNum;
    groupListType_ = params.gmmParams.groupListType;
    Get<MNK_M>(problemShape_) = params.gmmParams.m;
    Get<MNK_N>(problemShape_) = params.gmmParams.n;
    Get<MNK_K>(problemShape_) = params.gmmParams.k;

    if (groupListPtr_ != nullptr) {
        groupListGlobal_.SetGlobalBuffer((__gm__ int64_t *)groupListPtr_);
    }
    TupleShape l0Shape{static_cast<int64_t>(params.gmmParams.baseM), static_cast<int64_t>(params.gmmParams.baseN),
                       static_cast<int64_t>(params.gmmParams.baseK)};
    L1Params l1Params{static_cast<uint64_t>(params.gmmParams.kAL1), static_cast<uint64_t>(params.gmmParams.kBL1),
                      2UL}; // Enable double buffering by default.
    mmadOp_.Init(problemShape_, l0Shape, l1Params, false, params.gmmParams.dbL0C == DOUBLE_BUFFER_COUNT);
}

QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_TEM_PARAMS
__aicore__ inline bool KernelQGmmInplaceAddFixpipeDequant<
    QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_FUN_PARAMS>::IsLastGroupAndNeedSplit(const BlockSchedulerOp &bs, uint32_t groupIdx)
{
    // Consider tail split only when at least half of the cores are still available.
    return groupIdx == groupNum_ - 1 && (bs.GetEndBlockIdx() + 1) <= AscendC::GetBlockNum() >> 1;
}

QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_TEM_PARAMS
__aicore__ inline void KernelQGmmInplaceAddFixpipeDequant<QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_FUN_PARAMS>::UpdateOffset(
    uint32_t groupIdx)
{
    // baseOffset is 0 when groupIdx = 0
    if (groupIdx == 0) {
        return;
    }
    uint64_t m = Get<M_VALUE>(problemShape_);
    uint64_t n = Get<N_VALUE>(problemShape_);
    uint64_t k = Get<K_VALUE>(problemShape_);
    // aBaseOffset += m * k
    Get<IDX_A_OFFSET>(baseOffset_) = Get<IDX_A_OFFSET>(baseOffset_) + m * k;
    // bBaseOffset += n * k
    Get<IDX_B_OFFSET>(baseOffset_) = Get<IDX_B_OFFSET>(baseOffset_) + n * k;
    Get<IDX_X1SCALE_OFFSET>(baseOffset_) = groupIdx;
    Get<IDX_X2SCALE_OFFSET>(baseOffset_) = groupIdx;
    // yBaseOffset += m * n
    Get<IDX_C_OFFSET>(baseOffset_) = Get<IDX_C_OFFSET>(baseOffset_) + m * n;
}

QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_TEM_PARAMS
__aicore__ inline void
KernelQGmmInplaceAddFixpipeDequant<QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_FUN_PARAMS>::ProcessSingleGroup(
    const Params &params, BlockSchedulerOp &bs, uint32_t groupIdx)
{
    CoordClass coord(Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_),
                     static_cast<int64_t>(params.gmmParams.baseM), params.gmmParams.baseN, params.gmmParams.baseK);
    BlockCoord tileIdx;
    while (bs.GetTileIdx(tileIdx)) {
        BlockShape singleShape = bs.GetBlockShape(tileIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }
        blockOffset_ = coord.template GetQuantOffset<QuantMode::PERTENSOR_MODE>(
            Get<IDX_M_TILEIDX>(tileIdx), Get<IDX_N_TILEIDX>(tileIdx), Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
            Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
        Iterate(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape));
    }
}

QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_TEM_PARAMS
__aicore__ inline void KernelQGmmInplaceAddFixpipeDequant<QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_FUN_PARAMS>::Iterate(
    int64_t singleCoreM, int64_t singleCoreN)
{
    AscendC::Std::tuple<int64_t, int64_t, int64_t> blockShape{singleCoreM, singleCoreN,
                                                              static_cast<int64_t>(Get<MNK_K>(problemShape_))};

    mmadOp_(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)], scaleScalar_,
            yGlobal_[Get<IDX_C_OFFSET>(blockOffset_)], blockShape);
}

QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_TEM_PARAMS
__aicore__ inline void
KernelQGmmInplaceAddFixpipeDequant<QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_FUN_PARAMS>::UpdateMMGlobalAddr()
{
    // Update global tensor addresses for the current grouped matmul instance.
    aGlobal_.SetGlobalBuffer((__gm__ AType *)xTensorPtr_ + Get<IDX_A_OFFSET>(baseOffset_));
    bGlobal_.SetGlobalBuffer((__gm__ BType *)wTensorPtr_ + Get<IDX_B_OFFSET>(baseOffset_));
    x1ScaleGlobal_.SetGlobalBuffer((__gm__ float *)x1ScaleTensorPtr_ + Get<IDX_X1SCALE_OFFSET>(baseOffset_));
    x2ScaleGlobal_.SetGlobalBuffer((__gm__ float *)x2ScaleTensorPtr_ + Get<IDX_X2SCALE_OFFSET>(baseOffset_));
    float deqScale = x1ScaleGlobal_.GetValue(0) * x2ScaleGlobal_.GetValue(0);
    uint32_t uint32Scale = *(reinterpret_cast<uint32_t *>(&deqScale));
    scaleScalar_ = static_cast<uint64_t>(uint32Scale & DEQSCALE_HIGH19_MASK);

    yGlobal_.SetGlobalBuffer((__gm__ CType *)yTensorPtr_ + Get<IDX_C_OFFSET>(baseOffset_));
    yGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
}

QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_TEM_PARAMS
__aicore__ inline int32_t KernelQGmmInplaceAddFixpipeDequant<
    QGMM_INPLACE_ADD_FIXPIPE_DEQUANT_FUN_PARAMS>::GetSplitValueFromGroupList(uint32_t groupIdx)
{
    int32_t splitValue = 0;
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupListGlobal_.GetValue(groupIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else if (groupListType_ == 1) {
        splitValue = static_cast<int32_t>(groupListGlobal_.GetValue(groupIdx));
    }
    Get<MNK_K>(problemShape_) = splitValue;
    return splitValue;
}

} // namespace Kernel
} // namespace Gemm
} // namespace Cgmct

#endif
