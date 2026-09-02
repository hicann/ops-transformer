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
 * \file kernel_gmm_activation_mxquant.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_GMM_ACTIVATION_MXQUANT_H
#define MATMUL_KERNEL_KERNEL_GMM_ACTIVATION_MXQUANT_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "../block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "../block/block_scheduler_utils.h"
#include "../epilogue/block_epilogue_activation_mx_quant.h"
#include "../utils/cgmct_common_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/grouped_matmul_constant.h"
#include "../utils/cgmct_layout_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/tuple_utils.h"

namespace Cgmct {
namespace Gemm {
namespace Kernel {

namespace {
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_X1SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_X2SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_BIAS_OFFSET = 4UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint64_t SCALE_L1_BUFFER_NUM = 2UL;
} // namespace

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, typename Enable_ = void>
class KernelGmmActivationMixOnlineDynamic {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
                  "KernelGmmActivationMixOnlineDynamic is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class KernelGmmActivationMixOnlineDynamic<
    ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtWithTailSplitScheduler>>> {
public:
    __aicore__ inline KernelGmmActivationMixOnlineDynamic() {}
    __aicore__ inline ~KernelGmmActivationMixOnlineDynamic() {}

    using BlockEpilogue = BlockEpilogue_;
    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    using LayoutB = typename BlockMmad::LayoutB;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr bool isFp4 =
        AscendC::IsSameType<AType, fp4x2_e2m1_t>::value || AscendC::IsSameType<AType, fp4x2_e1m2_t>::value;
    static constexpr int32_t c0Size = isFp4 ? MATMUL_MNK_ALIGN_INT4 : MATMUL_MNK_ALIGN_INT8;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmad::L1TileShape,
                                                                    typename BlockMmad::L0TileShape, BlockScheduler,
                                                                    transA, transB>::SchedulerOp;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using DataTypeOut = typename BlockEpilogue::DataTypeOut;
    using L1Params = typename BlockMmad::L1Params;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using MmShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, formatB, CubeFormat::ND>;

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x1ScaleGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x2ScaleGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0, 0};
    uint64_t preOffset_ = 0;
    int64_t perGroupBOffset_ = 0;
    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    AscendC::LocalTensor<CType> l0cOutUb_;
    bool isVecSetSyncCom_ = false;

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
        uint32_t scaleKAL1;
        uint32_t scaleKBL1;
        uint8_t isBias;
        uint8_t dbL0C;
        int8_t groupType;
        uint8_t groupListType;

        __aicore__ GMMTiling() {}

        __aicore__ GMMTiling(uint32_t groupNum_, uint32_t m_, uint32_t n_, uint32_t k_, uint32_t baseM_,
                             uint32_t baseN_, uint32_t baseK_, uint32_t kAL1_, uint32_t kBL1_, uint32_t scaleKAL1_,
                             uint32_t scaleKBL1_, uint8_t isBias_, uint8_t dbL0C_, int8_t groupType_,
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
              scaleKAL1(scaleKAL1_),
              scaleKBL1(scaleKBL1_),
              isBias(isBias_),
              dbL0C(dbL0C_),
              groupType(groupType_),
              groupListType(groupListType_)
        {}
    };

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadParams mmadArgs;
        BlockEpilogueArguments epilogueArgs;
        GMMTiling gmmArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GMMTiling gmmParams;
        Params() = default;
    };

    __aicore__ inline void NotifyCube()
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
    }

    __aicore__ inline void WaitForVector()
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
    }

    __aicore__ inline void NotifyVector()
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    }

    __aicore__ inline void WaitForCube()
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                WaitForVector();
            }
        }
    }

    __aicore__ inline void SetL2CacheDisableIfNeeded(int64_t mSize, int64_t curBaseM)
    {
        if (curBaseM >= mSize) {
            bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            x2ScaleGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        } else {
            bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
            x2ScaleGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        }
    }

    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, uint8_t groupListType)
    {
        int32_t splitValue = 0;
        if (groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
            preOffset_ += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline void UpdateGlobalBuffer(const Params &params)
    {
        if ASCEND_IS_AIC {
            aGlobal_.SetGlobalBuffer((__gm__ AType *)params.mmadParams.aGmAddr + Get<IDX_A_OFFSET>(baseOffset_));
            bGlobal_.SetGlobalBuffer(GetTensorAddr<BType>(0, params.mmadParams.bGmAddr) +
                                     Get<IDX_B_OFFSET>(baseOffset_));
            x1ScaleGlobal_.SetGlobalBuffer((__gm__ AscendC::fp8_e8m0_t *)params.mmadParams.x1ScaleGmAddr +
                                           Get<IDX_X1SCALE_OFFSET>(baseOffset_));
            x2ScaleGlobal_.SetGlobalBuffer(GetTensorAddr<AscendC::fp8_e8m0_t>(0, params.mmadParams.x2ScaleGmAddr) +
                                           Get<IDX_X2SCALE_OFFSET>(baseOffset_));
            if (params.gmmParams.isBias == 1) {
                biasGlobal_.SetGlobalBuffer(GetTensorAddr<BiasType>(0, params.mmadParams.biasGmAddr) +
                                            Get<IDX_BIAS_OFFSET>(baseOffset_));
            }
        }
        if ASCEND_IS_AIV {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_), Get<IDX_C_SCALE_OFFSET>(baseOffset_), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    }

    __aicore__ inline void UpdateOffset(uint32_t groupIdx)
    {
        // baseOffset is 0 when groupIdx = 0
        if (groupIdx == 0) {
            return;
        }
        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);

        if constexpr (isFp4) {
            Get<IDX_A_OFFSET>(baseOffset_) = ((preOffset_ - m) * k) >> 1;
        } else {
            Get<IDX_A_OFFSET>(baseOffset_) = (preOffset_ - m) * k;
        }

        if constexpr (isFp4) {
            Get<IDX_B_OFFSET>(baseOffset_) = (perGroupBOffset_ * static_cast<int64_t>(groupIdx)) >> 1;
        } else {
            Get<IDX_B_OFFSET>(baseOffset_) = perGroupBOffset_ * static_cast<int64_t>(groupIdx);
        }

        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) = (preOffset_ - m) * scaleK;
        Get<IDX_X2SCALE_OFFSET>(baseOffset_) = static_cast<int64_t>(groupIdx) * n * scaleK;
        Get<IDX_BIAS_OFFSET>(baseOffset_) = n * static_cast<int64_t>(groupIdx);
        Get<IDX_C_OFFSET>(baseOffset_) = (preOffset_ - m) * n;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) = (preOffset_ - m) * CeilDiv(n, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    }

    __aicore__ inline bool UpdateGroupParams(const Params &params, uint32_t groupIdx)
    {
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.gmmParams.groupListType);
        Get<MNK_M>(problemShape_) = splitValue;
        if (Get<MNK_M>(problemShape_) == 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline void InitParamsAndTensor(const Params &params)
    {
        Get<MNK_M>(problemShape_) = params.gmmParams.m;
        Get<MNK_N>(problemShape_) = params.gmmParams.n;
        Get<MNK_K>(problemShape_) = params.gmmParams.k;
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        if constexpr (formatB == CubeFormat::NZ) {
            if constexpr (transB) {
                perGroupBOffset_ = Align16(n) * (isFp4 ? Align64(k) : Align32(k));
            } else {
                perGroupBOffset_ = (isFp4 ? Align64(n) : Align32(n)) * Align16(k);
            }
        } else {
            perGroupBOffset_ = n * k;
        }
        groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(params.mmadParams.groupListGmAddr));
    }

    __aicore__ inline void ProcessSingleGroup(const Params &params, BlockSchedulerOp &bs, uint32_t groupIdx)
    {
        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        CoordClass coord(m, n, k, params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        BlockCoord tileIdx;
        if (!bs.GetTileIdx(tileIdx)) {
            return;
        }
        UpdateOffset(groupIdx);
        UpdateGlobalBuffer(params);
        SetL2CacheDisableIfNeeded(Get<MNK_M>(problemShape_), static_cast<int64_t>(params.gmmParams.baseM));
        do {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
                return;
            }
            auto quantOffset = coord.template GetQuantOffset<GroupedMatmul::QuantMode::MX_PERGROUP_MODE, c0Size>(
                Get<IDX_M_TILEIDX>(tileIdx), Get<IDX_N_TILEIDX>(tileIdx), Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
                Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
            Get<IDX_A_OFFSET>(blockOffset_) = Get<IDX_A_OFFSET>(quantOffset);
            Get<IDX_B_OFFSET>(blockOffset_) = Get<IDX_B_OFFSET>(quantOffset);
            Get<IDX_X1SCALE_OFFSET>(blockOffset_) = Get<IDX_X1SCALE_OFFSET>(quantOffset);
            Get<IDX_X2SCALE_OFFSET>(blockOffset_) = Get<IDX_X2SCALE_OFFSET>(quantOffset);
            Get<IDX_BIAS_OFFSET>(blockOffset_) = Get<IDX_BIAS_OFFSET>(quantOffset);
            Get<IDX_C_OFFSET>(blockOffset_) = Get<IDX_C_OFFSET>(quantOffset);

            int64_t mOffset =
                Get<IDX_M_TILEIDX>(tileIdx) * params.gmmParams.baseM + Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape);
            int64_t nOffset =
                Get<IDX_N_TILEIDX>(tileIdx) * params.gmmParams.baseN + Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape);
            Get<IDX_C_SCALE_OFFSET>(blockOffset_) = mOffset * CeilDiv(n, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                                                    CeilDiv(nOffset, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) {
                    WaitForVector();
                }
                AscendC::Std::tuple<int64_t, int64_t, int64_t> mmSingleShape{Get<MNK_M>(singleShape),
                                                                             Get<MNK_N>(singleShape), k};
                mmadOp_(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)],
                        x1ScaleGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)],
                        x2ScaleGlobal_[Get<IDX_X2SCALE_OFFSET>(blockOffset_)],
                        biasGlobal_[Get<IDX_BIAS_OFFSET>(blockOffset_)], l0cOutUb_, mmSingleShape);
                NotifyVector();
            }
            isVecSetSyncCom_ = true;
            if ASCEND_IS_AIV {
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<MNK_M>(singleShape),
                                                                                      Get<MNK_N>(singleShape), 0, 0};
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    Get<IDX_C_OFFSET>(blockOffset_),       Get<IDX_C_SCALE_OFFSET>(blockOffset_),
                    Get<IDX_X2SCALE_OFFSET>(blockOffset_), Get<IDX_X1SCALE_OFFSET>(blockOffset_),
                    Get<IDX_BIAS_OFFSET>(blockOffset_),
                };
                WaitForCube();
                epilogueOp_(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        } while (bs.GetTileIdx(tileIdx));
    }

    __aicore__ inline void operator()(const Params &params)
    {
        InitParamsAndTensor(params);
        if ASCEND_IS_AIC {
            MmShape mmProblemShape{static_cast<int64_t>(params.gmmParams.m), static_cast<int64_t>(params.gmmParams.n),
                                   static_cast<int64_t>(params.gmmParams.k)};
            MmShape l0Shape{static_cast<int64_t>(params.gmmParams.baseM), static_cast<int64_t>(params.gmmParams.baseN),
                            static_cast<int64_t>(params.gmmParams.baseK)};
            L1Params l1Params{static_cast<uint64_t>(params.gmmParams.kAL1),
                              static_cast<uint64_t>(params.gmmParams.kBL1),
                              static_cast<uint64_t>(params.gmmParams.scaleKAL1), SCALE_L1_BUFFER_NUM};
            mmadOp_.Init(mmProblemShape, l0Shape, l1Params, params.gmmParams.isBias == 1,
                         params.gmmParams.dbL0C == DOUBLE_BUFFER_COUNT);
        }
        epilogueOp_.Init(params.epilogueParams);
        BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        if constexpr (formatB == CubeFormat::NZ) {
            if constexpr (transB) {
                bs.SetTailAlign(1, AscendC::BLOCK_CUBE);
            } else {
                bs.SetTailAlign(1, c0Size);
            }
        }
        l0cOutUb_ = epilogueOp_.GetL0c2UbTensor();
        for (uint32_t groupIdx = 0; groupIdx < params.gmmParams.groupNum; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }
            TupleShape resProblemShape{Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_)};
            bs.UpdateNextProblem(resProblemShape);
            epilogueOp_.UpdateNextProblem(resProblemShape);
            if ASCEND_IS_AIC {
                MmShape mmProblemShape{Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_)};
                mmadOp_.UpdateParamsForNextProblem(mmProblemShape);
            }
            ProcessSingleGroup(params, bs, groupIdx);
        }
        End();
    }
};
} // namespace Kernel
} // namespace Gemm
} // namespace Cgmct
#endif // MATMUL_KERNEL_KERNEL_GMM_ACTIVATION_MXQUANT_H
