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
 * \file grouped_matmul_mx_kernel.h
 */

#pragma once

#if defined(ASCENDC_CPU_DEBUG)
#include "kernel_operator.h"
#elif ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

namespace {
// group list 的 三种类型
constexpr uint64_t GROUP_LIST_TYPE_OFFSET = 0UL; // group list 累计offset
constexpr uint64_t GROUP_LIST_TYPE_LENGTH = 1UL; // group list 存每组 token 数
constexpr uint64_t GROUP_LIST_TYPE_SPARSE = 2UL; // [专家id， token数]， 组与专家可多对一，允许中间 0 token 组
constexpr uint64_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2UL;
constexpr uint64_t SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET = 1UL;
constexpr int64_t BLOCK_CUBE_MASK = BLOCK_CUBE - 1;
constexpr int64_t SCALE_CACHE_MASK = 0xff;

} // namespace

template <class ProblemShape, class BlockMmad, class BlockScheduler, class ScaleAType = fp8_e8m0_t,
          class ScaleBType = fp8_e8m0_t>
class GroupedMatmulMxKernel {
public:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutScaleB = typename BlockMmad::LayoutScaleB;
    using LayoutBias = typename BlockMmad::LayoutBias;

    static_assert((IsFp4<AType>() && IsFp4<BType>()) || (IsFp8<AType>() && IsFp8<BType>()),
                  "GroupedMatmulMx only supports same bit-width MXFP4/MXFP8 AType and BType.");
    static_assert(AscendC::Std::is_one_of_v<CType, half, bfloat16_t, float>,
                  "GroupedMatmulMx only supports half/bfloat16_t/float CType.");
    static_assert(AscendC::Std::is_same_v<BiasType, float>, "GroupedMatmulMx only supports float BiasType.");
    static_assert(AscendC::Std::is_same_v<LayoutA, asc::te::nd_ext_layout_ptn>,
                  "GroupedMatmulMx supports ND LayoutA only (A transpose unsupported).");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn,
                                            asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "GroupedMatmulMx only supports ND/DN/NZ/ZN LayoutB.");
    static_assert(!AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutBias, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "GroupedMatmulMx does not support NZ/ZN LayoutC or LayoutBias.");

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool SCALE_NZ = IsScaleNz<LayoutScaleB>::value;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockShape = typename BlockMmad::BlockShape;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    static constexpr uint32_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::scalea_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = asc::te::frame_layout_format<LayoutScaleB, AscendC::Std::Int<SCALE_C0>>;

    // 表按 grouplist 位置对齐（table[i] 对应 grouplist 第 i 项）
    struct GroupOffsetTables {
        const __gm__ uint64_t *aGroupAddrTable{nullptr};
        const __gm__ uint64_t *scaleAGroupAddrTable{nullptr};
        const __gm__ uint64_t *cGroupAddrTable{nullptr};
        const __gm__ uint64_t *bGroupAddrTable{nullptr};
        const __gm__ uint64_t *scaleBGroupAddrTable{nullptr};
    };

    struct GMMTiling {
        uint32_t groupNum;
        int64_t m;
        int64_t n;
        int64_t k;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1;      // A 一次搬进 L1 的 K 元素数；填 0 兜底为 baseK
        uint32_t kBL1;      // 同上，B 侧
        uint32_t scaleKAL1; // scale K 窗口；填 0 兜底为 baseK（MX 要求 kAL1 为 64 倍数）
        uint8_t isBias;
        uint8_t dbL0C{DOUBLE_BUFFER_COUNT};         // L0C 中的 buffer 个数（1/2）
        uint8_t l1BufferStage{DOUBLE_BUFFER_COUNT}; // L1 缓冲 stage 数（2/3，block 层硬编码消费）
        uint8_t groupListType;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        GM_ADDR groupListGmAddr; // groupList 每个 group 的 token 数
        GMMTiling gmmParams;
        GroupOffsetTables groupOffsetTables; // 各 group 相对基址偏移表
    };

    __aicore__ inline GroupedMatmulMxKernel() {}
    __aicore__ inline ~GroupedMatmulMxKernel() {}

    __aicore__ inline void Init(const Params &params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        tiling_ = params.gmmParams;
        const GMMTiling *gmmParams = &tiling_;
        aBasePtr_ = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
        cBasePtr_ = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
        scaleABasePtr_ = reinterpret_cast<__gm__ ScaleAType *>(params.mmadParams.scaleAGmAddr);
        scaleBBasePtr_ = reinterpret_cast<__gm__ ScaleBType *>(params.mmadParams.scaleBGmAddr);
        if (gmmParams->isBias == 1) {
            biasBasePtr_ = reinterpret_cast<__gm__ float *>(params.mmadParams.biasGmAddr);
            isBias_ = true;
        } else {
            biasBasePtr_ = nullptr;
            isBias_ = false;
        }
        const ProblemShape initProblemShape{gmmParams->m, gmmParams->n, gmmParams->k, 0};
        problemShape_ = initProblemShape;
        groupNum_ = gmmParams->groupNum;
        curBaseM_ = gmmParams->baseM;
        groupListGmAddr_ = params.groupListGmAddr;
        offsetTables_ = params.groupOffsetTables;

        groupState_ = {};
        groupOffsets_ = {};

        if constexpr (!WEIGHT_NZ) {
            groupState_.perGroupB = gmmParams->n * gmmParams->k;
        } else {
            if constexpr (TRANS_B) {
                groupState_.perGroupB = static_cast<int64_t>(
                    Align16(gmmParams->n) * (IsFp4<AType>() ? Align64(gmmParams->k) : Align32(gmmParams->k)));
            } else {
                groupState_.perGroupB = static_cast<int64_t>(
                    (IsFp4<AType>() ? Align64(gmmParams->n) : Align32(gmmParams->n)) * Align16(gmmParams->k));
            }
        }

        const BlockShape l0Shape{static_cast<int64_t>(gmmParams->baseM), static_cast<int64_t>(gmmParams->baseN),
                                 static_cast<int64_t>(gmmParams->baseK), 0};
        const uint64_t kAL1 = gmmParams->kAL1 != 0 ? gmmParams->kAL1 : gmmParams->baseK;
        const uint64_t kBL1 = gmmParams->kBL1 != 0 ? gmmParams->kBL1 : gmmParams->baseK;
        const uint64_t scaleKAL1 = gmmParams->scaleKAL1 != 0 ? gmmParams->scaleKAL1 : gmmParams->baseK;
        const L1Params l1Params{kAL1, kBL1, scaleKAL1};
        const typename BlockMmad::MmadParams mmadParams{
            l0Shape, l1Params, isBias_, gmmParams->dbL0C == DOUBLE_BUFFER_COUNT, gmmParams->l1BufferStage};
        mmadOp_.Init(initProblemShape, mmadParams);
    }

    __aicore__ inline void Run()
    {
        if ASCEND_IS_AIV {
            return;
        }
        if (groupNum_ == 0) {
            return;
        }
        const int64_t groupListSize =
            static_cast<int64_t>(groupNum_) *
            (tiling_.groupListType == GROUP_LIST_TYPE_SPARSE ? SPARSE_GROUP_LIST_ITEM_STRIDE : 1UL);
        const auto groupListLayout = asc::te::make_layout(asc::te::make_shape(groupListSize), asc::te::make_stride(1L));
        const auto gmGroupList = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ int64_t *>(groupListGmAddr_)),
            groupListLayout);
        BlockScheduler scheduler(tiling_.baseM, tiling_.baseN, tiling_.baseK);
        SetSchedulerTailAlign(scheduler); // 设置尾块调度器能切的最小 block 大小
        const uint32_t lastGroupIdx = groupNum_ - 1;
        for (uint32_t groupIdx = 0; groupIdx < lastGroupIdx; ++groupIdx) {
            SetMNK(gmGroupList, groupIdx); // 按当前 group 的 M 重新均衡 baseM
            const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
            const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
            const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
            if (problemM <= 0 || problemN <= 0 || problemK <= 0) {
                continue;
            }
            BaseMBalance(scheduler, problemM, tiling_.baseM); // 按当前 group 的 M 重新均衡 baseM
            scheduler.UpdateNextProblem(
                SchedulerProblemShape{problemM, problemN, problemK, 0}); // 对当前的 groupidx 更新调度器状态
            ProcessSingleGroup<false>(scheduler, gmGroupList, groupIdx);
        }

        SetMNK(gmGroupList, lastGroupIdx);
        const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
        if (problemM > 0 && problemN > 0 && problemK > 0) {
            BaseMBalance(scheduler, problemM, tiling_.baseM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{problemM, problemN, problemK, 0});
            if (IsLastGroupAndNeedSplit(scheduler)) {
                scheduler.UpdateTailTile(); // 尾部均衡
                ProcessSingleGroup<true>(scheduler, gmGroupList, lastGroupIdx);
            } else {
                ProcessSingleGroup<false>(scheduler, gmGroupList, lastGroupIdx);
            }
        }
    }

private:
    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler &scheduler)
    {
        // 调度器尾块切分，能切的最小block的大小
        constexpr uint32_t mTailAlign = 1;
        constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) : static_cast<uint32_t>(C0_SIZE);
        scheduler.SetTailAlign(mTailAlign, nTailAlign);
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2CacheHint(TensorB &gmB, TensorScaleB &gmScaleB, int64_t mSize, int64_t curBaseM,
                                          int64_t baseN)
    {
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
        if constexpr (WEIGHT_NZ) {
            if (curBaseM >= mSize) {
                gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
                gmScaleB.set_l2_cache_hint(asc::te::cache_mode::disable);
            } else {
                gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
                gmScaleB.set_l2_cache_hint(asc::te::cache_mode::normal);
            }
        } else {
            if constexpr (TRANS_B) {
                if (curBaseM >= mSize && (problemK & SCALE_CACHE_MASK) == 0) {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
                    gmScaleB.set_l2_cache_hint(asc::te::cache_mode::disable);
                } else {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
                    gmScaleB.set_l2_cache_hint(asc::te::cache_mode::normal);
                }
            } else {
                if (curBaseM >= mSize && (problemN & SCALE_CACHE_MASK) == 0 && (baseN & SCALE_CACHE_MASK) == 0) {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
                    gmScaleB.set_l2_cache_hint(asc::te::cache_mode::disable);
                } else {
                    gmB.set_l2_cache_hint(asc::te::cache_mode::normal);
                    gmScaleB.set_l2_cache_hint(asc::te::cache_mode::normal);
                }
            }
        }
    }

    __aicore__ inline void BaseMBalance(BlockScheduler &scheduler, int64_t m, int64_t baseM)
    {
        if (m <= 0) {
            return;
        }
        const int64_t safeBaseM = baseM > 0 ? baseM : static_cast<int64_t>(BLOCK_CUBE);
        const int64_t mCnt = (m + safeBaseM - 1) / safeBaseM;
        const int64_t balancedBaseM = (m + mCnt - 1) / mCnt;
        curBaseM_ = static_cast<uint32_t>((balancedBaseM + BLOCK_CUBE_MASK) & ~BLOCK_CUBE_MASK);
        scheduler.UpdateBaseM(curBaseM_);
    }

    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockScheduler &scheduler)
    {
        return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
    }

    template <typename GroupListTensor>
    __aicore__ inline void SetMNK(const GroupListTensor &gmGroupList, uint32_t groupIdx)
    {
        const int64_t splitValue = GetSplitValueFromGroupList(gmGroupList, groupIdx); // 当前 groupIdx 的 tokens 数量
        const int64_t n = asc::te::get<MNK_N>(problemShape_);
        problemShape_ = ProblemShape{splitValue, n, asc::te::get<MNK_K>(problemShape_), 0};
    }

    template <typename GroupListTensor>
    __aicore__ inline int64_t GetSplitValueFromGroupList(const GroupListTensor &gmGroupList, uint32_t groupIdx)
    {
        int64_t splitValue = 0;
        if (tiling_.groupListType == GROUP_LIST_TYPE_OFFSET) {
            const int64_t offset = gmGroupList[groupIdx];
            splitValue = offset - groupState_.preOffset;
            groupState_.preOffset = offset;
        } else if (tiling_.groupListType == GROUP_LIST_TYPE_LENGTH) {
            splitValue = gmGroupList[groupIdx];
            groupState_.preOffset += splitValue;
        } else {
            const uint32_t splitValueIdx =
                groupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE + SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET;
            splitValue = gmGroupList[splitValueIdx];
            groupState_.preOffset += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline int64_t GetScaleK(int64_t k) const
    {
        return ((k + MXFP_DIVISOR_SIZE - 1) >> MXFP_DIVISOR_SHIFT) << MXFP_MULTI_BASE_SHIFT;
    }

    template <typename GroupListTensor>
    __aicore__ inline void UpdateBaseOffsets(const GroupListTensor &gmGroupList, uint32_t groupIdx)
    {
        uint32_t weightIdx = groupIdx;
        if (tiling_.groupListType == GROUP_LIST_TYPE_SPARSE) {
            weightIdx = static_cast<uint32_t>(gmGroupList[groupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE]);
        }

        const int64_t m = asc::te::get<MNK_M>(problemShape_);
        const int64_t n = asc::te::get<MNK_N>(problemShape_);
        const int64_t k = asc::te::get<MNK_K>(problemShape_);
        const int64_t prevSplitOffset = groupState_.preOffset - m;

        if (offsetTables_.aGroupAddrTable != nullptr) {
            groupOffsets_.a = static_cast<int64_t>(offsetTables_.aGroupAddrTable[groupIdx]);
        } else {
            groupOffsets_.a = IsFp4<AType>() ? ((prevSplitOffset * k) >> 1) : (prevSplitOffset * k);
        }

        if (offsetTables_.scaleAGroupAddrTable != nullptr) {
            groupOffsets_.scaleA = static_cast<int64_t>(offsetTables_.scaleAGroupAddrTable[groupIdx]);
        } else {
            groupOffsets_.scaleA = prevSplitOffset * GetScaleK(k);
        }

        if (offsetTables_.cGroupAddrTable != nullptr) {
            groupOffsets_.c = static_cast<int64_t>(offsetTables_.cGroupAddrTable[groupIdx]);
        } else {
            groupOffsets_.c = prevSplitOffset * n;
        }

        if (offsetTables_.bGroupAddrTable != nullptr) {
            groupOffsets_.b = static_cast<int64_t>(offsetTables_.bGroupAddrTable[groupIdx]);
        } else {
            groupOffsets_.b = groupState_.perGroupB * static_cast<int64_t>(weightIdx);
            if constexpr (IsFp4<BType>()) {
                groupOffsets_.b >>= 1;
            }
        }

        if (offsetTables_.scaleBGroupAddrTable != nullptr) {
            groupOffsets_.scaleB = static_cast<int64_t>(offsetTables_.scaleBGroupAddrTable[groupIdx]);
        } else {
            const int64_t scaleK = GetScaleK(k);
            if constexpr (SCALE_NZ) {
                groupOffsets_.scaleB =
                    static_cast<int64_t>(CalScaleNZGmAddrOffset(TRANS_B, static_cast<int64_t>(weightIdx), n, scaleK));
            } else {
                groupOffsets_.scaleB = static_cast<int64_t>(weightIdx) * n * scaleK;
            }
        }
        groupOffsets_.bias = static_cast<int64_t>(weightIdx) * n;
    }

    template <bool isLastGroupAndNeedSplit, typename GroupListTensor>
    __aicore__ inline void ProcessSingleGroup(BlockScheduler &scheduler, const GroupListTensor &gmGroupList,
                                              uint32_t groupIdx)
    {
        BlockCoord blockCoord;
        if (!scheduler.GetNextBlockCoord(blockCoord)) {
            return;
        }
        UpdateBaseOffsets(gmGroupList, groupIdx);

        mmadOp_.UpdateParamsForNextProblem(problemShape_);

        const int64_t problemM = asc::te::get<MNK_M>(problemShape_);
        const int64_t problemN = asc::te::get<MNK_N>(problemShape_);
        const int64_t problemK = asc::te::get<MNK_K>(problemShape_);
        const int64_t scaleK = GetScaleK(problemK);
        const int64_t baseN = static_cast<int64_t>(tiling_.baseN);
        auto layoutA = MakeLayoutA{}(problemM, problemK);
        auto layoutScaleA = MakeLayoutScaleA{}(problemM, scaleK);
        auto layoutB = MakeLayoutB{}(problemK, problemN);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleK, problemN);
        auto layoutC = MakeLayoutC{}(problemM, problemN);
        auto gmA =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aBasePtr_ + groupOffsets_.a), layoutA);
        auto gmScaleA = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(scaleABasePtr_ + groupOffsets_.scaleA), layoutScaleA);
        auto gmB =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bBasePtr_ + groupOffsets_.b), layoutB);
        auto gmScaleB = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(scaleBBasePtr_ + groupOffsets_.scaleB), layoutScaleB);
        auto gmC =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cBasePtr_ + groupOffsets_.c), layoutC);
        auto biasPtr = isBias_ ? (biasBasePtr_ + groupOffsets_.bias) : nullptr;
        auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(1), problemN);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasPtr), layoutBias);

        if constexpr (!isLastGroupAndNeedSplit) {
            SetL2CacheHint(gmB, gmScaleB, problemM, static_cast<int64_t>(curBaseM_),
                           static_cast<int64_t>(tiling_.baseN));
        }

        do {
            const auto schedulerBlockShape = scheduler.GetBlockShape(blockCoord);
            const int64_t blockM = asc::te::get<MNK_M>(schedulerBlockShape);
            const int64_t blockN = asc::te::get<MNK_N>(schedulerBlockShape);
            if (blockM <= 0 || blockN <= 0) {
                continue;
            }
            const int64_t mSplitOffset = asc::te::get<MNK_K>(schedulerBlockShape); // 尾块切分时，被切块内部的偏移
            const int64_t nSplitOffset = asc::te::get<MNK_B>(schedulerBlockShape);
            const int64_t mBlockIdx = asc::te::get<MNK_M>(blockCoord);
            const int64_t nBlockIdx = asc::te::get<MNK_N>(blockCoord);
            const int64_t blockK = problemK;
            BlockShape blockShape{blockM, blockN, blockK, 0};
            const int64_t mPos = mBlockIdx * curBaseM_ + mSplitOffset;
            const int64_t nPos = nBlockIdx * baseN + nSplitOffset;

            auto gmBlockA =
                gmA.slice(asc::te::make_coord(mPos, static_cast<int64_t>(0)), asc::te::make_shape(blockM, blockK));
            auto gmBlockScaleA =
                gmScaleA.slice(asc::te::make_coord(mPos, static_cast<int64_t>(0)), asc::te::make_shape(blockM, scaleK));
            auto gmBlockB =
                gmB.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos), asc::te::make_shape(blockK, blockN));
            auto gmBlockScaleB =
                gmScaleB.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos), asc::te::make_shape(scaleK, blockN));
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(static_cast<int64_t>(0), nPos),
                                            asc::te::make_shape(static_cast<int64_t>(1), blockN));
            auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos), asc::te::make_shape(blockM, blockN));
            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, blockShape);
        } while (scheduler.GetNextBlockCoord(blockCoord));
    }

private:
    BlockMmad mmadOp_;
    ProblemShape problemShape_{};
    __gm__ AType *aBasePtr_{nullptr};
    __gm__ BType *bBasePtr_{nullptr};
    __gm__ CType *cBasePtr_{nullptr};
    __gm__ float *biasBasePtr_{nullptr};
    __gm__ ScaleAType *scaleABasePtr_{nullptr};
    __gm__ ScaleBType *scaleBBasePtr_{nullptr};
    GM_ADDR groupListGmAddr_{nullptr};
    GroupOffsetTables offsetTables_{};
    GMMTiling tiling_{};

    struct GroupState {
        int64_t preOffset{0}; // 已解析组的累计 token 总数（含当前组）
        int64_t perGroupB{0}; // 一组权重的步长（组间间隔）
    };
    struct GroupOffsets {
        int64_t a{0};
        int64_t scaleA{0};
        int64_t b{0};
        int64_t scaleB{0};
        int64_t c{0};
        int64_t bias{0};
    };

    GroupState groupState_{};
    GroupOffsets groupOffsets_{};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    bool isBias_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
