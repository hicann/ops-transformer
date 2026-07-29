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
 * \file mega_moe_impl.h
 * \brief
 */

#ifndef MEGA_MOE_IMPL_H
#define MEGA_MOE_IMPL_H
#include "kernel_operator.h"

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_base.h"

#include "tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_swizzle.h"
#include "blaze/gemm/block/block_mmad_mx_fp8fp4.h"
#include "blaze/prologue/block_prologue_mx_fp8fp4.h"

#include "mega_moe_impl_base.h"
#include "mega_moe_combine_send.h"

namespace MegaMoeImpl {
using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>; // 3: SwizzleOffset

struct GroupSyncSlotLayout {
    uint32_t baseSlotCountPerGroup; // 每个 group 固定分配的 slot 数。
    uint32_t extraSlotGroupCount;   // 前 extraSlotGroupCount 个 group 各多分配一个 slot。
};

/*
 * group 少于逻辑核时，将逻辑核 slot 尽量均匀地分给各 group；
 * group 多于逻辑核时，每个 group 获得一个独立 slot。
 */
__aicore__ inline GroupSyncSlotLayout CalcGroupSyncSlotLayout(uint32_t expertTokenCount, uint32_t logicalCoreCount)
{
    uint32_t groupCount = Ops::Base::CeilDiv(expertTokenCount, COMBINE_TOKEN_GROUP_SIZE);
    uint32_t totalSyncSlotCount = groupCount > logicalCoreCount ? groupCount : logicalCoreCount;
    return {totalSyncSlotCount / groupCount, totalSyncSlotCount % groupCount};
}

// 每个同步 slot 独占 INT_CACHELINE 个 int32_t（64B），避免不同 flag 共享 cacheline 产生读写竞争。
__aicore__ inline __gm__ int32_t *GetCombineSyncCounterAddress(__gm__ int32_t *expertCounterBase,
                                                               uint32_t localSyncSlotIndex)
{
    return expertCounterBase + static_cast<uint64_t>(localSyncSlotIndex) * INT_CACHELINE;
}

// =================================================================================================
// ComputeCoreGrouping：计算当前 core 所属的 group 及其在 group 内的位置
// =================================================================================================
// 将 totalCores 个 core 均匀分配到 numGroups 个 group 中，余数分配给前 remainder 个 group。
__aicore__ inline void ComputeCoreGrouping(uint32_t coreId, uint32_t numGroups, uint32_t totalCores, uint32_t &myGroup,
                                           uint32_t &myIdxInGrp, uint32_t &myGrpSize)
{
    uint32_t baseSize = totalCores / numGroups;     // 每个 group 的基础 core 数
    uint32_t remainder = totalCores % numGroups;    // 余数，前 remainder 个 group 多分配 1 个 core
    uint32_t boundary = remainder * (baseSize + 1); // 前 remainder 个 group 占用的 core 总数

    // 判断当前 core 是否在前 remainder 个 group 中（这些 group 有 baseSize+1 个 core）
    if (coreId < boundary) {
        myGroup = coreId / (baseSize + 1);    // 所属 group 索引
        myIdxInGrp = coreId % (baseSize + 1); // 在 group 内的索引
        myGrpSize = baseSize + 1;             // 当前 group 的 core 数
    } else {
        // 当前 core 在后面的 group 中（这些 group 只有 baseSize 个 core）
        uint32_t adjusted = coreId - boundary;     // 减去前 remainder 个 group 占用的 core 数
        myGroup = remainder + adjusted / baseSize; // 所属 group 索引 = remainder + 偏移
        myIdxInGrp = adjusted % baseSize;          // 在 group 内的索引
        myGrpSize = baseSize;                      // 当前 group 的 core 数
    }
}

/*
 * 获取一个 token group 对应的连续同步 slot 区间。每组的基础 slot 数和余数分配
 * 在 expert 的 tile loop 外计算，这里只计算当前 group 的起点和长度。
 */
__aicore__ inline void GetGroupSyncSlotRange(uint32_t groupIndex, const GroupSyncSlotLayout &slotLayout,
                                             uint32_t &firstSyncSlot, uint32_t &syncSlotCount)
{
    syncSlotCount = slotLayout.baseSlotCountPerGroup + (groupIndex < slotLayout.extraSlotGroupCount ? 1U : 0U);

    // 已被分配的基础槽位 + 已被分配的额外槽位 = 前面 group 的总槽位数
    uint32_t precedingExtraSlotCount =
        groupIndex < slotLayout.extraSlotGroupCount ? groupIndex : slotLayout.extraSlotGroupCount;
    firstSyncSlot = groupIndex * slotLayout.baseSlotCountPerGroup + precedingExtraSlotCount;
}

/*
 * 计算当前 logical core 负责的 group 序列。group 少时多核协作一个 group，一个 group 对应多个 slot；
 * group 多时，一核处理多个 group，一个 group 对应一个 slot。core c 处理 c、c + logicalCoreCount、... 对应的 group。
 */
__aicore__ inline void ComputeCombineGroupsForCore(uint32_t logicalCoreId, uint32_t groupCount,
                                                   uint32_t logicalCoreCount, uint32_t &firstGroupIndex,
                                                   uint32_t &groupStride, uint32_t &coreIndexWithinGroup,
                                                   uint32_t &coresAssignedToGroup)
{
    // group 少时 slot 数等于 core 数；group 多时 slot 数等于 group 数。
    uint32_t totalSyncSlotCount = groupCount > logicalCoreCount ? groupCount : logicalCoreCount;
    // 每个 group 至少获得 baseSlotCountPerGroup 个 slot，前 extraSlotGroupCount 个 group 多一个。
    uint32_t baseSlotCountPerGroup = totalSyncSlotCount / groupCount;
    uint32_t extraSlotGroupCount = totalSyncSlotCount % groupCount;
    // 前 extraSlotGroupCount 个 group 占用的 slot 结束位置。
    uint32_t slotBoundaryAfterExtraSlotGroups = extraSlotGroupCount * (baseSlotCountPerGroup + 1U);

    // logicalCoreId 对应该核负责的第一个 slot，用它反算首个 group 及 group 内编号。
    if (logicalCoreId < slotBoundaryAfterExtraSlotGroups) {
        coresAssignedToGroup = baseSlotCountPerGroup + 1U;
        firstGroupIndex = logicalCoreId / coresAssignedToGroup;
        coreIndexWithinGroup = logicalCoreId % coresAssignedToGroup;
    } else {
        uint32_t slotOffsetAfterExtraSlotGroups = logicalCoreId - slotBoundaryAfterExtraSlotGroups;
        coresAssignedToGroup = baseSlotCountPerGroup;
        firstGroupIndex = extraSlotGroupCount + slotOffsetAfterExtraSlotGroups / coresAssignedToGroup;
        coreIndexWithinGroup = slotOffsetAfterExtraSlotGroups % coresAssignedToGroup;
    }

    // group 少时只处理一个 group；group 多时按 logicalCoreCount 为步长轮转。
    groupStride = groupCount < logicalCoreCount ? groupCount : logicalCoreCount;
}

// 根据当前 token group 的 slot range 通知所有 Combine 消费核。
__aicore__ inline void NotifyCombineConsumersOfTileCompletion(uint32_t rowTileOffset,
                                                              const GroupSyncSlotLayout &slotLayout,
                                                              __gm__ int32_t *expertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    uint32_t firstSyncSlot = 0;
    uint32_t syncSlotCount = 0;
    // 找到负责消费当前 token group 的 logical-core slot 区间，逐 slot 通知该 GMM2 tile 已写回。
    GetGroupSyncSlotRange(tokenGroupIndex, slotLayout, firstSyncSlot, syncSlotCount);

    for (uint32_t syncSlot = firstSyncSlot; syncSlot < firstSyncSlot + syncSlotCount; ++syncSlot) {
        AscendC::AtomicAdd(GetCombineSyncCounterAddress(expertCounterBase, syncSlot), int32_t(1));
    }
}

// 共享专家专用: 每个 token group 独立 slot, 直接 AtomicAdd 通知对应 group 的 tile 完成
__aicore__ inline void NotifySharedExpertTileCompletion(uint32_t rowTileOffset,
                                                        __gm__ int32_t *sharedExpertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    AscendC::AtomicAdd(GetCombineSyncCounterAddress(sharedExpertCounterBase, tokenGroupIndex), int32_t(1));
}

// =================================================================================================
// WaitForUpstreamReady：等待上游 GMM 计算完成，GMM1/GMM2 分流（A8W8/A4W4 和 A8W4 共用）
// =================================================================================================
template <typename Policy, bool IsShared, typename Config>
__aicore__ inline void WaitForUpstreamReady(const GMMAddrInfo &gmmAddrInfo, const Config &config, uint32_t mLoc)
{
    if constexpr (IsShared) {
        return;
    }
    if constexpr (Policy::IS_GMM1) {
        uint32_t waveIdx = mLoc / config.tileM;
        uint32_t targetValue = (mLoc + config.tileM > config.m) ? (config.m - mLoc) : config.tileM;
        __gm__ int32_t *flagValueAddr = gmmAddrInfo.dispatchToGmm1Flag + waveIdx;
        while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    } else {
        BlockScheduler gmmBlockScheduler({config.m, config.k, config.n},
                                         BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(config.swigluTileM),
                                                                              static_cast<int64_t>(L1_TILE_N))});
        uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
        __gm__ int32_t *flagValueAddr = gmmAddrInfo.swigluToGmm2Flag;
        while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    }
}

// ==================================================================================
// 统一配置结构体 — 通过 IsA8W4 模板参数区分 A8W8/A4W4 和 A8W4 两条路径的配置
// ==================================================================================
namespace Detail {
struct Gmm1Policy {
    static constexpr bool IS_GMM1 = true;
};

struct Gmm2Policy {
    static constexpr bool IS_GMM1 = false;
};

// BlockMmadSelector — 通过偏特化处理 A8W8/A4W4 和 A8W4 的 BlockMmad 签名差异
template <bool IsA8W4, typename C>
struct BlockMmadSelector;

template <typename C>
struct BlockMmadSelector<false, C> {
    using type =
        Blaze::Gemm::Block::BlockMmad<typename C::DispatchPolicy, typename C::ElementAType, typename C::LayoutA,
                                      typename C::ElementBType, typename C::LayoutB, typename C::ElementCType,
                                      typename C::LayoutC, typename C::BiasType, typename C::LayoutBias>;
};

template <typename C>
struct BlockMmadSelector<true, C> {
    using type =
        Blaze::Gemm::Block::BlockMmad<typename C::DispatchPolicy,
                                      AscendC::Std::tuple<typename C::ElementAType, typename C::ElementMxScaleAType>,
                                      AscendC::Std::tuple<typename C::MakeLayoutA, typename C::MakeLayoutScaleA>,
                                      AscendC::Std::tuple<typename C::ElementBType, typename C::ElementMxScaleBType>,
                                      AscendC::Std::tuple<typename C::MakeLayoutB, typename C::MakeLayoutScaleB>,
                                      typename C::ElementCType, typename C::MakeLayoutC, void, void>;
};

// ==================================================================================
// 统一 Config — 通过 IsA8W4 模板参数区分 A8W8/A4W4 和 A8W4
// 含公共与差异类型别名，BlockMmad 通过 trait 选择
// ==================================================================================
template <bool IsA8W4, typename Policy, uint8_t CombineQuantMode, typename ElementA, typename ElementB,
          typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsLayered = false>
struct Config {
    static constexpr bool IS_SHARED = IsShared;
    using ElementAType = ElementA;
    using ElementBType = ElementB;
    using ElementCType = ElementC;
    using ElementMxScaleAType = ElementMxScaleA;
    using ElementMxScaleBType = ElementMxScaleB;

    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;

    static constexpr uint32_t C0_SIZE_B = IsA8W4 ? 32U : AuxGetC0Size<ElementB>();

    static constexpr bool TOPK_WEIGHTS_PREFETCH = TopkWeightsPrefetch;

    using LayoutA = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;
    using LayoutScaleA = Te::ScaleANDLayoutPtn;
    using LayoutScaleB = Te::ScaleBDNLayoutPtn;

    using BiasType = float;
    using LayoutBias = Te::NDExtLayoutPtn;
    using DispatchPolicy =
        Std::conditional_t<IsA8W4, Blaze::Gemm::MatmulMxFp8Fp4DynamicKL1TailResplit, Blaze::Gemm::MatmulWithScaleMx<>>;
    using LayoutB = Std::conditional_t<IsA8W4, Te::ZNLayoutPtn,
                                       Std::conditional_t<IsWeightNZ, Te::ZNLayoutPtn, Te::DNExtLayoutPtn>>;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE_A>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleA = Te::FrameLayoutFormat<LayoutScaleA, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutScaleB = Te::FrameLayoutFormat<LayoutScaleB, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE_C>>;

    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutAType = decltype(MakeLayoutA{}(uint32_t{}, uint32_t{}));
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t{}, uint32_t{}));
    using LayoutScaleAType = decltype(MakeLayoutScaleA{}(uint32_t{}, uint32_t{}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t{}, uint32_t{}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t{}, uint32_t{}));
    using LayoutBiasType = decltype(Te::MakeFrameLayout<LayoutBias>(uint32_t{}, uint32_t{}));

    using BlockMmad = typename BlockMmadSelector<IsA8W4, Config>::type;

    // BlockPrologue（仅 A8W4 使用；A8W8/A4W4 路径用 void 占位）
    using BlockPrologue =
        Std::conditional_t<IsA8W4, Blaze::Gemm::Prologue::BlockPrologue<DispatchPolicy, ElementA, ElementB>, void>;

    struct ProblemConfig {
        static __aicore__ inline typename BlockMmad::L1Params DefaultL1Params()
        {
            if constexpr (IsA8W4) {
                return typename BlockMmad::L1Params{.kL1 = L1_TILE_K, .scaleKL1 = 4096};
            } else {
                return typename BlockMmad::L1Params{
                    .kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE, .l1BufNum = 2};
            }
        }

        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        uint32_t outputN = 0;
        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t scaleK = 0;
        uint32_t tileM = 0;                   // A8W8/A4W4 路径用
        uint32_t swigluTileM = L1_TILE_M_256; // GMM1+SwiGLU 的 tileM，供 GMM2 WaitForUpstreamReady 使用
        typename BlockMmad::L1Params l1Params = DefaultL1Params();
    };

    struct LayoutBundle {
        LayoutAType a;
        LayoutBType b;
        LayoutScaleAType scaleA;
        LayoutScaleBType scaleB;
        LayoutCType c;
        LayoutBiasType bias; // A8W8/A4W4 路径用，A8W4 不使用
    };

    __aicore__ static inline ProblemConfig BuildProblemConfig(const ProblemShape &problemShape)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        if constexpr (Policy::IS_GMM1) {
            config.n = Get<N_VALUE>(problemShape);
            config.k = Get<K_VALUE>(problemShape);
        } else {
            config.n = Get<K_VALUE>(problemShape);
            config.k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;
        }
        config.outputN = Policy::IS_GMM1 ? config.n / SWIGLU_N_HALF : config.n;
        config.blockNum = GetBlockNum();
        config.blockIdx = GetBlockIdx() / GetTaskRation();
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        if constexpr (Policy::IS_GMM1) {
            config.tileM = Gmm1TileM;
        } else {
            if constexpr (IsA8W4 || IsLayered) {
                config.tileM = L1_TILE_M_256;
            } else {
                config.tileM = (CombineQuantMode == COMBINE_NO_QUANT && !IsShared) ? L1_TILE_M_128 : L1_TILE_M_256;
            }
        }
        // prefetch 路径 epilogue 按 128 子块逐个 AtomicAdd(swigluToGmm2Flag)，
        // GMM2 的 targetLoops 须按 128 基准计算才能与 epilogue 递增数一致（含尾块）。
        config.swigluTileM = TopkWeightsPrefetch ? L1_TILE_M_128 : Gmm1TileM;
        return config;
    }

    __aicore__ static inline LayoutBundle BuildLayouts(const ProblemConfig &config)
    {
        LayoutBundle layouts;
        layouts.a = MakeLayoutA{}(config.m, config.k);
        layouts.b = MakeLayoutB{}(config.k, config.n);
        layouts.scaleA = MakeLayoutScaleA{}(config.m, config.scaleK);
        layouts.scaleB = MakeLayoutScaleB{}(config.scaleK, config.n);
        if constexpr (IsA8W4) {
            layouts.c = MakeLayoutC{}(config.m, config.n);
        } else {
            layouts.bias = Te::MakeFrameLayout<LayoutBias>(1U, config.n);
            if constexpr (Policy::IS_GMM1) {
                if constexpr (TopkWeightsPrefetch) {
                    layouts.c = MakeLayoutC{}(config.m, config.n);
                } else {
                    layouts.c = MakeLayoutC{}(Gmm1TileM, L1_TILE_N);
                }
            } else {
                if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
                    layouts.c = MakeLayoutC{}(L1_TILE_M_128, L1_TILE_N);
                } else {
                    layouts.c = MakeLayoutC{}(config.m, config.n);
                }
            }
        }
        return layouts;
    }
};

template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename ElementC, bool IsShared,
          bool TopkWeightsPrefetch, bool IsLayered = false, typename WorkSet, typename ExtraArgs>
__aicore__ inline void AicComputeGeneric(BlockMmad &blockMmad, WorkSet &workSet, uint32_t startLoopIdx,
                                         uint32_t tileNum, ExtraArgs &args)
{
    const auto &config = workSet.config;
    uint32_t ubBufSize;
    if constexpr (Policy::IS_GMM1) {
        ubBufSize = (config.tileM == L1_TILE_M_128) ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
    } else {
        ubBufSize =
            (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) ? MAX_SINGLE_MN_ALIGN32_NUM_128 : 0;
    }
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), workSet.layouts.c);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), workSet.layouts.c);

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr (!Policy::IS_GMM1 && (CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
        uint32_t logicalCoreCount = config.blockNum * 2U;
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(config.m, logicalCoreCount);
    }

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        // Slice 只算数据视图不触发搬运)，前移到 sync/wait 之前统一组织：
        // 先定位本 tile 的数据视图 → 再等上游就绪 → 再算。
        auto gmBlockA = workSet.gmA.Slice(Te::MakeCoord(mLoc, kLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        auto gmBlockScaleA = workSet.gmScaleA.Slice(
            Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
            Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

        if constexpr (Policy::IS_GMM1) {
            uint32_t waveIdx = mLoc / config.tileM;
            if (waveIdx != lastWaveWaited) {
                WaitForUpstreamReady<Policy, IsShared>(workSet.gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
            if constexpr (!TopkWeightsPrefetch) {
                if (args.vecSetSyncCom) {
                    WaitForVector();
                }
            }
        } else {
            if (loopIdx == startLoopIdx) {
                WaitForUpstreamReady<Policy, IsShared>(workSet.gmmAddrInfo, config, mLoc);
            }
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsLayered) {
                if (args.vecSetSyncCom2 >= 2) {
                    WaitForVector(args.pingpongIdx);
                }
            }
        }

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                   Get<K_VALUE>(actualShape), 0};

        if constexpr (Policy::IS_GMM1) {
            if constexpr (TopkWeightsPrefetch) {
                // prefetch：GMM1 写 GM，参考 A8W4 路径
                auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                              reinterpret_cast<__gm__ ElementC *>(workSet.gmmAddrInfo.gmm1OutGlobal)),
                                          workSet.layouts.c);
                for (uint32_t weightBlock = 0; weightBlock < MegaMoeImpl::SWIGLU_N_HALF; ++weightBlock) {
                    auto nOffset = nLoc + weightBlock * config.outputN;
                    auto gmBlockB =
                        workSet.gmB.Slice(Te::MakeCoord(kLoc, nOffset),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                    auto gmBlockScaleB =
                        workSet.gmScaleB.Slice(Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nOffset),
                                               Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM),
                                                             Get<N_VALUE>(actualShape)));
                    auto tensorBlockGm = gmC.Slice(Te::MakeCoord(mLoc, nOffset),
                                                   Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                    blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, tensorBlockGm,
                              singleShape);
                }
                // prefetch 软同步：写 tile 状态位通知 AIV0（roundTag = expertIdx + 1）
                // FIX_S 同步确保 blockMmad 的 L0C→GM Fixpipe 传输完成后才写状态位
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                __gm__ int32_t *statusAddr =
                    reinterpret_cast<__gm__ int32_t *>(workSet.params.workspaceInfo.gmm1TileStatusPtr) +
                    static_cast<int64_t>(args.expertIdx) * workSet.params.tilingData->maxTilesPerExpert + loopIdx;
                AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(args.expertIdx + 1));
            } else {
                // 原始 UB ping-pong 路径
                auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
                    Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
                    Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                for (uint32_t weightBlock = 0; weightBlock < MegaMoeImpl::SWIGLU_N_HALF; ++weightBlock) {
                    auto gmBlockB =
                        workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc + weightBlock * config.outputN),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                    auto gmBlockScaleB = workSet.gmScaleB.Slice(
                        Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * config.outputN),
                        Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM),
                                      Get<N_VALUE>(actualShape)));
                    blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias,
                              weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
                }
                NotifyVector();
                args.vecSetSyncCom = 1;
            }
        } else {
            auto gmBlockB = workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                              Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = workSet.gmScaleB.Slice(
                Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
                auto tensorUb = args.pingpongIdx == 0 ? l0cOutUbFirst : l0cOutUbSecond;
                auto tensorBlockUb = tensorUb.Slice(
                    Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, tensorBlockUb, singleShape);
                NotifyVector(args.pingpongIdx);
                args.vecSetSyncCom2++;
                args.pingpongIdx = 1 - args.pingpongIdx;
            } else {
                auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                              reinterpret_cast<__gm__ ElementC *>(workSet.gmmAddrInfo.gmm2OutGlobal)),
                                          workSet.layouts.c);
                auto gmBlockC = gmC.Slice(Te::MakeCoord(mLoc, nLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
                if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
                    NotifyCombineConsumersOfTileCompletion(mLoc, groupSyncSlotLayout,
                                                           workSet.gmmAddrInfo.gmm2CombineSyncCounter);
                } else if constexpr (IsShared) {
                    NotifySharedExpertTileCompletion(mLoc, workSet.gmmAddrInfo.sharedExpertGmm2TileCounter);
                }
            }
        }
    }
}

template <typename MakeLayoutC, typename ElementC, bool TopkWeightsPrefetch, typename WorkSet, typename ExtraArgs>
__aicore__ inline void AivGmm1PostGeneric(WorkSet &workSet, ExtraArgs &args, uint32_t startLoopIdx, uint32_t tileNum)
{
    const auto &config = workSet.config;
    constexpr uint32_t SUB_TILE_M = MegaMoeImpl::L1_TILE_M_128;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t mLen = Get<M_VALUE>(actualShape);

        if constexpr (TopkWeightsPrefetch) {
            auto &swigluOp = args.swigluQuantOp;

            // prefetch 软同步：轮询 tile 状态位（roundTag = expertIdx + 1）
            // 必须在 DataCopy(weightUb_) 之前：statusAddr 由 AIC 在写完 GMM1 输出后设置，
            // 而该 status 的前置条件是 AIV1 已完成 metaInfo GM 写入（MTE3+S 流水排空）。
            // 轮询通过后 metaInfo GM 中的权重数据才确定可见。
            __gm__ int32_t *statusAddr =
                reinterpret_cast<__gm__ int32_t *>(workSet.params.workspaceInfo.gmm1TileStatusPtr) +
                static_cast<int64_t>(args.expertIdx) * workSet.params.tilingData->maxTilesPerExpert + loopIdx;
            int32_t roundTag = static_cast<int32_t>(args.expertIdx + 1);
            while (AscendC::ReadGmByPassDCache(statusAddr) != roundTag) {
                int64_t st = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - st < 100) {
                }
            }

            // 轮询通过后 metaInfo GM 已就绪，搬第一个 128 子块的权重到 weightUb_
            uint64_t metaInfoMLoc0 = args.expertBeforeCnt + mLoc;
            AscendC::DataCopy(swigluOp.weightUb_, swigluOp.metaInfoGm_[metaInfoMLoc0 * INT32_PER_256B],
                              static_cast<uint32_t>(mLen < SUB_TILE_M ? mLen : SUB_TILE_M) * INT32_PER_256B);

            auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                          reinterpret_cast<__gm__ ElementC *>(workSet.gmmAddrInfo.gmm1OutGlobal)),
                                      workSet.layouts.c);

            auto layoutL0cUB = MakeLayoutC{}(SUB_TILE_M, L1_TILE_N);
            int64_t ubOffsetFirst = 0;
            int64_t ubOffsetSecond = static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM_128) * sizeof(ElementC);
            auto tensorBlockUbFirst =
                Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUB);
            auto tensorBlockUbSecond =
                Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUB);
            auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});

            // V_MTE2 ping-pong：V 读完 UB 后通知 MTE2 可覆盖。
            // epilogue 内部 V(782) 读 UB → V(802) Wait<MTE3_V> → MTE3 写 GM → V 继续。
            // Set<V_MTE2> 排在 V(802) 之后，MTE2 只等 V 不等 MTE3_S/AtomicAdd，省 MTE3+S 延迟。
            // 严格配对：prime Set(循环前) → 每轮 Wait(循环头)+Set(epilogue后) → 末轮 Set 由循环后 Wait 清理。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            for (uint32_t subOff = 0; subOff < mLen; subOff += SUB_TILE_M) {
                uint32_t subM = (mLen - subOff < SUB_TILE_M) ? (mLen - subOff) : SUB_TILE_M;
                uint32_t subMLoc = mLoc + subOff;
                uint64_t metaInfoMLoc = args.expertBeforeCnt + subMLoc;

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);

                // 搬入当前 128 子块的权重（首个子块已在循环前预取，跳过）
                // WaitFlag<V_MTE2> 保证 V 已读完上一轮的 weightUb_，此处覆写安全
                if (subOff != 0) {
                    AscendC::DataCopy(swigluOp.weightUb_, swigluOp.metaInfoGm_[metaInfoMLoc * INT32_PER_256B],
                                      subM * INT32_PER_256B);
                }

                auto tensorBlockGmFirst =
                    gmC.Slice(Te::MakeCoord(subMLoc, nLoc), Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
                auto tensorBlockGmSecond = gmC.Slice(Te::MakeCoord(subMLoc, nLoc + config.outputN),
                                                     Te::MakeShape(subM, Get<N_VALUE>(actualShape)));

                AscendC::Te::Copy(copyGM2UB, tensorBlockUbFirst, tensorBlockGmFirst);
                AscendC::Te::Copy(copyGM2UB, tensorBlockUbSecond, tensorBlockGmSecond);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); // 权重+GMM1 都搬完才通知 V
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

                Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, Get<N_VALUE>(actualShape), 0, 0};
                Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    subMLoc * config.outputN + nLoc,
                    subMLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                        CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                    0,
                    0,
                    static_cast<int64_t>(metaInfoMLoc),
                    0};
                AscendC::SetCtrlSpr<60, 60>(0);
                args.swigluQuantOp(epilogueShape, epilogueOffset);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            // prefetch 路径无需 NotifyCube（单向软同步，AIC 不等 AIV0）
        } else {
            // 原始 256×256 一次处理路径
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape),
                                                                         Get<N_VALUE>(actualShape), 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                mLoc * config.outputN + nLoc,
                mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0,
                0,
                0,
                0};
            WaitForCube();
            AscendC::SetCtrlSpr<60, 60>(0);
            args.swigluQuantOp(epilogueShape, epilogueOffset);
            NotifyCube();
        }
    }
}

template <typename ElementC, typename WorkSet, typename ExtraArgs>
__aicore__ inline void AivGmm2PostGeneric(WorkSet &workSet, ExtraArgs &args, uint32_t startLoopIdx, uint32_t tileNum)
{
    constexpr uint32_t ubBufSize = MAX_SINGLE_MN_ALIGN32_NUM_128;
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    LocalTensor<ElementC> l0cOutUbGMM2First(TPosition::VECIN, ubOffsetFirst, L1_TILE_M_128 * L1_TILE_N);
    LocalTensor<ElementC> l0cOutUbGMM2Second(TPosition::VECIN, ubOffsetSecond, L1_TILE_M_128 * L1_TILE_N);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += workSet.config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        auto l0cOutUbGMM2 = args.pingpongIdx == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;
        WaitForCube(args.pingpongIdx);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::GlobalTensor<int32_t> metaInfoGm;
        int32_t lenTile = Get<M_VALUE>(actualShape);
        LocalTensor<int32_t> metaInfoTensor =
            LocalTensor<int32_t>(TPosition::VECCALC, META_INFO_TENSOR_ADDR, lenTile * META_INFO_SIZE);
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            workSet.params.workspaceInfo.metaInfoPtr + (args.groupCnt + mLoc) * META_INFO_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, lenTile * META_INFO_SIZE);
        MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(mLoc, nLoc, workSet.config.n, metaInfoTensor,
                                                                           l0cOutUbGMM2, actualShape, workSet.params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        NotifyCube(args.pingpongIdx);
        args.pingpongIdx = 1 - args.pingpongIdx;
    }
}

template <typename SwigluQuantOp>
struct Gmm1ArgsGeneric {
    SwigluQuantOp &swigluQuantOp;
    int32_t &vecSetSyncCom;
    uint32_t expertBeforeCnt{0};
    uint32_t expertIdx{0}; // prefetch 软同步：当前 expert 索引，roundTag = expertIdx + 1
};

struct Gmm2ArgsGeneric {
    int32_t &vecSetSyncCom2;
    uint32_t groupCnt;
    uint16_t &pingpongIdx;
};

template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename Config, typename LayoutBundle>
struct WorkSetGeneric {
    Scheduler &scheduler;
    TensorA &gmA;
    TensorB &gmB;
    TensorScaleA &gmScaleA;
    TensorScaleB &gmScaleB;
    TensorBias &gmBias;
    const GMMAddrInfo &gmmAddrInfo;
    const Params &params;
    const Config &config;
    const LayoutBundle &layouts;
};

template <typename Policy, uint8_t CombineQuantMode, typename BlockMmad, typename ElementC, typename MakeLayoutC,
          bool TopkWeightsPrefetch, bool IsLayered = false, bool IsShared, typename WorkSet, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecGeneric(WorkSet &workSet, uint32_t startLoopIdx, uint32_t tileNum,
                                              ExtraArgs &args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad;
        bool enableL0CPingPong = false;
        typename BlockMmad::BlockShape l0TileShape{workSet.config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{workSet.config.m, workSet.config.n, workSet.config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, workSet.config.l1Params, false, enableL0CPingPong);

        AicComputeGeneric<CombineQuantMode, Policy, BlockMmad, ElementC, IsShared, TopkWeightsPrefetch, IsLayered>(
            blockMmad, workSet, startLoopIdx, tileNum, args);
    } else {
        if constexpr (Policy::IS_GMM1) {
            AivGmm1PostGeneric<MakeLayoutC, ElementC, TopkWeightsPrefetch>(workSet, args, startLoopIdx, tileNum);
        } else if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
            AivGmm2PostGeneric<ElementC>(workSet, args, startLoopIdx, tileNum);
        }
    }
}

template <typename Policy, uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared, typename ExtraArgs>
__aicore__ inline void GroupMatmulImplGeneric(const Params &params,
                                              const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                              const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, ExtraArgs &args)
{
    using Config = Config<false, Policy, CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA,
                          ElementMxScaleB, IsWeightNZ, Gmm1TileM, TopkWeightsPrefetch, IsShared, IsLayered>;
    auto config = Config::BuildProblemConfig(problemShape);

    BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;

    auto layouts = Config::BuildLayouts(config);

    if constexpr (Policy::IS_GMM1) {
        if (GetSubBlockIdx() != 0) {
            startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
            return;
        }
        args.swigluQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    } else if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
        if (GetSubBlockIdx() != 0)
            return;
    }
    // GMM2 量化模式：两分支均不匹配，直接往下执行

    using BlockMmad = typename Config::BlockMmad;
    using BiasType = typename Config::BiasType;

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);

    using WorkSetType = WorkSetGeneric<BlockScheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                       decltype(gmScaleB), decltype(gmBias), decltype(config), decltype(layouts)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmmAddrInfo, params, config, layouts};

    using MakeLayoutC = typename Config::MakeLayoutC;
    GroupMatmulExecGeneric<Policy, CombineQuantMode, BlockMmad, ElementC, MakeLayoutC, TopkWeightsPrefetch, IsLayered,
                           IsShared>(workSet, startLoopIdx, tileNum, args);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}
} // namespace Detail
// =================================================================================================
template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false, bool IsShared = false>
__aicore__ inline void GroupMatmulSwigluQuant(
    BlockEpilogueSwigluMxQuant<EpilogueElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, EpilogueTileM,
                               L1_TILE_N, TopkWeightsPrefetch> &epilogueOp,
    const Params &params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &vecSetSyncCom, uint32_t expertBeforeCnt,
    uint32_t expertIdx = 0)
{
    using SwigluQuantOpType = std::remove_reference_t<decltype(epilogueOp)>;
    Detail::Gmm1ArgsGeneric<SwigluQuantOpType> args{epilogueOp, vecSetSyncCom, expertBeforeCnt, expertIdx};
    Detail::GroupMatmulImplGeneric<Detail::Gmm1Policy, COMBINE_NO_QUANT, ElementA, ElementB, ElementC, ElementMxScaleA,
                                   ElementMxScaleB, IsWeightNZ, false, Gmm1TileM, TopkWeightsPrefetch, IsShared>(
        params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

// =================================================================================================
// GroupMatmul2：GMM2 矩阵乘法，支持量化和非量化模式
// =================================================================================================
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool TopkWeightsPrefetch = false, bool IsShared = false>
__aicore__ inline void GroupMatmul2(const Params &params,
                                    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &vecSetSyncCom2,
                                    uint32_t groupCnt, uint16_t &pingpongIdx, uint32_t groupIdx = 0)
{
    (void)groupIdx;
    Detail::Gmm2ArgsGeneric args{vecSetSyncCom2, groupCnt, pingpongIdx};
    Detail::GroupMatmulImplGeneric<Detail::Gmm2Policy, CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA,
                                   ElementMxScaleB, IsWeightNZ, IsLayered, Gmm1TileM, TopkWeightsPrefetch, IsShared>(
        params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

// ==================================================================================
// A8W4 执行路径 — 共享骨架，基于 Policy 分派 GMM1 / GMM2
// ==================================================================================
namespace Detail {

template <typename SwigluQuantOp>
struct Gmm1ArgsA8W4 {
    SwigluQuantOp &swigluQuantOp;
    uint32_t expertBeforeCnt{0};
    uint32_t expertIdx{0}; // prefetch 软同步：当前 expert 索引，roundTag = expertIdx + 1
};

struct Gmm2ArgsA8W4 {
    uint32_t groupCnt;
    uint16_t &pingpongIdx;
};

template <uint8_t CombineQuantMode, typename Policy, bool IsShared, bool IsLayered = false, typename BlockMmad,
          typename Scheduler, typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC,
          typename Config, bool TopkWeightsPrefetch>
__aicore__ inline void AicComputeA8W4(BlockMmad &blockMmad, Scheduler &scheduler, TensorA &gmA, TensorScaleA &gmScaleA,
                                      TensorScaleB &gmScaleB, TensorC &l0cOutGm, const GMMAddrInfo &gmmAddrInfo,
                                      const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                      const Params &params, uint32_t expertIdx)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr (!Policy::IS_GMM1 && (CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
        // Specialized 路径每两个 AIV 只有 subBlockIdx=1 参与 Combine，逻辑核数等于 blockNum。
        uint32_t logicalCoreCount = config.blockNum;
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(config.m, logicalCoreCount);
    }
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        if constexpr (Policy::IS_GMM1) {
            uint32_t waveIdx = mLoc / config.tileM;
            if (waveIdx != lastWaveWaited) {
                WaitForUpstreamReady<Policy, IsShared>(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else {
            if (loopIdx == startLoopIdx) {
                WaitForUpstreamReady<Policy, IsShared>(gmmAddrInfo, config, mLoc);
            }
        }

        auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        if constexpr (Policy::IS_GMM1) {
            for (uint32_t weightBlock = 0; weightBlock < MegaMoeImpl::SWIGLU_N_HALF; ++weightBlock) {
                auto nOffset = nLoc + weightBlock * config.outputN;
                auto gmBlockScaleB =
                    gmScaleB.Slice(Te::MakeCoord(0, nOffset), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
                auto tensorBlockGm = l0cOutGm.Slice(
                    Te::MakeCoord(mLoc, nOffset), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
            }
        } else {
            auto gmBlockScaleB =
                gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
            auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                                Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
            if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
                NotifyCombineConsumersOfTileCompletion(mLoc, groupSyncSlotLayout, gmmAddrInfo.gmm2CombineSyncCounter);
            } else if constexpr (IsShared) {
                NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
            }
        }

        // Layered 路径 GMM2 输出由 ProcessCombine 通过 group sync counter 读取，不走 Aiv1 GM tile 通知。
        constexpr bool hasAiv1GmEpilogue =
            Policy::IS_GMM1 || (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered);
        if constexpr (hasAiv1GmEpilogue) {
            if constexpr (Policy::IS_GMM1 && TopkWeightsPrefetch) {
                // prefetch 软同步：写 tile 状态位通知 AIV1（roundTag = expertIdx + 1）
                // FIX_S 同步确保 blockMmad 的 L0C→GM Fixpipe 传输完成后才写状态位
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                __gm__ int32_t *statusAddr =
                    reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm1TileStatusPtr) +
                    static_cast<int64_t>(expertIdx) * params.tilingData->maxTilesPerExpert + loopIdx;
                AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(expertIdx + 1));
            } else {
                // Keep at most one AIC-to-AIV1 GM tile notification outstanding.
                NotifyAiv1GmTileReady();
                WaitForAiv1GmTileAccepted();
            }
        }
    }
}

template <typename Policy, typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void AivPrologueA8W4(BlockPrologue &blockPrologue, Scheduler &scheduler, TensorB &gmB,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);

        if constexpr (Policy::IS_GMM1) {
            for (uint32_t weightBlock = 0; weightBlock < MegaMoeImpl::SWIGLU_N_HALF; ++weightBlock) {
                auto nOffset = nLoc + weightBlock * config.outputN;
                blockPrologue(gmB, mL1Size, config.k, nL1Size, nOffset, config.n, config.l1Params.kL1);
            }
        } else {
            blockPrologue(gmB, mL1Size, config.k, nL1Size, nLoc, config.n, config.l1Params.kL1);
        }
    }
}

template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename SwigluQuantOp,
          typename Config, bool TopkWeightsPrefetch>
__aicore__ inline void AivGmm1PostA8W4(SwigluQuantOp &swigluQuantOp, Scheduler &scheduler, TensorC &l0cOutGm,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                       const Params &params, uint32_t expertBeforeCnt, uint32_t expertIdx)
{
    constexpr uint32_t SUB_TILE_M = MegaMoeImpl::L1_TILE_M_128;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t mLen = Get<M_VALUE>(actualShape);

        if constexpr (TopkWeightsPrefetch) {
            // prefetch 软同步：轮询 tile 状态位（roundTag = expertIdx + 1）
            // 必须在 DataCopy(weightUb_) 之前：statusAddr 由 AIC 在写完 GMM1 输出后设置，
            // 而该 status 的前置条件是 AIV1 已完成 metaInfo GM 写入（MTE3+S 流水排空）。
            // 轮询通过后 metaInfo GM 中的权重数据才确定可见。
            __gm__ int32_t *statusAddr = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm1TileStatusPtr) +
                                         static_cast<int64_t>(expertIdx) * params.tilingData->maxTilesPerExpert +
                                         loopIdx;
            int32_t roundTag = static_cast<int32_t>(expertIdx + 1);
            while (AscendC::ReadGmByPassDCache(statusAddr) != roundTag) {
                int64_t st = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - st < 100) {
                }
            }

            // 轮询通过后 metaInfo GM 已就绪，搬第一个 128 子块的权重到 weightUb_
            uint64_t metaInfoMLoc0 = expertBeforeCnt + mLoc;
            AscendC::DataCopy(swigluQuantOp.weightUb_, swigluQuantOp.metaInfoGm_[metaInfoMLoc0 * INT32_PER_256B],
                              static_cast<uint32_t>(mLen < SUB_TILE_M ? mLen : SUB_TILE_M) * INT32_PER_256B);

            auto layoutL0cUB = MakeLayoutC{}(SUB_TILE_M, L1_TILE_N);
            int64_t ubOffsetFirst = 0;
            int64_t ubOffsetSecond = static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM_128) * sizeof(ElementC);
            auto tensorBlockUbFirst =
                Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUB);
            auto tensorBlockUbSecond =
                Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUB);
            auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});

            // V_MTE2 ping-pong：V 读完 UB 后通知 MTE2 可覆盖
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            for (uint32_t subOff = 0; subOff < mLen; subOff += SUB_TILE_M) {
                uint32_t subM = (mLen - subOff < SUB_TILE_M) ? (mLen - subOff) : SUB_TILE_M;
                uint32_t subMLoc = mLoc + subOff;
                uint64_t metaInfoMLoc = expertBeforeCnt + subMLoc;

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);

                // 搬入当前 128 子块的权重（首个子块已在循环前预取，跳过）
                // WaitFlag<V_MTE2> 保证 V 已读完上一轮的 weightUb_，此处覆写安全
                if (subOff != 0) {
                    AscendC::DataCopy(swigluQuantOp.weightUb_, swigluQuantOp.metaInfoGm_[metaInfoMLoc * INT32_PER_256B],
                                      subM * INT32_PER_256B);
                }

                auto tensorBlockGmFirst =
                    l0cOutGm.Slice(Te::MakeCoord(subMLoc, nLoc), Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
                auto tensorBlockGmSecond = l0cOutGm.Slice(Te::MakeCoord(subMLoc, nLoc + config.outputN),
                                                          Te::MakeShape(subM, Get<N_VALUE>(actualShape)));

                AscendC::Te::Copy(copyGM2UB, tensorBlockUbFirst, tensorBlockGmFirst);
                AscendC::Te::Copy(copyGM2UB, tensorBlockUbSecond, tensorBlockGmSecond);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); // 权重+GMM1 都搬完才通知 V
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

                Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, Get<N_VALUE>(actualShape), 0, 0};
                Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    subMLoc * config.outputN + nLoc,
                    subMLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                        CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                    0,
                    0,
                    static_cast<int64_t>(metaInfoMLoc),
                    0};
                AscendC::SetCtrlSpr<60, 60>(0);
                swigluQuantOp(epilogueShape, epilogueOffset);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        } else {
            WaitForAicGmTileReady();
            // Acknowledge before processing so AIC can issue the next tile concurrently.
            NotifyAicGmTileAccepted();
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
            auto tensorBlockGmFirst = l0cOutGm.Slice(
                Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto tensorBlockGmSecond =
                l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc + config.outputN),
                               Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));

            auto layoutL0cUB = MakeLayoutC{}(config.tileM, L1_TILE_N);
            int64_t ubOffsetFirst = 0;
            uint32_t ubBufSizeA8W4 =
                (config.tileM == L1_TILE_M_128) ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
            int64_t ubOffsetSecond = ubOffsetFirst + static_cast<int64_t>(ubBufSizeA8W4) * sizeof(ElementC);
            auto tensorBlockUbFirst =
                Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUB);
            auto tensorBlockUbSecond =
                Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUB);
            auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
            AscendC::Te::Copy(copyGM2UB, tensorBlockUbFirst, tensorBlockGmFirst);
            AscendC::Te::Copy(copyGM2UB, tensorBlockUbSecond, tensorBlockGmSecond);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape),
                                                                         Get<N_VALUE>(actualShape), 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                mLoc * config.outputN + nLoc,
                mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0,
                0,
                0,
                0};

            AscendC::SetCtrlSpr<60, 60>(0);
            swigluQuantOp(epilogueShape, epilogueOffset);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
        }
    }
}

template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename Config>
__aicore__ inline void AivGmm2PostA8W4(Scheduler &scheduler, TensorC &l0cOutGm, const Params &params, uint32_t groupCnt,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        WaitForAicGmTileReady();
        // Acknowledge before processing so AIC can issue the next tile concurrently.
        NotifyAicGmTileAccepted();
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto layoutL0cUB = MakeLayoutC{}(config.tileM, L1_TILE_N);
        int64_t ubOffset = 0;
        auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutL0cUB);
        LocalTensor<ElementC> l0cOutUbGMM2 =
            LocalTensor<ElementC>(TPosition::VECIN, ubOffset, config.tileM * L1_TILE_N);
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBlockUb, tensorBlockGm);

        AscendC::GlobalTensor<int32_t> metaInfoGm;
        int32_t lenTile = Get<M_VALUE>(actualShape);
        LocalTensor<int32_t> metaInfoTensor =
            LocalTensor<int32_t>(TPosition::VECCALC, META_INFO_TENSOR_ADDR, lenTile * 8);
        metaInfoGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr + (groupCnt + mLoc) * 32));
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, lenTile * 8);
        MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(mLoc, nLoc, config.n, metaInfoTensor,
                                                                           l0cOutUbGMM2, actualShape, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
}

template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorC>
struct WorkSetA8W4 {
    Scheduler &scheduler;
    TensorA &gmA;
    TensorB &gmB;
    TensorScaleA &gmScaleA;
    TensorScaleB &gmScaleB;
    TensorC &l0cOutGm;
};

template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename BlockPrologue, typename ElementC,
          typename MakeLayoutC, bool IsShared, bool IsLayered = false, typename WorkSet, typename Config,
          bool TopkWeightsPrefetch, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecA8W4(WorkSet &workSet, const Params &params, const GMMAddrInfo &gmmAddrInfo,
                                           const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                           ExtraArgs &args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params);
        if constexpr (Policy::IS_GMM1) {
            AicComputeA8W4<CombineQuantMode, Policy, IsShared, false, BlockMmad, decltype(workSet.scheduler),
                           decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                           decltype(workSet.l0cOutGm), Config, TopkWeightsPrefetch>(
                blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.l0cOutGm,
                gmmAddrInfo, config, startLoopIdx, tileNum, params, args.expertIdx);
        } else {
            AicComputeA8W4<CombineQuantMode, Policy, IsShared, IsLayered, BlockMmad, decltype(workSet.scheduler),
                           decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                           decltype(workSet.l0cOutGm), Config, TopkWeightsPrefetch>(
                blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.l0cOutGm,
                gmmAddrInfo, config, startLoopIdx, tileNum, params, 0);
        }
    } else {
        if (GetSubBlockIdx() == 0) {
            BlockPrologue blockPrologue;
            AivPrologueA8W4<Policy>(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
        } else {
            if constexpr (Policy::IS_GMM1) {
                AivGmm1PostA8W4<ElementC, MakeLayoutC, decltype(workSet.scheduler), decltype(workSet.l0cOutGm),
                                decltype(args.swigluQuantOp), Config, TopkWeightsPrefetch>(
                    args.swigluQuantOp, workSet.scheduler, workSet.l0cOutGm, config, startLoopIdx, tileNum, params,
                    args.expertBeforeCnt, args.expertIdx);
            } else {
                if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
                    AivGmm2PostA8W4<ElementC, MakeLayoutC>(workSet.scheduler, workSet.l0cOutGm, params, args.groupCnt,
                                                           config, startLoopIdx, tileNum);
                }
            }
        }
    }
}

template <uint8_t CombineQuantMode, typename Policy, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool TopkWeightsPrefetch = false, bool IsShared, bool IsLayered = false, typename ExtraArgs>
__aicore__ inline void GroupMatmulImplA8W4(const Params &params,
                                           const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                           const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, ExtraArgs &args)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using Config = Config<true, Policy, 0, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                          Gmm1TileM, TopkWeightsPrefetch, IsShared, IsLayered>;
    auto config = Config::BuildProblemConfig(problemShape);

    if constexpr (Policy::IS_GMM1) {
        args.swigluQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    auto layouts = Config::BuildLayouts(config);
    using BlockMmad = typename Config::BlockMmad;
    using BlockPrologue = typename Config::BlockPrologue;
    using MakeLayoutC = typename Config::MakeLayoutC;

    auto l0cOutGm = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(
                                       Policy::IS_GMM1 ? gmmAddrInfo.gmm1OutGlobal : gmmAddrInfo.gmm2OutGlobal)),
                                   layouts.c);
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);

    BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    using WorkSetType = WorkSetA8W4<BlockScheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                    decltype(gmScaleB), decltype(l0cOutGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, l0cOutGm};
    GroupMatmulExecA8W4<CombineQuantMode, Policy, BlockMmad, BlockPrologue, ElementC, MakeLayoutC, IsShared, IsLayered,
                        WorkSetType, decltype(config), TopkWeightsPrefetch>(workSet, params, gmmAddrInfo, config,
                                                                            startLoopIdx, tileNum, args);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

} // namespace Detail

// GroupMatmulSwigluQuantA8W4 — A8W4 prologue（W4→W8）+ GMM1 + SwiGLU + 量化
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false,
          bool IsShared = false>
__aicore__ inline void GroupMatmulSwigluQuantA8W4(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true, EpilogueTileM, L1_TILE_N,
                               TopkWeightsPrefetch> &swigluQuantOp,
    const Params &params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &vecSetSyncCom, uint32_t expertBeforeCnt,
    uint32_t expertIdx = 0)
{
    (void)vecSetSyncCom;
    using SwigluQuantOpType = std::remove_reference_t<decltype(swigluQuantOp)>;
    Detail::Gmm1ArgsA8W4<SwigluQuantOpType> args{swigluQuantOp, expertBeforeCnt, expertIdx};
    Detail::GroupMatmulImplA8W4<COMBINE_NO_QUANT, Detail::Gmm1Policy, ElementA, ElementB, ElementC, ElementMxScaleA,
                                ElementMxScaleB, Gmm1TileM, TopkWeightsPrefetch, IsShared>(
        params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

// GroupMatmul2CombineA8W4 — A8W4 prologue（W4→W8）+ GMM2 + Combine
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsLayered = false>
__aicore__ inline void GroupMatmul2CombineA8W4(const Params &params,
                                               const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                               const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                               int32_t &vecSetSyncCom2, uint32_t groupCnt, uint16_t &pingpongIdx,
                                               uint32_t groupIdx = 0)
{
    (void)vecSetSyncCom2;
    (void)groupIdx;
    Detail::Gmm2ArgsA8W4 args{groupCnt, pingpongIdx};
    Detail::GroupMatmulImplA8W4<CombineQuantMode, Detail::Gmm2Policy, ElementA, ElementB, ElementC, ElementMxScaleA,
                                ElementMxScaleB, Gmm1TileM, TopkWeightsPrefetch, IsShared, IsLayered>(
        params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

} // namespace MegaMoeImpl

#endif
