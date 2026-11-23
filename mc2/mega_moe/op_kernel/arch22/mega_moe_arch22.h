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
 * \file mega_moe_arch22.h
 * \brief Unified MegaMoe implementation that merges the three legacy paths:
 *          - mega_moe          (INT8 weight)
 *          - mega_moe_bf16     (FP16 / BF16 weight)
 *          - mega_moe_w4_a8    (INT4 weight, W4A8)
 *        Differences from the legacy single-dtype implementations:
 *          - Inputs use the new unified naming (x / topk_ids / topk_weights / weight1
 *            / weight_scales1 / ... ) following the unified op def.
 *          - The first input is `context` (replaces the removed `group` attribute).
 *          - The single legacy `transB` attribute is split into `transpose_weight1`
 *            and `transpose_weight2`, surfaced via two template flags.
 *          - The legacy `M` attribute is renamed to `max_recv_token_num`.
 *          - Tiling fields are read in the order defined in the unified op def.
 */

#ifndef MEGA_MOE_ARCH22_H
#define MEGA_MOE_ARCH22_H

using namespace AscendC;

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../../../common/op_kernel/moe_distribute_base.h"

#include "mega_moe_tiling_a2a3.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "utils/select_helper.hpp"
#include "utils/const_args.hpp"
#include "utils/dispatch_policy_custom.hpp"
#include "mc2_moe_context.h"
#include "utils/block_epilogue_pertoken_v2.hpp"
#include "utils/block_epilogue_pertoken_swiglu.hpp"
#include "utils/block_epilogue_w4a8post_pertoken_v2.hpp"
#include "mega_moe_kernel_a2_int8.hpp"
#include "mega_moe_kernel_a2_bf16.hpp"
#include "mega_moe_kernel_a2_bf16_w4a8.hpp"
#include "mega_moe_kernel_a3.hpp"
#include "mega_moe_shared_expert.hpp"
#include "mega_moe_shared_output.hpp"
#include "mega_moe_independent_shared.hpp"

namespace MegaMoeImpl {
using namespace AscendC;
using namespace AscendC::HcclContextDef;
using namespace Catlass;
using namespace Mc2Tiling;

// Template parameter pack used by the unified MegaMoe class.
//   AType_ : input X element type (DTYPE_X)
//   BType_ : weight element type   (DTYPE_WEIGHT1)
//   CType_ : output  element type  (DTYPE_Y)
//   TB1_   : transpose_weight1 attribute
//   TB2_   : transpose_weight2 attribute
//   Nz_    : whether weight1/2 is in FRACTAL_NZ format (both are the same)
//   IS_A2_ : whether running on Atlas A2 (910B); false means A3 (910_93)
#define MegaMoeClass \
    typename AType_, typename BType_, typename CType_, bool TB1_, bool TB2_, bool Nz_, bool IS_A2_, \
        bool SharedGmm2Bf16_, typename SharedBType_, bool SharedW2Nz_, bool DedupOn_, bool MASK_ROUTING_, \
        bool BLOCKED_OUTPUT_
#define MegaMoeFunc \
    AType_, BType_, CType_, TB1_, TB2_, Nz_, IS_A2_, SharedGmm2Bf16_, SharedBType_, SharedW2Nz_, DedupOn_, \
        MASK_ROUTING_, BLOCKED_OUTPUT_

template <typename AType_, typename BType_, typename CType_, bool TB1_, bool TB2_, bool Nz_, bool IS_A2_,
          bool SharedGmm2Bf16_ = false, typename SharedBType_ = BType_, bool SharedW2Nz_ = Nz_, bool DedupOn_ = false,
          bool MASK_ROUTING_ = false, bool BLOCKED_OUTPUT_ = false>
class MegaMoe {
public:
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM, GM_ADDR topkWeightsGM,
                                GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR weightScales1GM, GM_ADDR weightScales2GM,
                                GM_ADDR bias1GM, GM_ADDR bias2GM, GM_ADDR xActiveMaskGM, GM_ADDR scalesGM,
                                GM_ADDR sharedWeight1GM, GM_ADDR sharedWeight2GM, GM_ADDR sharedWeightScales1GM,
                                GM_ADDR sharedWeightScales2GM, GM_ADDR sharedBias1GM, GM_ADDR sharedBias2GM,
                                GM_ADDR maskBufferGM, GM_ADDR yGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                                GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    GM_ADDR contextGM_;
    GM_ADDR xGM_;
    GM_ADDR topkIdsGM_;
    GM_ADDR topkWeightsGM_;
    GM_ADDR weight1GM_;
    GM_ADDR weight2GM_;
    GM_ADDR weightScales1GM_;
    GM_ADDR weightScales2GM_;
    GM_ADDR bias1GM_;
    GM_ADDR bias2GM_;
    GM_ADDR xActiveMaskGM_;
    GM_ADDR scalesGM_;
    GM_ADDR maskBufferGM_;
    GM_ADDR sharedWeight1GM_;
    GM_ADDR sharedWeight2GM_;
    GM_ADDR sharedWeightScales1GM_;
    GM_ADDR sharedWeightScales2GM_;
    GM_ADDR sharedBias1GM_;
    GM_ADDR sharedBias2GM_;
    GM_ADDR yGM_;
    GM_ADDR gmExpertTokenNums_;
    GM_ADDR workspaceGM_;
    GM_ADDR tilingGM_;
    MegaMoeA2A3TilingData common_{};

    GM_ADDR moeInitRoutingQuantV2Scale_ = nullptr;
    GM_ADDR moeInitRoutingQuantV2Offset_ = nullptr;
    GM_ADDR expertTokensBeforeCapacity_ = nullptr;

    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t aivNum_;

    static constexpr int32_t UB_MOVE_NUM = 16 * 1024;

    int32_t m_;
    int32_t k_;
    int32_t n_;
    int32_t topK_;
    int32_t expertPerRank_;
    uint64_t maxOutputSize_;
    // 接收侧轮表份数上限 = ceil(PERMUTE_CHUNK*EP*min(topK,epr) / recvRoundBudget)，A3 接收端轮次切分用
    uint64_t recvRoundsMax_;
    int32_t epWorldSize_;
    int32_t listLen_;
    uint32_t sharedExpertNum_;
    uint64_t sharedWorkspaceOffset_;
    uint32_t activationCode_;
    float activationClamp_;
    float activationParams1_;
    float activationParams2_;
    // 通信零冗余：dedupOn_=0 时下列字段全 0（基线布局）；=1 时偏移由 host dedup_layout
    // 单一口径透传（kernel 不重算）
    uint32_t dedupOn_{0};
    int64_t offsetExpandedDstToSrcRow_{0};
    int64_t dedupOffsetD_{0};
    int64_t distTokenFirstPosSize_{0};
    int64_t distTokenLastPosSize_{0};
    int64_t dedupRowMetaSize_{0};

    // Quant/NonQuant 两种 tiling 结构的 common 字段类型一致，此处统一加载公共字段
    __aicore__ inline void LoadCommonTiling(const MegaMoeA2A3TilingData& common)
    {
        common_ = common;
        aivNum_ = common.aivNum;
        m_ = common.M;
        k_ = common.K;
        n_ = common.N;
        epWorldSize_ = common.worldSize;
        topK_ = common.topK;
        expertPerRank_ = common.expertPerRank;
        // maxOutputSize 语义 = 单轮接收预算 B（recvRoundBudget）；接收行超出 B 由 kernel 接收端轮次切分处理
        maxOutputSize_ = common.recvRoundBudget;
        recvRoundsMax_ = common.recvRoundsMax;
        // Dedup 布局透传（全路径读取；dedupOn=0 时全 0，kernel 走基线布局）
        dedupOn_ = common.dedupOn;
        offsetExpandedDstToSrcRow_ = common.offsetExpandedDstToSrcRow;
        dedupOffsetD_ = common.dedupOffsetD;
        distTokenFirstPosSize_ = common.distTokenFirstPosSize;
        distTokenLastPosSize_ = common.distTokenLastPosSize;
        dedupRowMetaSize_ = common.dedupRowMetaSize;
        listLen_ = common.listLen;
        sharedExpertNum_ = common.sharedExpertNum;
        sharedWorkspaceOffset_ = common.sharedWorkspaceOffset;
        activationCode_ = common.activationCode;
        activationClamp_ = common.activationClamp;
        activationParams1_ = common.activationParams1;
        activationParams2_ = common.activationParams2;
    }
};

template <MegaMoeClass>
__aicore__ inline void MegaMoe<MegaMoeFunc>::Init(
    GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM, GM_ADDR topkWeightsGM, GM_ADDR weight1GM, GM_ADDR weight2GM,
    GM_ADDR weightScales1GM, GM_ADDR weightScales2GM, GM_ADDR bias1GM, GM_ADDR bias2GM, GM_ADDR xActiveMaskGM,
    GM_ADDR scalesGM, GM_ADDR sharedWeight1GM, GM_ADDR sharedWeight2GM, GM_ADDR sharedWeightScales1GM,
    GM_ADDR sharedWeightScales2GM, GM_ADDR sharedBias1GM, GM_ADDR sharedBias2GM, GM_ADDR maskBufferGM, GM_ADDR yGM,
    GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    constexpr bool kRoutingIsQuant =
        std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int32_t> || std::is_same_v<BType_, int8_t>;

    contextGM_ = contextGM;
    xGM_ = xGM;
    topkIdsGM_ = topkIdsGM;
    topkWeightsGM_ = topkWeightsGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    weightScales1GM_ = weightScales1GM;
    weightScales2GM_ = weightScales2GM;
    bias1GM_ = bias1GM;
    bias2GM_ = bias2GM;
    xActiveMaskGM_ = xActiveMaskGM;
    scalesGM_ = scalesGM;
    maskBufferGM_ = maskBufferGM;
    sharedWeight1GM_ = sharedWeight1GM;
    sharedWeight2GM_ = sharedWeight2GM;
    sharedWeightScales1GM_ = sharedWeightScales1GM;
    sharedWeightScales2GM_ = sharedWeightScales2GM;
    sharedBias1GM_ = sharedBias1GM;
    sharedBias2GM_ = sharedBias2GM;
    yGM_ = yGM;
    gmExpertTokenNums_ = expertTokenNumsGM;
    workspaceGM_ = workspaceGM;
    tilingGM_ = tilingGM;

    if constexpr (kRoutingIsQuant) {
        GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingDataQuant, tilingData, tilingGM);
        LoadCommonTiling(tilingData.common);
    } else {
        GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingDataNonQuant, tilingData, tilingGM);
        LoadCommonTiling(tilingData.common);
    }
}

template <MegaMoeClass>
__aicore__ inline void MegaMoe<MegaMoeFunc>::Process()
{
    if constexpr (!IS_A2_) {
        if (sharedExpertNum_ > MEGA_MOE_A3_MAX_SHARED_EXPERT_NUM) {
            return;
        }
    }
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;

    constexpr bool isW4A8 = std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int32_t>;
    constexpr bool isInt8 = std::is_same_v<BType_, int8_t>;

    using AElement = std::conditional_t<isW4A8, AscendC::int4b_t, std::conditional_t<isInt8, int8_t, AType_>>;
    using BElement = std::conditional_t<isW4A8, AscendC::int4b_t, BType_>;
    using CElement = std::conditional_t<isW4A8 || isInt8, float16_t, CType_>;
    using D1Element = std::conditional_t<isW4A8 || isInt8, int8_t, CType_>;

    uint32_t k2 = n_ / 2;
    uint32_t n2 = k_;

    using LayoutA = layout::RowMajor;
    using LayoutB =
        typename std::conditional<Nz_, layout::zN,
                                  typename std::conditional<TB1_, layout::ColumnMajor, layout::RowMajor>::type>::type;

    LayoutB layoutB1 = LayoutBInitializer<LayoutB, BElement>::create(k_, n_);
    LayoutB layoutB2 = LayoutBInitializer<LayoutB, BElement>::create(k2, n2);
    using LayoutC = layout::RowMajor;

    using L1TileShape =
        std::conditional_t<isW4A8, GemmShape<128, 256, 1024>,
                           std::conditional_t<isInt8, GemmShape<128, 256, 512>, GemmShape<128, 256, 256>>>;
    using L0TileShape =
        std::conditional_t<isW4A8, GemmShape<128, 256, 256>,
                           std::conditional_t<isInt8, GemmShape<128, 256, 128>, GemmShape<128, 256, 64>>>;

    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;

    using DispatchPolicy = Gemm::MmadDispatchPolicyFor_t<BElement, IS_A2_, preloadStages, l1Stages, l0AStages,
                                                         l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;

    using AType = Gemm::GemmType<AElement, layout::RowMajor>;
    using BType = Gemm::GemmType<BElement, LayoutB>;
    using CType = Gemm::GemmType<CElement, layout::RowMajor>;
    using D1Type = Gemm::GemmType<D1Element, layout::RowMajor>;
    using D2Type = std::conditional_t<std::is_same_v<CType_, bfloat16_t>, Gemm::GemmType<bfloat16_t, layout::RowMajor>,
                                      Gemm::GemmType<CType_, layout::RowMajor>>;

    using ScaleGranularity = Catlass::Gemm::Tile::ScaleGranularity;
    using TileCopyMmad = Gemm::Tile::QuantTileCopy<ArchTag, AType, BType, CType, void, ScaleGranularity::PER_CHANNEL>;

    using BlockMmad = std::conditional_t<
        isW4A8,
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopyMmad>,
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>>;

    constexpr uint32_t ubStages = 2;

    using A2EpiloguePolicy1 =
        std::conditional_t<isInt8, Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuantInt8<ubStages>,
                           Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuantBF16<ubStages>>;
    using EpilogueDispatchPolicy1 = std::conditional_t<
        isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<ubStages>,
        std::conditional_t<IS_A2_, A2EpiloguePolicy1, Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>>>;
    using A2EpiloguePolicy2 =
        std::conditional_t<isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantV2<ubStages>,
                           std::conditional_t<isInt8, Epilogue::EpilogueAtlasA2PerTokenDequantV2Int8<ubStages>,
                                              Epilogue::EpilogueAtlasA2PerTokenDequantV2BF16<ubStages>>>;
    using EpilogueDispatchPolicy2 =
        std::conditional_t<IS_A2_, A2EpiloguePolicy2, Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>>;
    using EpilogueDispatchPolicy3 =
        std::conditional_t<isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantV2<ubStages>,
                           Epilogue::EpilogueAtlasA2PerTokenDequantV2<ubStages>>;

    using ScaleType = Gemm::GemmType<uint64_t, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using ElementMulType = Gemm::GemmType<float, layout::RowMajor>;
    using TileElemWiseMuls = Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;

    using TileCopy1 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
    using BlockEpilogue1 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType, D1Type,
                                                          TileElemWiseMuls, TileCopy1>;

    using TileCopy2 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
    using IS_A2 = Epilogue::Block::IS_A2_T<IS_A2_>;
    using MaskRoute = Epilogue::Block::MaskIndexedRouteTagT<BLOCKED_OUTPUT_>;
    using BlockEpilogue2 = std::conditional_t<
        IS_A2_ && isW4A8,
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType, D2Type, TileCopy2, IS_A2>,
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType, D2Type, TileCopy2>>;
    using BlockEpilogue3 = std::conditional_t<
        isW4A8,
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy3, CType, PerTokenScaleType, D2Type, TileCopy2, IS_A2>,
        std::conditional_t<MASK_ROUTING_,
                           Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy3, CType, PerTokenScaleType, D2Type,
                                                          TileCopy2, MaskRoute>,
                           Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy3, CType, PerTokenScaleType, D2Type,
                                                          TileCopy2, IS_A2>>>;
    // Dedup combine 行级 epilogue，仅 DedupOn_=true 实例的 DedupCombineV2 使用
    using EpilogueDispatchPolicyDedup = Epilogue::EpilogueAtlasA2W4A8PostPerTokenDedup<ubStages>;
    using BlockEpilogueDedup =
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicyDedup, CType, PerTokenScaleType, D2Type, TileCopy2>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;

    // The mixed shared path retains INT8 GMM1, but does not quantize its
    // activation again. GMM2 selects its own BF16 ND/NZ weight layout.
    using SharedBf16Type = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
    using SharedBf16Layout = std::conditional_t<SharedGmm2Bf16_ && !SharedW2Nz_, layout::RowMajor, LayoutB>;
    using SharedBf16Weight = Gemm::GemmType<bfloat16_t, SharedBf16Layout>;
    using SharedBf16Dispatch = Gemm::MmadDispatchPolicyFor_t<bfloat16_t, IS_A2_, preloadStages, l1Stages, l0AStages,
                                                             l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;
    using SharedBf16Mmad = Gemm::Block::BlockMmad<SharedBf16Dispatch, GemmShape<128, 256, 256>, GemmShape<128, 256, 64>,
                                                  SharedBf16Type, SharedBf16Weight, SharedBf16Type>;
    using SharedTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, SharedBf16Type>;
    using SharedBf16Epilogue =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>, CType,
                                       PerTokenScaleType, SharedBf16Type, TileElemWiseMuls, SharedTileCopy>;

    // W4A8 routed FFNs and W8A8 shared FFNs reuse the existing A3 overlap
    // schedule. Their Cube tile K, NZ packing and vector epilogues are distinct.
    constexpr bool overlapW4W8 = !IS_A2_ && isW4A8 && std::is_same_v<SharedBType_, int8_t>;
    // Only the A3 shared W8/W8 pair spills raw accumulators. The routed
    // matmuls and the INT8/BF16 shared pair keep their existing C format.
    constexpr bool sharedW8Pair = !IS_A2_ && std::is_same_v<SharedBType_, int8_t> && !SharedGmm2Bf16_;
    using SharedW8C = Gemm::GemmType<int32_t, layout::RowMajor>;
    using SharedW8Type = Gemm::GemmType<int8_t, layout::RowMajor>;
    using SharedW8Weight = Gemm::GemmType<int8_t, LayoutB>;
    using SharedW8Dispatch = Gemm::MmadDispatchPolicyFor_t<int8_t, false, preloadStages, l1Stages, l0AStages, l0BStages,
                                                           l0CStages, enableUnitFlag, enableShuffleK>;
    using SharedW8Mmad = Gemm::Block::BlockMmad<SharedW8Dispatch, GemmShape<128, 256, 512>, GemmShape<128, 256, 128>,
                                                SharedW8Type, SharedW8Weight, SharedW8C>;
    using SharedW8Copy1 = Epilogue::Tile::TileCopy<ArchTag, SharedW8C, ScaleType, PerTokenScaleType, SharedW8Type>;
    using SharedW8Copy2 = Epilogue::Tile::TileCopy<ArchTag, SharedW8C, ScaleType, PerTokenScaleType, D2Type>;
    using SharedW8Epilogue1 =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>, SharedW8C,
                                       PerTokenScaleType, SharedW8Type, TileElemWiseMuls, SharedW8Copy1>;
    using SharedW8Epilogue2 = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>,
                                                             SharedW8C, PerTokenScaleType, D2Type, SharedW8Copy2>;
    using OverlapSharedMmad = std::conditional_t<sharedW8Pair, SharedW8Mmad, BlockMmad>;
    using OverlapSharedEpilogue1 = std::conditional_t<sharedW8Pair, SharedW8Epilogue1, BlockEpilogue1>;
    using OverlapSharedEpilogue2 = std::conditional_t<sharedW8Pair, SharedW8Epilogue2, BlockEpilogue2>;

    GM_ADDR originalX = xGM_;
    GM_ADDR originalY = yGM_;
    // The W4/W8 combination also overlaps when the two FFNs use different I.
    constexpr bool differentSharedType = !std::is_same_v<SharedBType_, BType_>;
    const bool separateShared = !IS_A2_ && !overlapW4W8 && (differentSharedType || common_.sharedN != common_.N);
    if (separateShared) {
        sharedExpertNum_ = 0;
    }

    // A2 W4A8 reuses the GMM1 input workspace; only this path pads its row stride.
    uint32_t layoutA1RowStride =
        IS_A2_ && isW4A8 ? static_cast<uint32_t>(k_ > k2 ? k_ : k2) : static_cast<uint32_t>(k_);
    LayoutA layoutA1{static_cast<uint32_t>(m_), static_cast<uint32_t>(k_), layoutA1RowStride};
    LayoutA layoutA2{static_cast<uint32_t>(m_), static_cast<uint32_t>(k2)};
    layout::VectorLayout layoutScale1{static_cast<uint32_t>(n_)};
    layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2)};
    layout::RowMajor layoutD1{static_cast<uint32_t>(maxOutputSize_), static_cast<uint32_t>(k2)};
    layout::RowMajor layoutD2{static_cast<uint32_t>(m_ * topK_), static_cast<uint32_t>(n2)};

    GemmCoord problemShape{static_cast<uint32_t>(m_), static_cast<uint32_t>(n_), static_cast<uint32_t>(k_)};

    uint32_t epilogueCoreNum;
    uint32_t epilogueGranularity;
    if constexpr (isW4A8) {
        epilogueCoreNum = static_cast<uint32_t>(aivNum_);
        epilogueGranularity = 0U;
    } else if constexpr (isInt8) {
        if constexpr (IS_A2_) {
            epilogueCoreNum = aivNum_ / 2;
            epilogueGranularity = (expertPerRank_ > 1) ? static_cast<uint32_t>(expertPerRank_ - 1) : 1u;
        } else {
            epilogueCoreNum = static_cast<uint32_t>(aivNum_);
            epilogueGranularity = (expertPerRank_ > 4) ? static_cast<uint32_t>(expertPerRank_ - 3) :
                                  (expertPerRank_ > 1) ? static_cast<uint32_t>(expertPerRank_ - 1) :
                                                         1u;
        }
    } else {
        epilogueCoreNum = static_cast<uint32_t>(aivNum_); // BF16/FP16：所有 AIV 参与
        epilogueGranularity = (expertPerRank_ > 2) ? static_cast<uint32_t>(expertPerRank_ - 2) : 1u;
    }

    using A2MatmulKernel = std::conditional_t<
        isW4A8,
        Gemm::Kernel::MegaMoeKernelA2Bf16W4A8<BlockMmad, BlockScheduler, ElementGroupList, BlockEpilogue1,
                                              BlockEpilogue2>,
        std::conditional_t<isInt8,
                           Gemm::Kernel::MegaMoeKernelA2Int8<BlockMmad, BlockScheduler, ElementGroupList,
                                                             BlockEpilogue1, BlockEpilogue2>,
                           Gemm::Kernel::MegaMoeKernelA2BF16<BlockMmad, BlockScheduler, ElementGroupList,
                                                             BlockEpilogue1, BlockEpilogue2>>>;
    using A3MatmulKernel =
        Gemm::Kernel::MegaMoeKernel<BlockMmad, BlockScheduler, ElementGroupList, BlockEpilogue1, BlockEpilogue2,
                                    BlockEpilogue3, SharedBf16Mmad, SharedBf16Epilogue, SharedGmm2Bf16_,
                                    OverlapSharedMmad, OverlapSharedEpilogue1, OverlapSharedEpilogue2,
                                    BlockEpilogueDedup, DedupOn_>;
    using MatmulKernel = std::conditional_t<IS_A2_, A2MatmulKernel, A3MatmulKernel>;

    typename MatmulKernel::Params params{problemShape,
                                         static_cast<uint32_t>(epWorldSize_),
                                         static_cast<uint32_t>(listLen_),
                                         static_cast<uint32_t>(expertPerRank_),
                                         static_cast<uint64_t>(maxOutputSize_),
                                         static_cast<uint32_t>(topK_),
                                         epilogueCoreNum,
                                         contextGM_,
                                         xGM_,
                                         layoutA1,
                                         layoutA2,
                                         weight1GM_,
                                         layoutB1,
                                         bias1GM_,
                                         weight2GM_,
                                         layoutB2,
                                         bias2GM_,
                                         weightScales1GM_,
                                         layoutScale1,
                                         weightScales2GM_,
                                         layoutScale2,
                                         yGM_,
                                         layoutD1,
                                         layoutD2,
                                         topkIdsGM_,
                                         moeInitRoutingQuantV2Scale_,
                                         moeInitRoutingQuantV2Offset_,
                                         expertTokensBeforeCapacity_,
                                         topkWeightsGM_,
                                         workspaceGM_,
                                         gmExpertTokenNums_,
                                         xActiveMaskGM_,
                                         scalesGM_,
                                         epilogueGranularity,
                                         activationClamp_,
                                         activationCode_,
                                         activationParams1_,
                                         activationParams2_,
                                         tilingGM_};
    if constexpr (!IS_A2_) {
        params.sharedExpertNum = sharedExpertNum_;
        params.ptrMaskBuffer = maskBufferGM_;
        params.recvRoundsMax = recvRoundsMax_;
        // Dedup 布局透传（仅 A3 Params 具备 Dedup 字段；dedupOn=0 时全 0，
        // PeermemInfo/WorkspaceInfo 走基线布局）
        params.dedupOn = dedupOn_;
        params.offsetExpandedDstToSrcRow = offsetExpandedDstToSrcRow_;
        params.dedupOffsetD = dedupOffsetD_;
        params.distTokenFirstPosSize = distTokenFirstPosSize_;
        params.distTokenLastPosSize = distTokenLastPosSize_;
        params.dedupRowMetaSize = dedupRowMetaSize_;
        using SharedElementB = typename A3MatmulKernel::SharedElementB;
        params.ptrSharedB1 = reinterpret_cast<__gm__ SharedElementB*>(sharedWeight1GM_);
        params.ptrSharedB2 = reinterpret_cast<__gm__ SharedElementB*>(sharedWeight2GM_);
        params.ptrSharedScale1 = reinterpret_cast<__gm__ uint64_t*>(sharedWeightScales1GM_);
        params.ptrSharedScale2 = reinterpret_cast<__gm__ uint64_t*>(sharedWeightScales2GM_);
        params.ptrSharedBias1 = reinterpret_cast<__gm__ float*>(sharedBias1GM_);
        params.ptrSharedBias2 = reinterpret_cast<__gm__ float*>(sharedBias2GM_);
        params.sharedWorkspaceOffset = sharedWorkspaceOffset_;
        params.sharedN = common_.sharedN;
        params.layoutSharedB1 = LayoutBInitializer<LayoutB, SharedElementB>::create(k_, common_.sharedN);
        params.layoutSharedWeight2 = LayoutBInitializer<LayoutB, SharedElementB>::create(common_.sharedN / 2U, n2);
        params.layoutSharedB2 = LayoutBInitializer<SharedBf16Layout, bfloat16_t>::create(k2, n2);
    }

    {
        MatmulKernel kernel(params);
        if constexpr (!IS_A2_) {
            // The independent shared stage must honor the routed rank's mask too.
            if (kernel.IsCurrentRankMasked()) {
                return;
            }
        }
        kernel(params);
    }

    if constexpr (!IS_A2_ && (std::is_same_v<SharedBType_, int8_t> || std::is_same_v<SharedBType_, AscendC::int4b_t> ||
                              std::is_same_v<SharedBType_, int32_t> || std::is_same_v<SharedBType_, bfloat16_t>)) {
        if (separateShared && common_.sharedExpertNum > 0U) {
            // Route-only unpermute must publish its final
            // output before the independent shared stage can reuse hardware flags.
            PublishRoutedOutput();
            const uint32_t h = common_.K;
            GM_ADDR result = IndependentShared<SharedBType_, Nz_, TB1_, SharedGmm2Bf16_, SharedW2Nz_>(
                common_, originalX, workspaceGM_, sharedWeight1GM_, sharedWeight2GM_, sharedWeightScales1GM_,
                sharedWeightScales2GM_, sharedBias1GM_, sharedBias2GM_, h, activationCode_, activationClamp_,
                activationParams1_, activationParams2_);
            AddIndependentShared(originalY, result, m_, h, common_.sharedExpertNum);
        }
    }
}

} // namespace MegaMoeImpl
#endif // MEGA_MOE_H
