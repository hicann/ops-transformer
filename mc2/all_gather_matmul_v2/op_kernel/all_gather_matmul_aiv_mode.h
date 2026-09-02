/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_matmul_aiv_mode.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gather_moe_distribute_base.h"
#include "all_gather_matmul_aiv_mode_tiling.h"
#include "all_gather_matmul_aiv_mode_util.h"
#include "all_gather_matmul_aiv_mode_padding.h"
#include "all_gather_matmul_aiv_mode_dequant.h"

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_arch.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_mmad.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_swizzle.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"

#include "gather_matmul.hpp"

using namespace AscendC;

constexpr static uint32_t BUFFER_NUM = 2U;     // 多buf
constexpr static uint32_t STATE_OFFSET = 512U; // 状态空间偏移地址
constexpr static uint32_t BLOCK_SIZE = 32U;
constexpr static uint32_t B32_PER_BLOCK = 8U;
constexpr static uint32_t B64_PER_BLOCK = 4U;
constexpr static int32_t CUBE_MATRIX_SIZE_B16 = 256;        // 16 * 16
constexpr static int32_t L0AB_PINGPONG_BUFFER_SIZE = 32768; // 32 KB
constexpr static int32_t FLAG_ZERO_IDX = 0;
constexpr static int32_t FLAG_ONE_IDX = 1;
constexpr static int32_t FLAG_VALUE = 1;
constexpr static int32_t FLAG_TWO_IDX = 2;
constexpr static int32_t FLAG_THREE_IDX = 3;
constexpr static int32_t USED_UB_SIZE = 160 * 1024;
constexpr static int32_t TILE_SHAPE_128 = 128;
constexpr static int32_t TILE_SHAPE_256 = 256;
constexpr static int32_t TILE_SHAPE_512 = 512;
constexpr static int32_t TILE_SHAPE_64 = 64;
constexpr static int32_t FLAG_OFFSET = 180 * 1024 * 1024 / sizeof(int32_t);

using namespace Catlass;

namespace AllGatherMatmulAIVModeImpl {

template <typename T>
using supportTypeForDataCopy = typename std::conditional<std::is_same<T, AscendC::int4b_t>::value, int8_t, T>::type;

template <typename T, typename ReturnType = size_t>
struct TILE_SHAPE_K_512B {
    static constexpr ReturnType value = Catlass::BytesToBits(512) / Catlass::SizeOfBits<T>::value;
};

template <typename T, typename ReturnType = size_t>
struct TILE_SHAPE_K_256B {
    static constexpr ReturnType value = Catlass::BytesToBits(256) / Catlass::SizeOfBits<T>::value;
};

template <typename T, typename ReturnType = size_t>
struct TILE_SHAPE_K_128B {
    static constexpr ReturnType value = Catlass::BytesToBits(128) / Catlass::SizeOfBits<T>::value;
};

// AGMM : AllGatherMatmulAIVMode
#define TemplateAGMMClass \
    typename X1Type, typename X2Type, typename BiasType, typename x2ScaleType, typename YType, bool weightNZ, bool TA, \
        bool TB
#define TemplateAGMMFunc X1Type, X2Type, BiasType, x2ScaleType, YType, weightNZ, TA, TB

using namespace AscendC;
template <TemplateAGMMClass>
class AllGatherMatmulAIVMode {
    constexpr static bool quantFlag =
        (std::is_same<X1Type, int8_t>::value && std::is_same<X2Type, int8_t>::value) ||
        (std::is_same<X1Type, AscendC::int4b_t>::value && std::is_same<X2Type, AscendC::int4b_t>::value);
    using supportX1Type = supportTypeForDataCopy<X1Type>;
    using supportX2Type = supportTypeForDataCopy<X2Type>;
    constexpr static uint32_t UB_OFFSET =
        Catlass::BytesToBits(97440) / Catlass::SizeOfBits<supportX1Type>::value; // 2 是 size of T

public:
    __aicore__ inline AllGatherMatmulAIVMode(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM,
                                GM_ADDR cGM, GM_ADDR allgatherGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AIVInit();
    __aicore__ inline void AICInit();
    __aicore__ inline void Padding();
    __aicore__ inline void Dequant(int32_t cal_idx);
    __aicore__ inline void AllGatherPerTokenScale(int64_t buff_st);
    __aicore__ inline void CatlassMatmul();
    __aicore__ inline void MoveWithSplit(__gm__ supportX1Type *gm_src, int64_t rank_offset, int64_t len);
    __aicore__ inline void MoveToOtherRankWithSkip(__gm__ supportX1Type *gm_src, int64_t rank_offset, int32_t len,
                                                   int32_t rank_st, int32_t skip_num, int32_t group_num,
                                                   int32_t rank_scope);
    __aicore__ inline void MoveResultFromPeerMemToOut(__gm__ supportX1Type *gm_src, __gm__ supportX1Type *gm_dst,
                                                      int32_t actual_m);
    __aicore__ inline void MoveResultToDst(__gm__ supportX1Type *gm_src, __gm__ supportX1Type *gm_dst, int32_t len);
    __aicore__ inline void MoveResultFromSrcToDst(__gm__ supportX1Type *gm_src, __gm__ supportX1Type *gm_dst,
                                                  int32_t len);
    __aicore__ inline void CrossRankSyncV1(int32_t flag_idx, int32_t flag_data);
    __aicore__ inline void CrossRankSyncV2(int32_t flag_idx, int32_t flag_data);

private:
    GlobalTensor<X1Type> aGMTensor_;
    GlobalTensor<X2Type> bGMTensor_;
    GlobalTensor<YType> cGMTensor_;
    GlobalTensor<YType> dataGMTensor_;
    GlobalTensor<int64_t> flagGMTensor_;

    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR x1ScaleGM_;
    GM_ADDR x2ScaleGM_;
    GM_ADDR allgatherGM_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR stateAddrPerRank[8];

    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t m;
    int32_t k;
    int32_t n;
    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t m_loop;
    int32_t k_loop;
    int32_t n_loop;
    int32_t pValue;
    int32_t aligned_a;
    int32_t aligned_b;
    int32_t cal_count;
    int64_t gm_a_pingpong_size;
    int32_t max_ub_ping_pong_size;
    int32_t m_align;
    int64_t k_align;
    int32_t n_align;
    int32_t comm_npu_split;
    int32_t comm_data_split;
    int32_t comm_direct;
    int32_t len_per_loop;
    int32_t core_count;
    int64_t data_len;
    int64_t num_per_rank_move;
    int64_t src_offset;
    int64_t rank_offset;
    int32_t swizzlCount;
    int32_t swizzlDirect;

    int32_t max_move_m;
    int32_t max_move_k = 20480;

    int32_t worldSize{0};
    int32_t rankId{0};
    int32_t coreIdx{0};
    int32_t aivIdx{0};
    int32_t coreNum{0};
    int32_t blockIdx{0};

    uint64_t aAlignSize{0};
    uint64_t bAlignSize{0};
    bool hasAAlign{false};
    bool hasBAlign{false};
    bool quanFlag{false};
    bool needAivDequant{false};
    bool isX2ScaleTypeInt64{false};
    DequantType dequantType;
    bool needPerChannel;
    bool needPerToken;
    bool accumWorkSpacePingPong{false};

    __gm__ supportX1Type *gm_peer_mem;

    GM_ADDR gm_a_align;
    GM_ADDR gm_b_align;
    __gm__ supportX1Type *gm_a_src;
    __gm__ supportX2Type *gm_b_src;
    __gm__ int32_t *gm_accum;
    GM_ADDR gm_scale_workspace;

    DequantRunner<YType> dequantRunner;
    Arch::Resource<Arch::AtlasA2> resource;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
};

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                                      GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM, GM_ADDR cGM,
                                                                      GM_ADDR allgatherGM, GM_ADDR workspaceGM,
                                                                      GM_ADDR tilingGM)
{
    auto tiling = (__gm__ AllGatherMatmulAIVModeTilingData *)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    allgatherGM_ = allgatherGM;
    x1ScaleGM_ = x1ScaleGM;
    x2ScaleGM_ = x2ScaleGM;

    m = tilingData.allGatherMatmulInfo.M;
    k = tilingData.allGatherMatmulInfo.K;
    n = tilingData.allGatherMatmulInfo.N;

    m0 = tilingData.cocTiling.m0; // tiling 寻优
    k0 = tilingData.cocTiling.k0; // tiling 寻优
    n0 = tilingData.cocTiling.n0; // tiling 寻优

    m_loop = tilingData.cocTiling.mLoop;
    n_loop = tilingData.cocTiling.nLoop;
    k_loop = tilingData.cocTiling.kLoop;
    pValue = tilingData.cocTiling.pValue;

    max_ub_ping_pong_size = tilingData.cocTiling.ubMoveNum / MAX_BLOCK_COUNT;
    comm_npu_split = tilingData.cocTiling.commNpuSplit;   // tiling 寻优
    comm_data_split = tilingData.cocTiling.commDataSplit; // tiling 寻优
    comm_direct = tilingData.cocTiling.commDirect;        // tiling 寻优
    len_per_loop = tilingData.cocTiling.lenPerLoop;       // tiling 寻优
    swizzlCount = tilingData.cocTiling.swizzlCount;       // 家里tiling写死
    swizzlDirect = tilingData.cocTiling.swizzlDirect;     // 家里tiling写死
    core_count = comm_npu_split * comm_data_split;

    coreIdx = GetBlockIdx();              // 0-48核
    coreNum = GetBlockNum();              // 24
    aivIdx = GetSubBlockIdx();            // 0-1
    blockIdx = coreIdx / GetTaskRation(); // 0-24核

    aAlignSize = tilingData.allGatherMatmulInfo.aAlignSize;
    bAlignSize = tilingData.allGatherMatmulInfo.bAlignSize;
    hasAAlign = tilingData.allGatherMatmulInfo.hasAAlign;
    hasBAlign = (tilingData.allGatherMatmulInfo.hasBAlign && !(weightNZ));
    gm_a_align = reinterpret_cast<GM_ADDR>(hasAAlign ? workspaceGM : 0);
    gm_b_align = reinterpret_cast<GM_ADDR>(hasBAlign ? workspaceGM + aAlignSize : 0);
    gm_a_src = reinterpret_cast<__gm__ supportX1Type *>(hasAAlign ? gm_a_align : aGM_);
    gm_b_src = reinterpret_cast<__gm__ supportX2Type *>(hasBAlign ? gm_b_align : bGM_);
    gm_accum = reinterpret_cast<__gm__ int32_t *>(quantFlag ? workspaceGM + aAlignSize + bAlignSize : 0);

    m_align = Block512B<X1Type>::AlignUp(m);
    k_align = Block512B<X1Type>::AlignUp(k);
    n_align = Block512B<X1Type>::AlignUp(n);
    aligned_a = hasAAlign;
    aligned_b = hasBAlign;

    isX2ScaleTypeInt64 = tilingData.allGatherMatmulInfo.isX2ScaleTypeInt64;
    dequantType = tilingData.allGatherMatmulInfo.dequantType;
    needAivDequant = quantFlag && (dequantType == PER_TOKEN || std::is_same<YType, bfloat16_t>::value);
    bool needPerChannelA8W8 = quantFlag && !(isX2ScaleTypeInt64 && std::is_same<YType, float16_t>::value);
    needPerChannel = std::is_same_v<X1Type, AscendC::int4b_t> || needPerChannelA8W8;
    needPerToken = quantFlag && dequantType == PER_TOKEN;

    bool is910C = tilingData.allGatherMatmulInfo.is910C;
    if (is910C) {
        __gm__ HcclOpResParam *winContext_{nullptr};
        winContext_ = (__gm__ HcclOpResParam *)contextGM0;
        rankId = winContext_->localUsrRankId;
        worldSize = winContext_->rankSize;
        for (int i = 0; i < worldSize; i++) {
            stateAddrPerRank[i] =
                (GM_ADDR)((i == rankId) ?
                              winContext_->localWindowsIn :
                              ((HcclRankRelationResV2 *)(winContext_->remoteRes[i].nextDevicePtr))->windowsIn);
        }
    } else {
        __gm__ HcclA2CombineOpParam *winContext_{nullptr};
        winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
        rankId = winContext_->rankId;
        worldSize = winContext_->rankNum;
        for (int i = 0; i < worldSize; i++) {
            stateAddrPerRank[i] = (GM_ADDR)winContext_->windowsIn[i];
        }
    }

    accumWorkSpacePingPong = tilingData.allGatherMatmulInfo.accumWorkSpacePingPong;
    int32_t workspaceM = accumWorkSpacePingPong ? MAX_BLOCK_COUNT * m0 * pValue : m;
    uint64_t gm_scale_workspace_st = static_cast<uint64_t>(aAlignSize) + bAlignSize +
                                     static_cast<uint64_t>(workspaceM) * n * worldSize * sizeof(int32_t);
    gm_scale_workspace = needPerToken ? workspaceGM + gm_scale_workspace_st : 0;

    AllGatherMatmulAIVMode<TemplateAGMMFunc>::AICInit();

    AllGatherMatmulAIVMode<TemplateAGMMFunc>::AIVInit();
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::AICInit()
{
    if ASCEND_IS_AIC {
        SetLoadDataPaddingValue(0);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        gm_peer_mem = reinterpret_cast<__gm__ supportX1Type *>(stateAddrPerRank[rankId]);
    }
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::CrossRankSyncV1(int32_t flag_idx, int32_t flag_data)
{
    if (aivIdx == 0 && blockIdx == rankId) {
        SetBuffFlagByAdd((__gm__ int32_t *)stateAddrPerRank[rankId] + FLAG_OFFSET + flag_idx, uBuf_, FLAG_VALUE);
    } else if (aivIdx == 0 && blockIdx < worldSize) {
        CheckBuffFlag((__gm__ int32_t *)stateAddrPerRank[blockIdx] + FLAG_OFFSET + flag_idx, uBuf_,
                      FLAG_VALUE * flag_data);
    }
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::CrossRankSyncV2(int32_t flag_idx, int32_t flag_data)
{
    if (aivIdx == 0 && blockIdx < worldSize) {
        SetBuffFlagByAdd((__gm__ int32_t *)stateAddrPerRank[blockIdx] + FLAG_OFFSET + flag_idx, uBuf_, FLAG_VALUE);
    }
    if (aivIdx == 0 && blockIdx == rankId) {
        CheckBuffFlag((__gm__ int32_t *)stateAddrPerRank[rankId] + FLAG_OFFSET + flag_idx, uBuf_,
                      FLAG_VALUE * worldSize * flag_data);
    }
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::AIVInit()
{
    if ASCEND_IS_AIV {
        SetAtomicNone();
        SetMaskNormImpl();
        SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1);
        __gm__ supportX1Type *buff[8];
        for (int i = 0; i < worldSize; ++i) {
            buff[i] = reinterpret_cast<__gm__ supportX1Type *>(stateAddrPerRank[i]);
        }

        cal_count = DivCeil(m_loop, pValue);
        gm_a_pingpong_size = m0 * k_align * pValue * worldSize;

        data_len = m * k_align;                    // 数据量
        num_per_rank_move = m0 * k_align * pValue; // 每轮搬运到其他卡的数据量
        src_offset = 0;                            // 当前份数据的起始位置
        rank_offset = rankId * num_per_rank_move;
    }
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        int64_t peer_mem_m = static_cast<int64_t>(m0) * pValue * worldSize;
        uint32_t layout_b_col = (TB || weightNZ) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(n_align);
        uint32_t layout_b_row = (TB && !weightNZ) ? static_cast<uint32_t>(k_align) : static_cast<uint32_t>(k);
        bool need_fixpipe = quantFlag && std::is_same<YType, half>::value && isX2ScaleTypeInt64 &&
                            (!std::is_same_v<X1Type, AscendC::int4b_t>);

        using ArchTag = Arch::AtlasA2;
        constexpr bool ENABLE_UNIT_FLAG = false;
        constexpr bool ENABLE_SHUFFLE_K = true;
        using ElementA = X1Type;
        using ElementB = X2Type;
        using ElementC = typename std::conditional<quantFlag, int32_t, YType>::type;
        using LayoutA = layout::RowMajor;
        using LayoutC = layout::RowMajor;
        using LayoutScale = layout::VectorLayout;
        LayoutA layoutA{static_cast<uint32_t>(m), static_cast<uint32_t>(k_align)};
        LayoutC layoutC{static_cast<uint32_t>(m * worldSize), static_cast<uint32_t>(n)};
        LayoutA layoutPeerMem{static_cast<uint32_t>(peer_mem_m * MAX_BLOCK_COUNT), static_cast<uint32_t>(k_align)};
        LayoutScale layoutScale{static_cast<uint32_t>(n)};
        GemmCoord processSize{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

        constexpr int32_t L1TileShapeK = TILE_SHAPE_K_512B<X1Type, int32_t>::value;
        constexpr int32_t L0TileShapeK = TILE_SHAPE_K_128B<X1Type, int32_t>::value;
        using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;
        using AType = Gemm::GemmType<ElementA, LayoutA>;
        using CType = Gemm::GemmType<ElementC, LayoutC>;

        if (weightNZ) {
            using LayoutNZ = typename std::conditional<TB, layout::nZ, layout::zN>::type;
            using BType = Gemm::GemmType<ElementB, LayoutNZ>;
            LayoutNZ layoutBNZ = LayoutNZ::template MakeLayout<ElementB>(layout_b_row, layout_b_col);

            struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, void> {
                using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, void>;
                using ElementA = typename Base::ElementA;
                using ElementB = typename Base::ElementB;
                using ElementAccumulator = typename Base::ElementAccumulator;
                using CopyGmToL1A = typename Base::CopyGmToL1A;
                using CopyGmToL1B = typename Base::CopyGmToL1B;
                using CopyL1ToL0A = typename Base::CopyL1ToL0A;
                using CopyL1ToL0B = typename Base::CopyL1ToL0B;
                using CopyL0CToGm = typename Base::CopyL0CToGm;
            };
            using TileCopy = TileCopyOpt;

            if (m0 == TILE_SHAPE_128) {
                using L1TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType,
                                                            CType, void, TileCopy>;
                using MatmulKernel = Gemm::Kernel::AllGatherMatmulV2<void, void, BlockMmadOpt>;
                typename MatmulKernel::Params params{processSize,
                                                     reinterpret_cast<GM_ADDR>(gm_a_src),
                                                     layoutA,
                                                     reinterpret_cast<GM_ADDR>(gm_b_src),
                                                     layoutBNZ,
                                                     reinterpret_cast<GM_ADDR>(cGM_),
                                                     layoutC,
                                                     reinterpret_cast<GM_ADDR>(x2ScaleGM_),
                                                     layoutScale,
                                                     reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                     layoutPeerMem,
                                                     reinterpret_cast<GM_ADDR>(gm_accum),
                                                     pValue,
                                                     swizzlCount,
                                                     swizzlDirect,
                                                     rankId,
                                                     worldSize,
                                                     need_fixpipe,
                                                     accumWorkSpacePingPong};
                MatmulKernel matmul_op;
                matmul_op(params);
            } else {
                using L1TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType,
                                                            CType, void, TileCopy>;
                using MatmulKernel = Gemm::Kernel::AllGatherMatmulV2<void, void, BlockMmadOpt>;
                typename MatmulKernel::Params params{processSize,
                                                     reinterpret_cast<GM_ADDR>(gm_a_src),
                                                     layoutA,
                                                     reinterpret_cast<GM_ADDR>(gm_b_src),
                                                     layoutBNZ,
                                                     reinterpret_cast<GM_ADDR>(cGM_),
                                                     layoutC,
                                                     reinterpret_cast<GM_ADDR>(x2ScaleGM_),
                                                     layoutScale,
                                                     reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                     layoutPeerMem,
                                                     reinterpret_cast<GM_ADDR>(gm_accum),
                                                     pValue,
                                                     swizzlCount,
                                                     swizzlDirect,
                                                     rankId,
                                                     worldSize,
                                                     need_fixpipe,
                                                     accumWorkSpacePingPong};
                MatmulKernel matmul_op;
                matmul_op(params);
            }
        } else {
            using LayoutB = typename std::conditional<TB, layout::ColumnMajor, layout::RowMajor>::type;
            LayoutB layoutB{layout_b_row, layout_b_col};
            using BType = Gemm::GemmType<ElementB, LayoutB>;

            struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, void> {
                using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, void>;
                using ElementA = typename Base::ElementA;
                using ElementB = typename Base::ElementB;
                using ElementAccumulator = typename Base::ElementAccumulator;
                using CopyGmToL1A = typename Base::CopyGmToL1A;
                using CopyGmToL1B = typename Base::CopyGmToL1B;
                using CopyL1ToL0A = typename Base::CopyL1ToL0A;
                using CopyL1ToL0B = typename Base::CopyL1ToL0B;
                using CopyL0CToGm = typename Base::CopyL0CToGm;
            };
            using TileCopy = TileCopyOpt;

            if (m0 == TILE_SHAPE_128) {
                using L1TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType,
                                                            CType, void, TileCopy>;
                using MatmulKernel = Gemm::Kernel::AllGatherMatmulV2<void, void, BlockMmadOpt>;
                typename MatmulKernel::Params params{processSize,
                                                     reinterpret_cast<GM_ADDR>(gm_a_src),
                                                     layoutA,
                                                     reinterpret_cast<GM_ADDR>(gm_b_src),
                                                     layoutB,
                                                     reinterpret_cast<GM_ADDR>(cGM_),
                                                     layoutC,
                                                     reinterpret_cast<GM_ADDR>(x2ScaleGM_),
                                                     layoutScale,
                                                     reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                     layoutPeerMem,
                                                     reinterpret_cast<GM_ADDR>(gm_accum),
                                                     pValue,
                                                     swizzlCount,
                                                     swizzlDirect,
                                                     rankId,
                                                     worldSize,
                                                     need_fixpipe,
                                                     accumWorkSpacePingPong};
                MatmulKernel matmul_op;
                matmul_op(params);
            } else {
                using L1TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType,
                                                            CType, void, TileCopy>;
                using MatmulKernel = Gemm::Kernel::AllGatherMatmulV2<void, void, BlockMmadOpt>;
                typename MatmulKernel::Params params{processSize,
                                                     reinterpret_cast<GM_ADDR>(gm_a_src),
                                                     layoutA,
                                                     reinterpret_cast<GM_ADDR>(gm_b_src),
                                                     layoutB,
                                                     reinterpret_cast<GM_ADDR>(cGM_),
                                                     layoutC,
                                                     reinterpret_cast<GM_ADDR>(x2ScaleGM_),
                                                     layoutScale,
                                                     reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                     layoutPeerMem,
                                                     reinterpret_cast<GM_ADDR>(gm_accum),
                                                     pValue,
                                                     swizzlCount,
                                                     swizzlDirect,
                                                     rankId,
                                                     worldSize,
                                                     need_fixpipe,
                                                     accumWorkSpacePingPong};
                MatmulKernel matmul_op;
                matmul_op(params);
            }
        }
    }
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::Padding()
{
    if (!aligned_a && !aligned_b) {
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreFlag flagAivFinishPadding{AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID};
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        return;
    }
    bool transA = false; // 当前暂未支持A矩阵转置
    bool transB = TB;
    bool alignedA = aligned_a;
    bool alignedB = aligned_b;
    uint32_t matrixAM = m;
    uint32_t matrixAK = k;
    uint32_t matrixBK = k;
    uint32_t matrixBN = n;
    uint32_t matrixAMAlign = m_align;
    uint32_t matrixAKAlign = k_align;
    uint32_t matrixBKAlign = k_align;
    uint32_t matrixBNAlign = n_align;
    GM_ADDR gmA = reinterpret_cast<GM_ADDR>(aGM_);
    GM_ADDR gmB = reinterpret_cast<GM_ADDR>(bGM_);
    GM_ADDR gmAAlign = reinterpret_cast<GM_ADDR>(gm_a_align);
    GM_ADDR gmBAlign = reinterpret_cast<GM_ADDR>(gm_b_align);
    PaddingRunner<X1Type, X2Type> padding_runner;
    padding_runner.Run(PADDING_ARGS_CALL());
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::MoveResultFromSrcToDst(__gm__ supportX1Type *gm_src,
                                                                                        __gm__ supportX1Type *gm_dst,
                                                                                        int32_t len)
{
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
    MoveResultToDst(gm_src, gm_dst, len);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::AllGatherPerTokenScale(int64_t buff_st)
{
    if (!needPerToken) {
        return;
    }

    int32_t multi = sizeof(float32_t) / sizeof(supportX1Type);
    int32_t scale_size = m * multi;
    int32_t scale_st = rankId * scale_size;
    __gm__ supportX1Type *scale = reinterpret_cast<__gm__ supportX1Type *>(x1ScaleGM_);
    __gm__ supportX1Type *scaleOut = reinterpret_cast<__gm__ supportX1Type *>(gm_scale_workspace);
    SetAndWaitAivSync(FLAG_VALUE);
    // 将本卡的scale拷贝到buff中
    if (aivIdx == 0 && rankId == blockIdx) {
        MoveResultFromSrcToDst(
            scale, reinterpret_cast<__gm__ supportX1Type *>(stateAddrPerRank[rankId]) + buff_st + scale_st, scale_size);
    }
    CrossRankSyncV1(FLAG_TWO_IDX, FLAG_VALUE);
    SetAndWaitAivSync(FLAG_VALUE);
    // 将其他卡的scale拷贝到buff中
    scale_st = blockIdx * scale_size;
    if (aivIdx == 0 && blockIdx < worldSize) {
        MoveResultFromSrcToDst(
            reinterpret_cast<__gm__ supportX1Type *>(stateAddrPerRank[blockIdx]) + buff_st + scale_st,
            scaleOut + scale_st, scale_size);
    }
    SetAndWaitAivSync(FLAG_VALUE);
    CrossRankSyncV2(FLAG_THREE_IDX, FLAG_VALUE);
    SetAndWaitAivSync(FLAG_VALUE);
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::Dequant(int32_t cal_idx)
{
    // per token 反量化实现
    if (!needPerChannel && !needPerToken) {
        return;
    }

    uint32_t rowNum = cal_idx == cal_count - 1 ? m - cal_idx * m0 * pValue : m0 * pValue;
    uint32_t colNum = n;
    uint32_t tileM0 = m0;
    uint32_t tileN0 = n0;

    int64_t blockSt = static_cast<int64_t>(cal_idx) * m0 * pValue * n;
    int64_t blockSize = static_cast<int64_t>(m) * n;
    int64_t blockStInWorkspace = static_cast<int64_t>(cal_idx % MAX_BLOCK_COUNT) * worldSize * m0 * pValue * n;
    int64_t blockSizeInWorkspace = static_cast<int64_t>(m0) * pValue * n;

    if (!accumWorkSpacePingPong) {
        blockStInWorkspace = blockSt;
        blockSizeInWorkspace = blockSize;
    }

    __gm__ float32_t *perChannelScale = needPerChannel ? reinterpret_cast<__gm__ float32_t *>(x2ScaleGM_) : nullptr;
    __gm__ float32_t *perTokenScale = needPerToken ? reinterpret_cast<__gm__ float32_t *>(gm_scale_workspace) : nullptr;
    __gm__ int32_t *workspace = needPerChannel ? reinterpret_cast<__gm__ int32_t *>(gm_accum) : nullptr;
    __gm__ YType *output = reinterpret_cast<__gm__ YType *>(cGM_);

    // 当 X1Type 为 int4_t 时，只让 subblockIdx == 1 的核参与计算
    constexpr bool isInt4Type = std::is_same_v<X1Type, AscendC::int4b_t>;
    if (isInt4Type && aivIdx != 1) {
        return;
    }

    dequantRunner.Run(DEQUANT_ARGS_CALL());
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::MoveToOtherRankWithSkip(
    __gm__ supportX1Type *gm_src, int64_t rank_offset, int32_t len, int32_t rank_st, int32_t skip_num,
    int32_t group_num, int32_t rank_scope)
{
    LocalTensor<supportX1Type> ubTensor = uBuf_.AllocTensor<supportX1Type>();
    LocalTensor<supportX1Type> copyTensor0 = ubTensor;
    LocalTensor<supportX1Type> copyTensor1 = ubTensor[UB_OFFSET];
    int32_t ping_pong_move_count = (len + max_ub_ping_pong_size - 1) / max_ub_ping_pong_size;
    for (int32_t move_idx = 0; move_idx < ping_pong_move_count; ++move_idx) {
        int32_t actual_move_size = max_ub_ping_pong_size;
        if (move_idx == ping_pong_move_count - 1) {
            actual_move_size = len - move_idx * max_ub_ping_pong_size;
        }
        int32_t block_len = actual_move_size * Catlass::SizeOfBits<X1Type>::value / 8;
        auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<supportX1Type> copyTensor = (move_idx & 1) ? copyTensor0 : copyTensor1;
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
        CopyGmToUbufAlignB16(copyTensor, gm_src, 1, block_len, 0, 0);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        int32_t dst_rank = rank_st % rank_scope;
        for (int32_t cycle_idx = 0; cycle_idx < group_num; ++cycle_idx) {
            if (dst_rank != rankId && dst_rank < worldSize) {
                if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                    CopyUbufToGmAlignB16((__gm__ int8_t *)stateAddrPerRank[dst_rank] + rank_offset / 2, copyTensor, 1,
                                         block_len, 0, 0);
                } else {
                    CopyUbufToGmAlignB16((__gm__ X1Type *)stateAddrPerRank[dst_rank] + rank_offset, copyTensor, 1,
                                         block_len, 0, 0);
                }
            }
            dst_rank = (dst_rank + skip_num) % rank_scope;
        }
        if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
            gm_src += (max_ub_ping_pong_size / 2);
        } else {
            gm_src += max_ub_ping_pong_size;
        }
        rank_offset += max_ub_ping_pong_size;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
    }
    uBuf_.FreeTensor<supportX1Type>(ubTensor);
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::MoveResultFromPeerMemToOut(
    __gm__ supportX1Type *gm_src, __gm__ supportX1Type *gm_dst, int32_t actual_m)
{
    LocalTensor<supportX1Type> ubTensor = uBuf_.AllocTensor<supportX1Type>();
    LocalTensor<supportX1Type> copyTensor0 = ubTensor;
    LocalTensor<supportX1Type> copyTensor1 = ubTensor[UB_OFFSET];
    max_move_m = max_ub_ping_pong_size > max_move_k ? max_ub_ping_pong_size / max_move_k : 1;
    int32_t ping_pong_move_count = (actual_m + max_move_m - 1) / max_move_m;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
    for (int32_t move_idx = 0; move_idx < ping_pong_move_count; ++move_idx) {
        int32_t actual_move_m = max_move_m;
        if (move_idx == ping_pong_move_count - 1) {
            actual_move_m = actual_m - move_idx * max_move_m;
        }
        auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<supportX1Type> ub_buff_st = (move_idx & 1) ? copyTensor0 : copyTensor1;
        int32_t k_move_count = (k_align + max_move_k - 1) / max_move_k;
        for (int32_t k_move_idx = 0; k_move_idx < k_move_count; ++k_move_idx) {
            int32_t actual_k_move_num_in_peer_mem = max_move_k;
            int32_t actual_k_move_num_in_out = max_move_k;
            if (k_move_idx == k_move_count - 1) {
                actual_k_move_num_in_peer_mem = k_align - k_move_idx * max_move_k;
                actual_k_move_num_in_out = k - k_move_idx * max_move_k;
            }
            WaitFlag<HardEvent::MTE3_MTE2>(event_id);
            int64_t gm_src_offset_k_align =
                static_cast<int64_t>(move_idx) * max_move_m * k_align + k_move_idx * max_move_k;
            if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                gm_src_offset_k_align = gm_src_offset_k_align / 2;
            }
            CopyGmToUbuf(ub_buff_st, gm_src + gm_src_offset_k_align, actual_move_m,
                         actual_k_move_num_in_peer_mem * Catlass::SizeOfBits<X1Type>::value / (8 * 32),
                         (k_align - actual_k_move_num_in_peer_mem) * Catlass::SizeOfBits<X1Type>::value / (8 * 32), 0);
            SetFlag<HardEvent::MTE2_MTE3>(event_id);
            WaitFlag<HardEvent::MTE2_MTE3>(event_id);
            int64_t gm_src_offset = static_cast<int64_t>(move_idx) * max_move_m * k + k_move_idx * max_move_k;
            if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                gm_src_offset = gm_src_offset / 2;
            }
            CopyUbufToGmAlignB16(gm_dst + gm_src_offset, ub_buff_st, actual_move_m,
                                 actual_k_move_num_in_out * Catlass::SizeOfBits<X1Type>::value / 8,
                                 (actual_k_move_num_in_peer_mem - actual_k_move_num_in_out) *
                                     Catlass::SizeOfBits<X1Type>::value / (8 * 32),
                                 (k - actual_k_move_num_in_out) * Catlass::SizeOfBits<X1Type>::value / (8 * 32));
            SetFlag<HardEvent::MTE3_MTE2>(event_id);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::MoveResultToDst(__gm__ supportX1Type *gm_src,
                                                                                 __gm__ supportX1Type *gm_dst,
                                                                                 int32_t len)
{
    LocalTensor<supportX1Type> ubTensor = uBuf_.AllocTensor<supportX1Type>();
    LocalTensor<supportX1Type> copyTensor0 = ubTensor;
    LocalTensor<supportX1Type> copyTensor1 = ubTensor[UB_OFFSET];
    int32_t ping_pong_move_count = (len + max_ub_ping_pong_size - 1) / max_ub_ping_pong_size;
    for (int32_t move_idx = 0; move_idx < ping_pong_move_count; ++move_idx) {
        int32_t actual_move_size = max_ub_ping_pong_size;
        if (move_idx == ping_pong_move_count - 1) {
            actual_move_size = len - move_idx * max_ub_ping_pong_size;
        }
        auto event_id = (move_idx & 1) ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<supportX1Type> copyTensor = (move_idx & 1) ? copyTensor0 : copyTensor1;
        WaitFlag<HardEvent::MTE3_MTE2>(event_id);
        CopyGmToUbufAlignB16(copyTensor, gm_src, 1, actual_move_size * sizeof(supportX1Type), 0, 0);
        SetFlag<HardEvent::MTE2_MTE3>(event_id);
        WaitFlag<HardEvent::MTE2_MTE3>(event_id);
        CopyUbufToGmAlignB16(gm_dst, copyTensor, 1, actual_move_size * sizeof(supportX1Type), 0, 0);
        gm_src += max_ub_ping_pong_size;
        gm_dst += max_ub_ping_pong_size;
        SetFlag<HardEvent::MTE3_MTE2>(event_id);
    }
    uBuf_.FreeTensor<supportX1Type>(ubTensor);
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::MoveWithSplit(__gm__ supportX1Type *gm_src,
                                                                               int64_t rank_offset, int64_t len)
{
    int64_t data_split = DivCeil(len, static_cast<int64_t>(len_per_loop));
    int32_t data_block = len_per_loop; // 每份数据量 len_per_loop = 2560
    int32_t rank_st = blockIdx;
    int32_t group_num = DivCeil(worldSize, comm_npu_split); // 1？ comm_npu_split=worldSize?
    int32_t scope = comm_npu_split * group_num;             // worldSize？
    int64_t data_offset = -data_block;                      // 当前份数据的起始位置
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);               // MTE2等MTE3
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);               // MTE2等MTE3
    for (int64_t data_block_idx = 0; data_block_idx < data_split; ++data_block_idx) {
        data_offset += data_block; // 当前份数据的起始位置
        data_block = data_block_idx == data_split - 1 ? static_cast<int32_t>(len - data_offset) : data_block;
        int32_t num_per_core = DivCeil(data_block, comm_data_split); // 2560

        int64_t data_src = data_offset + (blockIdx / comm_npu_split) * num_per_core;
        int32_t data_len = static_cast<int32_t>(data_block + data_offset - data_src);
        data_len = data_len >= num_per_core ? num_per_core : data_len;
        // npu 方向：一份数据先发送到所有目标卡，再发送下一份数据，以此类推
        if (comm_direct) { // comm_direct=0？
            if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                MoveToOtherRankWithSkip(gm_src + data_src / 2, rank_offset + data_src, data_len, rank_st,
                                        comm_npu_split, group_num, scope);
            } else {
                MoveToOtherRankWithSkip(gm_src + data_src, rank_offset + data_src, data_len, rank_st, comm_npu_split,
                                        group_num, scope);
            }
            continue;
        }
        // data len 方向：所有的数据先发送到目标卡0，再发送到目标卡1，以此类推
        int32_t dst_rank = rank_st % scope;
        for (int32_t rank_group_idx = 0; rank_group_idx < group_num; ++rank_group_idx) {
            if (dst_rank != rankId && dst_rank < worldSize) {
                if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                    MoveResultToDst(gm_src + data_src / 2,
                                    (__gm__ int8_t *)stateAddrPerRank[dst_rank] + (rank_offset + data_src) / 2,
                                    data_len / 2);
                } else {
                    MoveResultToDst(gm_src + data_src,
                                    (__gm__ X1Type *)stateAddrPerRank[dst_rank] + rank_offset + data_src, data_len);
                }
            }
            dst_rank = (dst_rank + comm_npu_split) % scope;
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
}

template <TemplateAGMMClass>
__aicore__ inline void AllGatherMatmulAIVMode<TemplateAGMMFunc>::Process()
{
    if ASCEND_IS_AIV {
        TPipe pipe;
        pipe.InitBuffer(uBuf_, USED_UB_SIZE);
        Padding();
        int32_t num_flags = 4;
        // flag[0] - flag[3] 清0
        for (int32_t idx = 0; idx < num_flags; ++idx) {
            if (blockIdx == 0 && aivIdx == 0) {
                SetBuffFlag((__gm__ int32_t *)stateAddrPerRank[rankId] + FLAG_OFFSET + idx, uBuf_, 0);
            }
        }
        PipeBarrier<PIPE_ALL>();

        for (int32_t cal_idx = 0; cal_idx < cal_count + MAX_BLOCK_COUNT; ++cal_idx) {
            uint64_t flag_idx = cal_idx % MAX_BLOCK_COUNT;

            if (cal_idx == cal_count - 1) {
                num_per_rank_move = data_len - src_offset;
            }

            if (cal_idx == 1) {
                // 聚合perToken scale
                AllGatherPerTokenScale(gm_a_pingpong_size);
            }

            // wait aic
            if (cal_idx >= MAX_BLOCK_COUNT) {
                WaitEvent(flag_idx);
            }

            SetAndWaitAivSync(flag_idx);
            if (cal_idx < cal_count) {
                // Step 2: Rank sync
                CrossRankSyncV1(FLAG_ZERO_IDX, cal_idx + 1);
                SetAndWaitAivSync(flag_idx);
            }

            if (cal_idx < cal_count && aivIdx == 0 && blockIdx < core_count) {
                int64_t gm_rank_offset = static_cast<int64_t>(flag_idx) * gm_a_pingpong_size + rank_offset;
                if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                    MoveWithSplit(reinterpret_cast<__gm__ int8_t *>(gm_a_src) + src_offset / 2, gm_rank_offset,
                                  num_per_rank_move);
                } else {
                    MoveWithSplit(reinterpret_cast<__gm__ X1Type *>(gm_a_src) + src_offset, gm_rank_offset,
                                  num_per_rank_move);
                }
                src_offset += num_per_rank_move;
            } else if (cal_idx > 0 && cal_idx < cal_count + 1 && aivIdx == 1 && blockIdx >= core_count &&
                       blockIdx < worldSize + core_count) { // peermem to out
                // 如果剩余的core数不够，则循环搬运
                int32_t other_core_num = coreNum - core_count;                         // 剩余的core数
                int32_t cycle_num = (other_core_num + worldSize - 1) / other_core_num; // 循环次数
                uint64_t s2_flag_idx = (cal_idx - 1) % MAX_BLOCK_COUNT;
                int64_t src_offset = static_cast<int64_t>(cal_idx - 1) * pValue * m0 * k_align;
                int32_t s2_actual_m = cal_idx == cal_count ? m - (cal_idx - 1) * pValue * m0 : pValue * m0;

                for (int32_t cycle_idx = 0; cycle_idx < cycle_num; ++cycle_idx) {
                    int32_t s2_other_rank = blockIdx - core_count + cycle_idx * other_core_num;
                    int64_t other_rank_offset = static_cast<int64_t>(s2_flag_idx) * gm_a_pingpong_size +
                                                static_cast<int64_t>(s2_other_rank) * pValue * m0 * k_align;
                    int64_t dst_offset = static_cast<int64_t>(s2_other_rank) * m * k +
                                         static_cast<int64_t>(cal_idx - 1) * pValue * m0 * k;
                    if (s2_other_rank >= worldSize) {
                        break;
                    }

                    if (s2_other_rank != rankId) {
                        if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                            MoveResultFromPeerMemToOut(
                                (__gm__ int8_t *)stateAddrPerRank[rankId] + other_rank_offset / 2,
                                reinterpret_cast<__gm__ int8_t *>(allgatherGM_) + dst_offset / 2, s2_actual_m);
                        } else {
                            MoveResultFromPeerMemToOut((__gm__ X1Type *)stateAddrPerRank[rankId] + other_rank_offset,
                                                       reinterpret_cast<__gm__ X1Type *>(allgatherGM_) + dst_offset,
                                                       s2_actual_m);
                        }
                    } else {
                        if constexpr (std::is_same_v<X1Type, AscendC::int4b_t>) {
                            MoveResultFromPeerMemToOut(reinterpret_cast<__gm__ int8_t *>(gm_a_src) + src_offset / 2,
                                                       reinterpret_cast<__gm__ int8_t *>(allgatherGM_) + dst_offset / 2,
                                                       s2_actual_m);
                        } else {
                            MoveResultFromPeerMemToOut(reinterpret_cast<__gm__ X1Type *>(gm_a_src) + src_offset,
                                                       reinterpret_cast<__gm__ X1Type *>(allgatherGM_) + dst_offset,
                                                       s2_actual_m);
                        }
                    }
                }
            }

            // dequant
            if (cal_idx >= MAX_BLOCK_COUNT) {
                Dequant(cal_idx - MAX_BLOCK_COUNT);
            }

            if (cal_idx < cal_count) {
                SetAndWaitAivSync(flag_idx);
                CrossRankSyncV2(FLAG_ONE_IDX, cal_idx + 1);
                SetAndWaitAivSync(flag_idx);
                // 发送aic同步
                SetAicSync(flag_idx);
            }
        }

        for (int32_t idx = 0; idx < num_flags; ++idx) {
            if (blockIdx == 0 && aivIdx == 0) {
                SetBuffFlag((__gm__ int32_t *)stateAddrPerRank[rankId] + FLAG_OFFSET + idx, uBuf_, 0);
            }
        }

        SetAndWaitAivSync(FLAG_ONE_IDX);

        if (blockIdx < worldSize && aivIdx == 1) {
            CheckBuffFlag((__gm__ int32_t *)stateAddrPerRank[blockIdx] + FLAG_OFFSET + FLAG_ZERO_IDX, uBuf_, 0);
        }

        PipeBarrier<PIPE_ALL>();
    }
    CatlassMatmul();

    SyncAll<false>();
}

} // namespace AllGatherMatmulAIVModeImpl
