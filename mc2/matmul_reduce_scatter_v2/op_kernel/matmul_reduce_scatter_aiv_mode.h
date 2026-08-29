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
 * \file matmul_reduce_scatter_aiv_mode.h
 * \brief
 */

#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_reduce_scatter_v2_aiv_mode_tiling.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "matmul_reduce_scatter_aiv_mode_util.h"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_arch.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_mmad.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/tla_gemm_block_swizzle.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_dispatch_policy.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tla_gemm_gemm_type.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "matmul_reduce_scatter_v2_matmul.hpp"
#include "matmul_reduce_scatter_aiv_mode_padding.h"
#include "matmul_reduce_scatter_aiv_mode_dequant.h"

using namespace Catlass;
using namespace AscendC;
using namespace matmulReduceScatterV2_aivmode_tiling;
using namespace matmulReduceScatterV2_util;
using namespace dequant;
using namespace padding;
namespace MatmulReduceScatterV2Impl {

// MMA2A : MatmulAllToAll
#define TemplateMMReduceScatterV2Class \
    typename AType, typename BType, typename biasType, typename x2ScaleType, typename cType, bool hasBias, \
        bool weight_nz, bool TA, bool TB
#define TemplateMMReduceScatterV2Func AType, BType, biasType, x2ScaleType, cType, hasBias, weight_nz, TA, TB

template <TemplateMMReduceScatterV2Class, typename Derived>
class MatmulReduceScatterAivMode : public CommBase {
public:
    __aicore__ inline MatmulReduceScatterAivMode(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR perTokenScale,
                                GM_ADDR perChannelScale, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

protected:
    static constexpr bool quantFlag = (std::is_same<AType, int8_t>::value) && (std::is_same<BType, int8_t>::value);
    static constexpr bool needCastBias = !quantFlag && hasBias && std::is_same_v<AType, bfloat16_t> &&
                                         std::is_same_v<BType, bfloat16_t> && std::is_same_v<biasType, bfloat16_t>;
    __aicore__ inline void Padding();
    __aicore__ inline void StartBeforeFirstStep();
    __aicore__ inline void EndFirstStep();

private:
    __aicore__ inline void AIVInit();
    __aicore__ inline void AICInit();
    __aicore__ inline void CatlassMatmul();
    __aicore__ inline void AIVComm();
    __aicore__ inline void Dequant(int32_t calIdx, uint64_t flagIdx);
    __aicore__ inline void FirstStepInOut(int32_t actual_move_size, __gm__ cType *input, int32_t move_offset,
                                          int32_t loop_idx_st);
    __aicore__ inline void FirstStepInOutWithSplit(int32_t rank_data_offset, int32_t loop_idx_st,
                                                   int32_t data_loop_idx);

protected:
    int32_t m_align;
    int64_t k_align;
    int32_t n_align;
    int32_t core_loop;
    int32_t cal_count;

    __gm__ cType *gm_peer_mem;
    __gm__ AType *gm_a_src;
    __gm__ BType *gm_b_src;
    __gm__ int32_t *gm_accum;

private:
    bool aligned_a;
    bool aligned_b;
    int32_t len_per_rank;
    int32_t pingpong_idx;
    int32_t tile_block_size;

    GM_ADDR gm_a_align;
    GM_ADDR gm_b_align;
    DequantRunner<biasType, cType> dequant_runner;
    Arch::Resource<Arch::AtlasA2> resource;
};

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR perTokenScale, GM_ADDR perChannelScale, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulReduceScatterV2AivModeTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasSrcGM_ = biasGM;
    perChannelScaleGM_ = perChannelScale;
    perTokenScaleGM_ = perTokenScale;
    CommBase::SetArgs(tilingData);
    biasCastGM_ = needCastBias ? workspaceGM + aAlignSize + bAlignSize + dequantSize : biasGM;
    biasGM_ = needCastBias ? biasCastGM_ : biasSrcGM_;
    hasBAlign = weight_nz ? false : hasBAlign;
    gm_a_align = reinterpret_cast<GM_ADDR>(hasAAlign ? workspaceGM : 0);
    gm_b_align = reinterpret_cast<GM_ADDR>(hasBAlign ? workspaceGM + aAlignSize : 0);
    gm_accum = reinterpret_cast<__gm__ int32_t *>(quantFlag ? workspaceGM + aAlignSize + bAlignSize : 0);
    gm_a_src = reinterpret_cast<__gm__ AType *>(hasAAlign ? gm_a_align : aGM_);
    gm_b_src = reinterpret_cast<__gm__ BType *>(hasBAlign ? gm_b_align : bGM_);

    if constexpr (!std::is_same_v<Derived, void>) {
        m_loop = CeilDev(m, m0);
    }

    m_align = Block512B<AType>::AlignUp(m);
    k_align = Block512B<AType>::AlignUp(k);
    n_align = Block512B<AType>::AlignUp(n);
    aligned_a = hasAAlign;
    aligned_b = hasBAlign;
    MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::AICInit();
    MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::AIVInit();
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::AICInit()
{
    if ASCEND_IS_AIC {
        SetLoadDataPaddingValue(0);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        gm_peer_mem = reinterpret_cast<__gm__ cType *>(buff[rank]);
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::AIVInit()
{
    if ASCEND_IS_AIV {
        SetAtomicNone();
        SetMaskNormImpl();
        SetVectorMask<int32_t>((uint64_t)-1, (uint64_t)-1);

        max_ub_ping_pong_size = max_ub_single_dma_size / 2; // double buffer的方式
        max_ub_ping_pong_size = max_ub_ping_pong_size / n0 * n0;
        core_loop = m_loop * n_loop;
        cal_count = (core_loop + loop_num_per_comm - 1) / loop_num_per_comm;
        tile_block_size = m0 * n0;
        pingpong_idx = 0;
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        bool need_fixpipe = quantFlag && isX2ScaleTypeInt64 && std::is_same<cType, half>::value;
        int32_t peer_mem_m = m0 * loop_num_per_comm * MAX_BLOCK_COUNT;
        uint32_t layout_b_row = (TB && !weight_nz) ? static_cast<uint32_t>(k_align) : static_cast<uint32_t>(k);
        uint32_t layout_b_col = (TB || weight_nz) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(n_align);
        using ArchTag = Arch::AtlasA2;
        constexpr bool ENABLE_UNIT_FLAG = false;
        constexpr bool ENABLE_SHUFFLE_K = true;
        constexpr bool aicCalBias = !quantFlag && hasBias; // 如果计算量化后的矩阵乘，bias不由CatlassMatmul负责
        using ElementA = AType;
        using ElementB = BType;
        using ElementC = typename std::conditional<quantFlag, int32_t, cType>::type;
        using ElementBias = std::conditional_t<needCastBias, float, biasType>;

        using LayoutA = typename std::conditional<TA, layout::ColumnMajor, layout::RowMajor>::type;
        using LayoutC = layout::RowMajor;
        using LayoutScale = layout::VectorLayout;
        using LayoutBias = layout::VectorLayout;

        LayoutA layoutA{TA ? static_cast<uint32_t>(m_align) : static_cast<uint32_t>(m),
                        TA ? static_cast<uint32_t>(k) : static_cast<uint32_t>(k_align)};
        LayoutC layoutC{static_cast<uint32_t>(m / rank_size), static_cast<uint32_t>(n)};
        LayoutC layoutPeerMem{static_cast<uint32_t>(peer_mem_m), static_cast<uint32_t>(n0)};
        LayoutScale layoutScale{static_cast<uint32_t>(n)};
        LayoutBias layoutBias{static_cast<uint32_t>(n)};
        GemmCoord processSize{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

        constexpr int32_t L1TileShapeK = quantFlag ? TILE_SHAPE_512 : TILE_SHAPE_256;
        constexpr int32_t L0TileShapeK = quantFlag ? TILE_SHAPE_128 : TILE_SHAPE_64;
        using DispatchPolicy = std::conditional_t<aicCalBias, Gemm::MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG>,
                                                  Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>>;
        using AType_ = Gemm::GemmType<ElementA, LayoutA>;
        using CType_ = Gemm::GemmType<ElementC, LayoutC>;
        using BiasType_ = std::conditional_t<aicCalBias, Gemm::GemmType<ElementBias, LayoutBias>, void>;
        if (weight_nz) {
            // B矩阵NZ格式
            using LayoutNZ = typename std::conditional<TB, layout::nZ, layout::zN>::type;
            using BType_ = Gemm::GemmType<ElementB, LayoutNZ>;
            LayoutNZ layoutBNZ = LayoutNZ::template MakeLayout<ElementB>(layout_b_row, layout_b_col);

            struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_> {
                using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_>;
                using ElementA = typename Base::ElementA;
                using ElementB = typename Base::ElementB;
                using ElementAccumulator = typename Base::ElementAccumulator;
                using CopyGmToL1A = typename Base::CopyGmToL1A;
                using CopyGmToL1B = typename Base::CopyGmToL1B;

                using CopyL1ToL0A = typename Base::CopyL1ToL0A;
                using CopyL1ToL0B = typename Base::CopyL1ToL0B;

                using CopyL0CToGm = typename Base::CopyL0CToGm;
                using BiasTypeSelector = typename Base::BiasTypeSelector;
                using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
                using CopyL1ToBT = typename Base::CopyL1ToBT;
            };
            using TileCopy = TileCopyOpt;

            if (m0 == TILE_SHAPE_128) {
                using L1TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_,
                                                            CType_, BiasType_, TileCopy>;
                using MatmulKernel =
                    Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt, void, void, aicCalBias>;
                typename MatmulKernel::Params params{processSize,
                                                     reinterpret_cast<GM_ADDR>(gm_a_src),
                                                     layoutA,
                                                     reinterpret_cast<GM_ADDR>(gm_b_src),
                                                     layoutBNZ,
                                                     biasGM_,
                                                     reinterpret_cast<GM_ADDR>(cGM_),
                                                     layoutC,
                                                     reinterpret_cast<GM_ADDR>(perChannelScaleGM_),
                                                     layoutScale,
                                                     reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                     layoutPeerMem,
                                                     reinterpret_cast<GM_ADDR>(gm_accum),
                                                     p_value,
                                                     swizzl_count,
                                                     swizzl_direct,
                                                     dequant_type,
                                                     rank,
                                                     rank_size,
                                                     need_fixpipe};
                MatmulKernel matmul_op;
                matmul_op(params);
            } else {
                using L1TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_,
                                                            CType_, BiasType_, TileCopy>;
                using MatmulKernel =
                    Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt, void, void, aicCalBias>;
                typename MatmulKernel::Params params{processSize,
                                                     reinterpret_cast<GM_ADDR>(gm_a_src),
                                                     layoutA,
                                                     reinterpret_cast<GM_ADDR>(gm_b_src),
                                                     layoutBNZ,
                                                     biasGM_,
                                                     reinterpret_cast<GM_ADDR>(cGM_),
                                                     layoutC,
                                                     reinterpret_cast<GM_ADDR>(perChannelScaleGM_),
                                                     layoutScale,
                                                     reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                     layoutPeerMem,
                                                     reinterpret_cast<GM_ADDR>(gm_accum),
                                                     p_value,
                                                     swizzl_count,
                                                     swizzl_direct,
                                                     dequant_type,
                                                     rank,
                                                     rank_size,
                                                     need_fixpipe};
                MatmulKernel matmul_op;
                matmul_op(params);
            }
        } else {
            // B矩阵ND格式
            using LayoutB = typename std::conditional<TB, layout::ColumnMajor, layout::RowMajor>::type;
            LayoutB layoutB{layout_b_row, layout_b_col};
            using BType_ = Gemm::GemmType<ElementB, LayoutB>;

            struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_> {
                using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_>;
                using ElementA = typename Base::ElementA;
                using ElementB = typename Base::ElementB;
                using ElementAccumulator = typename Base::ElementAccumulator;

                // When matrix A is row-major, if the number of rows in matrix A is less than 16,
                // using the CopyGmToL1IntervalDataCopy method can improve the transfer efficiency.
                // The situation is similar for matrix B. If the above conditions are met,
                // please uncomment the following and comment out the original matrix A transfer method

                using CopyGmToL1A = typename Base::CopyGmToL1A;
                using CopyGmToL1B = typename Base::CopyGmToL1B;
                using CopyL1ToL0A = typename Base::CopyL1ToL0A;
                using CopyL1ToL0B = typename Base::CopyL1ToL0B;
                using CopyL0CToGm = typename Base::CopyL0CToGm;
                using BiasTypeSelector = typename Base::BiasTypeSelector;
                using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
                using CopyL1ToBT = typename Base::CopyL1ToBT;
            };
            using TileCopy = TileCopyOpt;
            if (m0 == TILE_SHAPE_128) {
                using L1TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_128, TILE_SHAPE_256, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_,
                                                            CType_, BiasType_, TileCopy>;
                using MatmulKernel =
                    Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt, void, void, aicCalBias>;
                typename MatmulKernel::Params colMajorParams{processSize,
                                                             reinterpret_cast<GM_ADDR>(gm_a_src),
                                                             layoutA,
                                                             reinterpret_cast<GM_ADDR>(gm_b_src),
                                                             layoutB,
                                                             reinterpret_cast<GM_ADDR>(biasGM_),
                                                             reinterpret_cast<GM_ADDR>(cGM_),
                                                             layoutC,
                                                             reinterpret_cast<GM_ADDR>(perChannelScaleGM_),
                                                             layoutScale,
                                                             reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                             layoutPeerMem,
                                                             reinterpret_cast<GM_ADDR>(gm_accum),
                                                             p_value,
                                                             swizzl_count,
                                                             swizzl_direct,
                                                             dequant_type,
                                                             rank,
                                                             rank_size,
                                                             need_fixpipe};
                MatmulKernel colMajorMatmulOp;
                colMajorMatmulOp(colMajorParams);
            } else {
                using L1TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L1TileShapeK>; // m n k
                using L0TileShape = GemmShape<TILE_SHAPE_256, TILE_SHAPE_128, L0TileShapeK>;
                using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_,
                                                            CType_, BiasType_, TileCopy>;
                using MatmulKernel =
                    Gemm::Kernel::MatmulReduceScatterAivMode<void, void, BlockMmadOpt, void, void, aicCalBias>;
                typename MatmulKernel::Params colMajorParams{processSize,
                                                             reinterpret_cast<GM_ADDR>(gm_a_src),
                                                             layoutA,
                                                             reinterpret_cast<GM_ADDR>(gm_b_src),
                                                             layoutB,
                                                             reinterpret_cast<GM_ADDR>(biasGM_),
                                                             reinterpret_cast<GM_ADDR>(cGM_),
                                                             layoutC,
                                                             reinterpret_cast<GM_ADDR>(perChannelScaleGM_),
                                                             layoutScale,
                                                             reinterpret_cast<GM_ADDR>(gm_peer_mem),
                                                             layoutPeerMem,
                                                             reinterpret_cast<GM_ADDR>(gm_accum),
                                                             p_value,
                                                             swizzl_count,
                                                             swizzl_direct,
                                                             dequant_type,
                                                             rank,
                                                             rank_size,
                                                             need_fixpipe};
                MatmulKernel colMajorMatmulOp;
                colMajorMatmulOp(colMajorParams);
            }
        }
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::Dequant(int32_t calIdx,
                                                                                                   uint64_t flagIdx)
{
    // 仅在x2ScaleType为INT64,输出为FP16时走fixPipe，其余场景均需要AIV做反量化
    bool needPerChannel = quantFlag && !(isX2ScaleTypeInt64 && std::is_same<cType, float16_t>::value);
    // 可简化为只校验dequant_type；host侧校验，只有输入为quant场景，dequantType才能传有效值。
    bool needPerToken = quantFlag && dequant_type == DequantType::PER_TOKEN;
    if (!needPerChannel && !needPerToken) {
        return;
    }

    uint32_t pingpongSt = flagIdx * gm_c_pingpong_size;

    uint32_t rowNum = m;
    uint32_t colNum = n;
    uint32_t tileM0 = m0;
    uint32_t tileN0 = n0;
    uint32_t coreIdx = core_idx;
    uint32_t coreNum = core_num;
    uint32_t rankIdx = rank;
    uint32_t rankSize = rank_size;
    uint32_t pValue = p_value;
    uint32_t swizzlDirect = swizzl_direct;
    uint32_t swizzlCount = swizzl_count;
    __gm__ float32_t *perChannelScale =
        needPerChannel ? reinterpret_cast<__gm__ float32_t *>(perChannelScaleGM_) : nullptr;
    __gm__ float32_t *perTokenScale = needPerToken ? reinterpret_cast<__gm__ float32_t *>(perTokenScaleGM_) : nullptr;
    __gm__ int32_t *workspace = needPerChannel ? reinterpret_cast<__gm__ int32_t *>(gm_accum) + pingpongSt : nullptr;
    __gm__ cType *peerMem = reinterpret_cast<__gm__ cType *>(buff[rank]) + pingpongSt;
    __gm__ cType *output = reinterpret_cast<__gm__ cType *>(cGM_);
    __gm__ biasType *biasptr = reinterpret_cast<__gm__ biasType *>(biasGM_);

    dequant_runner.RunMatmulReduceScatter(DEQUANT_ARGS_CALL());
    SetAndWaitAivSync(flagIdx);
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::Padding()
{
    if (!aligned_a && !aligned_b && !needCastBias) {
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreFlag flagAivFinishPadding{AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID};
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        return;
    }
    bool transA = TA; // 当前暂未支持A矩阵转置
    bool transB = TB;
    bool alignedA = aligned_a;
    bool alignedB = aligned_b;
    uint32_t matrixAM = m;
    uint32_t matrixAK = k;
    uint32_t matrixBK = k;
    uint32_t matrixBN = n;
    uint32_t matrixAMAlign = static_cast<uint32_t>(m_align);
    uint32_t matrixAKAlign = static_cast<uint32_t>(k_align);
    uint32_t matrixBKAlign = static_cast<uint32_t>(k_align);
    uint32_t matrixBNAlign = static_cast<uint32_t>(n_align);
    GM_ADDR gmA = reinterpret_cast<GM_ADDR>(aGM_);
    GM_ADDR gmB = reinterpret_cast<GM_ADDR>(bGM_);
    GM_ADDR gmAAlign = reinterpret_cast<GM_ADDR>(gm_a_align);
    GM_ADDR gmBAlign = reinterpret_cast<GM_ADDR>(gm_b_align);
    bool castBias = needCastBias;
    uint32_t biasLength = n;
    GM_ADDR gmBias = biasSrcGM_;
    GM_ADDR gmBiasCast = biasCastGM_;
    PaddingRunner<AType, BType> padding_runner;
    padding_runner.Run(PADDING_ARGS_CALL());
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::StartBeforeFirstStep()
{
    SetAtomicAdd<cType>();
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::EndFirstStep()
{
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2等MTE3
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2等MTE3
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);     // Scalar等MTE3
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SetAtomicNone();
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::FirstStepInOut(
    int32_t actual_move_size, __gm__ cType *input, int32_t move_offset, int32_t loop_idx_st)
{
    auto ub_offset = USED_UB_SIZE / 2 / sizeof(cType);
    LocalTensor<cType> ubTensor = uBuf_.AllocTensor<cType>();
    LocalTensor<cType> copyTensor0 = ubTensor;
    LocalTensor<cType> copyTensor1 = ubTensor[ub_offset];

    auto event_id = (pingpong_idx & 1) ? EVENT_ID0 : EVENT_ID1;
    auto ub_buff_st = (pingpong_idx & 1) ? copyTensor0 : copyTensor1;
    WaitFlag<HardEvent::MTE3_MTE2>(event_id);
    // 读的matrix是多个小的m0*n0块顺序排布，写的时候需要重排
    CopyGmToUbuf(ub_buff_st, input, 1, actual_move_size * sizeof(cType) / 32, 0, 0);
    SetFlag<HardEvent::MTE2_MTE3>(event_id);
    WaitFlag<HardEvent::MTE2_MTE3>(event_id);

    int32_t move_num_offset = move_offset;
    auto ub_buff = ub_buff_st;
    int64_t ub_buff_offset = 0;
    int32_t left_m = actual_move_size / n0;
    while (left_m > 0) {
        int32_t loop_idx = loop_idx_st + (move_num_offset / tile_block_size) * rank_size;
        int64_t batch_idx = loop_idx / (m_loop * n_loop);
        int32_t in_batch_idx = loop_idx % (m_loop * n_loop);
        int32_t in_rank_idx = in_batch_idx / rank_size;
        int64_t m_idx, n_idx;
        GetSwizzledBlockIdx(in_rank_idx, m_loop / rank_size, n_loop, swizzl_direct, swizzl_count, m_idx, n_idx);
        int32_t actual_m = (m_idx == (m_loop / rank_size - 1)) ? (m / rank_size - m_idx * m0) : m0;
        int32_t actual_n = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
        int32_t m_offset = (move_num_offset % tile_block_size) / n0; // 当前一块起点对应的m，在当前块的位置
        int32_t actual_move_m = m0 < m_offset + left_m ? m0 - m_offset : left_m;
        // m0 - m_offset表示当前块剩下的一小段，跳过；
        if (m_offset < actual_m) {
            actual_move_m = actual_m < m_offset + left_m ? actual_m - m_offset : left_m;
            // left_m较大，则该块copy完，下次再copy下一块；
            // left_m较小，则只copy left_m的部分
            int64_t out_buff_offset =
                batch_idx * m * n / rank_size + (static_cast<int64_t>(m_idx) * m0 + m_offset) * n + n_idx * n0;
            CopyUbufToGmUnknown(nAlign16, reinterpret_cast<__gm__ cType *>(cGM_) + out_buff_offset,
                                ub_buff[ub_buff_offset], actual_move_m, actual_n * sizeof(cType),
                                (n0 - actual_n) * sizeof(cType) / 32, (n - actual_n) * sizeof(cType));
        }
        left_m -= actual_move_m;
        move_num_offset += actual_move_m * n0;
        ub_buff_offset += actual_move_m * n0;
    }
    SetFlag<HardEvent::MTE3_MTE2>(event_id);
    pingpong_idx = 1 - pingpong_idx;
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::FirstStepInOutWithSplit(
    int32_t rank_data_offset, int32_t loop_idx_st, int32_t data_loop_idx)
{
    int32_t rank_per_core = rank_size / comm_npu_split;
    // data_loop_idx：data方向的第几个loop；
    int32_t data_offset = (data_loop_idx * comm_data_split + core_idx % comm_data_split) * max_ub_ping_pong_size;
    int32_t core_rank_offset = (core_idx / comm_data_split) * rank_per_core;
    int32_t rank_buff_offset = rank_data_offset + data_offset;

    int32_t actual_move_num =
        (data_offset >= len_per_rank) ?
            0 :
            ((data_offset + max_ub_ping_pong_size) > len_per_rank ? len_per_rank - data_offset : max_ub_ping_pong_size);
    for (int32_t rank_idx = 0; rank_idx < rank_per_core; rank_idx++) {
        int32_t real_rank_idx = core_rank_offset + (rank_idx + core_idx) % rank_per_core;
        if (real_rank_idx == rank) {
            continue;
        }
        FirstStepInOut(actual_move_num, reinterpret_cast<__gm__ cType *>(buff[real_rank_idx]) + rank_buff_offset,
                       data_offset, loop_idx_st);
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::AIVComm()
{
    if ASCEND_IS_AIV {
        Padding();

        ResetIpcFlags(2);
        PipeBarrier<PIPE_ALL>();

        int32_t gm_c_block_size = gm_c_pingpong_size / rank_size;
        for (int32_t cal_idx = 0; cal_idx < cal_count; ++cal_idx) {
            uint64_t flag_idx = cal_idx % MAX_BLOCK_COUNT;
            int32_t loop_st = cal_idx * loop_num_per_comm;
            int32_t actual_loop_num = cal_idx == cal_count - 1 ? core_loop - loop_st : loop_num_per_comm;

            len_per_rank = actual_loop_num * tile_block_size / rank_size;
            // wait aic
            WaitEvent(flag_idx);

            SetAndWaitAivSync(flag_idx);
            Dequant(cal_idx, flag_idx);
            CrossRankSyncV1(FLAG_ZERO_IDX, cal_idx + 1);
            SetAndWaitAivSync(flag_idx);

            if (aiv_idx == 0 && core_idx < comm_npu_split * comm_data_split) {
                StartBeforeFirstStep();
                // 3层循环从外到内：data_loops*comm_data_split*max_ub_pingpong_size
                int32_t data_loops = DivCeil(len_per_rank, max_ub_ping_pong_size * comm_data_split);
                int32_t rank_offset = flag_idx * gm_c_pingpong_size + rank * gm_c_block_size;
                for (int32_t loop_idx = 0; loop_idx < data_loops; loop_idx++) {
                    FirstStepInOutWithSplit(rank_offset, loop_st, loop_idx);
                }
                EndFirstStep();
            }

            SetAndWaitAivSync(flag_idx);
            CrossRankSyncV2(FLAG_ONE_IDX, cal_idx + 1);
            SetAndWaitAivSync(flag_idx);
            if (cal_idx < cal_count - MAX_BLOCK_COUNT) { // 最后两次不进行同步
                // 发送aic同步
                SetAicSync(flag_idx);
            }
        }
        ResetIpcFlags(1);
        if (aiv_idx == 1 && core_idx < rank_size) {
            CheckBuffFlag((__gm__ int32_t *)buff[other_rank] + FLAG_OFFSET + FLAG_ZERO_IDX, 0);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

template <TemplateMMReduceScatterV2Class, typename Derived>
__aicore__ inline void MatmulReduceScatterAivMode<TemplateMMReduceScatterV2Func, Derived>::Process()
{
    if constexpr (std::is_same_v<Derived, void>) {
        CatlassMatmul();
        AIVComm();
    } else {
        static_cast<Derived *>(this)->CatlassMatmul();
        static_cast<Derived *>(this)->AIVComm();
    }
}
} // namespace MatmulReduceScatterV2Impl
#endif // MATMUL_REDUCE_SCATTER_V2_H
