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
 * \file grouped_matmul_activation_quant_mxquant.h
 * \brief
 */

#ifndef GROUPED_MATMUL_ACTIVATION_QUANT_MXQUANT_H
#define GROUPED_MATMUL_ACTIVATION_QUANT_MXQUANT_H

#include "blaze/epilogue/block/block_epilogue_gelu_tanh_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx_activation_quant.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "grouped_matmul_activation_quant_tiling_data.h"

namespace {
constexpr int64_t DEFAULT_PROBLEM_SHAPE_VALUE = 1;

template <typename T>
__aicore__ inline __gm__ T *GetFirstTensorAddr(GM_ADDR tensorListAddr)
{
    AscendC::GlobalTensor<uint64_t> tensorList;
    tensorList.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(tensorListAddr));
    const int64_t addressTableOffset = static_cast<int64_t>(tensorList.GetValue(0));
    return reinterpret_cast<__gm__ T *>(tensorList.GetValue(addressTableOffset >> 3));
}
} // namespace

namespace GroupedMatmulActivationQuant {
template <typename layoutA, typename layoutB>
__aicore__ inline void GmmActivationMxQuant(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                            GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace,
                                            GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA_WITH_STRUCT(GMMActivationQuantTilingDataParams, tilingData, tiling);
    const auto &gmmActivationQuantParams_ = tilingData.gmmActivationQuantParams;
    const auto &mmTilingData_ = tilingData.mmTilingData;
    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using OutputType = DTYPE_Y;
    using MmadCType = float;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockMmadPolicy =
        Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGroupedMmadWithScaleMxActivationQuant>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, MmadCType, LayoutC,
                                                    BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGeluTanhMxQuant<OutputType, MmadCType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using QGmmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    GMMTiling gmmParams{gmmActivationQuantParams_.groupNum,
                        mmTilingData_.m,
                        mmTilingData_.n,
                        mmTilingData_.k,
                        mmTilingData_.baseM,
                        mmTilingData_.baseN,
                        mmTilingData_.baseK,
                        mmTilingData_.kAL1,
                        mmTilingData_.kBL1,
                        mmTilingData_.scaleKAL1,
                        mmTilingData_.scaleKBL1,
                        mmTilingData_.isBias,
                        mmTilingData_.dbL0C,
                        static_cast<uint8_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT),
                        static_cast<int8_t>(0),
                        static_cast<uint8_t>(gmmActivationQuantParams_.groupListType),
                        static_cast<uint8_t>(1)};
    GM_ADDR weightData = reinterpret_cast<GM_ADDR>(GetFirstTensorAddr<BType>(weight));
    GM_ADDR weightScaleData = reinterpret_cast<GM_ADDR>(GetFirstTensorAddr<AscendC::fp8_e8m0_t>(weightScale));
    Params params = {{DEFAULT_PROBLEM_SHAPE_VALUE, DEFAULT_PROBLEM_SHAPE_VALUE, DEFAULT_PROBLEM_SHAPE_VALUE,
                      DEFAULT_PROBLEM_SHAPE_VALUE},
                     {x, weightData, y, nullptr, xScale, weightScaleData},
                     {y, yScale, static_cast<uint32_t>(mmTilingData_.baseM), static_cast<uint32_t>(mmTilingData_.baseN),
                      static_cast<uint32_t>(gmmActivationQuantParams_.scaleAlg), gmmActivationQuantParams_.dstTypeMax},
                     groupList,
                     gmmParams};
    QGmmKernel op;
    op(params);
}
} // namespace GroupedMatmulActivationQuant

#endif
