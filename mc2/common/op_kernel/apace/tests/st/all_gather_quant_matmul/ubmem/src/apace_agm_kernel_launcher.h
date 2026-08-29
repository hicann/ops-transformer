/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_launcher.h
 * \brief AllGatherQuantMatmul ubmem kernel launcher entry (<<<>>> direct launch).
 *
 * The HCCL context device pointer is obtained on the host side via
 * HcclAllocComResourceByTiling and passed as the first kernel parameter.
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_quant_matmul_ubmem_impl.h"
#include "apace/block/mmad/qmm_mx_block_mmad_tile_k_wait_flag.h"

using namespace AscendC;
using namespace Blaze::Gemm;

template <bool isX2Nz>
__aicore__ inline void AllGatherQuantMatmulUbmemKernelImpl(GM_ADDR hcclContext, GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale,
                                                           GM_ADDR x2Scale, GM_ADDR output, GM_ADDR allGatherDataOut,
                                                           GM_ADDR allGatherScalesOut, GM_ADDR tilingGM)
{
    using TypeA = fp8_e4m3fn_t;
    using TypeB = fp8_e4m3fn_t;
    using TypeC = bfloat16_t;
    using BiasType = float;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB =
        typename AscendC::Std::conditional_t<!isX2Nz, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;

    using DispatchPolicy = MatmulWithScaleMx<0, false>;
    using BlockMmad =
        Block::BlockMmad<DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutC>;
    using BlockScheduler =
        Block::BlockSchedulerQuantBatchMatmulV3<AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>, 0, LayoutA,
                                                LayoutB, TypeA>;
    using BlockEpilogue = Block::BlockEpilogueEmpty;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using Impl = Apace::AllGatherQuantMatmulUbmemImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelImpl = typename Impl::KernelImpl;

    __gm__ QuantMatmulTilingData *tilingData = (__gm__ QuantMatmulTilingData *)tilingGM;

    typename Impl::Params params;
    params.matmulKernelParams.problemShape =
        ProblemShape{static_cast<int64_t>(tilingData->m), static_cast<int64_t>(tilingData->n),
                     static_cast<int64_t>(tilingData->k), 1L};
    params.matmulKernelParams.mmadParams = typename BlockMmad::Params{nullptr, x2, output, nullptr, nullptr, x2Scale};
    params.matmulKernelParams.l1Params = typename BlockMmad::L1Params{
        static_cast<uint64_t>(tilingData->stepK) * tilingData->baseK, tilingData->scaleKL1, 2};
    params.matmulKernelParams.schParams = typename BlockScheduler::Params{tilingData->baseM,
                                                                          tilingData->baseN,
                                                                          tilingData->mTailTile,
                                                                          tilingData->nTailTile,
                                                                          tilingData->mBaseTailSplitCnt,
                                                                          tilingData->nBaseTailSplitCnt,
                                                                          tilingData->mTailMain,
                                                                          tilingData->nTailMain};
    params.matmulKernelParams.qbmmParams = typename KernelImpl::QBMMTiling{1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           1,
                                                                           0,
                                                                           tilingData->baseM,
                                                                           tilingData->baseN,
                                                                           tilingData->baseK,
                                                                           0,
                                                                           tilingData->dbL0c};
    params.hcclContext = hcclContext;
    params.tilingGM = tilingGM;
    params.x1Addr = x1;
    params.x1ScaleAddr = x1Scale;
    params.allGatherDataOutAddr = allGatherDataOut;
    params.allGatherScalesOutAddr = allGatherScalesOut;

    Impl impl;
    impl(params);
}

__global__ __aicore__ void AllGatherQuantMatmulUbmemKernelDN(GM_ADDR hcclContext, GM_ADDR x1, GM_ADDR x2,
                                                             GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR output,
                                                             GM_ADDR allGatherDataOut, GM_ADDR allGatherScalesOut,
                                                             GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AllGatherQuantMatmulUbmemKernelImpl<false>(hcclContext, x1, x2, x1Scale, x2Scale, output, allGatherDataOut,
                                               allGatherScalesOut, tilingGM);
}

__global__ __aicore__ void AllGatherQuantMatmulUbmemKernelNZ(GM_ADDR hcclContext, GM_ADDR x1, GM_ADDR x2,
                                                             GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR output,
                                                             GM_ADDR allGatherDataOut, GM_ADDR allGatherScalesOut,
                                                             GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AllGatherQuantMatmulUbmemKernelImpl<true>(hcclContext, x1, x2, x1Scale, x2Scale, output, allGatherDataOut,
                                              allGatherScalesOut, tilingGM);
}
