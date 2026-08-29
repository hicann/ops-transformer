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
 * \brief QuantMatmulReduceScatter ubmem kernel launcher entry (<<<>>> direct launch).
 *
 * The HCCL context device pointer is obtained on the host side via
 * HcclAllocComResourceByTiling and passed as the first kernel parameter.
 *
 * Flow:
 *   AIC: Matmul via Blaze GemmUniversal (MIX kernel) with AlltoAll epilogue
 *   AIV: BlockEpilogueAlltoAll (scatter matmul output rows to remote ranks) -> ReduceAdd
 *
 * ND layout (TransB=true): cGM = y (matmul writes directly to y, then ReduceAdd overwrites y).
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "apace/block/epilogue/block_epilogue_all_to_all.h"
#include "apace/kernel/fusions/quant_matmul_reduce_scatter/quant_matmul_reduce_scatter_impl.h"

using namespace AscendC;
using namespace Blaze::Gemm;

__global__ __aicore__ void QuantMatmulReduceScatterUbmemKernel(GM_ADDR hcclContext, GM_ADDR x1, GM_ADDR x2,
                                                               GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y,
                                                               GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    using TypeA = fp8_e4m3fn_t;
    using TypeB = fp8_e4m3fn_t;
    using TypeC = bfloat16_t;
    using BiasType = float;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::NZLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;

    using DispatchPolicy = MatmulWithScaleMx<0, 0>;
    using BlockMmad =
        Block::BlockMmad<DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutC>;
    using BlockScheduler =
        Block::BlockSchedulerQuantBatchMatmulV3<AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>, 0, LayoutA,
                                                LayoutB, TypeA>;
    using BlockEpilogue = Block::BlockEpilogueAlltoAll<TypeC, LayoutC>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using Impl = Apace::QuantMatmulReduceScatterImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelImpl = typename Impl::KernelImpl;

    __gm__ QuantMatmulTilingData *tilingData = (__gm__ QuantMatmulTilingData *)tilingGM;

    const uint64_t M = tilingData->m;
    const uint64_t N = tilingData->n;
    const uint64_t K = tilingData->k;

    // NZ layout (TransB=false): matmul output goes to workspaceGM.
    // y is only [M/R, N], but matmul produces [M, N] in NZ layout which needs workspace.
    // The AlltoAll epilogue reads from workspaceGM and scatters to remote windows.
    // ReduceAdd reads from windows and writes the final result to y.
    GM_ADDR cGM = workspaceGM;

    typename Impl::Params params;
    params.matmulKernelParams.problemShape =
        ProblemShape{static_cast<int64_t>(M), static_cast<int64_t>(N), static_cast<int64_t>(K), 1L};
    params.matmulKernelParams.mmadParams = typename BlockMmad::Params{x1, x2, cGM, nullptr, x1Scale, x2Scale};
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
    params.matmulKernelParams.epilogueParams = typename BlockEpilogue::Params{cGM, M, N, 0, hcclContext};
    params.yGM = y;
    params.hcclContext = hcclContext;

    Impl impl;
    impl(params);
}
