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
 * dtype 泛化（对齐 experimental/mc2/quant_matmul_reduce_scatter）：
 *   TypeA/TypeB: fp8_e4m3fn_t / fp8_e5m2_t / fp4x2_e2m1_t（fp8 可异构，fp4 仅 fp4+fp4）
 *   TypeC: bfloat16_t / half（y dtype）
 *   LayoutB: DNExt(转置 B [N,K]) / NDExt(非转置 [K,N]) / ZN(转置 NZ) / NZ(非转置 NZ)
 *
 * Flow:
 *   AIC: Matmul via Mc2QbmmMxUniversal (MIX kernel) with AlltoAll epilogue
 *   AIV: BlockEpilogueAlltoAll (scatter matmul output rows to remote ranks) -> ReduceAdd
 *
 * 统一 cGM = workspaceGM（[M, N] 缓冲）：DN/ND 连续布局与 ZN/NZ 分形布局均先落 workspace，
 * 再由 AlltoAll epilogue 分发、ReduceAdd 写最终 y[M/rankNum, N]。
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling_data.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "apace/block/epilogue/block_epilogue_all_to_all.h"
#include "apace/kernel/fusions/quant_matmul_reduce_scatter/quant_matmul_reduce_scatter_impl.h"
// 注：不包含 blaze/gemm/kernel/kernel_universal.h——新版 ops-tensor 会经它引入
// stock kernel_qbmm_mx.h 的 GemmUniversal 特化，与 apace 的 Mc2QbmmMxUniversal 冲突

using namespace AscendC;
using namespace Blaze::Gemm;

// dtype / 布局 枚举（host dispatch 用）
enum class STXDtype : int {
    kE4M3 = 0,
    kE5M2 = 1,
    kFP4 = 2
};
enum class STYDtype : int {
    kBF16 = 0,
    kFP16 = 1
};
enum class STX2Layout : int {
    kDN = 0,
    kND = 1,
    kZN = 2,
    kNZ = 3
};

template <typename TypeA, typename TypeB, typename TypeC, typename LayoutB>
__global__ __aicore__ void QuantMatmulReduceScatterUbmemKernel(GM_ADDR context, GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale,
                                                               GM_ADDR x2Scale, GM_ADDR y, GM_ADDR workspaceGM,
                                                               GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    using BiasType = float;
    using MatmulOutType = float; // L0C->UB fixpipe 落地类型（对齐 op 入口，DualDst 直写 UB 的 FP32 槽）
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;

    // L0C2UB_MODE_DUAL_DST_SPLIT_M：L0C 结果经 fixpipe 直写两个 AIV 子核的 UB（M 对半），
    // 不落本卡 GM；BlockEpilogueAlltoAll 从 UB 的 FP32 槽 cast 到 TypeC 后直发远端窗口（对齐 op 入口）
    using DispatchPolicy = MatmulWithScaleMx<0, 0, KernelMmadWithScaleMx, L0C2UB_MODE_DUAL_DST_SPLIT_M>;
    using BlockMmad =
        Block::BlockMmad<DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, MatmulOutType, LayoutC, BiasType, LayoutC>;
    using BlockScheduler = Block::BlockSchedulerQuantBatchMatmulV3<asc::te::shape<int64_t, int64_t, int64_t, int64_t>,
                                                                   0, LayoutA, LayoutB, TypeA>;
    using BlockEpilogue = Block::BlockEpilogueAlltoAll<TypeC, LayoutC>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using Impl = Apace::QuantMatmulReduceScatterImpl<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelImpl = typename Impl::KernelImpl;

    __gm__ QuantMatmulTilingData* tilingData = (__gm__ QuantMatmulTilingData*)tilingGM;

    const uint64_t M = tilingData->m;
    const uint64_t N = tilingData->n;
    const uint64_t K = tilingData->k;

    // DualDst 路径：matmul 输出经 fixpipe 直写 UB 的 FP32 槽，BlockEpilogueAlltoAll 从 UB cast 后分发，
    // 不落本卡 GM workspace（workspaceGM 保留分配但 kernel 不使用）
    (void)workspaceGM;

    typename Impl::Params params;
    params.matmulKernelParams.problemShape =
        ProblemShape{static_cast<int64_t>(M), static_cast<int64_t>(N), static_cast<int64_t>(K), 1L};
    params.matmulKernelParams.mmadParams = typename BlockMmad::Params{x1, x2, nullptr, nullptr, x1Scale, x2Scale};
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
    params.matmulKernelParams.epilogueParams = typename BlockEpilogue::Params{M, N, 0, context};
    params.yGM = y;
    params.ubmemContext = context;

    Impl impl;
    impl(params);
}

// ─────────────────────────────────────────────────────────────────────
// host dispatch：按 (x1dtype, x2dtype, ydtype, x2layout) 实例化模板 kernel
// ─────────────────────────────────────────────────────────────────────
template <typename TypeA, typename TypeB, typename TypeC, typename LayoutB>
inline void LaunchQmrsKernelTmpl(uint32_t blockDim, aclrtStream stream, GM_ADDR ctx, GM_ADDR x1, GM_ADDR x2, GM_ADDR sA,
                                 GM_ADDR sB, GM_ADDR y, GM_ADDR ws, GM_ADDR tiling)
{
    QuantMatmulReduceScatterUbmemKernel<TypeA, TypeB, TypeC, LayoutB>
        <<<blockDim, nullptr, stream>>>(ctx, x1, x2, sA, sB, y, ws, tiling);
}

template <typename TypeA, typename TypeB, typename TypeC>
inline void DispatchLayout(STX2Layout ly, uint32_t blockDim, aclrtStream stream, GM_ADDR ctx, GM_ADDR x1, GM_ADDR x2,
                           GM_ADDR sA, GM_ADDR sB, GM_ADDR y, GM_ADDR ws, GM_ADDR tiling)
{
    constexpr bool kIsFp4 = Std::is_same_v<TypeA, fp4x2_e2m1_t> || Std::is_same_v<TypeB, fp4x2_e2m1_t>;
    if (ly == STX2Layout::kDN) {
        LaunchQmrsKernelTmpl<TypeA, TypeB, TypeC, asc::te::dn_ext_layout_ptn>(blockDim, stream, ctx, x1, x2, sA, sB, y,
                                                                              ws, tiling);
    } else if (ly == STX2Layout::kND) {
        LaunchQmrsKernelTmpl<TypeA, TypeB, TypeC, asc::te::nd_ext_layout_ptn>(blockDim, stream, ctx, x1, x2, sA, sB, y,
                                                                              ws, tiling);
    } else if (ly == STX2Layout::kZN) {
        // fp4 不支持 ZN/NZ（ParseArgs 已拦截），用 if constexpr 避免实例化非法 mmad
        if constexpr (!kIsFp4) {
            LaunchQmrsKernelTmpl<TypeA, TypeB, TypeC, asc::te::zn_layout_ptn>(blockDim, stream, ctx, x1, x2, sA, sB, y,
                                                                              ws, tiling);
        }
    } else {
        if constexpr (!kIsFp4) {
            LaunchQmrsKernelTmpl<TypeA, TypeB, TypeC, asc::te::nz_layout_ptn>(blockDim, stream, ctx, x1, x2, sA, sB, y,
                                                                              ws, tiling);
        }
    }
}

template <typename TypeA, typename TypeB>
inline void DispatchY(STYDtype yt, STX2Layout ly, uint32_t blockDim, aclrtStream stream, GM_ADDR ctx, GM_ADDR x1,
                      GM_ADDR x2, GM_ADDR sA, GM_ADDR sB, GM_ADDR y, GM_ADDR ws, GM_ADDR tiling)
{
    if (yt == STYDtype::kBF16) {
        DispatchLayout<TypeA, TypeB, bfloat16_t>(ly, blockDim, stream, ctx, x1, x2, sA, sB, y, ws, tiling);
    } else {
        DispatchLayout<TypeA, TypeB, half>(ly, blockDim, stream, ctx, x1, x2, sA, sB, y, ws, tiling);
    }
}

template <typename TypeA>
inline void DispatchB(STXDtype xb, STYDtype yt, STX2Layout ly, uint32_t blockDim, aclrtStream stream, GM_ADDR ctx,
                      GM_ADDR x1, GM_ADDR x2, GM_ADDR sA, GM_ADDR sB, GM_ADDR y, GM_ADDR ws, GM_ADDR tiling)
{
    if (xb == STXDtype::kE4M3) {
        if constexpr (!Std::is_same_v<TypeA, fp4x2_e2m1_t>) {
            DispatchY<TypeA, fp8_e4m3fn_t>(yt, ly, blockDim, stream, ctx, x1, x2, sA, sB, y, ws, tiling);
        }
    } else if (xb == STXDtype::kE5M2) {
        if constexpr (!Std::is_same_v<TypeA, fp4x2_e2m1_t>) {
            DispatchY<TypeA, fp8_e5m2_t>(yt, ly, blockDim, stream, ctx, x1, x2, sA, sB, y, ws, tiling);
        }
    } else {
        // fp4 仅支持 fp4+fp4（TypeA 必须为 fp4）；非法组合已在 host ParseArgs 拦截，
        // 用 if constexpr 避免实例化非法 mmad（fp8×fp4）
        if constexpr (Std::is_same_v<TypeA, fp4x2_e2m1_t>) {
            DispatchY<TypeA, fp4x2_e2m1_t>(yt, ly, blockDim, stream, ctx, x1, x2, sA, sB, y, ws, tiling);
        }
    }
}

inline void LaunchQmrsKernel(STXDtype xa, STXDtype xb, STYDtype yt, STX2Layout ly, uint32_t blockDim,
                             aclrtStream stream, GM_ADDR ctx, GM_ADDR x1, GM_ADDR x2, GM_ADDR sA, GM_ADDR sB, GM_ADDR y,
                             GM_ADDR ws, GM_ADDR tiling)
{
    if (xa == STXDtype::kE4M3) {
        DispatchB<fp8_e4m3fn_t>(xb, yt, ly, blockDim, stream, ctx, x1, x2, sA, sB, y, ws, tiling);
    } else if (xa == STXDtype::kE5M2) {
        DispatchB<fp8_e5m2_t>(xb, yt, ly, blockDim, stream, ctx, x1, x2, sA, sB, y, ws, tiling);
    } else {
        DispatchB<fp4x2_e2m1_t>(xb, yt, ly, blockDim, stream, ctx, x1, x2, sA, sB, y, ws, tiling);
    }
}
