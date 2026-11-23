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
 * \file apace_mrs_urma_kernel_launcher.h
 * \brief quantMatmulReduceScatter kernel launcher 入口 (PUT 模式)
 *
 * A 为 ND [M,K], B 为 DN [N,K], 输出 C 为 ND [m/rankNum,n] BF16.
 * ScaleA/ScaleB 的形状分别为 [m,ceil(K/64),2] 和 [n,ceil(K/64),2], 类型为 E8M0,
 * 每个 scale 覆盖 32 个 K 元素.
 * workspaceGM 为 [M,N] BF16 缓冲区, 保存本 rank 的 C_i; AIV 将各 M 分片 PUT 到目标 rank win.
 * 支持 4 种 FP8 A/B 组合和 MXFP4 E2M1E2M1.
 */

#pragma once

#include "apace/kernel/fusions/matmul_reduce_scatter/quant_matmul_reduce_scatter_urma_impl.h"

// MIX kernel 的 AIC:AIV 配比为 1:2
#define QMRS_CV_RATIO KERNEL_TYPE_MIX_AIC_1_2

#define DECLARE_QMRS_KERNEL(name, typeA, typeB) \
    __global__ __aicore__ void name(__gm__ CommContext* hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, \
                                    GM_ADDR scaleBGM, GM_ADDR workspaceGM, GM_ADDR cGM, \
                                    QuantMatmulReduceScatterTilingData tilingData) \
    { \
        KERNEL_TASK_TYPE_DEFAULT(QMRS_CV_RATIO); \
        Apace::QuantMatmulReduceScatterUrmaImpl<typeA, typeB, bfloat16_t> impl; \
        impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, workspaceGM, cGM, &tilingData); \
        impl.Process(); \
    }

DECLARE_QMRS_KERNEL(QuantMatmulReduceScatterKernelE4M3E4M3, fp8_e4m3fn_t, fp8_e4m3fn_t)
DECLARE_QMRS_KERNEL(QuantMatmulReduceScatterKernelE5M2E5M2, fp8_e5m2_t, fp8_e5m2_t)
DECLARE_QMRS_KERNEL(QuantMatmulReduceScatterKernelE4M3E5M2, fp8_e4m3fn_t, fp8_e5m2_t)
DECLARE_QMRS_KERNEL(QuantMatmulReduceScatterKernelE5M2E4M3, fp8_e5m2_t, fp8_e4m3fn_t)
// MXFP4 E2M1: A/B 均使用 fp4x2_e2m1_t, 每字节打包两个元素
DECLARE_QMRS_KERNEL(QuantMatmulReduceScatterKernelE2M1E2M1, fp4x2_e2m1_t, fp4x2_e2m1_t)
#undef DECLARE_QMRS_KERNEL
