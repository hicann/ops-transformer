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
 * \file kernel_launcher.h
 * \brief MatmulAllToAll MX kernel launcher 入口（<<<>>> direct launch，非模板 __global__ 入口）
 */

#pragma once

#include "apace/kernel/fusions/quant_matmul_all_to_all/mx_quant_matmul_all_to_all_urma_impl.h"

#define DECLARE_MATMUL_A2A_MX_KERNEL(name, typeA, typeB, typeC, localDelay) \
    __global__ __aicore__ void name(__gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, \
                                    GM_ADDR scaleBGM, GM_ADDR biasGM, GM_ADDR cGM, \
                                    MatmulAllToAllTilingData tilingData) \
    { \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1); \
        Apace::MatmulAllToAllMxImpl<typeA, typeB, typeC, localDelay> impl; \
        impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, biasGM, cGM, &tilingData); \
        impl.Run(); \
    }

DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE2M1E2M1Bf16Delay, fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t, true)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE2M1E2M1Fp16Delay, fp4x2_e2m1_t, fp4x2_e2m1_t, half, true)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE2M1E2M1Bf16NoDelay, fp4x2_e2m1_t, fp4x2_e2m1_t, bfloat16_t, false)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE2M1E2M1Fp16NoDelay, fp4x2_e2m1_t, fp4x2_e2m1_t, half, false)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE4M3E4M3Bf16Delay, fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, true)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE4M3E4M3Fp16Delay, fp8_e4m3fn_t, fp8_e4m3fn_t, half, true)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE4M3E4M3Bf16NoDelay, fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE4M3E4M3Fp16NoDelay, fp8_e4m3fn_t, fp8_e4m3fn_t, half, false)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE5M2E5M2Bf16Delay, fp8_e5m2_t, fp8_e5m2_t, bfloat16_t, true)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE5M2E5M2Fp16Delay, fp8_e5m2_t, fp8_e5m2_t, half, true)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE5M2E5M2Bf16NoDelay, fp8_e5m2_t, fp8_e5m2_t, bfloat16_t, false)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE5M2E5M2Fp16NoDelay, fp8_e5m2_t, fp8_e5m2_t, half, false)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE4M3E5M2Bf16Delay, fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, true)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE4M3E5M2Fp16Delay, fp8_e4m3fn_t, fp8_e5m2_t, half, true)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE4M3E5M2Bf16NoDelay, fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, false)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE4M3E5M2Fp16NoDelay, fp8_e4m3fn_t, fp8_e5m2_t, half, false)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE5M2E4M3Bf16Delay, fp8_e5m2_t, fp8_e4m3fn_t, bfloat16_t, true)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE5M2E4M3Fp16Delay, fp8_e5m2_t, fp8_e4m3fn_t, half, true)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE5M2E4M3Bf16NoDelay, fp8_e5m2_t, fp8_e4m3fn_t, bfloat16_t, false)
DECLARE_MATMUL_A2A_MX_KERNEL(MatmulAllToAllMxKernelE5M2E4M3Fp16NoDelay, fp8_e5m2_t, fp8_e4m3fn_t, half, false)

#undef DECLARE_MATMUL_A2A_MX_KERNEL
