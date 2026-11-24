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
 * \file apace_a2a_kernel_launcher.h
 * \brief AllToAllMatmul UDMA kernel launcher 入口
 *
 * 量化/非量化共用同一个 urma_impl；
 * 差异仅体现在模板实例化的 QuantFlag 与 Init 参数。
 */

#pragma once

#include "apace/kernel/fusions/all_to_all_matmul/all_to_all_matmul_urma_impl.h"

// ---------------- 非量化（fp16/bf16）----------------

__global__ __aicore__ void AllToAllMmKernel_Udma(__gm__ CommContext* hcommCtx, GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                                 GM_ADDR biasGM, AllToAllMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMatmulUrmaImpl<half, half, half, half, false, true, Apace::QuantFlag::NON_QUANT> impl;
    impl.Init(hcommCtx, aGM, nullptr, bGM, nullptr, cGM, biasGM, nullptr, &tilingData);
    impl.Run();
}

__global__ __aicore__ void AllToAllMmKernel_Bf16_Udma(__gm__ CommContext* hcommCtx, GM_ADDR aGM, GM_ADDR bGM,
                                                      GM_ADDR cGM, GM_ADDR biasGM, AllToAllMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMatmulUrmaImpl<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true,
                                  Apace::QuantFlag::NON_QUANT>
        impl;
    impl.Init(hcommCtx, aGM, nullptr, bGM, nullptr, cGM, biasGM, nullptr, &tilingData);
    impl.Run();
}

// ---------------- 量化（fp8 E4M3/E5M2 组合）----------------

__global__ __aicore__ void AllToAllQuantMatmulKernelE4M3E4M3_Udma(__gm__ CommContext* hcommCtx, GM_ADDR aGM,
                                                                  GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM,
                                                                  GM_ADDR cGM, AllToAllQuantMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    // 模板参数 AType, BType, CType, BiasType, TransA, TransB, QuantFlag
    Apace::AllToAllMatmulUrmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, float, false, true,
                                  Apace::QuantFlag::MX_QUANT>
        impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, nullptr, nullptr, &tilingData);
    impl.Run();
}

__global__ __aicore__ void AllToAllQuantMatmulKernelE5M2E5M2_Udma(__gm__ CommContext* hcommCtx, GM_ADDR aGM,
                                                                  GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM,
                                                                  GM_ADDR cGM, AllToAllQuantMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMatmulUrmaImpl<fp8_e5m2_t, fp8_e5m2_t, bfloat16_t, float, false, true, Apace::QuantFlag::MX_QUANT>
        impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, nullptr, nullptr, &tilingData);
    impl.Run();
}

__global__ __aicore__ void AllToAllQuantMatmulKernelE4M3E5M2_Udma(__gm__ CommContext* hcommCtx, GM_ADDR aGM,
                                                                  GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM,
                                                                  GM_ADDR cGM, AllToAllQuantMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMatmulUrmaImpl<fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, float, false, true, Apace::QuantFlag::MX_QUANT>
        impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, nullptr, nullptr, &tilingData);
    impl.Run();
}

__global__ __aicore__ void AllToAllQuantMatmulKernelE5M2E4M3_Udma(__gm__ CommContext* hcommCtx, GM_ADDR aGM,
                                                                  GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM,
                                                                  GM_ADDR cGM, AllToAllQuantMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMatmulUrmaImpl<fp8_e5m2_t, fp8_e4m3fn_t, bfloat16_t, float, false, true, Apace::QuantFlag::MX_QUANT>
        impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, nullptr, nullptr, &tilingData);
    impl.Run();
}
