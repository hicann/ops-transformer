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
 * \file compressor_grad.cpp
 * \brief CompressorGrad arch22 单入口 MIX kernel（KERNEL_TYPE_MIX_AIC_1_2，blockDim=aicNum）：
 *        Phase0 cast(AIV) -> Phase1 scatter(AIV) -> Phase1.5 dApe reduce(AIV，与 gemm 并行)
 *        -> Phase2 gemm(AIC)
 *        相位屏障用 AscendC::SyncAll<false>()：AIC+AIV 全核、同序调用
 *        tiling 侧须 SetScheduleMode(BATCH_MODE_SCHEDULE)
 */

#if (__CCE_AICORE__ == 220)
#include "arch22/compressor_grad_cast_scatter.h"
#include "arch22/compressor_grad_gemm.h"
#include "arch22/compressor_grad_template_tiling_key.h"

using namespace AscendC;

template <uint8_t D_TYPE>
__global__ __aicore__ void compressor_grad(__gm__ uint8_t* x, __gm__ uint8_t* wkv, __gm__ uint8_t* wgate,
                                           __gm__ uint8_t* dCmpKv, __gm__ uint8_t* softmaxScore, __gm__ uint8_t* kv,
                                           __gm__ uint8_t* cuSeqlens, __gm__ uint8_t* seqUsed, __gm__ uint8_t* startPos,
                                           __gm__ uint8_t* dX, __gm__ uint8_t* dWkv, __gm__ uint8_t* dWgate,
                                           __gm__ uint8_t* dApe, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    REGISTER_TILING_DEFAULT(optiling::CompressorGradTilingDataA3);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorGradTilingDataA3, tilingDataIn, tiling);
    const optiling::CompressorGradTilingDataA3* __restrict tilingData = &tilingDataIn;
    GM_ADDR userWs = GetUserWorkspace(workspace);
    // D_TYPE tiling key 编码与 arch35 保持一致: 0 = BF16, 1 = FP16
    using ioT = typename AscendC::Conditional<(D_TYPE == 1), half, bfloat16_t>::type;

    if ASCEND_IS_AIV {
        {
            TPipe pipe;
            CompressorGradCastKernel<ioT> castKernel;
            castKernel.Init(x, wkv, wgate, dApe, userWs, tilingData, &pipe);
            castKernel.Process();
        }
        // 相位屏障 1：cast/zero-fill 的 GM 写入对全核可见后才能做 scatter
        AscendC::SyncAll<false>();
        {
            TPipe pipe;
            CompressorGradScatterKernel<ioT> scatterKernel;
            scatterKernel.Init(dCmpKv, softmaxScore, kv, cuSeqlens, seqUsed, startPos, userWs, tilingData, &pipe);
            scatterKernel.Process();
        }
        // 相位屏障 2：dn/d_ape partial 全部写完后 AIC 才能开始 gemm
        AscendC::SyncAll<false>();
        // d_ape 确定性归约：AIV 归并与 AIC gemm 并行（d_ape 不参与 gemm）
        {
            TPipe pipe;
            CompressorGradApeReduceKernel reduceKernel;
            reduceKernel.Init(dApe, userWs, tilingData, &pipe);
            reduceKernel.Process();
        }
    }
    if ASCEND_IS_AIC {
        // 与 AIV 侧等次数的屏障调用，保证 ffts flag 计数对齐
        AscendC::SyncAll<false>();
        AscendC::SyncAll<false>();
        TPipe pipe;
        CompressorGradGemmKernel<ioT> gemmKernel;
        gemmKernel.Init(dX, dWkv, dWgate, userWs, tilingData, &pipe);
        gemmKernel.Process();
    }
}
#endif
