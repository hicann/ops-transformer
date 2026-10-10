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
 * \file fa_kernel_interface.h
 * \brief Simplified FA kernel interface — no mask, no combine.
 */

#ifndef FA_KERNEL_INTERFACE_H
#define FA_KERNEL_INTERFACE_H

#include "kernel/fa_block_vec.h"
#include "kernel/fa_block_cube.h"

template <uint32_t M_BASE = 128, uint32_t N_BASE = 128, uint32_t D_SIZE = 128>
__global__ __aicore__ void FaKernel(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                    __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace, uint32_t blockDim,
                                    float softmaxScale, uint32_t B, uint32_t N1, uint32_t N2, uint32_t S1, uint32_t S2)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
#ifdef __DAV_C310_CUBE__
    CubeFunc<M_BASE, N_BASE, D_SIZE>(query, key, value, workspace, nullptr, blockDim, B, N1, N2, S1, S2);
#else
    VectorFunc<M_BASE, N_BASE, D_SIZE>(query, key, value, attentionOut, workspace, nullptr, blockDim, softmaxScale, B,
                                       N1, N2, S1, S2);
#endif
}

#endif // FA_KERNEL_INTERFACE_H
