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
 * \file rope_quant_kvcache.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "rope_quant_kvcache.h"
using namespace AscendC;
using namespace RopeQuantKvcacheND;

extern "C" __global__ __aicore__ void rope_quant_kvcache(GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR quant_scale,
                                                         GM_ADDR quant_offset, GM_ADDR k_cache, GM_ADDR v_cache,
                                                         GM_ADDR indice, GM_ADDR q_out, GM_ADDR k_out, GM_ADDR v_out,
                                                         GM_ADDR k_cache_out, GM_ADDR v_cache_out, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    RopeQuantKvcache op(&tilingData);

    op.Init(qkv, cos, sin, quant_scale, quant_offset, k_cache, v_cache, indice, q_out, k_cache_out, v_cache_out);
    op.Process();
}
