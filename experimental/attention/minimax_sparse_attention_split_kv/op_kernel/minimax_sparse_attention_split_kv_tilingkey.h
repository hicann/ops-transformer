/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MINIMAX_SPARSE_ATTENTION_SPLIT_KV_TILINGKEY_H
#define MINIMAX_SPARSE_ATTENTION_SPLIT_KV_TILINGKEY_H

#include "kernel_tiling/kernel_tiling.h"

#define MINIMAX_SA_SPLIT_KV_BASE_TILING 20000

// innerPrecise==4 (default): bf16 softmax S + fp32 O_partial.
#define MINIMAX_SA_SPLIT_KV_BF16_D128_TILING 20001

// innerPrecise==1: bf16 softmax S + bf16 O_partial (PV fixpipe F322BF16 + Phase2 regbase cast).
#define MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_LOW_TILING 20002

// innerPrecise==0: fp32 softmax S (QK NoQuant) + fp32 O_partial.
#define MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_HIGH_TILING 20003

// FP8 Q/K/V (and quantized P); PV accum / O_partial FP32; attentionOut BF16.
// Softmax is the same BF16 online template as 20001; innerPrecise must be 4
// (FP32 O_partial). innerPrecise=0/1 are not implemented for FP8.
#define MINIMAX_SA_SPLIT_KV_FP8_D128_BF16_TILING 20004

#endif // MINIMAX_SPARSE_ATTENTION_SPLIT_KV_TILINGKEY_H
