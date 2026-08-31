/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_TILINGKEY_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_TILINGKEY_H

#include "kernel_tiling/kernel_tiling.h"

/**
 * GBSA TilingKey — BSA-style decimal bitfields (64-bit):
 * - [0-1]   Q Layout（个位）           2=TND, 3=BNSD, 4=BSND
 * - [2-4]   Mask Type（千位）         0=NoMask, 1=maskType1 (current), 3=Causal(BSA)
 * - [5-7]   Softmax Precision（十万位） 0=Float, 1=Half
 * - [8-10]  PagedCache Flag（千万位） 0=NoCache, 1=WithCache
 * - [11-13] KV Layout（十亿位）        30=TND, 50=BNSD, 60=BSND, 70=PAGED_BBND
 * - [14-15] Data Type（百亿位附近）   00=FP16, 22=BF16 (via +22220); FP8 out: +10/+20
 * - LSE（亿位）                        +100000000
 * - [16-18] Op+Arch（千万亿位）       920=GBSA/aicore220, 925=GBSA/aicore310
 *
 * Stage1 path: Q=TND, KV=PAGED_BBND, paged=1, maskType=1.
 */

#define GBSA_OP_ARCH22_BASE 9200000000000000ULL
#define GBSA_OP_ARCH35_BASE 9250000000000000ULL

#if (__CCE_AICORE__ == 220)
#define GBSA_BASE_TILING GBSA_OP_ARCH22_BASE

// FP16, TND, PAGED_BBND+cache, Float SM, maskType=1
#define GBSA_FP16_TND_PAGED_BBND_TILING 9200000071001002ULL
// FP16, Half SM
#define GBSA_FP16_TND_PAGED_BBND_HALFSM_TILING 9200000071101002ULL
// BF16, Float SM (bf16+half SM rejected by host)
#define GBSA_BF16_TND_PAGED_BBND_TILING 9200000071023222ULL

#define GBSA_FP16_TND_PAGED_BBND_TILING_LSE_OUT 9200000171001002ULL
#define GBSA_FP16_TND_PAGED_BBND_HALFSM_TILING_LSE_OUT 9200000171101002ULL
#define GBSA_BF16_TND_PAGED_BBND_TILING_LSE_OUT 9200000171023222ULL

#elif (__CCE_AICORE__ == 310)
#define GBSA_BASE_TILING GBSA_OP_ARCH35_BASE

// Arch35 kernel does not branch on SoftmaxPrecision in the key (host still validates).
#define GBSA_FP16_TND_PAGED_BBND_TILING 9250000071001002ULL
#define GBSA_BF16_TND_PAGED_BBND_TILING 9250000071023222ULL
#define GBSA_FP8_TND_PAGED_BBND_TILING 9250000071001012ULL
#define GBSA_FP8_TND_PAGED_BBND_BF16_TILING 9250000071001022ULL

#define GBSA_FP16_TND_PAGED_BBND_TILING_LSE_OUT 9250000171001002ULL
#define GBSA_BF16_TND_PAGED_BBND_TILING_LSE_OUT 9250000171023222ULL

#endif

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_TILINGKEY_H
