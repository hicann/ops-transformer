/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RAIN_FUSION_ATTENTION_TILINGKEY_H_
#define RAIN_FUSION_ATTENTION_TILINGKEY_H_

/**
 * RainFusionAttention TilingKey 定义
 *
 * TilingKey编码规则 (64-bit):
 * 位域: AAAABBBBCCCCDDDDEEEE
 * - [0-1]   Q Layout: 2=TND, 3=BNSD
 * - [2-4]   Mask Type: 0=NoMask, 3=CausalMask
 * - [5-7]   Softmax Precision: 0=Float, 1=Half
 * - [8-10]  PagedCache Flag: 0=NoCache, 1=WithCache
 * - [11-13] KV Layout: 00=TND, 20=BNSD
 * - [14-15] Data Type: 00=FP16, 22=BF16
 * - [16-18] Operator Category: 900=RainFusionAttention
 */

#define RFA_BASE_TILING 9000000000000000

// FP16, Q=TND, KV=TND, No PagedCache, Float Softmax, No Mask
#define QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING 9000000030000002

// FP16, Q=TND, KV=TND, No PagedCache, Half Softmax, No Mask
#define QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_RFA_TILING 9000000030100002

// BF16, Q=TND, KV=TND, No PagedCache, Float Softmax, No Mask
#define QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_RFA_TILING 9000000030022222

// FP16, Q=BNSD, KV=BNSD, No PagedCache, Float Softmax, No Mask
#define QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING 9000000050000003

// FP16, Q=BNSD, KV=BNSD, No PagedCache, Half Softmax, No Mask
#define QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_RFA_TILING 9000000050100003

// BF16, Q=BNSD, KV=BNSD, No PagedCache, Float Softmax, No Mask
#define QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_RFA_TILING 9000000050022223

#endif // RAIN_FUSION_ATTENTION_TILINGKEY_H_
