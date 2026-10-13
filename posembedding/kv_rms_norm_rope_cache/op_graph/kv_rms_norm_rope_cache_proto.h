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
 * \file kv_rms_norm_rope_cache_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_KV_RMS_NORM_ROPE_CACHE_H_
#define OPS_OP_PROTO_INC_KV_RMS_NORM_ROPE_CACHE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief The fusion operator of RMSNorm, RotaryPositionEmbedding and Update KVCache.
 *
 * @par Inputs:
 * @li kv: A tensor. The type supports float32, float16 and bfloat16. Its shape is [Bkv, N, Skv, D] or [Bkv, N, Skv, Dk]
 * when v is not None. Format: ND.
 * @li gamma: A tensor, used in RMS Norm. Its type is consistent with kv. Its shape is [Dv,]. Format: ND.
 * @li cos: A tensor, from position embedding. Its type is consistent with kv. Its shape is [Bkv, 1, Skv, Dk] or
 * [Bkv, 1, 1, Dk]. Format: ND.
 * @li sin: A tensor, from position embedding. Its type is consistent with kv. Its shape is [Bkv, 1, Skv, Dk] or
 * [Bkv, 1, 1, Dk]. The shape of sin should be consistent with that of cos. Format: ND.
 * @li index: A tensor. The type supports int64. When the cache_mode is Norm, its shape is [Bkv, Skv]. When the
 * cache_mode is PA_BNSD or PA_NZ, its shape is [Bkv * Skv]. When the cache_mode is PA_BLK_BNSD or PA_BLK_NZ, its shape
 * is [Bkv * ceil_div(Skv, BlockSize)]. Format: ND.
 * @li k_cache: A tensor. Its type must be either consistent with kv, or one of: int8, hifloat8, float8_e5m2,
 * float8_e4m3fn. When the cache_mode is in PA scenario(cache_mode is PA_BNSD, PA_NZ, PA_BLK_BNSD or PA_BLK_NZ), its
 * shape is [BlockNum, BlockSize, N, Dk]. When the cache_mode is Norm, its shape is [Bcache, N, Scache, Dk]. Format: ND.
 * @li ckv_cache: A tensor. Its type must be either consistent with kv, or one of: int8, hifloat8, float8_e5m2,
 * float8_e4m3fn. When the cache_mode is in PA scenario, its shape is [BlockNum, BlockSize, N, Dv]. When the cache_mode
 * is Norm, its shape is [Bcache, N, Scache, Dv]. Format: ND.
 * @li k_rope_scale: A tensor. The type supports float32. This input parameter is required when the data type of k_cache
 * is int8, hifloat8, float8_e5m2, float8_e4m3fn. Its shape can be [N, Dk], [Dk,] or [1,]. Format: ND.
 * @li c_kv_scale: A tensor. The type supports float32. This input parameter is required when the data type of ckv_cache
 * is int8, hifloat8, float8_e5m2, float8_e4m3fn. Its shape can be [N, Dv], [Dv,] or [1,]. Format: ND.
 * @li k_rope_offset: A tensor. The type supports float32. This parameter is required when k_cache is quantized (int8,
 * hifloat8, float8_e5m2, float8_e4m3fn) and asymmetric quantization is used. Its shape can be [N, Dk], [Dk,] or [1,].
 * Format: ND.
 * @li c_kv_offset: A tensor. The type supports float32. This parameter is required when ckv_cache is quantized (int8,
 * hifloat8, float8_e5m2, float8_e4m3fn) and asymmetric quantization is used. Its shape can be [N, Dv], [Dv,] or [1,].
 * Format: ND.
 * @li v: A tensor. Its type is consistent with kv. Its shape is [Bkv, N, Skv, Dv]. Format: ND.
 *
 * @par Outputs:
 * @li k_cache: A tensor. Its type must be either consistent with kv, or one of: int8, hifloat8, float8_e5m2,
 * float8_e4m3fn. When the cache_mode is in PA scenario, its shape is [BlockNum, BlockSize, N, Dk]. When the cache_mode
 * is Norm, its shape is [Bcache, N, Scache, Dk]. Format: ND.
 * @li ckv_cache: A tensor. Its type must be either consistent with kv, or one of: int8, hifloat8, float8_e5m2,
 * float8_e4m3fn. When the cache_mode is in PA scenario, its shape is [BlockNum, BlockSize, N, Dv]. When the cache_mode
 * is Norm, its shape is [Bcache, N, Scache, Dv]. Format: ND.
 * @li k_rope: A tensor. Its type is consistent with kv. If is_output_kv is true, this parameter output is required.
 * Its shape is [Bkv, N, Skv, Dk]. Format: ND.
 * @li c_kv: A tensor. Its type is consistent with kv. If is_output_kv is true, this parameter output is required.
 * Its shape is [Bkv, N, Skv, Dv]. Format: ND.
 *
 * @par Attributes:
 * @li epsilon: A float32. The epsilon value for RMSNorm. Default: 1e-5.
 * @li cache_mode: A string. The cache dtype. There are five types: Norm, PA_BNSD, PA_NZ, PA_BLK_BNSD and PA_BLK_NZ.
 * Default: Norm.
 * @li is_output_kv: A bool. Control whether k_rope and c_kv are output. If it is true, then k_rope and c_kv need to be
 * output. If it is false, then k_rope and c_kv do not need to be output. Default: false.
 *
 * @attention Constraints:
 * @li The shape format description in the parameter description:
 *     - Bkv represents the batch size of the input kv, and Skv represents the sequence length of the input kv. The
 *     size is determined by the user's input scenario and has no explicit restrictions.
 *     - N is the head number of the input kv. This operator is strongly related to the DeepSeekV3 network structure
 *     and only supports scenarios where N = 1. There are no scenarios where N is not 1.
 *     - D is the head dim of the input kv. The data Dv required for rms norm calculation and the data Dk required
 *     for rope calculation are split from the D of the input kv. Therefore, the sizes of Dk and Dv must satisfy
 *     Dk + Dv = D. Meanwhile, Dk must comply with the rope rules. According to the rope rule, Dk is an even number.
 *     If cache_mode is an NZ scenario(cache_mode is PA_NZ or PA_BLK_NZ), Dk and Dv need to be 32B aligned.
 *     - If cache_mode is a PA scenario, the BlockSize needs to be 32B aligned.
 *     - Regarding the above-mentioned 32B alignment situation, the alignment value is determined by the data type
 *     of the cache. Take BlockSize as an example. If the data type of the cache is int8, hifloat8, float8_e5m2,
 * float8_e4m3fn then BlockSize % 32 = 0 is required. If the data type of the cache is flaot16, then BlockSize % 16=0 is
 * required.
 *     - Bcache is the batch size of the input cache, and Scache is the sequence length of the input cache. The size
 *     is input by the user.
 *     - BlockNum is the number of memory blocks written to the cache, and its size is determined by the user's input
 *     scenario without any definite limit.
 * @li The index-related constraints:
 *     - When cache_mode is Norm, the shape is [Bkv, Skv], and the value range of index must be [-1, Scache).
 *     Under different Bkv. The values can be repeated.
 *     - When cache_mode is PA_BNSD or PA_NZ, the shape is [Bkv * Skv], and the value range of the index must be
 *     [-1, BlockNum * BlockSize). The values cannot be repeated.
 *     - When cache_mode is PA_BLK_BNSD and PA_BLK_NZ, the shape is [Bkv * ceil_div(Skv, BlockSize)]. The value
 *     range of index should be [-1, BlockNum * BlockSize). The values of value / BlockSize cannot be repeated.
 * @li Constraints related to the quantization scenario:
 *     -Quantized scenario supported case 1: The data type of k_cache is float16 or bfloat16, and the data type of
 * ckv_cache is int8, hifloat8, float8_e5m2, float8_e4m3fn. -Quantized scenario supported case 2: The data type of
 * ckv_cache is float16 or bfloat16, and the data type of k_cache is int8, hifloat8, float8_e5m2, float8_e4m3fn.
 *     -Quantized scenario supported case 3: The data types of k_cache and ckv_cache are the same, being int8, hifloat8,
 * float8_e5m2, float8_e4m3fn.
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */

REG_OP(KvRmsNormRopeCache)
    .INPUT(kv, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(cos, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(sin, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(index, TensorType({DT_INT64}))
    .INPUT(k_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN}))
    .INPUT(ckv_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN}))
    .OPTIONAL_INPUT(k_rope_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(c_kv_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(k_rope_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(c_kv_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(v, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(k_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN}))
    .OUTPUT(ckv_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN}))
    .OUTPUT(k_rope, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(c_kv, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-5)
    .ATTR(cache_mode, String, "Norm")
    .ATTR(is_output_kv, Bool, false)
    .OP_END_FACTORY_REG(KvRmsNormRopeCache)
} // namespace ge

#endif // OPS_OP_PROTO_INC_KV_RMS_NORM_ROPE_CACHE_H_
