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
 * \file scatter_pa_kv_cache_with_k_scale_proto.h
 * \brief scatter_pa_kv_cache_with_k_scale operator prototype definition
 */

#ifndef SCATTER_PA_KV_CACHE_WITH_K_SCALE_PROTO_H_
#define SCATTER_PA_KV_CACHE_WITH_K_SCALE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Scatter KV cache with key scale. Updates key, value, and key_scale into corresponding caches. \n

 * @par Inputs:
 * 7 input tensors:
 * @li key: Key values to be updated, FP8 format. Shape is [num_tokens, num_head, k_head_size].
 *          Supported data types: DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN.
 * @li value: Value values to be updated, FP8 format. Shape is [num_tokens, num_head, v_head_size].
 *            Supported data types: DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN.
 * @li key_cache: Cache to be updated by key, FP8 format. Shape is [num_blocks, num_head, block_size, k_head_size].
 *                Supported data types: DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN.
 * @li value_cache: Cache to be updated by value, FP8 format. Shape is [num_blocks, num_head, block_size, v_head_size].
 *                  Supported data types: DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN.
 * @li slot_mapping: Offset of each token in the cache. Shape is [num_tokens].
 *                   Supported data types: DT_INT32 or DT_INT64.
 * @li key_scale: Key scaling factor to be updated, FP32 format. Shape is [num_tokens, num_head].
 *                Supported data types: float32.
 * @li key_scale_cache: Cache to be updated by key_scale, FP32 format.
 *                      Shape is [num_blocks, num_head, block_size, 1].
 *                      Supported data types: float32.
 *
 * @par Attributes:
 * @li cache_layout: Optional attribute, describes the cache layout format. Default value is "BNBD".
 *                   Current version only supports "BNBD" layout.
 *
 * @par Outputs:
 * 3 output tensors (inplace update):
 * @li key_cache: Updated key_cache, same shape and data type as input key_cache.
 * @li value_cache: Updated value_cache, same shape and data type as input value_cache.
 * @li key_scale_cache: Updated key_scale_cache, same shape and data type as input key_scale_cache.
 *
 * @attention Constraints:
 * @code{.c}
 * - key, value, key_cache, and value_cache must have the same data type.
 * - key_scale and key_scale_cache must be of float32 type.
 * - slot_mapping must be of DT_INT32 or DT_INT64 type.
 * - Values in slot_mapping must be in the range [0, num_blocks * block_size - 1].
 * - num_tokens <= num_blocks * block_size.
 * @endcode
 */

REG_OP(ScatterPaKvCacheWithKScale)
    .INPUT(key, "T")
    .INPUT(value, "T")
    .INPUT(key_cache, "T")
    .INPUT(value_cache, "T")
    .INPUT(slot_mapping, TensorType::IndexNumberType())
    .INPUT(key_scale, TensorType({DT_FLOAT}))
    .INPUT(key_scale_cache, TensorType({DT_FLOAT}))
    .OUTPUT(key_cache, "T")
    .OUTPUT(value_cache, "T")
    .OUTPUT(key_scale_cache, TensorType({DT_FLOAT}))
    .ATTR(cache_layout, String, "BNBD")
    .DATATYPE(T, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(ScatterPaKvCacheWithKScale)

} // namespace ge
#endif // SCATTER_PA_KV_CACHE_WITH_K_SCALE_PROTO_H_
