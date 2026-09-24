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
 * \file gather_pa_kv_cache_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_GatherPaKvCache_H_
#define OPS_OP_PROTO_INC_GatherPaKvCache_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Gather the key and value from the corresponding cache based on the block tables and sequence lengths. \n
 *        This operation is typically used in Paged Attention (PA) to retrieve cached key-value pairs for
 variable-length sequences. \n

 * @par Inputs:
 * Inputs including:
 * @li key_cache: Key cache tensor. The format and the shape depends on cache_mode. \n
 *   - The Atlas A2 products
 *     and Atlas A3 products: The format is ND.
 *   - Ascend 950PR&950DT products: Supports the ND (cache_mode = "Norm") and FRACTAL_NZ (cache_mode = "PA_NZ") data
 formats.
 * @li value_cache: Value cache tensor. The format and the shape depends on cache_mode.
 *   - The Atlas A2 products
 *     and Atlas A3 products: The format is ND.
 *   - Ascend 950PR&950DT products: Supports the ND (cache_mode = "Norm") and FRACTAL_NZ (cache_mode = "PA_NZ") data
 formats.
 * @li block_tables: Tensor mapping logical blocks to physical blocks in each batch. \n
 *   - Data types: int32, int64.
 *   - format: ND
 * @li seq_lens: Tensor for sequence lengths for each batch.
 *   - Data types: int32, int64.
 *   - format: ND
 * @li key: Tensor for gathered key values. Shape depends on cache_mode.
 *   - The Atlas A2 products
 *     and Atlas A3 products: The format is ND.
 *   - Ascend 950PR&950DT products: The format is ND.
 * @li value: Tensor for gathered value values. Shape depends on cache_mode.
 *   - The Atlas A2 products
 *     and Atlas A3 products: The format is ND.
 *   - Ascend 950PR&950DT products: The format is ND.
 * @li seq_offset: Optional. Starting offset for each sequence in the block_tables. Shape: [batch]. \n
 *   - Data types: int32, int64.
 *   - format: ND
 *
 * @par Attributes:
 * @li cache_mode: An optional attribute describing the format of key_cache/value_cache. Valid values: "Norm" (default)
 or "PA_NZ".
 *
 * @li is_seq_lens_cumsum: An optional boolean attribute indicating whether `seq_lens` is provided as cumulative sum.
 Defaults to true.
 *   - If true, `seq_lens` has shape [batch + 1], where `seq_lens[i+1] - seq_lens[i]` gives the length of the i-th
 sequence.
 *   - If false, `seq_lens` has shape [batch], directly providing the length of each sequence.
 *
 * @par Outputs:
 * @li key: Gathered key values from the cache, which is same as the input key.
 * @li value: Gathered value values from the cache, which is same as the input value.
 *
 * @attention Constraints:
 * @code{.c}
 *  - key_cache/value_cache/key/value share the same data type.
 *  - block_tables/seq_lens/seq_offset share the same data type.
 *
 * Shape constraints in "Norm" mode:
 *   key_cache:     [num_blocks, block_size, num_heads, head_size_k]
 *   value_cache:   [num_blocks, block_size, num_heads, head_size_v]
 *   block_tables:  [batch, block_indices]
 *   seq_lens:      [batch] (if is_seq_lens_cumsum=false) or [batch + 1] (if is_seq_lens_cumsum=true)
 *   key:           [num_tokens, num_heads, head_size_k]
 *   value:         [num_tokens, num_heads, head_size_v]
 *   seq_offset:    [batch] (optional)
 *
 * Shape constraints in "PA_NZ" mode:
 *   key_cache:     [num_blocks, (num_heads * head_size_k) / elenum_aligned, block_size, elenum_aligned]
 *   value_cache:   [num_blocks, (num_heads * head_size_v) / elenum_aligned, block_size, elenum_aligned]
 *   block_tables:  [batch, block_indices]
 *   seq_lens:      [batch] or [batch + 1]
 *   key:           [num_tokens, num_heads * head_size_k]
 *   value:         [num_tokens, num_heads * head_size_v]
 *   seq_offset:    [batch] (optional)
 *
 * Special note for PA_NZ mode: enum_aligned has 3 possible values, which depend on the bit-width of
 key_cache/value_cache data type.
 *       b8  (8 bits = 1 byte): elenum_aligned = 32
 *       b16 (16 bits = 2 bytes): elenum_aligned = 16
 *       b32 (32 bits = 4 bytes): elenum_aligned = 8
 *
 * block_tables constraints: All elements in block_tables must be in range [0, num_blocks).
 * seq_lens constraints: All elements in seq_lens must be ≥ 0.
 *
 * @endcode
 */
REG_OP(GatherPaKvCache)
    .INPUT(key_cache, "T")
    .INPUT(value_cache, "T")
    .INPUT(block_tables, TensorType::IndexNumberType())
    .INPUT(seq_lens, TensorType::IndexNumberType())
    .INPUT(key, "T")
    .INPUT(value, "T")
    .OPTIONAL_INPUT(seq_offset, TensorType::IndexNumberType())
    .OUTPUT(key, "T")
    .OUTPUT(value, "T")
    .ATTR(cache_mode, String, "Norm")
    .ATTR(is_seq_lens_cumsum, Bool, true)
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                             DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(GatherPaKvCache)

} // namespace ge

#endif // OPS_OP_PROTO_INC_GatherPaKvCache_H_
