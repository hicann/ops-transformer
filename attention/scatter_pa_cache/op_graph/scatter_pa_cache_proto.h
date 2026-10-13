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
 * \file scatter_pa_cache_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Update the key to the corresponding cache based on the index. \n

* @par Inputs:
* Inputs including:
* @li key: Key values to be updated, A 3D or 4D tensor. Must be one of the following types: float16, float, bfloat16,
int8, uint8,
*          int16, uint16, int32, uint32, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, float4_e1m2. When key is a
4D tensor,
*          its data type must not be hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1 or float4_e1m2.
* @li key_cache: Tensors that need to be updated by key. Data type must be the same as key.
* @li slot_mapping: The offset of each token of key in the cache. Must be one of the following types: int32 or int64.
* @li compress_lens: Compression amount. Data type must be the same as slot_mapping.
* @li compress_seq_offset: Compression starting point for each batch and head. Data type must be the same as
slot_mapping.
* @li seq_lens: Actual seq_len for each batch. Data type must be the same as slot_mapping.

* @par Attributes:
* cache_mode: An optional attribute. Reserved parameter, not effective now. Describing the format of cache. Defaults to
"Norm".

* @par Outputs:
* @li key_cache: Tensors that need to be updated by key. \n
* @attention Constraints:
* @code{.c}
  The dtype of key and key_cache must be same.
  When key is a 3D tensor, shape and value limit like:
    shape limit:
      key:[batch * seqLen, num_head, k_head_size]
      key_cache:[num_blocks, block_size, num_head, k_head_size]
      slot_mapping:[batch * seqLen]
    value limit:
      slot_mapping in range [0, num_blocks * block_size - 1]
      batch * seqLen <= num_blocks * block_size
      value in slot_mapping must be unique

  When key is a 4D tensor and compress_seq_offset is not None, shape and value limit like:
    shape limit:
      key:[batch, seqLen, num_head, k_head_size]
      key_cache:[num_blocks, block_size, 1, k_head_size]
      slot_mapping:[batch, num_head]
      compress_lens: [batch, num_head]
      compress_seq_offset:[batch*num_head]
      seq_lens:[batch]
    value limit:
      slot_mapping in range [0, num_blocks * block_size - 1]
      reduceSum(seqLen - compress_lens + 1) <= num_blocks * block_size
      compress_lens[b][i] + compress_seq_offset[b*i] < seq_lens[b]
      seq_lens[b] < seqLen
      0 < compress_lens[b][i] <= seq_lens[b] - compress_seq_offset[b*i]
      0 <= compress_seq_offset[b*i] < seq_lens[b] - compress_seq_offset[b*i]
      value in slot_mapping must be unique

  When key is a 4D tensor and compress_seq_offset is None, shape and value limit like:
    shape limit:
      key:[batch, seqLen, num_head, k_head_size]
      key_cache:[num_blocks, block_size, 1, k_head_size]
      slot_mapping:[batch, num_head]
      compress_lens: [batch * num_head]
      seq_lens:[batch]
    value limit:
      slot_mapping in range [0, num_blocks * block_size - 1]
      reduceSum(compress_lens) <= num_blocks * block_size
      compress_lens[b][i] < seq_lens[b]
      seq_lens[b] <= seqLen
      compress_lens[b][i] > 0
      value in slot_mapping must be unique

* @endcode
*/
REG_OP(ScatterPaCache)
    .INPUT(key, "T")
    .INPUT(key_cache, "T")
    .INPUT(slot_mapping, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(compress_lens, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(compress_seq_offset, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(seq_lens, TensorType::IndexNumberType())
    .OUTPUT(key_cache, "T")
    .ATTR(cache_mode, String, "Norm")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                             DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .OP_END_FACTORY_REG(ScatterPaCache)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
