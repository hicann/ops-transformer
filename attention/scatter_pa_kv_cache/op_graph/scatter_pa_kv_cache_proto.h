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
 * \file scatter_pa_kv_cache_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Update the key and value to the corresponding cache based on the index. \n

* @par Inputs:
* Inputs including:
* @li key: Key values to be updated,A 3D or 4D tensor.
* @li key_cache: Tensors that need to be updated by key.
* @li slot_mapping: The offset of each token, key, or value in the cache. Must be one of the following types: int32 or
* int64.
* @li value: Key values to be updated.
* @li value_cache: Tensors that need to be updated by value.
* @li compress_lens: Compression amount. Must be one of the following types: int32 or int64.
* @li compress_seq_offset: Compression starting point for each batch and head. Must be one of the following types: int32
* or int64.
* @li seq_lens: Actual seq_len for each batch. Must be one of the following types: int32 or int64.

* @par Attributes:
* @li cache_mode: An optional attribute. Describing the format of cache. Defaults to "Norm".
* This attribute field is only applicable to the Atlas A2 products,
* Atlas A3 products, and Ascend 950PR&950DT products.
* other products can only take the default value.
- If "PA_NZ"(Paged Attention NZ Format), the format of key_cache and value_cache is NZ.
- If "Norm" or None, the format of key_cache and value_cache is ND.
* @li scatter_mode: An optional attribute. Describing the format of cache. Defaults to "None".
* This attribute field is only applicable to the Atlas A2 products,
* Atlas A3 products, and Ascend 950PR&950DT products.
* other products can only take the default value.
- If "Alibi", key and value will compress by alibi mode.
- If "Rope", key and value will compress by Rope mode.
- If "Omni", key and value will compress by Omni mode.
- If "Nct", key and value will be uncontiguous.
- If "None" or None, key and value will update normally.
* @li strides: An optional attribute. A list of 2 integers. The stride of the key and value,
* its' shape is [stride_k, stride_v].
* This attribute field is only applicable to the Atlas A2 products,
* Atlas A3 products, and Ascend 950PR&950DT products.
* other products can only take the default value.
* @li offsets: An optional attribute. A list of 2 integers. The offsets of the key and value,
* its' shape is [offset_k, offset_v].
* This attribute field is only applicable to the Atlas A2 products,
* Atlas A3 products, and Ascend 950PR&950DT products.
* other products can only take the default value.

* @par Outputs:
* @li key_cache: Tensors that need to be updated by key. \n
* @li value_cache: Tensors that need to be updated by value. \n
* @attention Constraints:
* @code{.c}
  When cache_mode is "Norm" or None, the dtype of key, value, key_cache and value_cache dtype must be same.
  When key is a 3D tensor, shape and value limit like:
    shape limit:
      key:[batch * seqLen, num_head, k_head_size]
      key_cache:[num_blocks, block_size, num_head, k_head_size]
      slot_mapping:[batch * seqLen]
      value:[batch * seqLen, num_head, v_head_size]
      value_cache:[num_blocks, block_size, num_head, v_head_size]
    value limit:
      slot_mapping in range [0, num_blocks * block_size - 1]
      batch * seqLen <= num_blocks * block_size
      value in slot_mapping must be unique

  When key is a 4D tensor and compress_seq_offset is not None, shape and value limit like:
    shape limit:
      key:[batch, seqLen, num_head, k_head_size]
      key_cache:[num_blocks, block_size, 1, k_head_size]
      slot_mapping:[batch, num_head]
      value:[batch, seqLen, num_head, v_head_size]
      value_cache:[num_blocks, block_size, 1, v_head_size]
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
      value:[batch, seqLen, num_head, v_head_size]
      value_cache:[num_blocks, block_size, 1, v_head_size]
      compress_lens: [batch * num_head]
      seq_lens:[batch]
    value limit:
      slot_mapping in range [0, num_blocks * block_size - 1]
      reduceSum(compress_lens) <= num_blocks * block_size
      compress_lens[b][i] < seq_lens[b]
      seq_lens[b] <= seqLen
      compress_lens[b][i] > 0
      value in slot_mapping must be unique

  When cache_mode is "PA_NZ", the dtype of key and key_cache, value and value_cache dtype must be same,
  shape and value limit like:
    shape limit:
      key:[batch * seqLen, num_head, k_head_size]
      key_cache:[num_blocks, num_head * k_head_size / last_dim_k, block_size, last_dim_k]
      slot_mapping:[batch * seqLen]
      value:[batch * seqLen, num_head, v_head_size]
      value_cache:[num_blocks, num_head * v_head_size / last_dim_v, block_size, last_dim_v]
    value limit:
      slot_mapping in range [0, num_blocks * block_size - 1]
      batch * seqLen <= num_blocks * block_size
      value in slot_mapping must be unique
      last_dim_k and last_dim_v must be 32 Byte
      k_head_size and v_head_size should be align to 32 Byte

* @endcode
*/
REG_OP(ScatterPaKvCache)
    .INPUT(key, "T")
    .INPUT(key_cache, "T")
    .INPUT(slot_mapping, TensorType::IndexNumberType())
    .INPUT(value, "T")
    .INPUT(value_cache, "T")
    .OPTIONAL_INPUT(compress_lens, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(compress_seq_offset, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(seq_lens, TensorType::IndexNumberType())
    .OUTPUT(key_cache, "T")
    .OUTPUT(value_cache, "T")
    .ATTR(cache_mode, String, "Norm")
    .ATTR(scatter_mode, String, "None")
    .ATTR(strides, ListInt, {1, 1})
    .ATTR(offsets, ListInt, {0, 0})
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                             DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .OP_END_FACTORY_REG(ScatterPaKvCache)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
