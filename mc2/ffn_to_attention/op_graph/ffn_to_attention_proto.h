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
 * \file ffn_to_attention_proto.h
 * \brief
 */
#ifndef FFN_TO_ATTENTION_PROTO_H_
#define FFN_TO_ATTENTION_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Fusion op FFNToAttention.

 * @par Inputs:
 * @li x:
 * @li session_ids:
 * @li micro_batch_ids:
 * @li token_ids:
 * @li expert_offsets:
 * @li attn_rank_table:

 * @par Attributes:
 * @li group:
 * @li world_size:
 * @li quant_mode:
 * @li expert_tokens_cnt_flag_len:

 * @attention Constraints:
 */
REG_OP(FFNToAttention)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(session_ids, TensorType({DT_INT32}))
    .INPUT(micro_batch_ids, TensorType({DT_INT32}))
    .INPUT(token_ids, TensorType({DT_INT32}))
    .INPUT(expert_offsets, TensorType({DT_INT32}))
    .INPUT(actual_token_num, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(attn_rank_table, TensorType({DT_INT32}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .REQUIRED_ATTR(token_info_table_shape, ListInt)
    .REQUIRED_ATTR(token_data_shape, ListInt)
    .OP_END_FACTORY_REG(FFNToAttention)

} // namespace ge

#endif // FFN_TO_ATTENTION_PROTO_H_
