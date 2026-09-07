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
 * \file attention_to_ffn_proto.h
 * \brief
 */
#ifndef ATTENTION_TO_FFN_H_
#define ATTENTION_TO_FFN_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Fusion op AttentionToFFN.

 * @par Inputs:
 * @li x:
 * @li session_id:
 * @li micro_batch_id:
 * @li layer_id:
 * @li expert_ids:
 * @li expert_rank_table:
 * @li scales:
 * @li active_mask:

 * @par Attributes:
 * @li group:
 * @li world_size:
 * @li quant_mode:
 * @li expert_tokens_cnt_flag_len:

 * @attention Constraints:
 */
REG_OP(AttentionToFFN)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(session_id, TensorType({DT_INT32}))
    .INPUT(micro_batch_id, TensorType({DT_INT32}))
    .INPUT(layer_id, TensorType({DT_INT32}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(expert_rank_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(active_mask, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .REQUIRED_ATTR(ffn_token_info_table_shape, ListInt)
    .REQUIRED_ATTR(ffn_token_data_shape, ListInt)
    .REQUIRED_ATTR(attn_token_info_table_shape, ListInt)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(quant_mode, Int, 0)
    .ATTR(sync_flag, Int, 0)
    .ATTR(ffn_start_rank_id, Int, 0)
    .OP_END_FACTORY_REG(AttentionToFFN)

} // namespace ge

#endif // ATTENTION_TO_FFN_H_
