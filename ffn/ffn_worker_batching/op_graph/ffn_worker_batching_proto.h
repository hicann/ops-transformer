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
 * \file ffn_worker_batching_proto.h
 * \brief
 */
#ifndef OP_GRAPH_FFN_WORKER_BATCHING_PROTO_H
#define OP_GRAPH_FFN_WORKER_BATCHING_PROTO_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
 * @brief Fusion op of sort, gather_out and group_list
 *
 * @par Inputs:
 * one input, including:
 * @li schedule_context: A tensor, support INT8.
 *
 * @par Outputs:
 * @li y: A tensor, support INT8, FP16, BF16, shape [Y, H], Y=A*BS*(topk+1).
 * @li group_list: A tensor list, support INT64, formatted as [expert_id, expert_token_num]. Example: [[1,20], [10,40],
 * [22,15], …].
 * @li session_ids: A tensor, support INT32, the attention_worker_id to which the data is sent.
 * @li micro_batch_ids: A tensor, support INT32, the micro_batch_id that each token belongs to.
 * @li token_ids: A tensor, support INT32, the batch-wise offset of each token within its micro_batch.
 * @li expert_offsets: A tensor, support INT32, the index within the top-k selection.
 * @li dynamic_scale: A tensor, support FP32, dynamic_scale[i] = token_data[session_id, micro_batch_id, bs, k, H].
 * @li actual_token_num: A tensor, support INT64, the actual number of tokens processed.
 *
 * @par Attributes:
 * @li expert_num: A scalar, the total number of experts.
 * @li max_out_shape: A scalar list, [A, BS, topk+1, H], inferred output shape for Y and H, Y=A*BS*(topk+1).
 * @li token_dtype: A scalar, support FP16, BF16, quantization type of the input tokens.
 * @li need_schedule: A scalar, whether the op should be scheduled.
 * @li layer_num: A scalar, the number of layers.
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
#ifndef OPS_PROTO_DEF_FFNWORKERBATCHING
#define OPS_PROTO_DEF_FFNWORKERBATCHING
REG_OP(FfnWorkerBatching)
    .INPUT(schedule_context, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT16, DT_BFLOAT16}))
    .OUTPUT(group_list, TensorType({DT_INT64}))
    .OUTPUT(session_ids, TensorType({DT_INT32}))
    .OUTPUT(micro_batch_ids, TensorType({DT_INT32}))
    .OUTPUT(token_ids, TensorType({DT_INT32}))
    .OUTPUT(expert_offsets, TensorType({DT_INT32}))
    .OUTPUT(dynamic_scale, TensorType({DT_FLOAT32}))
    .OUTPUT(actual_token_num, TensorType({DT_INT64}))
    .REQUIRED_ATTR(expert_num, Int)
    .REQUIRED_ATTR(max_out_shape, ListInt)
    .ATTR(token_dtype, Int, 0)
    .ATTR(need_schedule, Int, 0)
    .ATTR(layer_num, Int, 0)
    .OP_END_FACTORY_REG(FfnWorkerBatching)
#endif // OPS_PROTO_DEF_FFNWORKERBATCHING

} // namespace ge

#endif // OP_GRAPH_FFN_WORKER_BATCHING_PROTO_H
