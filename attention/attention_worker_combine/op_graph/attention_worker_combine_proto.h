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
 * \file attention_worker_combine_proto.h
 * \brief
 */
#ifndef OP_GRAPH_ATTENTION_WORKER_COMBINE_PROTO_H
#define OP_GRAPH_ATTENTION_WORKER_COMBINE_PROTO_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
 * @brief The operator of AttentionWorkerCombine.
 *
 * @par Inputs:
 * @li schedule_context: A 1D tensor, represents struct contain token_data. Data type is DT_INT8. Format supports ND.
 * @li expert_scales: A 2D tensor, represents expert scales. Data type is DT_FLOAT32. Shape supports (BS, K). Format
 * supports ND.
 * @li layer_id: A 1D tensor, represents layer id. Data type is DT_INT32. Format supports ND. Shape supports (1,).
 *
 * @par Outputs:
 * @li y: A 2D tensor, represents the combine result of token_data and expert scales. Data type is DT_FLOAT16, DT_BF16.
 * Format supports ND. Shape supports (BS, H).
 * @li next_layer_id: A 1D tensor, represents the next layer id. Data type is DT_INT32. Format supports ND. Shape
 * supports (1,).
 *
 * @par Attributes:
 * @li hidden_size: Required Int, represents token_data hidden size.
 * @li token_dtype: Int, represents token data type. Value 0 represents token data type is FLOAT16, and value 1
 * represents BFLOAT16.
 * @li need_schedule: Int, Value 1 represents op waits until tokens are filled.
 */
REG_OP(AttentionWorkerCombine)
    .INPUT(schedule_context, TensorType({DT_INT8}))
    .INPUT(expert_scales, TensorType({DT_FLOAT}))
    .INPUT(layer_id, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(next_layer_id, TensorType({DT_INT32}))
    .REQUIRED_ATTR(hidden_size, Int)
    .ATTR(token_dtype, Int, 0)
    .ATTR(need_schedule, Int, 0)
    .OP_END_FACTORY_REG(AttentionWorkerCombine)
} // namespace ge

#endif // OP_GRAPH_ATTENTION_WORKER_COMBINE_PROTO_H
