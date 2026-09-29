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
 * \file moe_distribute_combine_teardown_proto.h
 * \brief 图模式原型定义
 */
#ifndef MOE_DISTRIBUTE_COMBINE_TEARDOWN_PROTO_H
#define MOE_DISTRIBUTE_COMBINE_TEARDOWN_PROTO_H

#include <graph/operator_reg.h>

namespace ge {
/**
 * @brief MoeDistributeCombineTeardown operator interface implementation.

 * @par Inputs
 * Eight inputs, including:
 * @li context: A tensor. Support dtype: int32, dimension must be 1. Shape supports (2052, ), support format: ND.
 * @li expand_x: A tensor. Support dtype: float16, bfloat16, dimension must be 2, Support Shape: (A, H),
 * support format: ND.
 * @li quant_expand_x: A tensor. Support dtype: int8, dimension must be 2, Support Shape: (A, tokenMsgSize), support
 * format: ND.
 * @li expert_ids: A tensor. Support dtype: int32, dimension must be 2, Support Shape: (BS, K), support format: ND.
 * @li expand_idx: A tensor. Support dtype: int32, dimension must be 1, Support Shape: (BS * K, ), support format: ND.
 * @li expert_scales: A tensor. Support dtype: float32, dimension must be 2, Support Shape: (BS, K),
 * support format: ND.
 * @li comm_cmd_info: A tensor. Support dtype: int32, dimension must be 1, Call the
 * `aclnnMoeDistributeCombineSetupTeardownCalcOutputSize` API to obtain the Shape, support format: ND.
 * @li x_active_mask: An optional tensor. Support dtype: bool, dimension must be 1, Support Shape: (BS, ) support
 * format: ND.
 * @li shared_expert_x: An optional tensor. Dtype should be same with expand_x, dimension must be 2 or 3,
 * Support Shape: (BS, H) or (a, b, H), where a * b = BS, support format: ND.

 * @par Attributes
 * @li ep_world_size: Input ep comm world size, dtype: Int64.
 * @li ep_rank_id: Input ep comm rank Id, dtype: Int64.
 * @li moe_expert_num: Input moe expert num, dtype: Int64.
 * @li ccl_buffer_size: Required, Input ccl buffer size, Support Range: [0, MAX_INT32)，MAX_INT32 = 2^31 - 1,
 * dtype: Int64.
 * @li expert_shard_type: Input moe shard type, dtype: Int64. Default: 0.
 * @li shared_expert_num: Input shared expert num, dtype: Int64. Default: 1.
 * @li shared_expert_rank_num: Input shared expert rank num, dtype: Int64. Default: 0.
 * @li global_bs: Input global batch size, dtype: Int64. Default: 0.
 * @li comm_quant_mode: communication quantization mode, 1 for enable, 0 for disable, dtype: Int64. Default: 0.
 * @li comm_type: type of communication, dtype: Int64. Default: 0.
 * @li comm_alg: Input comm alg type, dtype: String. Default: "".

 * @par Outputs
 * One outputs, including:
 * @li x_out: A tensor. Result of combine. Dtype should be same with expand_x, Support Shape: (BS, H), support format:
 * ND.
 */
REG_OP(MoeDistributeCombineTeardown)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(expand_x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(quant_expand_x, TensorType({DT_INT8}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(expand_idx, TensorType({DT_INT32}))
    .INPUT(expert_scales, TensorType({DT_FLOAT}))
    .INPUT(comm_cmd_info, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(shared_expert_x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(x_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .REQUIRED_ATTR(ccl_buffer_size, Int)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(comm_type, Int, 0)
    .ATTR(comm_alg, String, "")
    .OP_END_FACTORY_REG(MoeDistributeCombineTeardown)
} // namespace ge

#endif // MOE_DISTRIBUTE_COMBINE_TEARDOWN_PROTO_H
