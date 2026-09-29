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
 * \file moe_distribute_dispatch_teardown_proto.h
 * \brief 算子图模式原型定义
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief MoeDistributeDispatchTeardown operator interface implementation.

* @par Inputs
* Four inputs, including:
* @li context: A tensor. Support dtype: int32, dimension must be 1. Shape supports (2052, ), support format: ND.
* @li x: A tensor. Support dtype: float16, bfloat16, dimension must be 2. Shape supports (BS, H), support format: ND.
* @li y: A tensor. Support dtype: float16, bfloat16, dimension must be 2. Shape supports (BS * (K + sharedExpertNum),
tokenMsgSize), support format: ND.
* @li expert_ids: A tensor. Support dtype: int32, indicates top k experts of each token, dimension must be 2. Shape
supports (BS, K), support format: ND.
* @li comm_cmd_info: An optional tensor. Support dtype: int32, dimension must be 2. Shape supports ((BS * (K +
sharedExpertNum) + epWorldSize * localExpertNum) * 16), support format: ND.

* @par Attributes
* @li ep_world_size: Required. Input ep comm world size, value range: [2, 384], dtype: int64.
* @li ep_rank_id: Required. Input ep comm rank Id, value range: [0, epWorldSize), dtype: int64.
* @li moe_expert_num: Required. Input number of moe experts, value range: (0, 512] and must satisfy moeExpertNum %
(epWorldSize - SharedExpertRankNum) = 0, dtype: int64.
* @li ccl_buffer_size: Required, Input ccl buffer size, Support Range: [0, MAX_INT32)，MAX_INT32 = 2^31 - 1,
dtype: Int64.
* @li expert_shard_type: The shard type of shared expert rank. Only 0 (the shared expert rank is placed
before moe expert rank) is supported currently, dtype: int64. Default: 0.
* @li shared_expert_num: Input shared expert num, value range: [0, 4], dtype: int64. Default: 1.
* @li shared_expert_rank_num: Input shared expert rank num, value range: [0, epWorldSize / 2], dtype: int64.
Default: 0.
* @li quant_mode: Input quant mode. The options are 0 (non-quantization), and 2 (dynamic quantization). dtype: int64.
Default: 0.
* @li global_bs: Input global batch size, dtype: int64. Default: 0.
  When the number of Bs at each rank is the same, globalBs = Bs * epWorldSize or globalBs = 0. When the number of Bs at
each rank is different, globalBs = maxBs * epWorldSize.
* @li expert_token_nums_type: The semantic type of the value in expertTokenNums.
  The options are 0 (expertTokenNums is the prefix sum of the number of tokens processed by each expert), and 1
(expertTokenNums is the number of tokens processed by each expert), dtype: int64. Default: 1.
* @li comm_type: Communication scheme selection. The options are 0 (AICPU-SDMA), 1 (CCU), and 2 (URMA), only 0 is
supported currently. dtype: int64. Default: 0.
* @li comm_alg: Communication affinity and memory layout algorithm. Only nullptr is supported currently. dtype: String.
Default: "".

* @par Outputs
* Four outputs, including:
* @li expand_x: A tensor. Result of each expert after dispatching. Support dtype: float16, bfloat16, int8. Shape
supports (A, H), support format: ND.
* @li dynamic_scales: A tensor. Scale value of each token if quantization is enabled. Support dtype: float32. Shape
supports (A, ), support format: ND.
* @li assist_info_for_combine: A tensor. Support dtype: int32. Shape supports (A * 128, ), support format: ND.
* @li expert_token_nums: A tensor. Tokens nums of expand_x. Support dtype: int64. Shape supports (localExpertNum, ),
support format: ND.
*/
REG_OP(MoeDistributeDispatchTeardown)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(comm_cmd_info, TensorType({DT_INT32}))
    .OUTPUT(expand_x, TensorType({DT_BF16, DT_INT8, DT_FLOAT16}))
    .OUTPUT(dynamic_scales, TensorType({DT_FLOAT}))
    .OUTPUT(assist_info_for_combine, TensorType({DT_INT32}))
    .OUTPUT(expert_token_nums, TensorType({DT_INT64}))
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .REQUIRED_ATTR(ccl_buffer_size, Int)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(expert_token_nums_type, Int, 1)
    .ATTR(comm_type, Int, 0)
    .ATTR(comm_alg, String, "")
    .OP_END_FACTORY_REG(MoeDistributeDispatchTeardown)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
