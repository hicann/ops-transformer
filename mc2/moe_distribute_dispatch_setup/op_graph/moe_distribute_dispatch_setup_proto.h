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
 * \file moe_distribute_dispatch_setup_proto.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_SETUP_PROTO_H
#define MOE_DISTRIBUTE_DISPATCH_SETUP_PROTO_H

#include <graph/operator_reg.h>

namespace ge {
/**
* @brief MoeDistributeDispatchSetup operator interface implementation.
* @par Inputs
* Four inputs, including:
* @li x: A tensor. Support dtype: float16, bfloat16, dimension must be 2. Shape supports (BS, H), support format: ND.
* @li expert_ids: A tensor. Support dtype: int32, indicates top k experts of each token, dimension must be 2. Shape
supports (BS, K), support format: ND.
* @li scales: An optional tensor. Support dtype: float32, dimension must be 2. Shape supports (sharedExpertNum +
moeExpertNum, H)， support format: ND. In non-quantization scenario, the value in scales is nullptr. In dynamic
quantization scenario, the value in scales is nullptr or valid data (indicating the quantization smoothing parameter of
each expert).
* @li x_active_mask: An optional tensor. Support dtype: bool. Shape supports (BS, ), support format: ND.
  The value in x_active_mask can be valid data or nullptr. If the value is nullptr, all tokens participate in
communication. If the value is valid data, tokens where x_active_mask is true participate in communication.

* @par Attributes
* @li group_ep: Required. Input ep comm group name, ep means experts parallelism. The string length range: [1, 128),
dtype: String.
* @li ep_world_size: Required. Input ep comm world size, value range: [2, 384], dtype: int64.
* @li ep_rank_id: Required. Input ep comm rank Id, value range: [0, epWorldSize), dtype: int64.
* @li moe_expert_num: Required. Input moe expert num, value range: (0, 512] and must satisfy moeExpertNum %
(epWorldSize - SharedExpertRankNum) = 0, dtype: int64.
* @li expert_shard_type: The shard type of shared expert rank. Only 0 (the shared expert rank is placed before moe
expert rank) is supported currently, dtype: int64. Default: 0.
* @li shared_expert_num: Input shared expert num, value range: [0, 4], dtype: int64. Default: 1.
* @li shared_expert_rank_num: Input shared expert rank num, value range: [0, epWorldSize / 2], dtype: int64.
Default: 0.
* @li quant_mode: Input quant mode. The options are 0 (non-quantization), and 2 (dynamic quantization). dtype: int64.
Default: 0.
* @li global_bs: Input global batch size, dtype: int64. Default: 0.
  When the number of Bs at each rank is the same, globalBs = Bs * epWorldSize or globalBs = 0. When the number of Bs at
each rank is different, globalBs = maxBs * epWorldSize.
* @li comm_type: Communication scheme selection. The options are 0 (AICPU-SDMA), 1 (CCU), and 2 (URMA), only 0 is
supported currently. dtype: int64. Default: 0.
* @li comm_alg: Communication affinity and memory layout algorithm. Only nullptr is supported currently. dtype: String.
Default: "".

* @par Outputs
* Three outputs, including:
* @li y: A tensor. Support dtype: float16, bfloat16, int8. Shape supports (BS * K, ), support format: ND.
* @li expand_idx: A tensor. Support dtype: int32. Shape supports (BS * K, ), support format: ND.
* @li comm_cmd_info: A tensor. Support dtype: int32. Shape supports (BS * K, ), support format: ND.
*/
REG_OP(MoeDistributeDispatchSetup)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BF16, DT_INT8, DT_FLOAT16}))
    .OUTPUT(expand_idx, TensorType({DT_INT32}))
    .OUTPUT(comm_cmd_info, TensorType({DT_INT32}))
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .REQUIRED_ATTR(ccl_buffer_size, Int)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(comm_type, Int, 0)
    .ATTR(comm_alg, String, "")
    .ATTR(y_dtype, Int, DT_UNDEFINED)
    .OP_END_FACTORY_REG(MoeDistributeDispatchSetup)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
