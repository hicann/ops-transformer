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
 * \file moe_distribute_combine_setup_proto.h
 * \brief 图模式原型定义
 */

#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_PROTO_H_
#define MOE_DISTRIBUTE_COMBINE_SETUP_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief MoeDistributeCombineSetup operator interface implementation.
 * @par Inputs
 * Three inputs, including:
 * @li context: A tensor. Support dtype: int32, dimension must be 1. Shape supports (2052, ), support format: ND.
 * @li expand_x: A tensor. Support dtype: float16, bfloat16, dimension must be 2. Shape supports (A, H), support format:
 * ND.
 * @li expert_ids: A tensor. Support dtype: int32, indicates top k experts of each token, dimension must be 2. Shape
 * supports (BS, K), support format: ND.
 * @li assist_info_for_combine: A tensor. Support dtype: int32, dimension must be 1. Shape supports (N, ), support
 * format: ND.
 *
 * @par Attributes
 * @li ep_world_size: Required. Input ep comm world size, dtype: int64.
 * @li ep_rank_id: Required. Input ep comm rank Id, dtype: int64.
 * @li moe_expert_num: Required. Input moe expert num, dtype: int64.
 * @li ccl_buffer_size: Required, Input ccl buffer size, Support Range: [0, MAX_INT32)，MAX_INT32 = 2^31 - 1, dtype:
 * Int64.
 * @li expert_shard_type: Input moe shard type, dtype: int64. Default: 0.
 * @li shared_expert_num: Input shared expert num, dtype: int64. Default: 1.
 * @li shared_expert_rank_num: Input shared expert rank num, dtype: int64. Default: 0.
 * @li comm_quant_mode: Input quant mode. The options are 0 (non-quantization), 1 (static quantization), and 2 (dynamic
 * quantization). dtype: int64. Default: 0.
 * @li global_bs: Input global batch size, dtype: int64. Default: 0.
 * @li comm_type: Input comm type, dtype: int64. Default: 0.
 * @li comm_alg: Input comm algorithm, dtype: String. Default: "".
 *
 * @par Outputs
 * Two outputs, including:
 * @li quant_expand_x: A tensor. Result of quantized expand_x. Support dtype: int8, dimension must be 2. Shape supports
 * (A, H), support format: ND.
 * @li comm_cmd_info: A tensor. Communication command info. Support dtype: int32, dimension must be 1. Shape supports
 * (N, ), support format: ND.
 */
REG_OP(MoeDistributeCombineSetup)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(expand_x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(assist_info_for_combine, TensorType({DT_INT32}))
    .OUTPUT(quant_expand_x, TensorType({DT_INT8}))
    .OUTPUT(comm_cmd_info, TensorType({DT_INT32}))
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
    .OP_END_FACTORY_REG(MoeDistributeCombineSetup)

} // namespace ge

#endif // MOE_DISTRIBUTE_COMBINE_SETUP_PROTO_H_
