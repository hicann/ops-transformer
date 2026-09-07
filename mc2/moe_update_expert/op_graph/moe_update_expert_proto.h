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
 * \file moe_update_expert_proto.h
 * \brief
 */
#ifndef MOE_UPDATE_EXPERT_PROTO_H_
#define MOE_UPDATE_EXPERT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief MoeUpdateExpert operator interface implementation.
 *
 * @par Inputs:
 * @li expert_ids: A tensor. The topK expert index for each token. Support dtype: int32, int64. dimension must be 2,
 * support format: ND.
 * @li eplb_table: A tensor. Mapping table of logical rank_id to physical rank_id. Support dtype: int32. dimension must
 * be 2, support format: ND.
 * @li expert_scales: A tensor. Scales for top k experts for each token. Support dtype: float16, bfloat16, float32.
 * dimension must be 2, support format: ND.
 * @li pruning_threshold: A tensor. Threshold for expert scales. Support dtype: float32. dimension must be 1 or 2,
 * support format: ND.
 * @li active_mask: A tensor. Indicates if token is involved in communication. Support dtype: bool. dimension must be 1,
 * support format: ND.
 *
 * @par Attributes:
 * @li local_rank_id: Required. The rank id in the current communication domain. dtype: Int64.
 * @li world_size: Required. The num of physical rank in the current communication domain. dtype: Int64.
 * @li balance_mode: Optional. Balanced by rank(0) or by token(1), and the current default value is 0. dtype: Int64
 *
 * @par Output:
 * balance_expert_ids: A tensor. Convert the logical rank_id in expert_ids to physical rank_id using the eplb_table.
 * Support dtype: int32, int64. Support format: ND. balanced_active_mask: A tensor. Get balanced active mask through
 * tailored expert scales and pruning threshold. Support dtype: bool. Support format: ND.
 */
REG_OP(MoeUpdateExpert)
    .INPUT(expert_ids, TensorType({DT_INT32, DT_INT64}))
    .INPUT(eplb_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(expert_scales, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pruning_threshold, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(active_mask, TensorType({DT_BOOL}))
    .OUTPUT(balanced_expert_ids, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(balanced_active_mask, TensorType({DT_BOOL}))
    .ATTR(local_rank_id, Int, -1)
    .ATTR(world_size, Int, -1)
    .ATTR(balance_mode, Int, 0)
    .OP_END_FACTORY_REG(MoeUpdateExpert)

} // namespace ge

#endif // MOE_UPDATE_EXPERT_PROTO_H_
