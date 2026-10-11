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
 * \file moe_init_routing_quant_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOE_INIT_ROUTING_QUANT_V2_OPS_H_
#define OPS_OP_PROTO_INC_MOE_INIT_ROUTING_QUANT_V2_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief compute init routing quant for moe input.
 * @par Inputs:
 * @li x: A 2D Tensor. represents the input token. shape is (num_rows, H), num_rows is the number of tokens, H is the
 *        length of each token. Type is:BFloat16, Float16 or Float32. Format support ND.
 * @li expert_idx: A 2D Tensor. represents the id of top k experts for each tokens. shape is (num_row, K).
 *                 When drop_pad_mode is 1, or drop_pad_mode is 0 but expert_tokens_count_or_cumsum_flag is not 0,
 *                 The range of value is [0, expert_num). Other scenario the value of expert_idx cannot be less than
 *                 0. Type is:Int32. Format support ND.
 * @li scale: An Optional Tensor. represents the parameters used to compute the result of the quant, which is required
 *            to be a 1D shape (1,) for the static quant scenario. The dynamic quant scene if not input, indicates
 *            that scale is not used in the calculation process; if a 2D Tensor is input, the shape is required to be
 *            (expertNum, H) or (1, H). Type is:Float32. Format support ND.
 * @li offset: An Optional Tensor. Indicates the offset value used to compute the result, it`s only used in the static
 *             quant scenario, which is required to be a 1D shape (1,). Type is:Float32. Format support ND.
 * @par Outputs:
 * @li expanded_x: A 2D or 3D Tensor. Type is:Int8. Format support ND.
 *                 dropless scenario:
 *                 the first dim is val=min(num_rows * K, active_num), so shape is (val, H)
 *                 dropPad scenario:
 *                 shape is (expert_num, expert_capacity, H)
 * @li expanded_row_idx:A 1D Tensor. Type is:Int32. shape is (num_rows * K,). Format support ND.
 * @li expert_tokens_count_or_cumsum: A 1D Tensor. represents the number of tokens processed by each expert and the
 * cumulative value. The value is controlled by expert_tokens_count_or_cumsum_flag to output. Type is:Int32. shape
 * is (expert_num,). Format support ND.
 * @li expert_tokens_before_capacity: A 1D Tensor. represents the number of tokens processed by each expert before
 * drop. The value is controlled by expert_tokens_before_capacity_flag to output. Type is:Int32. shape is
 * (expert_num,). Format support ND.
 * @li dynamic_quant_scale: A 1D Tensor. represents tthe outputs the intermediate value of the dynamic quant
 *                          computation process, which is only output in dynamic quant scenarios.Type is:Float32.
 *                          The shape is expanded_x_shape[:-1]. Format support ND.
 * @par Attributes:
 * @li active_num: Optional parameter. Type is:Int32. identify activate scenario. The value 0 indicates a non-active
 *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size
 *                 of axis 0 of grad_expanded_x must be equal to the value of active_num. Default: 0.
 * @li expert_capacity: Optional parameter. Type is:Int32. The max tokens count of every expert. Default: 0.
 * @li expert_num: Optional parameter. Type is:Int32. Default: 0.
 * @li drop_pad_mode: Optional parameter. Type is:Int32. The value is 0(dropless) or 1(dropPad). Default: 0.
 * @li expert_tokens_count_or_cumsum_flag: Optional parameter. Type is:Int32. The value is 0 (no token count),
 *                                         1(compute token count) or 2(compute token cumsum), which in dropless
 *                                         scenario. Default: 0.
 * @li expert_tokens_before_capacity_flag: Optional parameter. Type is:Bool. The value is true (no tokens count) or
 *                                         1(compute token count), which in dropPad scenario. Default: false.
 * @li quant_mode: Optional parameter. Type is:Int32. The value is 0(static quant) or 1(dynamic quant). Default: 0.
 */
REG_OP(MoeInitRoutingQuantV2)
    .INPUT(x, "T1")
    .INPUT(expert_idx, "T2")
    .OPTIONAL_INPUT(scale, "T3")
    .OPTIONAL_INPUT(offset, "T3")
    .OUTPUT(expanded_x, "T4")
    .OUTPUT(expanded_row_idx, "T2")
    .OUTPUT(expert_tokens_count_or_cumsum, "T2")
    .OUTPUT(expert_tokens_before_capacity, "T2")
    .OUTPUT(dynamic_quant_scale, "T3")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .DATATYPE(T3, TensorType({DT_FLOAT}))
    .DATATYPE(T4, TensorType({DT_INT8}))
    .ATTR(active_num, Int, 0)
    .ATTR(expert_capacity, Int, 0)
    .ATTR(expert_num, Int, 0)
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(expert_tokens_count_or_cumsum_flag, Int, 0)
    .ATTR(expert_tokens_before_capacity_flag, Bool, false)
    .ATTR(quant_mode, Int, 0)
    .OP_END_FACTORY_REG(MoeInitRoutingQuantV2)

} // namespace ge

#endif
