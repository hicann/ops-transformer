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
 * \file moe_finalize_routing_v2_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOEFINALIZE_ROUTEV2GRAD_H_
#define OPS_OP_PROTO_INC_MOEFINALIZE_ROUTEV2GRAD_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Backwards calculation of MoeFinalizeRoutingV2.

* @par Inputs:
* @li grad_y: A 2D Tensor, represents the gradient of output of MoeFinalizeRoutingV2. Type is BFloat16, Float16 or
      Float32. Shape supports (R, H). Format supports ND.
* @li expanded_row_idx: A 1D Tensor, represents the token indexes of expanded_x. Type is Int32. Shape supports (R * K),
      when scales is not passed in, K must be 1. If drop_pad_mode is 0, the value range is [0, R * K - 1], and there are
      no duplicate indexes. When drop_pad_mode is 1, the value range is [-1, expert_num * expert_capacity - 1], and
      duplicate indexes are not allowed except -1. Format supports ND.
* @li expanded_x: An optional 2D or 3D Tensor, represents the token sequences. Type should be the same as the type of
      grad_y. When scales is passed in, it should be passed in. When drop_pad_mode is 0, it should be a 2D Tensor, and
      when active_num is between (0, R * K), the shape is (active_num, H), otherwise the shape is (R * K, H). When
      drop_pad_mode is 1, it should be a 3D Tensor, the shape is (expert_num, expert_capacity, H). Format supports ND.
* @li scales: An optional 2D Tensor, represents the scale of expanded_x. Type should be the same as the type of grad_y
      except in Ascend 950PR&950DT products. Shape supports (R, K). Format supports ND.
* @li expert_idx: An optional 2D Tensor, represents the indexes of bias. Type should be the same as the type of
      expanded_row_idx. When bias is passed in, it should be passed in. Shape supports (R, K). the value range is
      [0, E - 1], E >= 1, and duplicate indexes are allowed. Format supports ND.
* @li bias: An optional 2D Tensor, represents the bias of expanded_x. Type should be the same as the type of grad_y.
      Shape supports (E, H). Format supports ND.

* @par Outputs:
* @li grad_expanded_x: A 2D or 3D Tensor, represents the gradient of expanded_x. Type should be the same as the type of
      grad_y. When drop_pad_mode is 0, it should be a 2D Tensor, when active_num is between (0, R * K), the shape is
      (active_num, H), otherwise the shape is (R * K, H). When drop_pad_mode is 1, it should be a 3D Tensor, the shape
is (expert_num, expert_capacity, H). Format supports ND.
* @li grad_scales: A 2D Tensor, represents the gradient of scales. Type should be the same as the type of grad_y except
      in Ascend 950PR&950DT products. Shape supports (R, K). This output only makes sense when scales is passed in.
      Format supports ND.

* @par Attributes:
* @li drop_pad_mode: An optional integer, represents the dropless or drop/pad mode. Type is Int32. Default: 0. Value
      supports 0 or 1.
* @li active_num: An optional integer, represents the active tokens of expanded_x. Type is Int32. Default: 0. When
      drop_pad_mode is 0, it takes effect only when it is between (0, R * K). When drop_pad_mode is 1, it does not take
      effect.
* @li expert_num: An optional integer, represents the number of expert. Type is Int32. Default: 0. When drop_pad_mode
      is 0, it does not take effect. When drop_pad_mode is 1, it should be equal to E when bias is passed in, otherwise
      it should be greater than 0.
* @li expert_capacity: An optional integer, represents the capacity of expert. Type is Int32. Default: 0. When
      drop_pad_mode is 0, it does not take effect. When drop_pad_mode is 1, it should be greater than 0.
*/
REG_OP(MoeFinalizeRoutingV2Grad)
    .INPUT(grad_y, "T1")
    .INPUT(expanded_row_idx, "T2")
    .OPTIONAL_INPUT(expanded_x, "T1")
    .OPTIONAL_INPUT(scales, "T1")
    .OPTIONAL_INPUT(expert_idx, "T2")
    .OPTIONAL_INPUT(bias, "T1")
    .OUTPUT(grad_expanded_x, "T1")
    .OUTPUT(grad_scales, "T1")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(active_num, Int, 0)
    .ATTR(expert_num, Int, 0)
    .ATTR(expert_capacity, Int, 0)
    .OP_END_FACTORY_REG(MoeFinalizeRoutingV2Grad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOEFINALIZE_ROUTEV2GRAD_H_
