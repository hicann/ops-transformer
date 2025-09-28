/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_softmax_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOEGATINGTOPKSOFTMAX_H_
#define OPS_OP_PROTO_INC_MOEGATINGTOPKSOFTMAX_H_

#include "graph/operator_reg.h"

namespace ge {

/**
   * @brief compute softmax and topk for moe input.
   * @par Inputs:
   * @li x: A 2D or 3D Tensor. Type is:BFloat16, Float16 or Float32. Format support ND.
   * @li finished: A Tensor. Type is:Bool. Shape is x_shape[:-1]. Format support ND.
   * @par Outputs:
   * @li y: A Tensor. Type is:BFloat16, Float16 or Float32. The data type must be the same as that of x.
         The size of the non-1 axis must be the same as that of the corresponding axis of x.
         The size of the -1 axis must be the same as that of k. Format support ND.
   * @li expert_idx: A Tensor. Type is:Int32. The shape must be the same as that of y. Format support ND.
   * @li row_idx: A Tensor. Type is:Int32. The shape must be the same as that of y. Format support ND.
   * @par Attributes:
   * @li k: Required parameter. Type is:Int32. The value must greater than 0 and less than or equal to the size
         of the -1 axis of x, and k must not greater than 1024.
   */
REG_OP(MoeGatingTopKSoftmax)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(finished, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(expert_idx, TensorType({DT_INT32}))
    .OUTPUT(row_idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(k, Int)
    .OP_END_FACTORY_REG(MoeGatingTopKSoftmax)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOEGATINGTOPKSOFTMAX_H_