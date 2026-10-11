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
 * \file moe_gating_top_k_softmax_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOEGATINGTOPKSOFTMAX2_H_
#define OPS_OP_PROTO_INC_MOEGATINGTOPKSOFTMAX2_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief compute softmax and topk for moe input.
 * @par Inputs:
 * @li x: A 2D or 3D tensor, the size of each dimensioin should not exceed the maximum value of int32, which is
 2147483647. Type is:BFloat16, Float16 or Float32. Format supports ND.
 * @li finished: An optional tensor, the size of each dimensioin should not exceed the maximum value of int32, which is
 2147483647. Type is:Bool. Shape is x_shape[:-1]. Format supports ND.
 * @par Outputs:
 * @li y: A tensor. Type is:BFloat16, Float16 or Float32. The data type must be the same as that of x.
       The size of the non-1 axis must be the same as that of the corresponding axis of x.
       The size of the -1 axis must be the same as that of k. Format supports ND.
 * @li expert_idx: A tensor. Type is:Int32. The shape must be the same as that of y. Format supports ND.
 * @li softmax_result: A tensor. Type is:Float32. The shape must be the same as that of x. Format supports ND.
 * @par Attributes:
 * @li k: Required parameter. Type is:Int32. The value must greater than 0 and less than or equal to the size
       of the -1 axis of x, and k must not greater than 1024.
 * @li renorm: Optional parameter. Type is:Int32. The value must be 0(non-renorm) or 1(renorm)
 * @li output_softmax_result_flag: Optional parameter. Type is:Bool. The value must be true or false.
       When renorm is 0, output_softmax_result_flag is true indicates that the Softmax result is output.
       When renorm is 0, output_softmax_result_flag is false indicates that the Softmax result is not output.
       When renorm is 1, this parameter does not take effect, the Softmax result is not output.
 */
REG_OP(MoeGatingTopKSoftmaxV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(finished, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(expert_idx, TensorType({DT_INT32}))
    .OUTPUT(softmax_result, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(k, Int)
    .ATTR(renorm, Int, 0)
    .ATTR(output_softmax_result_flag, Bool, false)
    .OP_END_FACTORY_REG(MoeGatingTopKSoftmaxV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOEGATINGTOPKSOFTMAX2_H_
