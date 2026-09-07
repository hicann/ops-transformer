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
 * \file grouped_mat_mul_allto_allv_proto.h
 * \brief
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_PROTO_H_
#define GROUPED_MAT_MUL_ALLTO_ALLV_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Fusion of grouped matmul and alltoallv.

* @par Inputs:
* @li gmm_x: A matrix tensor of shape [A, H1]. The data type of elements supports float16 or bfloat16; the format
supports ND.
* @li gmm_weight: A matrix tensor of shape [e, H1, N1]. The data type of elements supports float16 or bfloat16 and
should match that of gmm_x; the format supports ND.
* @li send_counts_tensor: A tensor of shape [e * ep]. The data type of elements supports int32 or int64; the format
supports ND.
* @li recv_counts_tensor: A tensor of shape [e * ep]. The data type of elements supports int32 or int64; the format
supports ND.
* Data type of elements is int64. Format: ND.
* @li mm_x: A matrix tensor of shape [BS, H2]. The data type of elements supports float16 or bfloat16; the format
supports ND.
* @li mm_weight: gmm_weight: A matrix tensor of shape [H2, N2]. The data type of elements supports float16 or bfloat16
and should match that of mm_x; the format supports ND.

* @par Attributes:
* @li group: A required String identifying the expert group of ranks
* @li ep_world_size: A required int identifying the number of expert parallel group rank num.
* @li send_counts: An int list. A list containing amount of data to be sent.
* @li recv_counts: An int list. A list containing amount of data to be received.
* @li trans_gmm_weight: A boolean value. Indicating whether gmm_weight is transposed.
* @li trans_mm_weight: A boolean value. Indicating whether mm_weight is transposed.
* @li comm_mode: A String value. Indicating communication type.

* @par Outputs:
* @li y: A matrix tensor of shape [BSK, N1] containing result of grouped matmul. The data type of elements supports
float16 or bfloat16; the format supports ND.
* @li mm_y_optional: A matrix tensor of shape [BS, N2] containing result of matmul. The data type of elements supports
float16 or bfloat16; the format supports ND.
*/
REG_OP(GroupedMatMulAlltoAllv)
    .INPUT(gmm_x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gmm_weight, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(send_counts_tensor, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(recv_counts_tensor, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(mm_x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(mm_weight, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(mm_y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(send_counts, ListInt)
    .REQUIRED_ATTR(recv_counts, ListInt)
    .ATTR(trans_gmm_weight, Bool, false)
    .ATTR(trans_mm_weight, Bool, false)
    .ATTR(comm_mode, String, "ai_cpu")
    .OP_END_FACTORY_REG(GroupedMatMulAlltoAllv)

} // namespace ge

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_PROTO_H_
