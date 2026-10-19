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
 * \file quant_grouped_matmul_inplace_add_proto.h
 * \brief Graph IR definition for QuantGroupedMatmulInplaceAdd.
 */
#ifndef OPS_TRANSFORMER_QUANT_GROUPED_MATMUL_INPLACE_ADD_PROTO_H_
#define OPS_TRANSFORMER_QUANT_GROUPED_MATMUL_INPLACE_ADD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
// clang-format off
/**
* @brief Function QuantGroupedMatmulInplaceAdd. This op fuse GroupedMatmul with InplaceAdd.

* @par Inputs:
* @li x1: A tensor. Format supports ND. Data type supports hifloat8, float8_e5m2, float8_e4m3fn.
* @li x2: A tensor. Format supports ND. Data type supports hifloat8, float8_e5m2, float8_e4m3fn.
* @li scale2: A tensor. Indicating scaling factor of quantization parameter, introduced by x2 quantization.
* Format supports ND. Data type supports float32, float8_e8m0.
* @li group_list: A tensor. Indicating the matmul size distribution along separated dimension.
* Format supports ND. Data type supports int64. The first dimension of group_list should not be greater than 1024, meaning it supports up to 1024 groups.
* @li y: A tensor. Format supports ND. Data type supports float32.
* @li scale1: A tensor. Indicating the scaling factor of the quantization parameter, introduced by x1 quantization.
* Format supports ND. The data type supports float32, float8_e8m0.

* @par Attributes:
* @li group_list_type: An int. Indicates whether the value in group_list is cumsum or count.
* 0, value in group_list is cumsum. 1, value in group_list is count. Default: 0.
* @li group_size: An int. Default: 0. Value only supports 0 for now.

* @par Outputs:
* y: A Tensor. Format supports ND.
* The data type supports float32.

*\n
*\n
* - The following are the supported quantization modes, data types and shapes(for Ascend950PR/Ascend950DT):
*\n
| quantization         | x1 type                   | x2 type                   | scale1 type                    | scale2 type                    | x1 shape  | x2 shape            | y shape   | scale2 shape                  | scale1 shape          |
|----------------------|---------------------------|---------------------------|--------------------------------|--------------------------------|-----------|---------------------|-----------|-------------------------------|-----------------------|
| pertensor-pertensor  | hifloat8                  | hifloat8                  | float32                        | float32                        | (K,M)     | (K,N)               | (B,M,N)   | (B,)/(B,1)                    | (B,)/(B,1)            |
| pertensor-perchannel | hifloat8                  | hifloat8                  | float32                        | float32                        | (K,M)     | (K,N)               | (B,M,N)   | (B,N)                         | (B,)/(B,1)            |
| mx                   | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float8_e8m0                    | float8_e8m0                    | (K,M)     | (K,N)               | (B,M,N)   | (K/64+B,N,2)                  | (K/64+B,M,2)          |

*\n
* @attention Constraints
* @li When group_list_type is 0, group_list must be a non-negative monotone non-decreasing array, when group_list_type is 1, it must be a non-negative array.
* @li Each axis for tensor x1 and tensor x2 should be less than or equal to 2147483647 (maximum of data type int32) after aligning to 32 byte.
* @li Input tensor x1 and x2 should have dim num 2, input tensor y should have dim num 3.
* @li The group_list must be passed, if group_list_type equals to 0, the last value in group_list must be no greater than the first dimension of tensor x1,
* if group_list_type equals to 1, sum of values in group_list must be no greater than the first dimension of tensor x1. \n
*/
// clang-format on
REG_OP(QuantGroupedMatmulInplaceAdd)
    .INPUT(x1, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(scale2, TensorType({DT_FLOAT8_E8M0, DT_FLOAT}))
    .INPUT(group_list, TensorType({DT_INT64}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(scale1, TensorType({DT_FLOAT8_E8M0, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(group_list_type, Int, 0)
    .ATTR(group_size, Int, 0)
    .OP_END_FACTORY_REG(QuantGroupedMatmulInplaceAdd)

} // namespace ge

#endif
