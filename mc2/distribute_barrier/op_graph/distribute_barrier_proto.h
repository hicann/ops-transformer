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
 * \file fusion_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief DistributeBarrier operator interface implementation.

* @par Inputs
* Three inputs, including:
* @li x_ref: An optional tensor, reserved. Support dtype:bfloat16, float16, float32, bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64. Support format: ND.
* @li time_out: An optional tensor, reserved. Support dtype:int32. Support format: ND.
* @li elastic_info: An optional tensor, reserved. Support dtype:int32. Support format: ND.

* @par Attributes
* @li group: Required. Input comm group name, means experts parallelism, dtype: String.
* @li world_size: Required. Input comm world size, dtype: int64.

* @par Outputs
* One outputs, including:
* @li x_ref: A tensor. reserved. Support dtype:bfloat16, float16, float32, bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64. Support format: ND.
*/
REG_OP(DistributeBarrier)
    .INPUT(x_ref, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OPTIONAL_INPUT(time_out, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))
    .OUTPUT(x_ref, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .OP_END_FACTORY_REG(DistributeBarrier)

}  // namespace ge


#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
