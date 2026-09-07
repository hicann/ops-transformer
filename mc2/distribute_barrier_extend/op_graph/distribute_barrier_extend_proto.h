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
 * \file distribute_barrier_extend_proto.h
 * \brief
 */
#ifndef DISTRIBUTE_BARRIER_EXTEND_PROTO_H_
#define DISTRIBUTE_BARRIER_EXTEND_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief DistributeBarrierExtend operator interface implementation.

* @par Inputs
* Three inputs, including:
* @li context: A tensor. Support dtype: int32, dimension must be 1, Support Shape (2052, ), support format: ND.
* @li x_ref: An tensor, reserved. Support dtype:bfloat16, float16, float32, bool, int8, int16, int32, int64, uint8,
  uint16, uint32, uint64, float8_e5m2, float_8e4m3fn, float4_e1m2, float4_e2m1, hifloat8, int4. Support format: ND.
* @li time_out: An optional tensor. Support dtype:int32. Support format: ND.
* @li elastic_info: An optional tensor. Support dtype:int32. Support format: ND.

* @par Attributes
* @li group: Required. Input comm group name, means experts parallelism, dtype: String.
* @li world_size: Required. Input comm world size, dtype: int64.

* @par Outputs
* One outputs, including:
* @li x_ref: A tensor. reserved. Support dtype:bfloat16, float16, float32, bool, int8, int16, int32, int64, uint8,
  uint16, uint32, uint64, float8_e5m2, float_8e4m3fn, float4_e1m2, float4_e2m1, hifloat8, int4.Support format: ND.
*/
REG_OP(DistributeBarrierExtend)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(x_ref, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                              DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2,
                              DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_INT4}))
    .OPTIONAL_INPUT(time_out, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))
    .OUTPUT(x_ref, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_BOOL, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                               DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2,
                               DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_INT4}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .OP_END_FACTORY_REG(DistributeBarrierExtend)

} // namespace ge

#endif // DISTRIBUTE_BARRIER_EXTEND_PROTO_H_
