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
 * \file quant_all_reduce_proto.h
 * \brief 图模式原型定义
 */
#ifndef QUANT_ALL_REDUCE_PROTO_H_
#define QUANT_ALL_REDUCE_PROTO_H_

#include <graph/operator_reg.h>

namespace ge {
/**
 * @brief Fusion op of quant all reduce.
 * @par Inputs:
 * three inputs, including:
 * @li context: A matrix tensor. The type support int32. The format supports ND.
 * @li x: A matrix tensor. The type support int8, hifloat8, float8_e4m3fn, float8_e5m2, float4_e1m2, float4_e2m1. The
 * format supports ND. In graph mode, the format supports NCHW/NHWC.
 * @li scales: A matrix tensor. The type support float32, float8_e8m0. The format supports ND.
 * In graph mode, the format supports NCHW/NHWC.
 *
 * @par Outputs:
 * out_put: A matrix tensor. The type support float16, bfloat16, float32. The format supports ND.
 * In graph mode, the format supports NCHW/NHWC.
 *
 * @par Attributes:
 * @li hccl_buffer_size: A required string identifying the ccl buffer size in the op.
 * @li reduce_op: An optional string identifying the reduction operation to perform. Default: "sum".
 * @li output_dtype: An optional int identifying the data type of output. The type support 0(float), 1(float16),
 * 27(bfloat16). Default: 27(bfloat16).
 * @li world_size: A required int identifying the rank size.
 */
REG_OP(QuantAllReduce)
    .INPUT(context, "T0")
    .INPUT(x, "T1")
    .INPUT(scales, "T2")
    .OUTPUT(out_put, "T3")
    .DATATYPE(T0, TensorType({DT_INT32}))
    .DATATYPE(T1, TensorType({DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .DATATYPE(T2, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .DATATYPE(T3, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .REQUIRED_ATTR(hccl_buffer_size, Int)
    .ATTR(reduce_op, String, "sum")
    .ATTR(output_dtype, Int, DT_BF16)
    .REQUIRED_ATTR(world_size, Int)
    .OP_END_FACTORY_REG(QuantAllReduce)

} // namespace ge

#endif // QUANT_ALL_REDUCE_PROTO_H
