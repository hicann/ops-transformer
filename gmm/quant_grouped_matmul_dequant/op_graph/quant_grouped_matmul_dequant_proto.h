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
 * \file quant_grouped_matmul_dequant_proto.h
 * \brief
 */
#ifndef OPS_TRANSFORMER_QUANT_GROUPED_MATMUL_DEQUANT_PROTO_H_
#define OPS_TRANSFORMER_QUANT_GROUPED_MATMUL_DEQUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief QuantGroupedMatmulDequant operator interface implementation.

* @par Inputs
* @li x: A tensor. Quantized input data in combination operations. Support dtype: float16, dimension must be 2, support
format: ND.
* @li quantized_weight: A tensor. Weights used for quantitative calculations. Support dtype: int8, dimension must be 3,
support format: NZ.
* @li weight_scale: A tensor. Quantization coefficient for weight. Support dtype: float32 and int64, support format: ND.
* @li group_list: A tensor. The cumsum result (cumulative sum) of the matmul size distribution representing the x and
out group axis direction.
* Support dtype: int64, support format: ND.
* @li bias: An optional input bias tensor. Support dtype: int32, support format: ND.
* @li x_scale: A optional tensor. Indicates the quantization coefficient of x. Support dtype: float32 and float16,
support format: ND.
* @li x_offset: A optional tensor. Indicates the offset of the input x. Support dtype: float32 and float16, support
format: ND.
* @li smooth_scale: A optional tensor. The smoothing coefficient for x. Support dtype: float16, support format: ND.

* @par Attributes
* @li x_quant_mode: dtype: String. Quantization mode for input x.
* @li transpose_weight: dtype: Bool. Indicates whether the input weight is transposed.

* @par Outputs
* y: A tensor. The result of the combination operation. Has the same dtype and format as x.
* The shape supports 2D, with each dimension representing: (m, n). Where m is consistent with the m of x,
* and n is consistent with the n of quantized_weight.
*/
REG_OP(QuantGroupedMatmulDequant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(quantized_weight, TensorType({DT_INT8}))
    .INPUT(weight_scale, TensorType({DT_FLOAT, DT_INT64}))
    .INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_scale, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(x_offset, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(smooth_scale, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(x_quant_mode, String, "pertoken")
    .ATTR(transpose_weight, Bool, true)
    .OP_END_FACTORY_REG(QuantGroupedMatmulDequant)

} // namespace ge

#endif
