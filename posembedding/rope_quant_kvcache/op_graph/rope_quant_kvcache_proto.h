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
 * \file rope_quant_kvcache_proto.cpp
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_REPO_QUANT_KVCACHE_H_
#define OPS_OP_PROTO_INC_REPO_QUANT_KVCACHE_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Fusion ops for splitvd rope quantize scatter.

* @par Inputs:
* eight inputs, including:
* @li qkv: A 3D Tensor of type float16 with shape (B, S, H), H is (Nq+Nkv+Nkv)*D, format support ND
* @li cos: A 4D Tensor of type float16 with shape (B, S, 1, D), shape must same with k, format support ND
* @li sin: A 4D Tensor of type float16 with shape (B, S, 1, D), shape must same with k, format support ND
* @li quant_scale: A 1D Tensor of type float with shape (D), shape D must same with k, format support ND
* @li quant_offset: A 1D Tensor of type int32 with shape (D), shape D must same with k, format support ND
* @li k_cache: A 4D Tensor of type int8 with shape (B, S, Nkv, D), shape B/N/D must same with k,
* S must large than k, format support ND
* @li v_cache: A 4D Tensor of type int8 with shape (B, S, Nkv, D), shape B/N/D must same with v,
* S must large than k, format support ND
* @li indice: A 1D Tensor of type int32 with shape (B), shape must same with qkv, format support ND

* @par Attributes:
* @li size_splits: list to split qkv input tensor to q/k/v, default is null.
* @li layout: qkv input tensor layout, like BSND/BNSD, default is BSND.
* @li kv_output: control origin k/v output or not, default is false.

* @par Outputs:
* @li q: A 4D Tensor of type float16 with shape (B, S, Nq, D), split from qkv, format support ND
* @li k: A 4D Tensor of type float16 with shape (B, S, Nkv, D), split from qkv, N must same with v, format support ND
* @li v: A 4D Tensor of type float16 with shape (B, S, Nkv, D), split from qkv, N must same with k, format support ND
* @li k_cache: A 4D Tensor of type int8 with shape (B, S, Nkv, D), shape B/N/D must same with k, S must large than k,
* format support ND
* @li v_cache: A 4D Tensor of type int8 with shape (B, S, Nkv, D), shape B/N/D must same with v, S must large than v,
* format support ND
* It is a custom operator.
*/
REG_OP(RopeQuantKvcache)
    .INPUT(qkv, TensorType({DT_FLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16}))
    .INPUT(quant_scale, TensorType({DT_FLOAT32}))
    .INPUT(quant_offset, TensorType({DT_INT32}))
    .INPUT(k_cache, TensorType({DT_INT8}))
    .INPUT(v_cache, TensorType({DT_INT8}))
    .INPUT(indice, TensorType({DT_INT32}))
    .OUTPUT(q, TensorType({DT_FLOAT16}))
    .OUTPUT(k, TensorType({DT_FLOAT16}))
    .OUTPUT(v, TensorType({DT_FLOAT16}))
    .OUTPUT(k_cache, TensorType({DT_INT8}))
    .OUTPUT(v_cache, TensorType({DT_INT8}))
    .ATTR(size_splits, ListInt, {})
    .ATTR(layout, String, "BSND")
    .ATTR(kv_output, Bool, false)
    .OP_END_FACTORY_REG(RopeQuantKvcache)
} // namespace ge

#endif // OPS_OP_PROTO_INC_REPO_QUANT_KVCACHE_H_
