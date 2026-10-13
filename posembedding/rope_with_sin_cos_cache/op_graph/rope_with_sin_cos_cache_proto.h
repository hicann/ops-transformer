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
 * \file apply_rotary_pos_emb_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ROPE_WITH_SIN_COS_CACHE_OPS_H_
#define OPS_OP_PROTO_INC_ROPE_WITH_SIN_COS_CACHE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief RopeWithSinCosCache.

* @par Inputs:
* @li positions: A tensor of type int32 of int64.
* @li queryIn: A tensor of type float, bf16, float16.
* @li keyIn: A tensor of type float, bf16, float16.
* @li cosSinCache: A tensor of type float, bf16, float16.

* @par Attributes:
* @li numQHeads: A int attr.
* @li numKHeads: A int attr.
* @li headSize: A int attr.
* @li mropeSection: A ListInt attr.
* @li qStride: A int attr.
* @li KStride: A int attr.
* @li isNeoxStyle: A bool attr.

* @par Outputs:
* @li queryOut: A tensor of type FP16/FP32/BF16.
* @li keyOut: A tensor of type FP16/FP32/BF16.
*/

REG_OP(RopeWithSinCosCache)
    .INPUT(positions, TensorType({DT_INT32, DT_INT64}))
    .INPUT(queryIn, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(keyIn, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(cosSinCache, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(queryOut, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(keyOut, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .REQUIRED_ATTR(numQHeads, Int)
    .REQUIRED_ATTR(numKHeads, Int)
    .REQUIRED_ATTR(headSize, Int)
    .ATTR(mropeSection, ListInt, {0, 0, 0})
    .ATTR(qStride, Int, 0)
    .ATTR(kStride, Int, 0)
    .ATTR(isNeoxStyle, Bool, true)
    .OP_END_FACTORY_REG(RopeWithSinCosCache)

} // namespace ge

#endif
