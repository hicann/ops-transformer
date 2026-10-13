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
 * \file attention_update_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_ATTENTION_UPDATE_OPS_H_
#define OPS_OP_PROTO_INC_ATTENTION_UPDATE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief  Function AttentionUpdate.

* @par Inputs:
* Two inputs, including:
* @li lse: Tensor list. Type is float32. The input of lse.
* @li go: Tensor list. Type is float32, float16, bfloat16. The input of attentionout.

* @par Outputs:
* Two outputs, including:
* @li output: A tensor. Type is float32, float16, bfloat16.
* @li lse_m: A tensor. Type is float32.

* @par Attributes:
* Two attributes, including:
* @li update_type: An int. The update type, value is 0 or 1. 0 means the output of lse_m is invalid, 1 means valid.
* @li sp: An int. The sp num, value is [1..16].
*/
REG_OP(AttentionUpdate)
    .DYNAMIC_INPUT(lse, TensorType({DT_FLOAT}))
    .DYNAMIC_INPUT(go, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(lse_m, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(update_type, Int)
    .REQUIRED_ATTR(sp, Int)
    .OP_END_FACTORY_REG(AttentionUpdate)

} // namespace ge

#endif
