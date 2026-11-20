/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ENGRAM_GATE_PROTO_H
#define ENGRAM_GATE_PROTO_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(EngramGate)
    .INPUT(hidden_states, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(kv, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight_fused, TensorType({DT_FLOAT}))
    .OUTPUT(output, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(dot, TensorType({DT_FLOAT}))
    .OUTPUT(gate_score, TensorType({DT_FLOAT}))
    .OUTPUT(rstd_x, TensorType({DT_FLOAT}))
    .OUTPUT(rstd_k, TensorType({DT_FLOAT}))
    .ATTR(eps, Float, 1e-6)
    .ATTR(clamp_value, Float, 1e-6)
    .ATTR(save_for_backward, Bool, true)
    .OP_END_FACTORY_REG(EngramGate)
}
#endif
