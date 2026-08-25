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
 * \file recurrent_gated_delta_rule_proto.h
 * \brief recurrent_gated_delta_rule operator prototype definition
 */

#ifndef RECURRENT_GATED_DELTA_RULE_PROTO_H_
#define RECURRENT_GATED_DELTA_RULE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Recurrent Gated Delta Rule operator interface implementation.
 *
 * @par Inputs:
 * @li query: queries of shape [T, Nk, Dk]. Support dtype: bfloat16. Support format: ND.
 * @li key: keys of shape [T, Nk, Dk]. Support dtype: bfloat16. Support format: ND.
 * @li value: values of shape [T, Nv, Dv]. Support dtype: bfloat16. Support format: ND.
 * @li beta: betas of shape [T, Nv]. Support dtype: bfloat16. Support format: ND.
 * @li state: initial states of shape [BlockNum, Nv, Dv, Dk]. Support dtype: bfloat16, float32. Support format: ND.
 * @li actual_seq_lengths: actual sequence length of shape [B,] used for variable-length training. Support dtype: int32.
 * Support format: ND.
 * @li ssm_state_indices: indices to map the input sequences to the states of shape [T,]. Support dtype: int32. Support
 * format: ND.
 * @li g: decays of shape [T, Nv], alpha = e^g. Support dtype: float32. Support format: ND.
 * @li gk: key decays of shape [T, Nv, Dk], alpha = e^g. Support dtype: float32. Support format: ND.
 * @li num_accepted_tokens: number of accepted tokens for each sequence during decoding of shape (B,). Support dtype:
 * int32. Support format: ND.
 *
 * @par Attributes:
 * scale_value: scale factor for the RetNet attention scores, usually 1/sqrt(Dk), the default value is 1.0. dtype:
 * float32.
 *
 * @par Output:
 * @li out: outputs of shape [T, Nv,Dv]. Support dtype: bfloat16. Support format: ND.
 * @li state: final states of shape [BlockNum, Nv, Dv, Dk]. Support dtype: bfloat16, float32. Support format: ND.
 */
#ifndef OPS_PROTO_DEF_RECURRENTGATEDDELTARULE
#define OPS_PROTO_DEF_RECURRENTGATEDDELTARULE
REG_OP(RecurrentGatedDeltaRule)
    .INPUT(query, TensorType({DT_BF16}))
    .INPUT(key, TensorType({DT_BF16}))
    .INPUT(value, TensorType({DT_BF16}))
    .INPUT(beta, TensorType({DT_BF16}))
    .INPUT(state, TensorType({DT_BF16, DT_FLOAT}))
    .INPUT(actual_seq_lengths, TensorType({DT_INT32}))
    .INPUT(ssm_state_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(g, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(gk, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(num_accepted_tokens, TensorType({DT_INT32}))
    .OUTPUT(out, TensorType({DT_BF16}))
    .OUTPUT(state, TensorType({DT_BF16, DT_FLOAT}))
    .ATTR(scale_value, Float, 1.0)
    .OP_END_FACTORY_REG(RecurrentGatedDeltaRule)
#endif
} // namespace ge

#endif // RECURRENT_GATED_DELTA_RULE_PROTO_H_
