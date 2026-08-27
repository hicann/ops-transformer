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
 * \file fused_causal_conv1d_.cpp
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
static int64_t parseActivation(const c10::optional<c10::string_view> &activation)
{
    c10::string_view activation_str = activation.value_or("None");
    std::string input_activation = std::string(activation_str);
    if (input_activation == "silu") {
        return 1;
    } else if (input_activation == "swish") {
        return 2;
    }
    return 0;
}

void InplaceFusedCausalConv1d(
    const at::Tensor &x, const at::Tensor &weight, at::Tensor &conv_states,
    const c10::optional<at::Tensor> &query_start_loc, const c10::optional<at::Tensor> &cache_indices,
    const c10::optional<at::Tensor> &initial_state_mode, const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &num_accepted_tokens, const c10::optional<at::Tensor> &num_computed_tokens,
    const c10::optional<at::Tensor> &block_idx_first_scheduled_token,
    const c10::optional<at::Tensor> &block_idx_last_scheduled_token, const c10::optional<at::Tensor> &initial_state_idx,
    c10::optional<c10::string_view> activation, c10::optional<int64_t> pad_slot_id,
    c10::optional<int64_t> max_query_len, c10::optional<int64_t> residual_connection, c10::optional<int64_t> block_size,
    c10::optional<int64_t> conv_mode, c10::optional<int64_t> max_draft_tokens)
{
    int64_t activation_value = parseActivation(activation);
    int64_t pad_slot_id_value = pad_slot_id.value_or(-1);
    int64_t run_mode_value = 0;
    int64_t max_query_len_value = max_query_len.value_or(-1);
    int64_t residual_connection_value = residual_connection.value_or(0);
    int64_t block_size_value = block_size.value_or(128);
    int64_t conv_mode_value = conv_mode.value_or(1);
    int64_t max_draft_tokens_value = max_draft_tokens.value_or(7);
    ACLNN_CMD(aclnnInplaceFusedCausalConv1dV2, x, weight, conv_states, query_start_loc, cache_indices,
              initial_state_mode, bias, num_accepted_tokens, num_computed_tokens, block_idx_first_scheduled_token,
              block_idx_last_scheduled_token, initial_state_idx, activation_value, pad_slot_id_value, run_mode_value,
              max_query_len_value, residual_connection_value, block_size_value, conv_mode_value,
              max_draft_tokens_value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("inplace_fused_causal_conv1d", &InplaceFusedCausalConv1d, "inplace_fused_causal_conv1d");
}

} // namespace op_api
