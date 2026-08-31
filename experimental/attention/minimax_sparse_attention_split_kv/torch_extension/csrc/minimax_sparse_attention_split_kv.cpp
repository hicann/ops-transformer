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
 * \file minimax_sparse_attention_split_kv.cpp
 * \brief aclnn binding for MinimaxSparseAttentionSplitKv.
 */

#include <torch/extension.h>
#include <string>
#include "aclnn_common.h"

namespace op_api {
namespace {

constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;

void CheckInt32Tensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " dtype must be int32, got ", tensor.scalar_type());
}

} // namespace

std::tuple<at::Tensor, at::Tensor> minimax_sparse_attention_split_kv(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &block_table, const at::Tensor &k2q_row_ptr, const at::Tensor &k2q_q_indices,
    const at::Tensor &k2q_slot_indices, const at::Tensor &actual_seq_lengths, const at::Tensor &actual_seq_lengths_kv,
    int64_t num_key_value_heads, double scale_value, int64_t block_size, int64_t top_k, int64_t inner_precise,
    bool softmax_lse_flag, std::string input_layout)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(value.numel() > 0, "Tensor value is empty.");
    TORCH_CHECK(inner_precise == 0 || inner_precise == 1 || inner_precise == 4, "inner_precise must be 0, 1 or 4, got ",
                inner_precise);
    TORCH_CHECK(input_layout == "TND" || input_layout == "BNSD" || input_layout == "BSND",
                "input_layout must be TND, BNSD or BSND, got ", input_layout);
    TORCH_CHECK(num_key_value_heads > 0, "num_key_value_heads must be > 0, got ", num_key_value_heads);
    TORCH_CHECK(block_size > 0, "block_size must be > 0, got ", block_size);
    TORCH_CHECK(top_k > 0, "top_k must be > 0, got ", top_k);

    if (block_table.has_value() && block_table.value().defined()) {
        CheckInt32Tensor(block_table.value(), "block_table");
    }
    CheckInt32Tensor(k2q_row_ptr, "k2q_row_ptr");
    CheckInt32Tensor(k2q_q_indices, "k2q_q_indices");
    CheckInt32Tensor(k2q_slot_indices, "k2q_slot_indices");
    CheckInt32Tensor(actual_seq_lengths, "actual_seq_lengths");
    CheckInt32Tensor(actual_seq_lengths_kv, "actual_seq_lengths_kv");

    if (input_layout == "TND") {
        TORCH_CHECK(query.dim() == DIM_THREE, "TND query must be rank 3 [T, N, D], got ", query.dim());
    } else {
        TORCH_CHECK(query.dim() == DIM_FOUR, input_layout, " query must be rank 4, got ", query.dim());
    }

    at::Tensor attention_out{nullptr};
    at::Tensor softmax_lse{nullptr};
    {
        auto local_device = c10::Device(query.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        // Zero-init so BNSD/BSND padding tokens (kernel leaves them unwritten) stay 0.
        // FP8 Q/K/V still writes BF16 attentionOut (see infershape / aclnn).
        auto out_opts = query.options();
        if (query.scalar_type() == at::kFloat8_e4m3fn) {
            out_opts = out_opts.dtype(at::kBFloat16);
        }
        attention_out = at::zeros(query.sizes(), out_opts);
        if (softmax_lse_flag) {
            if (input_layout == "TND") {
                softmax_lse = at::zeros({query.size(0), query.size(1), 1}, query.options().dtype(at::kFloat));
            } else if (input_layout == "BNSD") {
                softmax_lse =
                    at::zeros({query.size(0), query.size(1), query.size(2), 1}, query.options().dtype(at::kFloat));
            } else {
                softmax_lse =
                    at::zeros({query.size(0), query.size(1), query.size(2), 1}, query.options().dtype(at::kFloat));
            }
        } else {
            softmax_lse = at::empty({0}, query.options().dtype(at::kFloat));
        }
    }

    char *layout_ptr = const_cast<char *>(input_layout.c_str());
    ACLNN_CMD(aclnnMinimaxSparseAttentionSplitKv, query, key, value, block_table, k2q_row_ptr, k2q_q_indices,
              k2q_slot_indices, actual_seq_lengths, actual_seq_lengths_kv, num_key_value_heads, scale_value, block_size,
              top_k, inner_precise, softmax_lse_flag, layout_ptr, attention_out, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(attention_out, softmax_lse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("minimax_sparse_attention_split_kv", &minimax_sparse_attention_split_kv,
          "MinimaxSparseAttentionSplitKv forward");
}

} // namespace op_api
