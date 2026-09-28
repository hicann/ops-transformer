/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <torch/extension.h>
#include <vector>
#include <string>
#include <cstring>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;
const int LAYOUT_MAX_LENGTH = 20;

namespace {
at::Tensor format_trans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
                    "Expected all tensors to be on the same device. "
                    "Expected NPU tensor, please check whether the input tensor device is correct.");
        // The Python adapter uses the public torch_npu format-cast API.
        return at_tensor;
    }
    return at_tensor;
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> lightning_indexer_grad_kl_loss(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &query_index, const at::Tensor &key_index,
    const at::Tensor &weights, const at::Tensor &sparse_indices, const at::Tensor &softmax_max,
    const at::Tensor &softmax_sum, const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, const c10::optional<std::vector<int64_t>> &cur_seq_lengths_query,
    const c10::optional<std::vector<int64_t>> &cur_seq_lengths_key, double scale_value,
    const c10::optional<std::string> &layout, c10::optional<int64_t> sparse_mode, c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens, int64_t block_size, c10::optional<bool> deterministic)
{
    TORCH_CHECK(query.is_contiguous(), "query should be contiguous.");
    TORCH_CHECK(key.is_contiguous(), "key should be contiguous.");
    TORCH_CHECK(query_index.is_contiguous(), "query_index should be contiguous.");
    TORCH_CHECK(weights.is_contiguous(), "weights should be contiguous.");
    TORCH_CHECK(sparse_indices.is_contiguous(), "sparse_indices should be contiguous.");
    TORCH_CHECK(softmax_max.is_contiguous(), "softmax_max should be contiguous.");
    TORCH_CHECK(softmax_sum.is_contiguous(), "softmax_sum should be contiguous.");
    const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
    const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());
    const at::Tensor &mask_const = at::Tensor();
    at::IntArrayRef cur_seq_lengths_query_const =
        cur_seq_lengths_query.has_value() ? at::IntArrayRef(*cur_seq_lengths_query) : at::IntArrayRef{};
    at::IntArrayRef cur_seq_lengths_key_const =
        cur_seq_lengths_key.has_value() ? at::IntArrayRef(*cur_seq_lengths_key) : at::IntArrayRef{};
    std::string layout_str = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str.data());
    int64_t sparse_mode_const = sparse_mode.value_or(3);
    int64_t pre_tokens_const = pre_tokens.value_or(2147483647);
    int64_t next_tokens_const = next_tokens.value_or(2147483647);
    bool deterministic_const = deterministic.value_or(false);
    int64_t valid_token_num_const = -1;
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
                "The shapes of the input query should be 3 or 4 dimensional, but got ", query.dim(), "-dimensional");
    if (query_rope_const.defined()) {
        TORCH_CHECK(query_rope_const.dim() == DIMENSION_3D || query_rope_const.dim() == DIMENSION_4D,
                    "The shapes of the input query_rope should be 3 or 4 dimensional, but got ", query_rope_const.dim(),
                    "-dimensional");
    }
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
                "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(), "-dimensional");
    if (key_rope_const.defined()) {
        TORCH_CHECK(key_rope_const.dim() == DIMENSION_3D || key_rope_const.dim() == DIMENSION_4D,
                    "The shapes of the input key_rope should be 3 or 4 dimensional, but got ", key_rope_const.dim(),
                    "-dimensional");
    }
    at::Tensor format_query = format_trans(query);
    at::Tensor format_key = format_trans(key);
    at::Tensor format_query_index = format_trans(query_index);
    at::Tensor format_key_index = format_trans(key_index);
    at::Tensor format_weights = format_trans(weights);
    at::Tensor format_sparse_indices = format_trans(sparse_indices);
    at::Tensor format_softmax_max = format_trans(softmax_max);
    at::Tensor format_softmax_sum = format_trans(softmax_sum);
    at::Tensor format_query_rope = format_trans(query_rope_const);
    at::Tensor format_key_rope = format_trans(key_rope_const);
    at::Tensor format_mask = format_trans(mask_const);
    at::Tensor d_query_index = at::empty(format_query_index.sizes(), format_query_index.options());
    at::Tensor d_key_index = at::empty(format_key_index.sizes(), format_key_index.options());
    at::Tensor d_weights = at::empty(format_weights.sizes(), format_weights.options());
    at::Tensor loss = at::empty({1}, query.options().dtype(at::kFloat));

    ACLNN_CMD(aclnnLightningIndexerGradKLLoss, format_query, format_key, format_query_index, format_key_index,
              format_weights, format_sparse_indices, format_softmax_max, format_softmax_sum, format_query_rope,
              format_key_rope, format_mask, cur_seq_lengths_query_const, cur_seq_lengths_key_const, scale_value,
              layout_ptr, sparse_mode_const, pre_tokens_const, next_tokens_const, block_size, deterministic_const,
              valid_token_num_const, d_query_index, d_key_index, d_weights, loss);

    return std::make_tuple(d_query_index, d_key_index, d_weights, loss);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> lightning_indexer_grad_kl_loss_skip_padding(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &query_index, const at::Tensor &key_index,
    const at::Tensor &weights, const at::Tensor &sparse_indices, const at::Tensor &softmax_max,
    const at::Tensor &softmax_sum, const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, const c10::optional<at::Tensor> &mask,
    const c10::optional<std::vector<int64_t>> &cur_seq_lengths_query,
    const c10::optional<std::vector<int64_t>> &cur_seq_lengths_key, double scale_value,
    const c10::optional<std::string> &layout, c10::optional<int64_t> sparse_mode, c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens, int64_t block_size, c10::optional<bool> deterministic,
    c10::optional<int64_t> valid_token_num)
{
    TORCH_CHECK(query.is_contiguous(), "query should be contiguous.");
    TORCH_CHECK(key.is_contiguous(), "key should be contiguous.");
    TORCH_CHECK(query_index.is_contiguous(), "query_index should be contiguous.");
    TORCH_CHECK(weights.is_contiguous(), "weights should be contiguous.");
    TORCH_CHECK(sparse_indices.is_contiguous(), "sparse_indices should be contiguous.");
    TORCH_CHECK(softmax_max.is_contiguous(), "softmax_max should be contiguous.");
    TORCH_CHECK(softmax_sum.is_contiguous(), "softmax_sum should be contiguous.");
    const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
    const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());
    const at::Tensor &mask_const = mask.value_or(at::Tensor());
    at::IntArrayRef cur_seq_lengths_query_const =
        cur_seq_lengths_query.has_value() ? at::IntArrayRef(*cur_seq_lengths_query) : at::IntArrayRef{};
    at::IntArrayRef cur_seq_lengths_key_const =
        cur_seq_lengths_key.has_value() ? at::IntArrayRef(*cur_seq_lengths_key) : at::IntArrayRef{};
    int64_t valid_token_num_const = valid_token_num.value_or(-1);
    std::string layout_str = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str.data());
    int64_t sparse_mode_const = sparse_mode.value_or(3);
    int64_t pre_tokens_const = pre_tokens.value_or(2147483647);
    int64_t next_tokens_const = next_tokens.value_or(2147483647);
    bool deterministic_const = deterministic.value_or(false);
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
                "The shapes of the input query should be 3 or 4 dimensional, but got ", query.dim(), "-dimensional");
    if (query_rope_const.defined()) {
        TORCH_CHECK(query_rope_const.dim() == DIMENSION_3D || query_rope_const.dim() == DIMENSION_4D,
                    "The shapes of the input query_rope should be 3 or 4 dimensional, but got ", query_rope_const.dim(),
                    "-dimensional");
    }
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
                "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(), "-dimensional");
    if (key_rope_const.defined()) {
        TORCH_CHECK(key_rope_const.dim() == DIMENSION_3D || key_rope_const.dim() == DIMENSION_4D,
                    "The shapes of the input key_rope should be 3 or 4 dimensional, but got ", key_rope_const.dim(),
                    "-dimensional");
    }
    at::Tensor format_query = format_trans(query);
    at::Tensor format_key = format_trans(key);
    at::Tensor format_query_index = format_trans(query_index);
    at::Tensor format_key_index = format_trans(key_index);
    at::Tensor format_weights = format_trans(weights);
    at::Tensor format_sparse_indices = format_trans(sparse_indices);
    at::Tensor format_softmax_max = format_trans(softmax_max);
    at::Tensor format_softmax_sum = format_trans(softmax_sum);
    at::Tensor format_query_rope = format_trans(query_rope_const);
    at::Tensor format_key_rope = format_trans(key_rope_const);
    at::Tensor format_mask = format_trans(mask_const);
    at::Tensor d_query_index = ((valid_token_num_const > 0) || (mask.has_value())) ?
                                   at::zeros_like(format_query_index) :
                                   at::empty(format_query_index.sizes(), format_query_index.options());
    at::Tensor d_key_index = at::empty(format_key_index.sizes(), format_key_index.options());
    at::Tensor d_weights = ((valid_token_num_const > 0) || (mask.has_value())) ?
                               at::zeros_like(format_weights) :
                               at::empty(format_weights.sizes(), format_weights.options());
    at::Tensor loss = at::empty({1}, query.options().dtype(at::kFloat));

    ACLNN_CMD(aclnnLightningIndexerGradKLLoss, format_query, format_key, format_query_index, format_key_index,
              format_weights, format_sparse_indices, format_softmax_max, format_softmax_sum, format_query_rope,
              format_key_rope, format_mask, cur_seq_lengths_query_const, cur_seq_lengths_key_const, scale_value,
              layout_ptr, sparse_mode_const, pre_tokens_const, next_tokens_const, block_size, deterministic_const,
              valid_token_num_const, d_query_index, d_key_index, d_weights, loss);

    return std::make_tuple(d_query_index, d_key_index, d_weights, loss);
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lightning_indexer_grad_kl_loss", &op_api::lightning_indexer_grad_kl_loss);
    m.def("lightning_indexer_grad_kl_loss_skip_padding", &op_api::lightning_indexer_grad_kl_loss_skip_padding);
}
