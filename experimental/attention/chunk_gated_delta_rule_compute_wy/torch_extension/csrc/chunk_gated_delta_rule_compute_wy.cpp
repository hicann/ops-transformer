/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {

constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;
constexpr int64_t FIXED_CHUNK_SIZE = 64;
constexpr int64_t HEAD_DIM_ALIGNMENT = 16;
constexpr int64_t MAX_HEAD_DIM = 128;
constexpr int64_t MAX_BATCH_SIZE = 32;
constexpr int64_t MAX_VALUE_HEADS = 64;

void check_inputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                  const at::Tensor& beta, int64_t chunk_size)
{
    TORCH_CHECK(q.dim() == DIM_FOUR, "q must have shape [B, T, Hk, K], but got ", q.sizes(), ".");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must have the same shape as q, but got k=", k.sizes(), " and q=", q.sizes(),
                ".");
    TORCH_CHECK(v.dim() == DIM_FOUR, "v must have shape [B, T, Hv, V], but got ", v.sizes(), ".");
    TORCH_CHECK(g.dim() == DIM_THREE, "g must have shape [B, T, Hv], but got ", g.sizes(), ".");
    TORCH_CHECK(beta.sizes() == g.sizes(), "beta must have the same shape as g, but got beta=", beta.sizes(),
                " and g=", g.sizes(), ".");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf &&
                    beta.scalar_type() == at::kHalf,
                "q, k, v and beta must have dtype float16.");
    TORCH_CHECK(g.scalar_type() == at::kFloat, "g must have dtype float32, but got ", g.scalar_type(), ".");
    TORCH_CHECK(
        q.device() == k.device() && q.device() == v.device() && q.device() == g.device() && q.device() == beta.device(),
        "all inputs must be on the same device.");
    TORCH_CHECK(chunk_size == FIXED_CHUNK_SIZE, "only chunk_size=64 is supported, but got ", chunk_size, ".");

    const int64_t batch_size = q.size(0);
    const int64_t sequence_length = q.size(1);
    const int64_t num_key_heads = q.size(2);
    const int64_t key_dim = q.size(3);
    const int64_t num_value_heads = v.size(2);
    const int64_t value_dim = v.size(3);
    TORCH_CHECK(v.size(0) == batch_size && v.size(1) == sequence_length, "v must share the B and T dimensions with q.");
    TORCH_CHECK(g.size(0) == batch_size && g.size(1) == sequence_length && g.size(2) == num_value_heads,
                "g must have shape [B, T, Hv].");
    TORCH_CHECK(num_key_heads > 0 && num_value_heads > 0 && num_value_heads % num_key_heads == 0,
                "Hv must be divisible by Hk, but got Hv=", num_value_heads, " and Hk=", num_key_heads, ".");
    TORCH_CHECK(sequence_length > 0 && sequence_length % chunk_size == 0,
                "T must be a positive multiple of chunk_size, but got T=", sequence_length, ".");
    TORCH_CHECK(key_dim > 0 && key_dim % HEAD_DIM_ALIGNMENT == 0 && key_dim <= MAX_HEAD_DIM,
                "K must be a positive multiple of 16 and no greater than 128, but got ", key_dim, ".");
    TORCH_CHECK(value_dim > 0 && value_dim % HEAD_DIM_ALIGNMENT == 0 && value_dim <= MAX_HEAD_DIM,
                "V must be a positive multiple of 16 and no greater than 128, but got ", value_dim, ".");
    TORCH_CHECK(batch_size > 0 && batch_size <= MAX_BATCH_SIZE, "B must be in [1, 32], but got ", batch_size, ".");
    TORCH_CHECK(num_value_heads <= MAX_VALUE_HEADS, "Hv must be no greater than 64, but got ", num_value_heads, ".");
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_compute_wy(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g, const at::Tensor& beta,
    int64_t chunk_size)
{
    check_inputs(q, k, v, g, beta, chunk_size);

    const int64_t batch_size = q.size(0);
    const int64_t sequence_length = q.size(1);
    const int64_t num_key_heads = q.size(2);
    const int64_t key_dim = q.size(3);
    const int64_t num_value_heads = v.size(2);
    const int64_t value_dim = v.size(3);

    at::Tensor q_kernel;
    at::Tensor k_kernel;
    at::Tensor w_kernel;
    at::Tensor u_kernel;
    at::Tensor g_kernel;
    {
        const c10::OptionalDeviceGuard device_guard(c10::Device(q.device()));
        q_kernel = at::empty({batch_size, num_key_heads, sequence_length, key_dim}, q.options());
        k_kernel = at::empty({batch_size, num_key_heads, sequence_length, key_dim}, k.options());
        w_kernel = at::empty({batch_size, num_value_heads, sequence_length, key_dim}, q.options());
        u_kernel = at::empty({batch_size, num_value_heads, sequence_length, value_dim}, v.options());
        g_kernel = at::empty({batch_size, num_value_heads, sequence_length}, g.options().dtype(at::kFloat));
    }

    ACLNN_CMD(aclnnChunkGatedDeltaRuleComputeWy, q, k, v, g, beta, chunk_size, q_kernel, k_kernel, w_kernel, u_kernel,
              g_kernel);
    return {q_kernel, k_kernel, w_kernel, u_kernel, g_kernel};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("chunk_gated_delta_rule_compute_wy", &chunk_gated_delta_rule_compute_wy, "chunk_gated_delta_rule_compute_wy");
}

} // namespace op_api
