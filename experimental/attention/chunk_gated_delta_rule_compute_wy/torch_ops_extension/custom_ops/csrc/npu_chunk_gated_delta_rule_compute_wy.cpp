/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>

#include <torch/library.h>

#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

namespace {
constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;
constexpr int64_t FIXED_CHUNK = 64;
constexpr int64_t HEAD_DIM_ALIGN = 16;
// 192KB UB on Atlas 推理系列产品; the two-pass W/U solve fits head dim 128.
constexpr int64_t MAX_HEAD_DIM = 128;
constexpr int64_t MAX_BATCH = 32;
constexpr int64_t MAX_VALUE_HEADS = 64;

void check_chunk_gated_delta_rule_compute_wy(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                                             const at::Tensor &g, const at::Tensor &beta, int64_t chunk_size)
{
    TORCH_CHECK(q.dim() == DIM_FOUR, "q must be [B, T, Hk, K], got ", q.sizes());
    TORCH_CHECK(k.sizes() == q.sizes(), "k must have the same shape as q, got k=", k.sizes(), " q=", q.sizes());
    TORCH_CHECK(v.dim() == DIM_FOUR, "v must be [B, T, Hv, V], got ", v.sizes());
    TORCH_CHECK(g.dim() == DIM_THREE, "g must be [B, T, Hv], got ", g.sizes());
    TORCH_CHECK(beta.sizes() == g.sizes(), "beta must have the same shape as g, got beta=", beta.sizes(),
                " g=", g.sizes());
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf &&
                    beta.scalar_type() == at::kHalf,
                "q/k/v/beta must be float16.");
    TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be float32.");
    TORCH_CHECK(chunk_size == FIXED_CHUNK, "only chunk_size=64 is supported, got ", chunk_size);

    const int64_t b = q.size(0);
    const int64_t t = q.size(1);
    const int64_t hk = q.size(2);
    const int64_t kDim = q.size(3);
    const int64_t hv = v.size(2);
    const int64_t vDim = v.size(3);
    TORCH_CHECK(v.size(0) == b && v.size(1) == t, "v must share B/T with q.");
    TORCH_CHECK(g.size(0) == b && g.size(1) == t && g.size(2) == hv, "g must match [B, T, Hv].");
    TORCH_CHECK(hk > 0 && hv % hk == 0, "Hv must be divisible by Hk, got Hv=", hv, " Hk=", hk);
    TORCH_CHECK(t % chunk_size == 0, "T must be padded to a multiple of chunk_size, got T=", t);
    TORCH_CHECK(kDim % HEAD_DIM_ALIGN == 0, "K must be a multiple of 16, got ", kDim);
    TORCH_CHECK(vDim % HEAD_DIM_ALIGN == 0, "V must be a multiple of 16, got ", vDim);
    TORCH_CHECK(kDim <= MAX_HEAD_DIM, "K must be <= 128, got ", kDim);
    TORCH_CHECK(vDim <= MAX_HEAD_DIM, "V must be <= 128, got ", vDim);
    TORCH_CHECK(b <= MAX_BATCH, "B must be <= 32, got ", b);
    TORCH_CHECK(hv <= MAX_VALUE_HEADS, "Hv must be <= 64, got ", hv);
}

// q/k -> [B, Hk, T, K]; w -> [B, Hv, T, K]; u -> [B, Hv, T, V]; g -> [B, Hv, T] fp32.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> construct_compute_wy_output_tensors(
    const at::Tensor &q, const at::Tensor &v, const at::Tensor &g)
{
    const int64_t b = q.size(0);
    const int64_t t = q.size(1);
    const int64_t hk = q.size(2);
    const int64_t kDim = q.size(3);
    const int64_t hv = v.size(2);
    const int64_t vDim = v.size(3);

    at::Tensor qKernel = at::empty({b, hk, t, kDim}, q.options());
    at::Tensor kKernel = at::empty({b, hk, t, kDim}, q.options());
    at::Tensor wKernel = at::empty({b, hv, t, kDim}, q.options());
    at::Tensor uKernel = at::empty({b, hv, t, vDim}, v.options());
    at::Tensor gKernel = at::empty({b, hv, t}, g.options().dtype(at::kFloat));
    return std::make_tuple(qKernel, kKernel, wKernel, uKernel, gKernel);
}
} // namespace

// 为NPU设备实现前向接口（函数形参顺序 = schema 顺序）
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_chunk_gated_delta_rule_compute_wy_npu(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g, const at::Tensor &beta,
    int64_t chunk_size)
{
    check_chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, chunk_size);
    auto outputs = construct_compute_wy_output_tensors(q, v, g);
    at::Tensor &qKernel = std::get<0>(outputs);
    at::Tensor &kKernel = std::get<1>(outputs);
    at::Tensor &wKernel = std::get<2>(outputs);
    at::Tensor &uKernel = std::get<3>(outputs);
    at::Tensor &gKernel = std::get<4>(outputs);

    EXEC_NPU_CMD_V1(aclnnChunkGatedDeltaRuleComputeWy, q, k, v, g, beta, chunk_size, qKernel, kKernel, wKernel, uKernel,
                    gKernel);
    return outputs;
}

// 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_chunk_gated_delta_rule_compute_wy_meta(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g, const at::Tensor &beta,
    int64_t chunk_size)
{
    check_chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, chunk_size);
    return construct_compute_wy_output_tensors(q, v, g);
}
} // namespace custom

// 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_chunk_gated_delta_rule_compute_wy", &custom::npu_chunk_gated_delta_rule_compute_wy_npu);
}

// 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_chunk_gated_delta_rule_compute_wy", &custom::npu_chunk_gated_delta_rule_compute_wy_meta);
}
