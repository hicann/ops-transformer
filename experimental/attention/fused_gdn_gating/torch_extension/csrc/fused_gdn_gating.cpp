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

constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;

} // namespace

std::tuple<at::Tensor, at::Tensor> fused_gdn_gating(const at::Tensor& a_log, const at::Tensor& a, const at::Tensor& b,
                                                    const at::Tensor& dt_bias, double beta, double threshold)
{
    TORCH_CHECK(a_log.dim() == DIM_ONE, "a_log must be 1D [num_heads], but got ", a_log.dim(), "D.");
    TORCH_CHECK(dt_bias.dim() == DIM_ONE, "dt_bias must be 1D [num_heads], but got ", dt_bias.dim(), "D.");
    TORCH_CHECK(a.dim() == DIM_TWO, "a must be 2D [batch, num_heads], but got ", a.dim(), "D.");
    TORCH_CHECK(b.dim() == DIM_TWO, "b must be 2D [batch, num_heads], but got ", b.dim(), "D.");
    TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0, "a dimensions must be greater than zero, but got ", a.sizes(), ".");
    TORCH_CHECK(b.sizes() == a.sizes(), "b must have the same shape as a, but got b=", b.sizes(), " and a=", a.sizes(),
                ".");
    TORCH_CHECK(dt_bias.sizes() == a_log.sizes(),
                "dt_bias must have the same shape as a_log, but got dt_bias=", dt_bias.sizes(),
                " and a_log=", a_log.sizes(), ".");
    TORCH_CHECK(a.size(1) == a_log.size(0), "a.shape[1] must equal a_log.shape[0], but got ", a.size(1), " and ",
                a_log.size(0), ".");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
                "a dtype must be float16 or bfloat16, but got ", a.scalar_type(), ".");
    TORCH_CHECK(b.scalar_type() == a.scalar_type(), "b dtype must match a, but got b=", b.scalar_type(),
                " and a=", a.scalar_type(), ".");
    TORCH_CHECK(
        a_log.scalar_type() == at::kFloat || a_log.scalar_type() == at::kHalf || a_log.scalar_type() == at::kBFloat16,
        "a_log dtype must be float32, float16 or bfloat16, but got ", a_log.scalar_type(), ".");
    TORCH_CHECK(dt_bias.scalar_type() == a_log.scalar_type(),
                "dt_bias dtype must match a_log, but got dt_bias=", dt_bias.scalar_type(),
                " and a_log=", a_log.scalar_type(), ".");
    TORCH_CHECK(a.device() == a_log.device() && b.device() == a_log.device() && dt_bias.device() == a_log.device(),
                "all inputs must be on the same device.");

    at::Tensor g;
    at::Tensor beta_output;
    {
        const c10::OptionalDeviceGuard device_guard(c10::Device(a.device()));
        g = at::empty({1, a.size(0), a.size(1)}, a.options().dtype(at::kFloat));
        beta_output = at::empty({1, a.size(0), a.size(1)}, b.options());
    }

    // The aclnn ABI expects float attributes while a Torch schema float is received as double in C++.
    const float beta_value = static_cast<float>(beta);
    const float threshold_value = static_cast<float>(threshold);
    ACLNN_CMD(aclnnFusedGdnGating, a_log, a, b, dt_bias, beta_value, threshold_value, g, beta_output);
    return {g, beta_output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_gdn_gating", &fused_gdn_gating, "fused_gdn_gating");
}

} // namespace op_api
