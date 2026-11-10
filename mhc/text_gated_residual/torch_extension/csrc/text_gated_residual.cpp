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
using OptionalTensor = c10::optional<at::Tensor>;
std::tuple<at::Tensor, at::Tensor, at::Tensor, OptionalTensor, OptionalTensor> TextGatedResidualBackward(
    const at::Tensor& grad_mixed, const OptionalTensor& grad_inject, const at::Tensor& x, const at::Tensor& w_down,
    const at::Tensor& w_up, const OptionalTensor& w_gate, const OptionalTensor& gamma, const at::Tensor& inv_rms,
    const OptionalTensor& inject, const at::Tensor& down_pre_act, const at::Tensor& up, int64_t channels)
{
    constexpr int64_t INPUT_RANK = 3;
    TORCH_CHECK(x.dim() == INPUT_RANK, "x must be rank-3 [B,S,D]");
    TORCH_CHECK(channels > 0, "channels must be positive");
    TORCH_CHECK(x.size(-1) % channels == 0, "D must be divisible by channels");
    TORCH_CHECK(grad_inject.has_value() == w_gate.has_value() && inject.has_value() == w_gate.has_value(),
                "grad_inject, w_gate and inject must be provided or omitted together");
    const c10::OptionalDeviceGuard device_guard(x.device());
    auto dx = at::empty_like(x);
    auto dw_down = at::empty_like(w_down);
    auto dw_up = at::empty_like(w_up);
    OptionalTensor dw_gate = c10::nullopt;
    OptionalTensor dgamma = c10::nullopt;
    if (w_gate.has_value()) {
        dw_gate = at::empty_like(*w_gate);
    }
    if (gamma.has_value()) {
        dgamma = at::empty_like(*gamma);
    }
    ACLNN_CMD(aclnnTextGatedResidualGrad, grad_mixed, x, w_down, w_up, inv_rms, down_pre_act, up, grad_inject, w_gate,
              gamma, inject, channels, dx, dw_down, dw_up, dw_gate, dgamma);
    return {dx, dw_down, dw_up, dw_gate, dgamma};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> text_gated_residual(
    const at::Tensor& x, const at::Tensor& w_down, const at::Tensor& w_up, const c10::optional<at::Tensor>& w_gate,
    int64_t channels, const c10::optional<at::Tensor>& gamma, double eps, bool need_backward)
{
    constexpr int64_t RANK_THREE = 3;
    TORCH_CHECK(x.dim() == RANK_THREE, "x must be rank-3 [B,S,D], got ", x.dim());
    TORCH_CHECK(channels > 0, "channels must be > 0, got ", channels);
    TORCH_CHECK(x.size(2) % channels == 0, "D must be divisible by channels");
    // gamma is the FP32 [D] raw RMSNorm weight; the kernel applies 1 + gamma.
    if (gamma.has_value()) {
        TORCH_CHECK(gamma->dim() == 1 && gamma->numel() == x.size(2), "gamma must have shape [D=C*H]");
        TORCH_CHECK(gamma->scalar_type() == at::kFloat, "gamma must be FLOAT32");
    }
    TORCH_CHECK(w_down.dim() == 2, "w_down must be rank-2 [R,D]");
    const int64_t hidden = x.size(2) / channels;
    const int64_t rank = w_down.size(0);

    at::Tensor mixed;
    at::Tensor inject;
    at::Tensor inv_rms_t;
    at::Tensor down_pre_act_t;
    at::Tensor up_pre_act_t;
    {
        auto local_device = c10::Device(x.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        mixed = at::empty({x.size(0), x.size(1), hidden}, x.options());
        inject =
            w_gate.has_value() ? at::empty({x.size(0), x.size(1), channels}, x.options()) : at::empty({0}, x.options());
        inv_rms_t = need_backward ? at::empty({x.size(0), x.size(1), channels}, x.options().dtype(at::kFloat)) :
                                    at::empty({0}, x.options().dtype(at::kFloat));
        up_pre_act_t = need_backward ? at::empty(x.sizes(), x.options()) : at::empty({0}, x.options());
        down_pre_act_t =
            need_backward ? at::empty({x.size(0), x.size(1), rank}, x.options()) : at::empty({0}, x.options());
    }

    c10::optional<at::Tensor> inject_opt = c10::nullopt;
    c10::optional<at::Tensor> inv_opt = c10::nullopt;
    c10::optional<at::Tensor> down_opt = c10::nullopt;
    c10::optional<at::Tensor> up_opt = c10::nullopt;
    if (w_gate.has_value()) {
        inject_opt = inject;
    }
    if (need_backward) {
        inv_opt = inv_rms_t;
        down_opt = down_pre_act_t;
        up_opt = up_pre_act_t;
    }
    ACLNN_CMD(aclnnTextGatedResidual, x, w_down, w_up, w_gate, gamma, channels, eps, need_backward, mixed, inject_opt,
              inv_opt, down_opt, up_opt);
    return std::make_tuple(mixed, inject, inv_rms_t, down_pre_act_t, up_pre_act_t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_text_gated_residual_backward", &TextGatedResidualBackward, "TextGatedResidual backward");
    m.def("text_gated_residual", &text_gated_residual, "text_gated_residual");
}
} // namespace op_api
