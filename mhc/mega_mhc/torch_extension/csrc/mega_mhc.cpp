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
#include <c10/core/DeviceGuard.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
// Public output modes and OCP scale packing; this wrapper is packaged as an independent source file.
constexpr int64_t kOutputModeUnquantized = 0;
constexpr int64_t kOutputModeMxFp8 = 1;
constexpr int64_t kQuantGroupElements = 32;
constexpr int64_t kQuantScalesPerPair = 2;
constexpr int64_t kQuantPairElements = kQuantGroupElements * kQuantScalesPerPair;
} // namespace
/** Allocate the active outputs, including current pre coefficients when external mixing is supplied. */
std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>, c10::optional<at::Tensor>>
mega_mhc(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& post_mix, const at::Tensor& fn,
         const at::Tensor& mix_scales, const at::Tensor& mix_bases, const at::Tensor& rms_norm_1_weight,
         const c10::optional<at::Tensor>& comb_res_mix, const c10::optional<at::Tensor>& shifted_prev_mix,
         const c10::optional<at::Tensor>& rms_norm_0_weight, const c10::optional<at::Tensor>& rms_norm_mid_weight,
         double rms_norm_eps, double hc_norm_eps, double hc_pre_eps, int64_t inner_precise, int64_t hc_mult,
         double hc_post_scale, double sinkhorn_eps, int64_t num_sinkhorn_iters, double rms_norm_scale,
         int64_t output_mode)
{
    constexpr int64_t HEADS = 4;
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be an NPU tensor");
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be [T,D] or [B,S,D]");
    TORCH_CHECK(output_mode == kOutputModeUnquantized || output_mode == kOutputModeMxFp8,
                "output_mode must be 0 or 1; mode 2 is reserved");
    const int64_t d = x.size(-1);
    // Device ownership is a Torch concern; tensor metadata and attributes are checked by Host.
    const auto checkDevice = [&](const at::Tensor& value) {
        TORCH_CHECK(value.device() == x.device(), "all inputs must be on x.device");
    };
    for (const auto* value : {&residual, &post_mix, &fn, &mix_scales, &mix_bases, &rms_norm_1_weight}) {
        checkDevice(*value);
    }
    for (const auto* value : {&comb_res_mix, &shifted_prev_mix, &rms_norm_0_weight, &rms_norm_mid_weight}) {
        if (value->has_value()) {
            checkDevice(**value);
        }
    }
    auto leading = x.sizes().vec();
    leading.pop_back();
    auto post_shape = leading;
    post_shape.push_back(HEADS);
    auto res_shape = post_shape;
    res_shape.push_back(HEADS);
    auto x2_shape = post_shape;
    x2_shape.push_back(d);
    post_shape.push_back(1);
    at::Tensor h_post_new, h_res_new, x_2_out;
    c10::optional<at::Tensor> h_in_prime = c10::nullopt;
    c10::optional<at::Tensor> y_quant = c10::nullopt;
    c10::optional<at::Tensor> y_scale = c10::nullopt;
    c10::optional<at::Tensor> h_pre_new = c10::nullopt;
    {
        const c10::OptionalDeviceGuard guard(x.device());
        if (output_mode == kOutputModeUnquantized) {
            h_in_prime = at::empty(x.sizes(), x.options());
        } else {
            auto scale_shape = leading;
            scale_shape.push_back((d + kQuantPairElements - 1) / kQuantPairElements);
            scale_shape.push_back(kQuantScalesPerPair);
            y_quant = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
            y_scale = at::empty(scale_shape, x.options().dtype(at::ScalarType::Float8_e8m0fnu));
        }
        h_post_new = at::empty(post_shape, x.options().dtype(at::kFloat));
        h_res_new = at::empty(res_shape, x.options().dtype(at::kFloat));
        x_2_out = at::empty(x2_shape, x.options());
        if (shifted_prev_mix.has_value()) {
            h_pre_new = at::empty(post_shape, x.options().dtype(at::kFloat));
        }
        ACLNN_CMD(aclnnMegaMhc, x, residual, post_mix, fn, mix_scales, mix_bases, rms_norm_1_weight, comb_res_mix,
                  shifted_prev_mix, rms_norm_0_weight, rms_norm_mid_weight, rms_norm_eps, hc_norm_eps, hc_pre_eps,
                  inner_precise, hc_mult, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters, rms_norm_scale, output_mode,
                  x_2_out, h_post_new, h_res_new, h_pre_new, h_in_prime, y_quant, y_scale);
    }
    return {x_2_out, h_post_new, h_res_new, h_pre_new, h_in_prime, y_quant, y_scale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mega_mhc", &mega_mhc, "Ascend 950 mega_mhc forward");
}
} // namespace op_api
