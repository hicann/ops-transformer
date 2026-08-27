/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_FOUR = 4;
constexpr int64_t MIN_SUPPORTED_K = 64;
constexpr int64_t MAX_SUPPORTED_K = 2032;

void check_positive_shape(const at::Tensor &tensor, const char *name)
{
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        TORCH_CHECK(tensor.size(i) > 0, name, " shape[", i, "] must be greater than 0, but got ", tensor.size(i), ".");
    }
}

void check_fused_gdn_decode_dtype(const at::Tensor &mixed_qkv, const at::Tensor &a, const at::Tensor &b,
                                  const at::Tensor &a_log, const at::Tensor &dt_bias, const at::Tensor &state_ref,
                                  const at::Tensor &ssm_state_indices)
{
    const auto mixed_dtype = mixed_qkv.scalar_type();
    TORCH_CHECK(mixed_dtype == at::ScalarType::BFloat16 || mixed_dtype == at::ScalarType::Half,
                "mixed_qkv dtype must be bfloat16 or float16, but got ", mixed_dtype, ".");
    TORCH_CHECK(
        a.scalar_type() == mixed_dtype && b.scalar_type() == mixed_dtype && dt_bias.scalar_type() == mixed_dtype,
        "a, b and dt_bias dtype must be the same as mixed_qkv.");
    TORCH_CHECK(a_log.scalar_type() == at::ScalarType::Float, "a_log dtype must be float32, but got ",
                a_log.scalar_type(), ".");
    TORCH_CHECK(state_ref.scalar_type() == at::ScalarType::Float || state_ref.scalar_type() == mixed_dtype,
                "state_ref dtype must be float32 or the same as mixed_qkv, but got ", state_ref.scalar_type(), ".");
    TORCH_CHECK(ssm_state_indices.scalar_type() == at::ScalarType::Int,
                "ssm_state_indices dtype must be int32, but got ", ssm_state_indices.scalar_type(), ".");
}

at::Tensor construct_fused_gdn_decode_output_tensor(const at::Tensor &mixed_qkv, const at::Tensor &a,
                                                    const at::Tensor &b, const at::Tensor &a_log,
                                                    const at::Tensor &dt_bias, const at::Tensor &state_ref,
                                                    const at::Tensor &ssm_state_indices)
{
    TORCH_CHECK(mixed_qkv.dim() == DIM_TWO, "mixed_qkv must be a 2D tensor [B, packed_dim], but got ", mixed_qkv.dim(),
                "D.");
    TORCH_CHECK(a.dim() == DIM_TWO && b.dim() == DIM_TWO, "a and b must be 2D tensors [B, Hv], but got ", a.dim(),
                "D and ", b.dim(), "D.");
    TORCH_CHECK(a_log.dim() == DIM_ONE && dt_bias.dim() == DIM_ONE,
                "a_log and dt_bias must be 1D tensors [Hv], but got ", a_log.dim(), "D and ", dt_bias.dim(), "D.");
    TORCH_CHECK(state_ref.dim() == DIM_FOUR, "state_ref must be a 4D tensor [BlockNum, Hv, V, K], but got ",
                state_ref.dim(), "D.");
    TORCH_CHECK(ssm_state_indices.dim() == DIM_ONE, "ssm_state_indices must be a 1D tensor [B], but got ",
                ssm_state_indices.dim(), "D.");

    check_positive_shape(mixed_qkv, "mixed_qkv");
    check_positive_shape(a, "a");
    check_positive_shape(b, "b");
    check_positive_shape(a_log, "a_log");
    check_positive_shape(dt_bias, "dt_bias");
    check_positive_shape(state_ref, "state_ref");
    check_positive_shape(ssm_state_indices, "ssm_state_indices");
    check_fused_gdn_decode_dtype(mixed_qkv, a, b, a_log, dt_bias, state_ref, ssm_state_indices);

    TORCH_CHECK(state_ref.is_contiguous(), "state_ref must be contiguous.");
    TORCH_CHECK(mixed_qkv.size(0) == a.size(0) && a.sizes() == b.sizes(),
                "mixed_qkv, a and b batch or shape mismatch.");
    TORCH_CHECK(ssm_state_indices.size(0) == mixed_qkv.size(0),
                "ssm_state_indices shape[0] must equal mixed_qkv batch.");

    const int64_t hv = state_ref.size(1);
    const int64_t v = state_ref.size(2);
    const int64_t k = state_ref.size(3);
    TORCH_CHECK(k >= MIN_SUPPORTED_K && k <= MAX_SUPPORTED_K, "K must be in [", MIN_SUPPORTED_K, ", ", MAX_SUPPORTED_K,
                "], but got ", k, ".");
    TORCH_CHECK(a.size(1) == hv && a_log.size(0) == hv && dt_bias.size(0) == hv,
                "a/b/a_log/dt_bias Hv must match state_ref.shape[1].");

    const int64_t qk_dim = mixed_qkv.size(1) - hv * v;
    TORCH_CHECK(qk_dim > 0 && qk_dim % (2 * k) == 0, "mixed_qkv packed_dim is inconsistent with state_ref shape.");
    const int64_t h = qk_dim / (2 * k);
    TORCH_CHECK(h > 0 && hv % h == 0, "invalid H/Hv relation derived from mixed_qkv and state_ref.");

    at::SmallVector<int64_t, SIZE> out_size = {mixed_qkv.size(0), 1, hv, v};
    return at::empty(out_size, mixed_qkv.options());
}

// step2, 为NPU设备实现前向接口（函数形参顺序 = schema 顺序）
at::Tensor npu_fused_gdn_decode_npu(const at::Tensor &mixed_qkv, const at::Tensor &a, const at::Tensor &b,
                                    const at::Tensor &a_log, const at::Tensor &dt_bias, at::Tensor &state_ref,
                                    const at::Tensor &ssm_state_indices, double scale, double softplus_threshold)
{
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(softplus_threshold),
                "scale and softplus_threshold must be finite.");
    at::Tensor out =
        construct_fused_gdn_decode_output_tensor(mixed_qkv, a, b, a_log, dt_bias, state_ref, ssm_state_indices);
    const float scale_value = static_cast<float>(scale);
    const float softplus_threshold_value = static_cast<float>(softplus_threshold);

    EXEC_NPU_CMD_V1(aclnnFusedGdnDecode, mixed_qkv, a, b, a_log, dt_bias, state_ref, ssm_state_indices, scale_value,
                    softplus_threshold_value, out);
    return out;
}

// step3, 为META设备实现前向接口
at::Tensor npu_fused_gdn_decode_meta(const at::Tensor &mixed_qkv, const at::Tensor &a, const at::Tensor &b,
                                     const at::Tensor &a_log, const at::Tensor &dt_bias, at::Tensor &state_ref,
                                     const at::Tensor &ssm_state_indices, double scale, double softplus_threshold)
{
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(softplus_threshold),
                "scale and softplus_threshold must be finite.");
    return construct_fused_gdn_decode_output_tensor(mixed_qkv, a, b, a_log, dt_bias, state_ref, ssm_state_indices);
}
} // namespace custom

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_fused_gdn_decode", &custom::npu_fused_gdn_decode_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_fused_gdn_decode", &custom::npu_fused_gdn_decode_meta);
}
