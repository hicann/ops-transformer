/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t CONTEXT_BYTES = 1024;
constexpr int64_t MAX_K = 64;
constexpr int64_t MX_GROUP_SIZE = 32;
constexpr int64_t SCALE_ALIGNMENT = 2;
constexpr int64_t NONQUANT_SCALE_RANK = 2;
constexpr int64_t MXFP_SCALE_RANK = 3;
enum TokenKind : int64_t {
    FP16,
    BF16,
    MXFP8_E5M2,
    MXFP8_E4M3,
    MXFP4_E2M1
};
} // namespace

std::tuple<at::Tensor, at::Tensor> attention_worker_combine(const at::Tensor& schedule_context,
                                                            const at::Tensor& expert_scales, const at::Tensor& layer_id,
                                                            int64_t hidden_size, int64_t token_dtype,
                                                            int64_t need_schedule)
{
    TORCH_CHECK(schedule_context.device().type() == c10::DeviceType::PrivateUse1,
                "schedule_context must be on NPU, got ", schedule_context.device());
    TORCH_CHECK(expert_scales.device() == schedule_context.device() && layer_id.device() == schedule_context.device(),
                "inputs must be on the same device, got schedule_context=", schedule_context.device(),
                ", expert_scales=", expert_scales.device(), ", layer_id=", layer_id.device());
    TORCH_CHECK(schedule_context.is_contiguous() && expert_scales.is_contiguous() && layer_id.is_contiguous(),
                "inputs must be contiguous, got strides: schedule_context=", schedule_context.strides(),
                ", expert_scales=", expert_scales.strides(), ", layer_id=", layer_id.strides());
    TORCH_CHECK(schedule_context.scalar_type() == at::kChar && schedule_context.dim() == 1 &&
                    schedule_context.numel() == CONTEXT_BYTES,
                "schedule_context must be int8 (1024,), got dtype=", schedule_context.scalar_type(),
                ", shape=", schedule_context.sizes());
    TORCH_CHECK(layer_id.scalar_type() == at::kInt && layer_id.dim() == 1 && layer_id.numel() == 1,
                "layer_id must be int32 (1,), got dtype=", layer_id.scalar_type(), ", shape=", layer_id.sizes());
    TORCH_CHECK(token_dtype >= FP16 && token_dtype <= MXFP4_E2M1, "token_dtype must be in [0, 4], got ", token_dtype);
    TORCH_CHECK(need_schedule == 0 || need_schedule == 1, "need_schedule must be 0 or 1, got ", need_schedule);
    TORCH_CHECK(hidden_size > 0, "hidden_size must be positive, got ", hidden_size);
    const bool is_mxfp = token_dtype >= MXFP8_E5M2;
    const int64_t expected_rank = is_mxfp ? MXFP_SCALE_RANK : NONQUANT_SCALE_RANK;
    TORCH_CHECK(expert_scales.dim() == expected_rank, "expert_scales rank must be ", expected_rank,
                " for token_dtype=", token_dtype, ", got shape=", expert_scales.sizes());
    const int64_t k = expert_scales.size(1) - (is_mxfp ? 1 : 0);
    TORCH_CHECK(expert_scales.size(0) > 0 && k > 0 && k <= MAX_K,
                "R must be positive and K must be in [1, 64], got R=", expert_scales.size(0), ", K=", k);
    if (is_mxfp) {
        TORCH_CHECK(
            expert_scales.scalar_type() == at::kByte || expert_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu,
            "MXFP scales must contain E8M0 encodings (uint8 or float8_e8m0fnu), got ", expert_scales.scalar_type());
        const int64_t groups = hidden_size / MX_GROUP_SIZE + (hidden_size % MX_GROUP_SIZE != 0);
        const int64_t aligned_groups = (groups + SCALE_ALIGNMENT - 1) / SCALE_ALIGNMENT * SCALE_ALIGNMENT;
        TORCH_CHECK(expert_scales.size(2) == aligned_groups, "MXFP scale S must be ", aligned_groups,
                    " (ceil(H/32) rounded up to even) for H=", hidden_size, ", got ", expert_scales.size(2));
    } else {
        TORCH_CHECK(expert_scales.scalar_type() == at::kFloat, "nonquant scales must be float32, got ",
                    expert_scales.scalar_type());
    }
    const c10::OptionalDeviceGuard device_guard(schedule_context.device());
    auto y = at::empty({expert_scales.size(0), hidden_size},
                       schedule_context.options().dtype(token_dtype == FP16 ? at::kHalf : at::kBFloat16));
    auto next_layer_id = at::empty({1}, layer_id.options());
    // uint8 stores E8M0 bytes; reinterpret its dtype without numeric conversion.
    TensorWrapper scales = {expert_scales, is_mxfp ? ACL_FLOAT8_E8M0 : ACL_FLOAT};
    ACLNN_CMD(aclnnAttentionWorkerCombine, schedule_context, scales, layer_id, hidden_size, token_dtype, need_schedule,
              y, next_layer_id);
    return std::make_tuple(y, next_layer_id);
}
} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention_worker_combine", &op_api::attention_worker_combine, "AttentionWorkerCombine");
}
