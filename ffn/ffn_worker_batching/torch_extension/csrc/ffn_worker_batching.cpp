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
#include <cstring>
#include <limits>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t CONTEXT_BYTES = 1024;
constexpr int64_t SHAPE_DIMS = 4;
constexpr int64_t GROUP_LIST_COLUMNS = 2;
constexpr int64_t MAX_SESSIONS = 1024;
constexpr int64_t MAX_EXPERTS = 8192;
constexpr int64_t MAX_K = 64;
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t FP4_ELEMENTS_PER_BYTE = 2;
enum TokenKind : int64_t {
    FP16,
    BF16,
    INT8,
    FP8_E5M2,
    FP8_E4M3,
    FP4_E2M1
};
using Outputs =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
} // namespace

Outputs ffn_worker_batching(const at::Tensor& schedule_context, int64_t expert_num,
                            const std::vector<int64_t>& max_out_shape, int64_t token_dtype, int64_t need_schedule,
                            int64_t layer_num, bool sync_flag, bool mx_output_uint8)
{
    // Inspect metadata only. Copying a noncontiguous context to a temporary
    // would change the object whose polling_index is updated by RECV.
    TORCH_CHECK(schedule_context.device().type() == c10::DeviceType::PrivateUse1,
                "schedule_context must be on NPU, got ", schedule_context.device());
    TORCH_CHECK(schedule_context.scalar_type() == at::kChar && schedule_context.dim() == 1 &&
                    schedule_context.is_contiguous() && schedule_context.numel() >= CONTEXT_BYTES,
                "schedule_context must be contiguous 1D int8 with at least ", CONTEXT_BYTES, " bytes");
    TORCH_CHECK(max_out_shape.size() == SHAPE_DIMS, "max_out_shape must be [A, BS, K, H]");
    const auto a = max_out_shape[0], bs = max_out_shape[1], k = max_out_shape[2], h = max_out_shape[3];
    TORCH_CHECK(a > 0 && a <= MAX_SESSIONS && bs > 0 && k > 0 && k <= MAX_K && h > 0, "invalid max_out_shape: A=", a,
                ", BS=", bs, ", K=", k, ", H=", h);
    // Guard each product before evaluating it; host tiling runs after allocation.
    TORCH_CHECK(a <= std::numeric_limits<int64_t>::max() / bs, "max_out_shape A*BS overflows int64: ", a, "*", bs);
    const int64_t a_bs = a * bs;
    TORCH_CHECK(a_bs <= std::numeric_limits<int64_t>::max() / k, "max_out_shape A*BS*K overflows int64: ", a, "*", bs,
                "*", k);
    const int64_t y_rows = a_bs * k;
    TORCH_CHECK(expert_num > 0 && expert_num <= MAX_EXPERTS, "invalid expert_num: ", expert_num);
    TORCH_CHECK(token_dtype >= FP16 && token_dtype <= FP4_E2M1, "invalid token_dtype: ", token_dtype);
    TORCH_CHECK(token_dtype != FP4_E2M1 || h % FP4_ELEMENTS_PER_BYTE == 0, "FP4 requires even H, got ", h);
    TORCH_CHECK(need_schedule == 0 || need_schedule == 1, "invalid need_schedule: ", need_schedule);
    if (need_schedule == 1 && sync_flag) {
        TORCH_CHECK(layer_num > 0 && expert_num % layer_num == 0,
                    "async RECV requires layer_num > 0 and expert_num divisible by layer_num");
    }
    TORCH_CHECK(layer_num >= 0 && layer_num <= expert_num, "invalid layer_num: ", layer_num);

    Outputs outputs;
    {
        // Guard allocations AND ACLNN_CMD: the macro allocates workspace and
        // submits work to the current device's current stream.
        const c10::OptionalDeviceGuard device_guard(schedule_context.device());
        const char* soc = aclrtGetSocName();
        TORCH_CHECK(soc != nullptr && std::strncmp(soc, "Ascend950", std::strlen("Ascend950")) == 0,
                    "ffn_worker_batching supports arch35/Ascend950 only, got ", soc == nullptr ? "unknown" : soc);
        const at::ScalarType token_types[] = {at::kHalf,
                                              at::kBFloat16,
                                              at::kChar,
                                              at::ScalarType::Float8_e5m2,
                                              at::ScalarType::Float8_e4m3fn,
                                              at::ScalarType::Float4_e2m1fn_x2};
        const bool mx = token_dtype >= FP8_E5M2;
        const int64_t y_cols = token_dtype == FP4_E2M1 ? h / FP4_ELEMENTS_PER_BYTE : h;
        const int64_t scales = h / MX_BLOCK_SIZE + (h % MX_BLOCK_SIZE != 0);
        auto options = schedule_context.options();
        // Native FP4 x2 counts packed bytes. The common converter expands the
        // final dimension and strides to logical FP4 elements for aclTensor.
        auto y = at::empty({y_rows, y_cols}, options.dtype(token_types[token_dtype]));
        if (token_dtype == FP4_E2M1 && y_rows == 1 && y_cols == 1) {
            // Both default strides would be 1, which the shared converter
            // interprets as packing along the row axis. A singleton row can
            // use stride 2 without extra storage; this identifies H as the
            // packed axis and gives ACLNN the intended logical shape [1,2].
            y = y.as_strided({1, 1}, {FP4_ELEMENTS_PER_BYTE, 1});
        }
        auto group_list = at::empty({expert_num, GROUP_LIST_COLUMNS}, options.dtype(at::kLong));
        auto session_ids = at::empty({y_rows}, options.dtype(at::kInt));
        auto micro_batch_ids = at::empty({y_rows}, options.dtype(at::kInt));
        auto token_ids = at::empty({y_rows}, options.dtype(at::kInt));
        auto expert_offsets = at::empty({y_rows}, options.dtype(at::kInt));
        auto scale_shape = mx ? std::vector<int64_t>{y_rows, scales} : std::vector<int64_t>{y_rows};
        auto dynamic_scale = at::empty(scale_shape, options.dtype(mx ? at::ScalarType::Float8_e8m0fnu : at::kFloat));
        auto actual_token_num = at::empty({1}, options.dtype(at::kLong));
        // V2 exposes sync_flag; the legacy ACLNN symbol retains its original argument layout.
        at::IntArrayRef shape_attr(max_out_shape);
        ACLNN_CMD(aclnnFfnWorkerBatchingV2, schedule_context, expert_num, shape_attr, token_dtype, need_schedule,
                  layer_num, sync_flag, y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets,
                  dynamic_scale, actual_token_num);
        // Expose raw storage to graph passes that cannot handle FP4/E8M0.
        // Keep native dtype descriptors above for ACLNN; view does not cast data.
        if (mx_output_uint8 && mx) {
            dynamic_scale = dynamic_scale.view(at::kByte);
            if (token_dtype == FP4_E2M1) {
                y = y.view(at::kByte);
            }
        }
        outputs = std::make_tuple(y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale,
                                  actual_token_num);
    }
    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ffn_worker_batching", &ffn_worker_batching, "ffn_worker_batching (arch35)");
}
} // namespace op_api
