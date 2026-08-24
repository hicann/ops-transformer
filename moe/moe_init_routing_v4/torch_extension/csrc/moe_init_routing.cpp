/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing.cpp
 * \brief MoE init routing operator, bridging PyTorch to aclnnMoeInitRoutingV4.
 *        Structure mirrors op-plugin MoeInitRoutingV2KernelNpuOpApi.cpp for maintainability,
 *        with V4 additions (topk_weight / expanded_topk_weight).
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

// ---- Constants (from MoeInitRoutingV2KernelNpuOpApi.cpp) ----
constexpr int64_t DIM_X = 2;
constexpr int64_t DIM_EXPERT_IDX = 2;
constexpr int64_t MXQUANT_BLOCK_SIZE = 32;
constexpr int64_t FP8_QUANT_BLOCK_SIZE = 128;
constexpr int64_t PAD_TO_EVEN_FACTOR = 2;
constexpr int64_t INT4_NUMS_IN_INT8 = 2;
constexpr int64_t SCALE_BLOCK_SIZE = 64;
constexpr int64_t SCALE_THIRD_DIM_SIZE = 2;
constexpr int64_t EXPERT_TOKENS_KEY_VALUE = 2;
constexpr int64_t ACTIVE_EXPERT_RANGE_SIZE = 2;

// ---- Quant mode enum (from MoeInitRoutingV2KernelNpuOpApi.cpp) ----
enum MoeQuantMode {
    QUANT_MODE_UNQUANT = -1,
    QUANT_MODE_STATIC = 0,
    QUANT_MODE_DYNAMIC = 1,
    QUANT_MODE_MXFP8_E5M2 = 2,
    QUANT_MODE_MXFP8_E4M3FN = 3,
    QUANT_MODE_FP8_GROUP_E5M2 = 4,
    QUANT_MODE_FP8_GROUP_E4M3FN = 5,
    QUANT_MODE_HIF8_CAST = 6,
    QUANT_MODE_HIF8_PERTENSOR = 7,
    QUANT_MODE_HIF8_PER_TOKEN_DIM = 8,
    QUANT_MODE_MXFP4_E2M1 = 9,
    QUANT_MODE_FP8_PERBLOCK_E5M2 = 11,
    QUANT_MODE_FP8_PERBLOCK_E4M3FN = 12,
    QUANT_MODE_INT4_DYNAMIC = 13,
    QUANT_MODE_FP8_GROUP_AMAX_E5M2 = 14,
    QUANT_MODE_FP8_GROUP_AMAX_E4M3FN = 15,
    QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 = 16,
    QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN = 17,
};

// ---- Helper functions (from MoeInitRoutingV2KernelNpuOpApi.cpp) ----
inline bool IsQuantModeMXFP4(int64_t quantMode) { return quantMode == QUANT_MODE_MXFP4_E2M1; }

inline bool IsQuantModeMXFP8(int64_t quantMode)
{
    return quantMode == QUANT_MODE_MXFP8_E5M2 || quantMode == QUANT_MODE_MXFP8_E4M3FN ||
           quantMode == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 || quantMode == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN;
}

inline bool IsQuantModeFP8Group(int64_t quantMode)
{
    return quantMode == QUANT_MODE_FP8_GROUP_E5M2 || quantMode == QUANT_MODE_FP8_GROUP_E4M3FN ||
           quantMode == QUANT_MODE_FP8_GROUP_AMAX_E5M2 || quantMode == QUANT_MODE_FP8_GROUP_AMAX_E4M3FN;
}

inline bool IsQuantModeFP8(int64_t quantMode)
{
    return quantMode == QUANT_MODE_FP8_PERBLOCK_E5M2 || quantMode == QUANT_MODE_FP8_PERBLOCK_E4M3FN;
}

inline bool IsQuantModeHIF8(int64_t quantMode)
{
    return quantMode == QUANT_MODE_HIF8_CAST || quantMode == QUANT_MODE_HIF8_PERTENSOR ||
           quantMode == QUANT_MODE_HIF8_PER_TOKEN_DIM;
}

inline bool IsInt4OutputDtype(const c10::optional<int64_t> &xDtype)
{
    return xDtype.has_value() && xDtype.value() == static_cast<int64_t>(DType::INT4);
}

inline bool IsDynamicQuantInt4Output(int64_t quantMode, const c10::optional<int64_t> &xDtype)
{
    return quantMode == QUANT_MODE_INT4_DYNAMIC && (!xDtype.has_value() || IsInt4OutputDtype(xDtype));
}

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline int64_t CeilAlign(int64_t a, int64_t b) { return CeilDiv(a, b) * b; }

// ---- Main function ----
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> moe_init_routing(
    const at::Tensor &x, const at::Tensor &expert_idx, const c10::optional<at::Tensor> &scale,
    const c10::optional<at::Tensor> &offset, const c10::optional<at::Tensor> &topk_weight, int64_t active_num,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, const c10::optional<std::vector<int64_t>> &active_expert_range,
    int64_t row_idx_type, const c10::optional<int64_t> &x_dtype)
{
    // ---- Input validation ----
    TORCH_CHECK(x.dim() == DIM_X, "x must be 2D, but got ", x.dim(), "D.");
    TORCH_CHECK(expert_idx.dim() == DIM_EXPERT_IDX, "expert_idx must be 2D, but got ", expert_idx.dim(), "D.");
    TORCH_CHECK(expert_num > 0, "expert_num must be greater than 0, current is: ", expert_num);
    TORCH_CHECK((drop_pad_mode == 0 || drop_pad_mode == 1),
                "drop_pad_mode must be 0 or 1, current is: ", drop_pad_mode);
    TORCH_CHECK((expert_tokens_num_type >= 0 && expert_tokens_num_type <= EXPERT_TOKENS_KEY_VALUE),
                "expert_tokens_num_type must be 0, 1 or 2, current is: ", expert_tokens_num_type);
    TORCH_CHECK((row_idx_type == 0 || row_idx_type == 1), "row_idx_type must be 0 or 1, current is: ", row_idx_type);
    if (drop_pad_mode == 1) {
        TORCH_CHECK(expert_capacity > 0,
                    "expert_capacity must be greater than 0 when drop_pad_mode=1, current is: ", expert_capacity);
    }
    bool has_topk_weight = topk_weight.has_value() && topk_weight.value().defined();
    if (has_topk_weight) {
        TORCH_CHECK(topk_weight.value().dim() == DIM_X, "topk_weight must be 2D, but got ", topk_weight.value().dim(),
                    "D.");
        TORCH_CHECK(topk_weight.value().size(0) == x.size(0), "topk_weight row count must equal x row count, got ",
                    topk_weight.value().size(0), " vs ", x.size(0));
        TORCH_CHECK(topk_weight.value().size(1) == expert_idx.size(1),
                    "topk_weight col count must equal expert_idx col count, got ", topk_weight.value().size(1), " vs ",
                    expert_idx.size(1));
        TORCH_CHECK(topk_weight.value().scalar_type() == at::kFloat, "topk_weight dtype must be float32, got ",
                    topk_weight.value().scalar_type());
    }

    // ---- Dimension computation (mirrors MoeInitRoutingV2KernelNpuOpApi.cpp) ----
    auto x_size = x.sizes();
    auto expert_idx_size = expert_idx.sizes();
    const at::Tensor &p_scale = c10::value_or_else(scale, [] { return at::Tensor(); });
    const at::Tensor &p_offset = c10::value_or_else(offset, [] { return at::Tensor(); });

    int64_t bs = x_size[0];
    int64_t h = x_size[1];
    int64_t k = expert_idx_size[1];
    bool has_scale = p_scale.defined();

    // x_dtype: determine actual ACL dtype and adjust h for float4 (mirrors MoeInitRoutingV2KernelNpuOpApi.cpp)
    aclDataType x_acl_type =
        x_dtype.has_value() ? GetAclDataType(x_dtype.value()) : ConvertToAclDataType(x.scalar_type());
    if (x_acl_type == aclDataType::ACL_FLOAT4_E2M1) {
        h = h * INT4_NUMS_IN_INT8;
    }

    // ---- Allocate output tensors under DeviceGuard (critical for multi-card) ----
    at::Tensor expanded_x;
    at::Tensor expanded_row_idx;
    at::Tensor expert_tokens_count_or_cumsum;
    at::Tensor expanded_scale;
    at::Tensor expanded_topk_weight;
    c10::optional<at::Tensor> expanded_topk_weight_opt;

    {
        auto local_device = c10::Device(x.device());
        const c10::OptionalDeviceGuard device_guard(local_device);

        // expanded_x and expanded_scale_len (mirrors MoeInitRoutingV2KernelNpuOpApi.cpp)
        int64_t expanded_scale_len = 0;

        if (drop_pad_mode == 1) {
            expanded_scale_len = expert_num * expert_capacity;
            if (quant_mode == QUANT_MODE_UNQUANT) {
                expanded_x = at::empty({expert_num, expert_capacity, h}, x.options());
            } else {
                at::ScalarType output_dtype = at::kChar;
                if (IsQuantModeMXFP8(quant_mode)) {
                    output_dtype = (quant_mode == QUANT_MODE_MXFP8_E5M2) ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
                } else if (IsQuantModeFP8(quant_mode)) {
                    output_dtype = (quant_mode == QUANT_MODE_FP8_PERBLOCK_E5M2) ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
                } else if (IsQuantModeHIF8(quant_mode)) {
                    output_dtype = at::kByte;
                } else if (IsQuantModeMXFP4(quant_mode) || quant_mode == QUANT_MODE_INT4_DYNAMIC) {
                    output_dtype = at::kByte;
                }
                expanded_x = at::empty({expert_num, expert_capacity, h}, x.options().dtype(output_dtype));
            }
        } else {
            expanded_scale_len = bs * k;
            switch (quant_mode) {
                case QUANT_MODE_MXFP8_E5M2:
                case QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2:
                    expanded_x = at::empty({expanded_scale_len, h}, x.options().dtype(at::kFloat8_e5m2));
                    break;
                case QUANT_MODE_MXFP8_E4M3FN:
                case QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN:
                    expanded_x = at::empty({expanded_scale_len, h}, x.options().dtype(at::kFloat8_e4m3fn));
                    break;
                case QUANT_MODE_MXFP4_E2M1:
                    expanded_x = at::empty({expanded_scale_len, h / INT4_NUMS_IN_INT8}, x.options().dtype(at::kByte));
                    break;
                case QUANT_MODE_STATIC:
                case QUANT_MODE_DYNAMIC:
                    expanded_x = at::empty({expanded_scale_len, h}, x.options().dtype(at::kChar));
                    break;
                case QUANT_MODE_INT4_DYNAMIC:
                    expanded_x = at::empty({expanded_scale_len, h / INT4_NUMS_IN_INT8}, x.options().dtype(at::kByte));
                    break;
                case QUANT_MODE_HIF8_CAST:
                case QUANT_MODE_HIF8_PERTENSOR:
                case QUANT_MODE_HIF8_PER_TOKEN_DIM:
                    expanded_x = at::empty({expanded_scale_len, h}, x.options().dtype(at::kByte));
                    break;
                case QUANT_MODE_FP8_PERBLOCK_E5M2:
                case QUANT_MODE_FP8_GROUP_E5M2:
                case QUANT_MODE_FP8_GROUP_AMAX_E5M2:
                    expanded_x = at::empty({expanded_scale_len, h}, x.options().dtype(at::kFloat8_e5m2));
                    break;
                case QUANT_MODE_FP8_PERBLOCK_E4M3FN:
                case QUANT_MODE_FP8_GROUP_E4M3FN:
                case QUANT_MODE_FP8_GROUP_AMAX_E4M3FN:
                    expanded_x = at::empty({expanded_scale_len, h}, x.options().dtype(at::kFloat8_e4m3fn));
                    break;
                default:
                    expanded_x = at::empty({expanded_scale_len, x_size[1]}, x.options());
                    break;
            }
        }

        // expanded_row_idx (mirrors MoeInitRoutingV2KernelNpuOpApi.cpp)
        expanded_row_idx = at::empty({bs * k}, x.options().dtype(at::kInt));

        // expert_tokens_count_or_cumsum (mirrors MoeInitRoutingV2KernelNpuOpApi.cpp)
        int64_t expert_range_end = expert_num;
        int64_t expert_range_start = 0;
        if (active_expert_range.has_value() && active_expert_range.value().size() >= ACTIVE_EXPERT_RANGE_SIZE) {
            expert_range_start = active_expert_range.value()[0];
            expert_range_end = active_expert_range.value()[1];
        }
        int64_t expert_length = expert_range_end - expert_range_start;

        if (expert_tokens_num_type < EXPERT_TOKENS_KEY_VALUE) {
            expert_tokens_count_or_cumsum = at::empty({expert_length}, x.options().dtype(at::kLong));
        } else {
            expert_tokens_count_or_cumsum = at::empty({expert_num, 2}, x.options().dtype(at::kLong));
        }

        // expanded_scale (mirrors MoeInitRoutingV2KernelNpuOpApi.cpp)
        if (IsQuantModeMXFP8(quant_mode)) {
            int64_t scale_cols = CeilAlign(CeilDiv(h, MXQUANT_BLOCK_SIZE), PAD_TO_EVEN_FACTOR);
            expanded_scale = at::empty({expanded_scale_len, scale_cols}, x.options().dtype(at::kFloat8_e8m0fnu));
        } else if (IsQuantModeFP8(quant_mode)) {
            int64_t block_size = FP8_QUANT_BLOCK_SIZE * 2;
            expanded_scale = at::empty({expanded_scale_len, CeilDiv(h, block_size), SCALE_THIRD_DIM_SIZE},
                                       x.options().dtype(at::kFloat));
        } else if (IsQuantModeFP8Group(quant_mode)) {
            expanded_scale =
                at::empty({expanded_scale_len, CeilDiv(h, FP8_QUANT_BLOCK_SIZE)}, x.options().dtype(at::kFloat));
        } else if (quant_mode == QUANT_MODE_UNQUANT && has_scale &&
                   (x.scalar_type() == at::kFloat8_e5m2 || x.scalar_type() == at::kFloat8_e4m3fn ||
                    x_acl_type == aclDataType::ACL_FLOAT4_E2M1)) {
            expanded_scale = at::empty({expanded_scale_len, CeilDiv(h, SCALE_BLOCK_SIZE), SCALE_THIRD_DIM_SIZE},
                                       x.options().dtype(at::kFloat8_e8m0fnu));
        } else if (IsQuantModeMXFP4(quant_mode)) {
            expanded_scale = at::empty({expanded_scale_len, CeilDiv(h, SCALE_BLOCK_SIZE), SCALE_THIRD_DIM_SIZE},
                                       x.options().dtype(at::kFloat8_e8m0fnu));
        } else {
            expanded_scale = at::empty({expanded_scale_len}, x.options().dtype(at::kFloat));
        }

        // expanded_topk_weight (V4 addition over V3)
        if (has_topk_weight) {
            expanded_topk_weight = at::empty({expanded_scale_len, 1}, x.options().dtype(at::kFloat));
            expanded_topk_weight_opt = expanded_topk_weight;
        } else {
            expanded_topk_weight = at::empty({0}, x.options().dtype(at::kFloat));
        }
    }

    // ---- Early return for empty input (mirrors MoeInitRoutingV2KernelNpuOpApi.cpp) ----
    if (bs == 0) {
        expert_tokens_count_or_cumsum.zero_();
        return std::tie(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale,
                        expanded_topk_weight);
    }

    // ---- TensorWrapper for dtype overrides (mirrors MoeInitRoutingV2KernelNpuOpApi.cpp) ----
    bool non_quant_fp8_or_fp4_input = (quant_mode == QUANT_MODE_UNQUANT) &&
                                      (x.scalar_type() == at::kFloat8_e5m2 || x.scalar_type() == at::kFloat8_e4m3fn ||
                                       x_acl_type == aclDataType::ACL_FLOAT4_E2M1);

    auto scale_acl_dtype = non_quant_fp8_or_fp4_input ?
                               ACL_FLOAT8_E8M0 :
                               ConvertToAclDataType(p_scale.defined() ? p_scale.scalar_type() : at::kFloat);
    TensorWrapper scale_wrapper = {p_scale, scale_acl_dtype};

    auto expanded_scale_acl_dtype =
        non_quant_fp8_or_fp4_input ? ACL_FLOAT8_E8M0 : ConvertToAclDataType(expanded_scale.scalar_type());
    TensorWrapper expanded_scale_wrapper = {expanded_scale, expanded_scale_acl_dtype};

    aclDataType x_wrapper_acl_dtype =
        (quant_mode == QUANT_MODE_UNQUANT && x_dtype.has_value()) ? x_acl_type : ConvertToAclDataType(x.scalar_type());
    TensorWrapper x_wrapper = {x, x_wrapper_acl_dtype};
    TensorWrapper expanded_x_wrapper = {expanded_x, ConvertToAclDataType(expanded_x.scalar_type())};

    if (quant_mode == QUANT_MODE_UNQUANT && x_dtype.has_value()) {
        expanded_x_wrapper.dtype = x_acl_type;
    } else if (IsDynamicQuantInt4Output(quant_mode, x_dtype)) {
        expanded_x_wrapper.dtype = ACL_INT4;
    } else if (IsQuantModeHIF8(quant_mode)) {
        expanded_x_wrapper.dtype = ACL_HIFLOAT8;
    } else if (IsQuantModeMXFP4(quant_mode)) {
        expanded_x_wrapper.dtype = ACL_FLOAT4_E2M1;
    }

    // ---- active_expert_range: std::vector -> at::IntArrayRef ----
    c10::optional<at::IntArrayRef> active_expert_range_opt;
    if (active_expert_range.has_value()) {
        active_expert_range_opt = at::IntArrayRef(active_expert_range.value());
    }

    // ---- active_num: int64_t -> aclTensor* (V4 changed active_num from attr to optional input) ----
    c10::optional<at::Tensor> active_num_tensor_opt;
    if (active_num > 0) {
        active_num_tensor_opt = at::full({}, active_num, at::TensorOptions().dtype(at::kLong).device(x.device()));
    }

    // ---- Call aclnnMoeInitRoutingV4 ----
    ACLNN_CMD(aclnnMoeInitRoutingV4, x_wrapper, expert_idx, scale_wrapper, p_offset, active_num_tensor_opt, topk_weight,
              expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag, quant_mode,
              active_expert_range_opt, row_idx_type, expanded_x_wrapper, expanded_row_idx,
              expert_tokens_count_or_cumsum, expanded_scale_wrapper, expanded_topk_weight_opt);

    return std::tie(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale, expanded_topk_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe_init_routing", &moe_init_routing, "moe_init_routing"); }

} // namespace op_api
