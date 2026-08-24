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
 * \file moe_re_routing.cpp
 * \brief MoE re routing operator, bridging PyTorch to aclnnMoeReRoutingV2.
 *        Structure mirrors op-plugin MoeReRoutingOpApi.cpp for maintainability,
 *        with V2 additions (expert_topk_weight / permute_topk_weight).
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

inline at::ScalarType MapDTypeToAtenScalarType(int64_t dtype_val)
{
    aclDataType acl_dtype = GetAclDataType(dtype_val);
    switch (acl_dtype) {
        case ACL_FLOAT8_E5M2:
            return at::kFloat8_e5m2;
        case ACL_FLOAT8_E4M3FN:
            return at::kFloat8_e4m3fn;
        case ACL_HIFLOAT8:
            return at::kByte;
        case ACL_FLOAT4_E2M1:
        case ACL_FLOAT4_E1M2:
            return at::kByte;
        default:
            return at::kByte;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> moe_re_routing(
    const at::Tensor &tokens, const at::Tensor &expert_token_num_per_rank,
    const c10::optional<at::Tensor> &per_token_scales_opt, const c10::optional<at::Tensor> &expert_topk_weight_opt,
    int64_t expert_token_num_type, int64_t idx_type, c10::optional<int64_t> tokens_dtype)
{
    TORCH_CHECK(tokens.dim() > 1, "tokens dim should larger than 1");
    TORCH_CHECK(expert_token_num_per_rank.dim() > 1, "expert_token_num_per_rank dim should larger than 1");

    if (tokens_dtype.has_value()) {
        auto td = tokens_dtype.value();
        TORCH_CHECK(td == static_cast<int64_t>(DType::HIFLOAT8) || td == static_cast<int64_t>(DType::FLOAT4_E2M1) ||
                        td == static_cast<int64_t>(DType::FLOAT4_E1M2),
                    "tokens_dtype only supports hifloat8, float4_e2m1fn_x2, float4_e1m2fn_x2 or None, but got ", td);
    }

    const at::Tensor &per_token_scales = c10::value_or_else(per_token_scales_opt, [] { return at::Tensor(); });
    auto per_token_scales_dtype = per_token_scales.defined() ? per_token_scales.scalar_type() : at::kFloat;

    at::SmallVector<int64_t, op_infer::SIZE> permute_tokens_size;
    at::SmallVector<int64_t, op_infer::SIZE> permute_per_token_scales_size;
    at::SmallVector<int64_t, op_infer::SIZE> permute_token_idx_size;
    at::SmallVector<int64_t, op_infer::SIZE> expert_token_num_size;

    for (int i = 0; i < tokens.dim(); i++) {
        permute_tokens_size.push_back(tokens.size(i));
    }
    if (per_token_scales.defined()) {
        for (int i = 0; i < per_token_scales.dim(); i++) {
            permute_per_token_scales_size.push_back(per_token_scales.size(i));
        }
    } else {
        permute_per_token_scales_size.push_back(tokens.size(0));
    }
    permute_token_idx_size.push_back(tokens.size(0));
    expert_token_num_size.push_back(expert_token_num_per_rank.size(1));

    bool has_topk_weight = expert_topk_weight_opt.has_value() && expert_topk_weight_opt.value().defined();

    at::Tensor permute_tokens;
    at::Tensor permute_per_token_scales;
    at::Tensor permute_token_idx;
    at::Tensor expert_token_num;
    at::Tensor permute_topk_weight;
    c10::optional<at::Tensor> permute_topk_weight_opt;

    {
        auto local_device = c10::Device(tokens.device());
        const c10::OptionalDeviceGuard device_guard(local_device);

        at::TensorOptions permute_tokens_options = tokens.options();
        if (tokens_dtype.has_value()) {
            aclDataType acl_dtype = GetAclDataType(tokens_dtype.value());
            if (acl_dtype == aclDataType::ACL_FLOAT4_E2M1 || acl_dtype == aclDataType::ACL_FLOAT4_E1M2) {
                permute_tokens_options = permute_tokens_options.dtype(at::kByte);
            } else {
                permute_tokens_options = permute_tokens_options.dtype(MapDTypeToAtenScalarType(tokens_dtype.value()));
            }
        }
        permute_tokens = at::empty(permute_tokens_size, permute_tokens_options);

        if (per_token_scales.defined()) {
            permute_per_token_scales = at::empty(permute_per_token_scales_size, per_token_scales.options());
        } else {
            permute_per_token_scales = at::empty(permute_per_token_scales_size, tokens.options().dtype(at::kFloat));
        }

        permute_token_idx = at::empty(permute_token_idx_size, tokens.options().dtype(at::kInt));
        expert_token_num = at::empty(expert_token_num_size, expert_token_num_per_rank.options());

        if (has_topk_weight) {
            permute_topk_weight =
                at::empty(expert_topk_weight_opt.value().sizes(), expert_topk_weight_opt.value().options());
            permute_topk_weight_opt = permute_topk_weight;
        } else {
            permute_topk_weight = at::empty({0}, tokens.options().dtype(at::kFloat));
        }
    }

    aclDataType tokens_acl_dtype =
        tokens_dtype.has_value() ? GetAclDataType(tokens_dtype.value()) : ConvertToAclDataType(tokens.scalar_type());
    TensorWrapper tokens_wrapper = {tokens, tokens_acl_dtype};
    aclDataType permute_tokens_acl_dtype = tokens_dtype.has_value() ?
                                               GetAclDataType(tokens_dtype.value()) :
                                               ConvertToAclDataType(permute_tokens.scalar_type());
    TensorWrapper permute_tokens_wrapper = {permute_tokens, permute_tokens_acl_dtype};

    if (per_token_scales_dtype == at::kByte) {
        TensorWrapper per_token_scales_wrapper = {per_token_scales, aclDataType::ACL_FLOAT8_E8M0};
        TensorWrapper permute_per_token_scales_wrapper = {permute_per_token_scales, aclDataType::ACL_FLOAT8_E8M0};
        ACLNN_CMD(aclnnMoeReRoutingV2, tokens_wrapper, expert_token_num_per_rank, per_token_scales_wrapper,
                  expert_topk_weight_opt, expert_token_num_type, idx_type, permute_tokens_wrapper,
                  permute_per_token_scales_wrapper, permute_token_idx, expert_token_num, permute_topk_weight_opt);
    } else {
        ACLNN_CMD(aclnnMoeReRoutingV2, tokens_wrapper, expert_token_num_per_rank, per_token_scales,
                  expert_topk_weight_opt, expert_token_num_type, idx_type, permute_tokens_wrapper,
                  permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight_opt);
    }

    return std::tie(permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe_re_routing", &moe_re_routing, "moe_re_routing"); }

} // namespace op_api
