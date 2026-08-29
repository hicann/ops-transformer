/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_grad.cpp
 * \brief PyTorch extension wrapper for aclnnApplyRotaryPosEmbGrad.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t MIN_INPUT_DIM = 3;
constexpr int64_t MAX_INPUT_DIM = 4;
constexpr int64_t LAYOUT_BSND = 1;
constexpr int64_t LAYOUT_SBND = 2;
constexpr int64_t LAYOUT_BNSD = 3;
constexpr int64_t LAYOUT_TND = 4;
constexpr const char *ROTARY_MODE_HALF = "half";

void CheckDimRange(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.dim() >= MIN_INPUT_DIM && tensor.dim() <= MAX_INPUT_DIM, "apply_rotary_pos_emb_grad: ", name,
                " dim must be 3 or 4, but got ", tensor.dim(), ".");
}

void CheckSameDtype(const at::Tensor &tensor, const at::Tensor &reference, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == reference.scalar_type(), "apply_rotary_pos_emb_grad: ", name,
                " dtype must be same as grad_query_embed, but got ", tensor.scalar_type(), " vs ",
                reference.scalar_type(), ".");
}

void CheckRequiredTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.defined(), "apply_rotary_pos_emb_grad: ", name, " must be defined.");
    TORCH_CHECK(tensor.numel() > 0, "apply_rotary_pos_emb_grad: ", name, " must not be empty.");
}

void CheckInputs(const at::Tensor &grad_query_embed, const at::Tensor &grad_key_embed, const at::Tensor &cos,
                 const at::Tensor &sin, const at::Tensor &query, const at::Tensor &key, const std::string &rotary_mode,
                 int64_t layout)
{
    CheckRequiredTensor(grad_query_embed, "grad_query_embed");
    CheckRequiredTensor(grad_key_embed, "grad_key_embed");
    CheckRequiredTensor(cos, "cos");
    CheckRequiredTensor(sin, "sin");

    CheckDimRange(grad_query_embed, "grad_query_embed");
    CheckDimRange(grad_key_embed, "grad_key_embed");
    CheckDimRange(cos, "cos");
    CheckDimRange(sin, "sin");

    TORCH_CHECK(grad_query_embed.scalar_type() == at::ScalarType::Half ||
                    grad_query_embed.scalar_type() == at::ScalarType::Float ||
                    grad_query_embed.scalar_type() == at::ScalarType::BFloat16,
                "apply_rotary_pos_emb_grad: dtype only supports float16, float32, and bfloat16.");
    CheckSameDtype(grad_key_embed, grad_query_embed, "grad_key_embed");
    CheckSameDtype(cos, grad_query_embed, "cos");
    CheckSameDtype(sin, grad_query_embed, "sin");

    TORCH_CHECK(rotary_mode == ROTARY_MODE_HALF, "apply_rotary_pos_emb_grad: rotary_mode only supports 'half', got '",
                rotary_mode, "'.");
    TORCH_CHECK(layout == LAYOUT_BSND || layout == LAYOUT_SBND || layout == LAYOUT_BNSD || layout == LAYOUT_TND,
                "apply_rotary_pos_emb_grad: layout must be one of 1(BSND), 2(SBND), 3(BNSD), 4(TND), got ", layout,
                ".");
    if (layout == LAYOUT_TND) {
        TORCH_CHECK(grad_query_embed.dim() == MIN_INPUT_DIM,
                    "apply_rotary_pos_emb_grad: TND(4) layout requires 3D "
                    "inputs, but got ",
                    grad_query_embed.dim(), "D.");
    } else {
        TORCH_CHECK(grad_query_embed.dim() == MAX_INPUT_DIM,
                    "apply_rotary_pos_emb_grad: BSND(1)/SBND(2) layout "
                    "requires 4D inputs, but got ",
                    grad_query_embed.dim(), "D.");
    }

    if (query.defined()) {
        CheckDimRange(query, "query");
        CheckSameDtype(query, grad_query_embed, "query");
        TORCH_CHECK(query.sizes() == grad_query_embed.sizes(),
                    "apply_rotary_pos_emb_grad: query shape must equal grad_query_embed shape.");
    }
    if (key.defined()) {
        CheckDimRange(key, "key");
        CheckSameDtype(key, grad_query_embed, "key");
        TORCH_CHECK(key.sizes() == grad_key_embed.sizes(),
                    "apply_rotary_pos_emb_grad: key shape must equal grad_key_embed shape.");
    }
    TORCH_CHECK(query.defined() == key.defined(),
                "apply_rotary_pos_emb_grad: query and key must both be provided or both be None.");
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ApplyRotaryPosEmbGrad(
    const at::Tensor &grad_query_embed, const at::Tensor &grad_key_embed, const at::Tensor &cos, const at::Tensor &sin,
    const c10::optional<at::Tensor> &query, const c10::optional<at::Tensor> &key,
    c10::optional<c10::string_view> rotary_mode, c10::optional<int64_t> layout)
{
    const at::Tensor &query_const = query.value_or(at::Tensor());
    const at::Tensor &key_const = key.value_or(at::Tensor());
    const std::string rotary_mode_str = (!rotary_mode.has_value() || rotary_mode.value().empty()) ?
                                            std::string(ROTARY_MODE_HALF) :
                                            std::string(rotary_mode.value());
    const int64_t layout_value = layout.value_or(LAYOUT_BSND);
    CheckInputs(grad_query_embed, grad_key_embed, cos, sin, query_const, key_const, rotary_mode_str, layout_value);

    at::Tensor grad_query = at::empty(grad_query_embed.sizes(), grad_query_embed.options());
    at::Tensor grad_key = at::empty(grad_key_embed.sizes(), grad_key_embed.options());
    at::Tensor grad_cos = at::empty(cos.sizes(), cos.options());
    at::Tensor grad_sin = at::empty(sin.sizes(), sin.options());

    char *rotary_mode_ptr = const_cast<char *>(rotary_mode_str.c_str());
    ACLNN_CMD(aclnnApplyRotaryPosEmbGrad, grad_query_embed, grad_key_embed, cos, sin, query_const, key_const,
              rotary_mode_ptr, layout_value, grad_query, grad_key, grad_cos, grad_sin);

    return std::make_tuple(grad_query, grad_key, grad_cos, grad_sin);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("apply_rotary_pos_emb_grad", &ApplyRotaryPosEmbGrad, "apply_rotary_pos_emb_grad");
}
} // namespace op_api
