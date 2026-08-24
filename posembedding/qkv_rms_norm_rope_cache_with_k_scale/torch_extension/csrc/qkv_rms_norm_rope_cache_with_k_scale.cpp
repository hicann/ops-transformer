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
 * \file qkv_rms_norm_rope_cache_with_k_scale.cpp
 * \brief PyTorch extension wrapper for aclnnQkvRmsNormRopeCacheWithKScale.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_THREE = 3;
constexpr const char *QKV_LAYOUT_TND = "TND";
constexpr const char *QKV_LAYOUT_NTD = "NTD";
constexpr const char *DEFAULT_QKV_LAYOUT = QKV_LAYOUT_TND;
constexpr const char *DEFAULT_Q_OUT_LAYOUT = QKV_LAYOUT_NTD;
constexpr const char *Q_QUANT_PER_TOKEN_PER_HEAD = "PerTokenPerHead";

struct QkvKScaleParams {
    int64_t n_q;
    int64_t token_num;
    int64_t head_size;
    bool q_out_is_tnd;
    bool q_scale_required;
    at::ScalarType q_out_dtype;
};

std::string ResolveLayout(const c10::optional<c10::string_view> &layout, const char *defaultLayout)
{
    if (!layout.has_value() || layout.value().empty()) {
        return std::string(defaultLayout);
    }
    return std::string(layout.value());
}

std::string ResolveQQuantMode(const std::string &q_quant_mode)
{
    return q_quant_mode.empty() ? std::string(Q_QUANT_PER_TOKEN_PER_HEAD) : q_quant_mode;
}

void CheckOptionalTensorDefined(const c10::optional<at::Tensor> &tensor, const char *name)
{
    if (tensor.has_value()) {
        TORCH_CHECK(tensor.value().defined(), "qkv_rms_norm_rope_cache_with_k_scale: ", name,
                    " is present but undefined.");
    }
}

QkvKScaleParams ResolveOutputParams(const at::Tensor &qkv, at::IntArrayRef head_nums, const std::string &layout_qkv,
                                    const std::string &layout_q_out, const c10::optional<at::Tensor> &mrope_position,
                                    const c10::optional<std::vector<int64_t>> &mrope_section,
                                    at::ScalarType q_out_dtype)
{
    TORCH_CHECK(!head_nums.empty(),
                "qkv_rms_norm_rope_cache_with_k_scale: head_nums must contain Nq for output allocation.");
    const int64_t n_q = head_nums[0];
    TORCH_CHECK(qkv.dim() >= DIM_THREE,
                "qkv_rms_norm_rope_cache_with_k_scale: qkv must have at least 3 dimensions for output allocation.");
    const bool is_tnd = layout_qkv == QKV_LAYOUT_TND;
    const bool is_ntd = layout_qkv == QKV_LAYOUT_NTD;
    TORCH_CHECK(is_tnd || is_ntd,
                "qkv_rms_norm_rope_cache_with_k_scale: layout_qkv must be TND or NTD for output allocation, but got ",
                layout_qkv, ".");
    const bool q_out_is_tnd = layout_q_out == QKV_LAYOUT_TND;
    const bool q_out_is_ntd = layout_q_out == QKV_LAYOUT_NTD;
    TORCH_CHECK(q_out_is_tnd || q_out_is_ntd,
                "qkv_rms_norm_rope_cache_with_k_scale: layout_q_out must be TND or NTD for output allocation, but got ",
                layout_q_out, ".");

    const int64_t token_num = is_tnd ? qkv.size(DIM_0) : qkv.size(DIM_1);
    const int64_t head_size = qkv.size(DIM_2);
    const bool has_mrope_section = mrope_section.has_value() && !mrope_section.value().empty();
    const bool is_mrope = mrope_position.has_value() || has_mrope_section;
    return {n_q, token_num, head_size, q_out_is_tnd, !is_mrope, q_out_dtype};
}

std::tuple<at::Tensor, c10::optional<at::Tensor>> MakeOutputs(const at::Tensor &qkv, const QkvKScaleParams &params)
{
    c10::SmallVector<int64_t, DIM_THREE> q_out_shape =
        params.q_out_is_tnd ? c10::SmallVector<int64_t, DIM_THREE>{params.token_num, params.n_q, params.head_size} :
                              c10::SmallVector<int64_t, DIM_THREE>{params.n_q, params.token_num, params.head_size};
    c10::SmallVector<int64_t, DIM_2> q_scale_shape =
        params.q_out_is_tnd ? c10::SmallVector<int64_t, DIM_2>{params.token_num, params.n_q} :
                              c10::SmallVector<int64_t, DIM_2>{params.n_q, params.token_num};

    at::Tensor q_out;
    c10::optional<at::Tensor> q_scale = c10::nullopt;
    q_out = at::empty(q_out_shape, qkv.options().dtype(params.q_out_dtype));
    if (params.q_scale_required) {
        q_scale = at::empty(q_scale_shape, qkv.options().dtype(at::kFloat));
    }
    return {q_out, q_scale};
}

void RunQkvKScaleAclnn(const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma,
                       const at::Tensor &cos_sin, const at::Tensor &slot_mapping, const at::Tensor &k_cache,
                       const at::Tensor &v_cache, const at::Tensor &k_scale_cache,
                       const c10::optional<at::Tensor> &query_start_loc, const c10::optional<at::Tensor> &seq_lens,
                       at::IntArrayRef head_nums, const std::string &layout_qkv, const std::string &layout_q_out,
                       const c10::optional<at::Tensor> &rotation, const c10::optional<at::Tensor> &v_scale,
                       const c10::optional<at::Tensor> &mrope_position,
                       const c10::optional<std::vector<int64_t>> &mrope_section, const std::string &q_quant_mode,
                       double epsilon, at::Tensor &q_out, const c10::optional<at::Tensor> &q_scale)
{
    const char *layout_qkv_ptr = layout_qkv.c_str();
    const char *layout_q_out_ptr = layout_q_out.c_str();
    const char *q_quant_mode_ptr = q_quant_mode.c_str();
    float epsilon_value = static_cast<float>(epsilon);
    c10::optional<at::IntArrayRef> mrope_section_ref = c10::nullopt;
    if (mrope_section.has_value()) {
        mrope_section_ref = at::IntArrayRef(mrope_section.value());
    }

    // M-RoPE positions use token-major logical shape [T, 3] (T/H/W columns).
    // Keep the tensor untouched here; ACLNN owns scene and shape validation.
    ACLNN_CMD(aclnnQkvRmsNormRopeCacheWithKScale, qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
              k_scale_cache, query_start_loc, seq_lens, rotation, v_scale, mrope_position, head_nums, layout_qkv_ptr,
              layout_q_out_ptr, epsilon_value, mrope_section_ref, q_quant_mode_ptr, q_out, q_scale);
}
} // namespace

std::tuple<at::Tensor, c10::optional<at::Tensor>> qkv_rms_norm_rope_cache_with_k_scale_(
    const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma, const at::Tensor &cos_sin,
    const at::Tensor &slot_mapping, const at::Tensor &k_cache, const at::Tensor &v_cache,
    const at::Tensor &k_scale_cache, const c10::optional<at::Tensor> &query_start_loc,
    const c10::optional<at::Tensor> &seq_lens, at::IntArrayRef head_nums,
    const c10::optional<c10::string_view> &layout_qkv, const c10::optional<c10::string_view> &layout_q_out,
    const c10::optional<at::Tensor> &rotation, const c10::optional<at::Tensor> &v_scale, double epsilon,
    const c10::optional<at::Tensor> &mrope_position, const c10::optional<std::vector<int64_t>> &mrope_section,
    const std::string &q_quant_mode, at::ScalarType q_out_dtype)
{
    const c10::OptionalDeviceGuard device_guard(qkv.device());
    const std::string layout_qkv_str = ResolveLayout(layout_qkv, DEFAULT_QKV_LAYOUT);
    const std::string layout_q_out_str = ResolveLayout(layout_q_out, DEFAULT_Q_OUT_LAYOUT);
    const std::string q_quant_mode_str = ResolveQQuantMode(q_quant_mode);
    CheckOptionalTensorDefined(query_start_loc, "query_start_loc");
    CheckOptionalTensorDefined(seq_lens, "seq_lens");
    CheckOptionalTensorDefined(rotation, "rotation");
    CheckOptionalTensorDefined(v_scale, "v_scale");
    CheckOptionalTensorDefined(mrope_position, "mrope_position");
    const QkvKScaleParams params = ResolveOutputParams(qkv, head_nums, layout_qkv_str, layout_q_out_str, mrope_position,
                                                       mrope_section, q_out_dtype);
    auto outputs = MakeOutputs(qkv, params);
    at::Tensor q_out = std::get<0>(outputs);
    c10::optional<at::Tensor> q_scale = std::get<1>(outputs);

    RunQkvKScaleAclnn(qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache, k_scale_cache, query_start_loc,
                      seq_lens, head_nums, layout_qkv_str, layout_q_out_str, rotation, v_scale, mrope_position,
                      mrope_section, q_quant_mode_str, epsilon, q_out, q_scale);
    return {q_out, q_scale};
}

std::tuple<at::Tensor, c10::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor>
qkv_rms_norm_rope_cache_with_k_scale(
    const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma, const at::Tensor &cos_sin,
    const at::Tensor &slot_mapping, const at::Tensor &k_cache, const at::Tensor &v_cache,
    const at::Tensor &k_scale_cache, const c10::optional<at::Tensor> &query_start_loc,
    const c10::optional<at::Tensor> &seq_lens, at::IntArrayRef head_nums,
    const c10::optional<c10::string_view> &layout_qkv, const c10::optional<c10::string_view> &layout_q_out,
    const c10::optional<at::Tensor> &rotation, const c10::optional<at::Tensor> &v_scale, double epsilon,
    const c10::optional<at::Tensor> &mrope_position, const c10::optional<std::vector<int64_t>> &mrope_section,
    const std::string &q_quant_mode, at::ScalarType q_out_dtype)
{
    const c10::OptionalDeviceGuard device_guard(qkv.device());
    const std::string layout_qkv_str = ResolveLayout(layout_qkv, DEFAULT_QKV_LAYOUT);
    const std::string layout_q_out_str = ResolveLayout(layout_q_out, DEFAULT_Q_OUT_LAYOUT);
    const std::string q_quant_mode_str = ResolveQQuantMode(q_quant_mode);
    CheckOptionalTensorDefined(query_start_loc, "query_start_loc");
    CheckOptionalTensorDefined(seq_lens, "seq_lens");
    CheckOptionalTensorDefined(rotation, "rotation");
    CheckOptionalTensorDefined(v_scale, "v_scale");
    CheckOptionalTensorDefined(mrope_position, "mrope_position");
    const QkvKScaleParams params = ResolveOutputParams(qkv, head_nums, layout_qkv_str, layout_q_out_str, mrope_position,
                                                       mrope_section, q_out_dtype);

    at::Tensor k_cache_clone = k_cache.clone();
    at::Tensor v_cache_clone = v_cache.clone();
    at::Tensor k_scale_cache_clone = k_scale_cache.clone();
    auto outputs = MakeOutputs(qkv, params);
    at::Tensor q_out = std::get<0>(outputs);
    c10::optional<at::Tensor> q_scale = std::get<1>(outputs);

    RunQkvKScaleAclnn(qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache_clone, v_cache_clone, k_scale_cache_clone,
                      query_start_loc, seq_lens, head_nums, layout_qkv_str, layout_q_out_str, rotation, v_scale,
                      mrope_position, mrope_section, q_quant_mode_str, epsilon, q_out, q_scale);
    return {q_out, q_scale, k_cache_clone, v_cache_clone, k_scale_cache_clone};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("qkv_rms_norm_rope_cache_with_k_scale_", &qkv_rms_norm_rope_cache_with_k_scale_,
          "qkv_rms_norm_rope_cache_with_k_scale_");
    m.def("qkv_rms_norm_rope_cache_with_k_scale", &qkv_rms_norm_rope_cache_with_k_scale,
          "qkv_rms_norm_rope_cache_with_k_scale");
}
} // namespace op_api
