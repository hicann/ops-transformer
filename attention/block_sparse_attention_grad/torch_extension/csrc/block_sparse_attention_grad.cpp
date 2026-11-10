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
 * \file block_sparse_attention_grad.cpp
 * \brief
 */
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

const int64_t MAX_HEAD_DIM = 128;
const int64_t kDefaultBlockShape[2] = {128, 128};
const int64_t kMaxWindow = 2147483647;

static void check_params(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                         const c10::optional<std::vector<int64_t>>& actual_seq_lengths,
                         const c10::optional<std::vector<int64_t>>& actual_seq_lengths_kv,
                         const std::string& q_input_layout, const std::string& kv_input_layout)
{
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
                "query, key, value must have the same dtype, got query=", query.scalar_type(),
                ", key=", key.scalar_type(), ", value=", value.scalar_type());
    TORCH_CHECK(query.size(-1) <= MAX_HEAD_DIM, "head_dim must be <= ", MAX_HEAD_DIM, ", but got ", query.size(-1));
    if (q_input_layout == "TND") {
        TORCH_CHECK(actual_seq_lengths.has_value() && actual_seq_lengths->size() > 0,
                    "actual_seq_lengths must be specified when q_input_layout is TND");
    }
    if (kv_input_layout == "TND") {
        TORCH_CHECK(actual_seq_lengths_kv.has_value() && actual_seq_lengths_kv->size() > 0,
                    "actual_seq_lengths_kv must be specified when kv_input_layout is TND");
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_block_sparse_attention_backward(
    const at::Tensor& d_out, const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& attention_out, const at::Tensor& softmax_lse, const at::Tensor& block_sparse_mask,
    const c10::optional<std::vector<int64_t>>& block_shape, c10::string_view q_input_layout,
    c10::string_view kv_input_layout, int64_t num_key_value_heads, double scale_value,
    const c10::optional<std::vector<int64_t>>& actual_seq_lengths,
    const c10::optional<std::vector<int64_t>>& actual_seq_lengths_kv, const c10::optional<at::Tensor>& atten_mask,
    int64_t mask_type, const c10::optional<at::Tensor>& q_dequant_scale,
    const c10::optional<at::Tensor>& k_dequant_scale, const c10::optional<at::Tensor>& v_dequant_scale,
    const c10::optional<at::Tensor>& do_dequant_scale, const c10::optional<at::Tensor>& p_quant_scale,
    const c10::optional<at::Tensor>& ds_quant_scale, const c10::optional<int64_t>& quant_mode)
{
    const bool is_hif8 = query.scalar_type() == at::kByte;
    const int64_t resolved_quant_mode = quant_mode.value_or(is_hif8 ? 5 : 0);
    TORCH_CHECK(resolved_quant_mode == 0 || resolved_quant_mode == 5, "quant_mode only supports 0 or 5, got ",
                resolved_quant_mode);
    TORCH_CHECK(
        (resolved_quant_mode == 5 && is_hif8) ||
            (resolved_quant_mode == 0 && (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16)),
        "quant_mode must match query dtype: 0=FP16/BF16, 5=HiF8-encoded uint8");
    if (is_hif8) {
        TORCH_CHECK(
            key.scalar_type() == at::kByte && value.scalar_type() == at::kByte && d_out.scalar_type() == at::kByte,
            "HiF8 requires HiF8-encoded uint8 query/key/value/d_out.");
    }
    // mask_type 契约校验（只支持 0/1；0 ↔ 无 counts；1 ↔ 必传 counts）
    TORCH_CHECK(mask_type == 0 || mask_type == 1, "mask_type only supports 0 or 1, got ", mask_type);
    if (mask_type == 0) {
        TORCH_CHECK(!atten_mask.has_value(), "atten_mask must be None when mask_type is 0");
    } else {
        TORCH_CHECK(atten_mask.has_value(), "atten_mask is required when mask_type is 1");
        TORCH_CHECK(atten_mask->scalar_type() == at::kInt,
                    "atten_mask must be int32 valid-counts (1D prefix counts or 4D [B,N,>=maxBlocks,2]), got ",
                    atten_mask->scalar_type());
    }

    std::string q_layout(q_input_layout);
    std::string kv_layout(kv_input_layout);
    check_params(query, key, value, actual_seq_lengths, actual_seq_lengths_kv, q_layout, kv_layout);

    const auto grad_dtype = is_hif8 ? at::kBFloat16 : query.scalar_type();
    at::Tensor d_query = at::empty(query.sizes(), query.options().dtype(grad_dtype));
    at::Tensor d_key = at::empty(key.sizes(), key.options().dtype(grad_dtype));
    at::Tensor d_value = at::empty(value.sizes(), value.options().dtype(grad_dtype));

    // block_shape 缺省 [128, 128]（与 op_plugin 参考实现一致）
    at::IntArrayRef block_shape_value = (block_shape.has_value() && block_shape->size() >= 2) ?
                                            at::IntArrayRef(*block_shape) :
                                            at::IntArrayRef(kDefaultBlockShape, 2);

    // optional<vector> → optional<IntArrayRef>（引用本作用域存活数据，避免 pybind 悬垂）
    c10::optional<at::IntArrayRef> actual_q_ref = c10::nullopt;
    c10::optional<at::IntArrayRef> actual_kv_ref = c10::nullopt;
    if (actual_seq_lengths.has_value()) {
        actual_q_ref = at::IntArrayRef(*actual_seq_lengths);
    }
    if (actual_seq_lengths_kv.has_value()) {
        actual_kv_ref = at::IntArrayRef(*actual_seq_lengths_kv);
    }

    char* q_layout_ptr = const_cast<char*>(q_layout.c_str());
    char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());

    // 滑动窗口参数：当前不支持自定义，按参考实现填默认值
    const int64_t pre_tokens = kMaxWindow;
    const int64_t next_tokens = kMaxWindow;

    if (is_hif8) {
        TensorWrapper dout_hif8{d_out, ACL_HIFLOAT8};
        TensorWrapper query_hif8{query, ACL_HIFLOAT8};
        TensorWrapper key_hif8{key, ACL_HIFLOAT8};
        TensorWrapper value_hif8{value, ACL_HIFLOAT8};
        ACLNN_CMD(aclnnBlockSparseAttentionGradV2, dout_hif8, query_hif8, key_hif8, value_hif8, attention_out,
                  softmax_lse, block_sparse_mask, atten_mask, block_shape_value, actual_q_ref, actual_kv_ref,
                  q_dequant_scale, k_dequant_scale, v_dequant_scale, do_dequant_scale, p_quant_scale, ds_quant_scale,
                  q_layout_ptr, kv_layout_ptr, num_key_value_heads, mask_type, scale_value, pre_tokens, next_tokens,
                  resolved_quant_mode, d_query, d_key, d_value);
    } else {
        TORCH_CHECK(!q_dequant_scale.has_value() && !k_dequant_scale.has_value() && !v_dequant_scale.has_value() &&
                        !do_dequant_scale.has_value() && !p_quant_scale.has_value() && !ds_quant_scale.has_value(),
                    "All six scales must be None for FP16/BF16 inputs.");
        ACLNN_CMD(aclnnBlockSparseAttentionGrad, d_out, query, key, value, attention_out, softmax_lse,
                  block_sparse_mask, atten_mask, block_shape_value, actual_q_ref, actual_kv_ref, q_layout_ptr,
                  kv_layout_ptr, num_key_value_heads, mask_type, scale_value, pre_tokens, next_tokens, d_query, d_key,
                  d_value);
    }

    return std::make_tuple(d_query, d_key, d_value);
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_block_sparse_attention_backward", &op_api::npu_block_sparse_attention_backward,
          "block_sparse_attention_backward");
}
