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

const int64_t FLOAT4_E2M1 = static_cast<int64_t>(296);
const int64_t FLOAT8_E8M0 = static_cast<int64_t>(293);

TensorWrapper make_wrapper(const at::Tensor &tensor, const int64_t real_dtype)
{
    TensorWrapper wrapper = {tensor, ACL_DT_UNDEFINED};
    if (kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())] == ACL_UINT8) {
        if (real_dtype == FLOAT4_E2M1) {
            wrapper.dtype = ACL_FLOAT4_E2M1;
        } else if (real_dtype == FLOAT8_E8M0) {
            wrapper.dtype = ACL_FLOAT8_E8M0;
        }
    } else {
        wrapper.dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())];
    }
    return wrapper;
}

enum QuantMode : int64_t {
    NO_QUANT = 0,
    FP8_QUANT = 1,
    MXFP4_OCP_QUANT = 2,
    MXFP4_CX_QUANT = 3,
};

std::tuple<at::Tensor, at::Tensor> npu_block_sparse_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &block_sparse_mask,
    std::vector<int64_t> block_shape, c10::string_view q_input_layout, c10::string_view kv_input_layout,
    int64_t num_key_value_heads, double scale_value, int64_t inner_precise,
    const c10::optional<std::vector<int64_t>> &actual_seq_lengths,
    const c10::optional<std::vector<int64_t>> &actual_seq_lengths_kv, int64_t softmax_lse_flag, int64_t mask_type,
    int64_t quant_mode, int64_t block_size, int64_t pre_tokens, int64_t next_tokens, double dst_type_max,
    const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &q_dequant_scale, const c10::optional<at::Tensor> &k_dequant_scale,
    const c10::optional<at::Tensor> &v_dequant_scale, const c10::optional<at::Tensor> &p_quant_scale,
    c10::optional<at::ScalarType> attention_out_dtype)
{
    // pybind 按值收下 std::vector<int64_t>（真实拷贝），再在本函数作用域内构建
    // IntArrayRef，避免直接收 optional<IntArrayRef> 时 pybind 的悬垂指针问题
    // （inner caster 的 vector 在 load() 返回后即被释放）。
    at::IntArrayRef block_shape_ref(block_shape);
    c10::optional<at::IntArrayRef> actual_seq_lengths_ref = c10::nullopt;
    c10::optional<at::IntArrayRef> actual_seq_lengths_kv_ref = c10::nullopt;
    if (actual_seq_lengths.has_value()) {
        actual_seq_lengths_ref = at::IntArrayRef(*actual_seq_lengths);
    }
    if (actual_seq_lengths_kv.has_value()) {
        actual_seq_lengths_kv_ref = at::IntArrayRef(*actual_seq_lengths_kv);
    }

    // Resolve output dtype: an explicit value passes through unchanged.
    // A missing value (None) only falls back to the input dtype for
    // quant_mode=0 (no quant); quantized paths require it explicitly.
    at::ScalarType out_dtype;
    if (attention_out_dtype.has_value()) {
        out_dtype = *attention_out_dtype;
    } else if (quant_mode == NO_QUANT) {
        out_dtype = query.scalar_type();
    } else {
        TORCH_CHECK(false, "attention_out_dtype must be specified when quant_mode != 0");
    }

    // FP4: two values pack into one uint8 byte, so last dim is 2*D
    at::Tensor attention_out;
    if (quant_mode == MXFP4_OCP_QUANT || quant_mode == MXFP4_CX_QUANT) {
        auto out_sizes = query.sizes().vec();
        out_sizes.back() *= 2;
        attention_out = at::empty(out_sizes, query.options().dtype(out_dtype));
    } else {
        attention_out = at::empty(query.sizes(), query.options().dtype(out_dtype));
    }

    at::Tensor softmax_lse;
    auto opts_f32 = query.options().dtype(at::kFloat);
    if (softmax_lse_flag) {
        if (q_input_layout == "TND") {
            softmax_lse = at::empty({query.size(0), query.size(1), 1}, opts_f32);
        } else {
            softmax_lse = at::empty({query.size(0), query.size(1), query.size(2), 1}, opts_f32);
        }
    } else {
        softmax_lse = at::empty({0}, opts_f32);
    }

    std::string q_layout(q_input_layout);
    std::string kv_layout(kv_input_layout);
    char *q_layout_ptr = const_cast<char *>(q_layout.c_str());
    char *kv_layout_ptr = const_cast<char *>(kv_layout.c_str());

    // block_shape_ref / actual_seq_lengths_ref / actual_seq_lengths_kv_ref 为指向
    // 本作用域存活数据的 IntArrayRef，ACLNN_CMD 内部 ConvertType 会转成 aclIntArray。
    if (quant_mode == MXFP4_OCP_QUANT || quant_mode == MXFP4_CX_QUANT) {
        TensorWrapper q_wrapper = make_wrapper(query, FLOAT4_E2M1);
        TensorWrapper k_wrapper = make_wrapper(key, FLOAT4_E2M1);
        TensorWrapper v_wrapper = make_wrapper(value, FLOAT4_E2M1);
        TensorWrapper q_descale_wrapper = make_wrapper(*q_dequant_scale, FLOAT8_E8M0);
        TensorWrapper k_descale_wrapper = make_wrapper(*k_dequant_scale, FLOAT8_E8M0);
        TensorWrapper v_descale_wrapper = make_wrapper(*v_dequant_scale, FLOAT8_E8M0);
        ACLNN_CMD(aclnnBlockSparseAttentionV3, q_wrapper, k_wrapper, v_wrapper, block_sparse_mask, atten_mask,
                  block_shape_ref, actual_seq_lengths_ref, actual_seq_lengths_kv_ref, block_table, q_descale_wrapper,
                  k_descale_wrapper, v_descale_wrapper, p_quant_scale, q_layout_ptr, kv_layout_ptr, num_key_value_heads,
                  mask_type, scale_value, inner_precise, block_size, pre_tokens, next_tokens, softmax_lse_flag,
                  quant_mode, dst_type_max, attention_out, softmax_lse);
    } else {
        // FP8 / FP16 / BF16
        ACLNN_CMD(aclnnBlockSparseAttentionV3, query, key, value, block_sparse_mask, atten_mask, block_shape_ref,
                  actual_seq_lengths_ref, actual_seq_lengths_kv_ref, block_table, q_dequant_scale, k_dequant_scale,
                  v_dequant_scale, p_quant_scale, q_layout_ptr, kv_layout_ptr, num_key_value_heads, mask_type,
                  scale_value, inner_precise, block_size, pre_tokens, next_tokens, softmax_lse_flag, quant_mode,
                  dst_type_max, attention_out, softmax_lse);
    }

    return std::make_tuple(attention_out, softmax_lse);
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_block_sparse_attention", &op_api::npu_block_sparse_attention, "block_sparse_attention");
}
