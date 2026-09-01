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
 * \file quant_flash_attn.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;
const int64_t QUANT_MODE_MXFP4 = 5;
const int64_t QUANT_MODE_HIF8 = 0;
const int64_t FLOAT4_E2M1 = 296;
const int64_t FLOAT8_E8M0 = 293;
const int64_t HIFLOAT8 = 34;

inline int64_t GetQkvDtypeRatio(int64_t quant_mode)
{
    // mxfp4 时 q/k/v 在 Python 端被 pack 成 uint8（每 byte 2 个 fp4 元素），
    // 因此 v.size(-1) 是 packed 维度（d/2），output 需要展开为逻辑 d（即 2 倍）。
    if (quant_mode == QUANT_MODE_MXFP4) {
        return 2;
    }
    return 1;
}

TensorWrapper make_wrapper(const at::Tensor &tensor, const int64_t real_dtype)
{
    TensorWrapper wrapper = {tensor, ACL_DT_UNDEFINED};
    int64_t temp = static_cast<int64_t>(tensor.scalar_type());
    if (kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())] == ACL_UINT8) {
        if (real_dtype == FLOAT4_E2M1) {
            wrapper.dtype = ACL_FLOAT4_E2M1;
        } else if (real_dtype == FLOAT8_E8M0) {
            wrapper.dtype = ACL_FLOAT8_E8M0;
        } else if (real_dtype == HIFLOAT8) {
            wrapper.dtype = ACL_HIFLOAT8;
        }
    } else {
        wrapper.dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())];
    }
    return wrapper;
}

at::Tensor quant_flash_attn_metadata(const c10::optional<at::Tensor> &cu_seqlens_q,
                                     const c10::optional<at::Tensor> &cu_seqlens_kv,
                                     const c10::optional<at::Tensor> &seqused_q,
                                     const c10::optional<at::Tensor> &seqused_kv, int64_t batch_size,
                                     int64_t max_seqlen_q, int64_t max_seqlen_kv, int64_t num_heads_q,
                                     int64_t num_heads_kv, int64_t head_dim, int64_t head_dim_v, int64_t quant_mode,
                                     int64_t mask_mode, int64_t win_left, int64_t win_right, std::string layout_q,
                                     std::string layout_q_descale, std::string layout_kv, std::string layout_out,
                                     bool is_grad_enabled, const at::Tensor &output)
{
    ACLNN_CMD(aclnnQuantFlashAttnMetadata, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, batch_size, max_seqlen_q,
              max_seqlen_kv, num_heads_q, num_heads_kv, head_dim, head_dim_v, quant_mode, mask_mode, win_left,
              win_right, layout_q, layout_q_descale, layout_kv, layout_out, is_grad_enabled, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> quant_flash_attn(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &q_descale,
    const at::Tensor &k_descale, const at::Tensor &v_descale, int64_t quant_mode,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &p_scale,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &attn_mask,
    const c10::optional<at::Tensor> &metadata, double softmax_scale, int64_t mask_mode, int64_t win_left,
    int64_t win_right, int64_t max_seqlen_q, int64_t max_seqlen_kv, std::string layout_q, std::string layout_q_descale,
    std::string layout_kv, std::string layout_out, bool return_softmax_lse)
{
    const c10::string_view device = "npu";
    at::Device outputDevice = at::Device(std::string(device));
    int64_t tSize = 0;
    int64_t nSize = 0;
    int64_t dSize = 0;
    int64_t sSize = 0;
    int64_t bSize = 0;
    at::SmallVector<int64_t, MAX_DIM_SIZE> attentionOutSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> softmaxOutSize;

    if (layout_q == "TND") {
        tSize = q.size(0);
        nSize = q.size(1);
    } else if (layout_q == "NTD") {
        nSize = q.size(0);
        tSize = q.size(1);
    } else if (layout_q == "BSND") {
        bSize = q.size(0);
        sSize = q.size(1);
        nSize = q.size(2);
    } else {
        bSize = q.size(0);
        nSize = q.size(1);
        sSize = q.size(2);
    }
    if (layout_kv == "TND") {
        dSize = v.size(2);
    } else if (layout_kv == "PA_NZ") {
        dSize = v.size(2) * v.size(4);
    } else {
        dSize = v.size(v.dim() - 1);
    }
    if (return_softmax_lse) {
        if (q.dim() == DIM_THREE) {
            softmaxOutSize = {nSize, tSize};
        } else {
            softmaxOutSize = {bSize, nSize, sSize};
        }
    } else {
        softmaxOutSize = {0};
    }
    at::Tensor softmaxLse = at::empty(softmaxOutSize, torch::dtype(at::kFloat).device(outputDevice));
    int64_t qDtypeRatio = GetQkvDtypeRatio(quant_mode);
    if (layout_out == "TND") {
        attentionOutSize = {tSize, nSize, qDtypeRatio * dSize};
    } else if (layout_out == "BNSD") {
        attentionOutSize = {bSize, nSize, sSize, qDtypeRatio * dSize};
    } else {
        attentionOutSize = {bSize, sSize, nSize, qDtypeRatio * dSize};
    }
    at::Tensor attentionOutput = at::empty(attentionOutSize, torch::dtype(at::kBFloat16).device(outputDevice));

    char *layout_q_ptr = const_cast<char *>(layout_q.c_str());
    char *layout_q_descale_ptr = const_cast<char *>(layout_q_descale.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv.c_str());
    char *layout_out_ptr = const_cast<char *>(layout_out.c_str());
    if (quant_mode == QUANT_MODE_MXFP4) {
        TensorWrapper q_wrapper = make_wrapper(q, FLOAT4_E2M1);
        TensorWrapper k_wrapper = make_wrapper(k, FLOAT4_E2M1);
        TensorWrapper v_wrapper = make_wrapper(v, FLOAT4_E2M1);
        TensorWrapper q_descale_wrapper = make_wrapper(q_descale, FLOAT8_E8M0);
        TensorWrapper k_descale_wrapper = make_wrapper(k_descale, FLOAT8_E8M0);
        TensorWrapper v_descale_wrapper = make_wrapper(v_descale, FLOAT8_E8M0);
        ACLNN_CMD(aclnnQuantFlashAttn, q_wrapper, k_wrapper, v_wrapper, q_descale_wrapper, k_descale_wrapper,
                  v_descale_wrapper, block_table, p_scale, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks,
                  attn_mask, metadata, quant_mode, softmax_scale, mask_mode, win_left, win_right, max_seqlen_q,
                  max_seqlen_kv, layout_q_ptr, layout_q_descale_ptr, layout_kv_ptr, layout_out_ptr, return_softmax_lse,
                  attentionOutput, softmaxLse);
    } else if (quant_mode == QUANT_MODE_HIF8) {
        TensorWrapper q_wrapper = make_wrapper(q, HIFLOAT8);
        TensorWrapper k_wrapper = make_wrapper(k, HIFLOAT8);
        TensorWrapper v_wrapper = make_wrapper(v, HIFLOAT8);
        ACLNN_CMD(aclnnQuantFlashAttn, q_wrapper, k_wrapper, v_wrapper, q_descale, k_descale, v_descale, block_table,
                  p_scale, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask, metadata, quant_mode,
                  softmax_scale, mask_mode, win_left, win_right, max_seqlen_q, max_seqlen_kv, layout_q_ptr,
                  layout_q_descale_ptr, layout_kv_ptr, layout_out_ptr, return_softmax_lse, attentionOutput, softmaxLse);
    } else {
        ACLNN_CMD(aclnnQuantFlashAttn, q, k, v, q_descale, k_descale, v_descale, block_table, p_scale, cu_seqlens_q,
                  cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask, metadata, quant_mode, softmax_scale,
                  mask_mode, win_left, win_right, max_seqlen_q, max_seqlen_kv, layout_q_ptr, layout_q_descale_ptr,
                  layout_kv_ptr, layout_out_ptr, return_softmax_lse, attentionOutput, softmaxLse);
    }

    return std::tuple<at::Tensor, at::Tensor>(attentionOutput, softmaxLse);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_flash_attn_metadata", &quant_flash_attn_metadata, "quant_flash_attn_metadata");
    m.def("quant_flash_attn", &quant_flash_attn, "quant_flash_attn");
}
} // namespace op_api
