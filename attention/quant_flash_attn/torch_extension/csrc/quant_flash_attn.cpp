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
const int64_t QUANT_MODE_HIF4 = 10;
const int64_t QUANT_MODE_A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR_SOFTMAX_FP16 = 12;
const int64_t QUANT_MODE_HIF4_P_HIF4_LEVEL1 = 7;
const int64_t FLOAT4_E2M1 = 296;
const int64_t FLOAT8_E8M0 = 293;
const int64_t HIFLOAT8 = 34;
const int64_t HIFLOAT4 = 42;
const int64_t HIFLOAT4_SCALE = 43;

inline int64_t GetQkvDtypeRatio(int64_t quant_mode)
{
    // mxfp4 时 q/k/v 在 Python 端被 pack 成 uint8（每 byte 2 个 fp4 元素），
    // 因此 v.size(-1) 是 packed 维度（d/2），output 需要展开为逻辑 d（即 2 倍）。
    if (quant_mode == QUANT_MODE_MXFP4 || quant_mode == QUANT_MODE_HIF4 ||
        quant_mode == QUANT_MODE_HIF4_P_HIF4_LEVEL1) {
        return 2;
    }
    return 1;
}

TensorWrapper make_wrapper(const at::Tensor& tensor, const int64_t real_dtype)
{
    TensorWrapper wrapper = {tensor, ACL_DT_UNDEFINED};
    if (kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())] == ACL_UINT8) {
        if (real_dtype == FLOAT4_E2M1) {
            wrapper.dtype = ACL_FLOAT4_E2M1;
        } else if (real_dtype == FLOAT8_E8M0) {
            wrapper.dtype = ACL_FLOAT8_E8M0;
        } else if (real_dtype == HIFLOAT8) {
            wrapper.dtype = ACL_HIFLOAT8;
        } else if (real_dtype == HIFLOAT4) {
            wrapper.dtype = cann_ops_transformer::ACL_HIFLOAT4;
        }
    } else if (kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())] == ACL_FLOAT &&
               real_dtype == HIFLOAT4_SCALE) {
        wrapper.dtype = cann_ops_transformer::ACL_HIFLOAT4_SCALE;
    } else {
        wrapper.dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())];
    }
    return wrapper;
}

void CheckInt8Window(int64_t mode, int64_t left, int64_t right)
{
    TORCH_CHECK((mode == 0 || mode == 3) && left >= -1 && right >= -1 && left <= INT32_MAX && right <= INT32_MAX,
                "INT8 GQA requires mask 0/3 and windows in [-1, INT32_MAX]");
}

// max_seqlen is a static upper bound. Cumulative length values stay on device.
std::vector<int64_t> ValidateInt8Inputs(const at::Tensor& q, const at::Tensor& k, const c10::optional<at::Tensor>& cuq,
                                        const c10::optional<at::Tensor>& cuk, const c10::optional<at::Tensor>& metadata,
                                        int64_t maxQ, int64_t maxK, const std::string& layoutQ,
                                        const std::string& layoutK, const std::string& layoutOut, int64_t maskMode,
                                        int64_t winLeft, int64_t winRight)
{
    TORCH_CHECK(!metadata, "A2/A3 INT8 Prefill requires metadata=None");
    const bool packed = layoutQ == "TND";
    int64_t batch = q.size(0);
    if (packed) {
        TORCH_CHECK(cuq && cuk && cuq->scalar_type() == at::kInt && cuk->scalar_type() == at::kInt && cuq->dim() == 1 &&
                        cuk->dim() == 1 && cuq->numel() == cuk->numel() && cuq->numel() >= 2 && cuq->is_contiguous() &&
                        cuk->is_contiguous(),
                    "TND requires contiguous INT32 cumulative lengths [B+1]");
        batch = cuq->numel() - 1;
        TORCH_CHECK(maxQ != 0 && maxK != 0 && q.size(0) > 0 && k.size(0) > 0,
                    "TND max_seqlen bounds and total token counts must be positive");
        maxQ = maxQ < 0 ? q.size(0) : std::min(maxQ, q.size(0));
        maxK = maxK < 0 ? k.size(0) : std::min(maxK, k.size(0));
    } else {
        maxQ = q.size(layoutQ == "BNSD" ? 2 : 1);
        maxK = k.size(layoutK == "BNSD" ? 2 : 1);
    }
    return {batch, maxQ, maxK};
}

at::Tensor quant_flash_attn_metadata(const c10::optional<at::Tensor>& cu_seqlens_q,
                                     const c10::optional<at::Tensor>& cu_seqlens_kv,
                                     const c10::optional<at::Tensor>& seqused_q,
                                     const c10::optional<at::Tensor>& seqused_kv, int64_t batch_size,
                                     int64_t max_seqlen_q, int64_t max_seqlen_kv, int64_t num_heads_q,
                                     int64_t num_heads_kv, int64_t head_dim, int64_t head_dim_v, int64_t quant_mode,
                                     int64_t mask_mode, int64_t win_left, int64_t win_right, std::string layout_q,
                                     std::string layout_q_descale, std::string layout_kv, std::string layout_out,
                                     bool is_grad_enabled, const at::Tensor& output)
{
    TORCH_CHECK(quant_mode != QUANT_MODE_A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR_SOFTMAX_FP16,
                "quant_flash_attn_metadata does not support A2/A3 INT8 Prefill");
    ACLNN_CMD(aclnnQuantFlashAttnMetadata, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, batch_size, max_seqlen_q,
              max_seqlen_kv, num_heads_q, num_heads_kv, head_dim, head_dim_v, quant_mode, mask_mode, win_left,
              win_right, layout_q, layout_q_descale, layout_kv, layout_out, is_grad_enabled, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> quant_flash_attn_impl(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& q_descale,
    const at::Tensor& k_descale, const at::Tensor& v_descale, int64_t quant_mode,
    const c10::optional<at::Tensor>& block_table, const c10::optional<at::Tensor>& p_scale,
    const c10::optional<at::Tensor>& cu_seqlens_q, const c10::optional<at::Tensor>& cu_seqlens_kv,
    const c10::optional<at::Tensor>& seqused_q, const c10::optional<at::Tensor>& seqused_kv,
    const c10::optional<at::Tensor>& sinks, const c10::optional<at::Tensor>& attn_mask,
    const c10::optional<at::Tensor>& metadata, const c10::optional<at::Tensor>& v_tail,
    const c10::optional<at::Tensor>& block_table_tail, const c10::optional<at::Tensor>& seqused_v_tail,
    double softmax_scale, int64_t mask_mode, int64_t win_left, int64_t win_right, int64_t max_seqlen_q,
    int64_t max_seqlen_kv, std::string layout_q, std::string layout_q_descale, std::string layout_kv,
    std::string layout_out, bool return_softmax_lse, int64_t output_dim)
{
    TORCH_CHECK(output_dim >= 0, "Internal output_dim must be nonnegative");
    if (output_dim != 0) {
        // Only the private alignment adapter requests a compact output. Q/K/V
        // retain their physical aligned D; all internal math is unchanged.
        TORCH_CHECK(
            quant_mode == QUANT_MODE_A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR_SOFTMAX_FP16 &&
                layout_q == layout_kv && layout_q == layout_out &&
                (layout_q == "BNSD" || layout_q == "BSND" || layout_q == "TND"),
            "Compact INT8 output requires mode12 with matching dense/TND layouts");
        const int64_t rank = layout_q == "TND" ? 3 : 4;
        TORCH_CHECK(q.dim() == rank && k.dim() == rank && v.dim() == rank && q.scalar_type() == at::kChar &&
                        k.scalar_type() == at::kChar && v.scalar_type() == at::kChar && q.is_contiguous() &&
                        k.is_contiguous() && v.is_contiguous() && q.device() == k.device() && q.device() == v.device(),
                    "Compact INT8 output requires contiguous, same-device INT8 Q/K/V");
        const int64_t physical_dim = q.size(-1);
        TORCH_CHECK(physical_dim >= 64 && physical_dim <= 256 && physical_dim % 32 == 0 && k.size(-1) == physical_dim &&
                        v.size(-1) == physical_dim && output_dim >= 64 && output_dim <= physical_dim &&
                        output_dim % 8 == 0 && ((output_dim + 31) / 32) * 32 == physical_dim,
                    "Compact INT8 output must remain within its physical 32-channel alignment bucket");
    }
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1 || quant_mode == 3 || quant_mode == 5 || quant_mode == 6 ||
                    quant_mode == 10 || quant_mode == QUANT_MODE_HIF4_P_HIF4_LEVEL1 ||
                    quant_mode == QUANT_MODE_A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR_SOFTMAX_FP16,
                "Unsupported quant_mode");
    const bool pagedInt8 =
        quant_mode == QUANT_MODE_A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR_SOFTMAX_FP16 &&
        (layout_kv == "PA_BNBD" || layout_kv == "PA_BBND" || layout_kv == "PA_NZ");
    TORCH_CHECK(!pagedInt8 || !metadata, "A2/A3 paged INT8 QFA does not support metadata");
    if (quant_mode == QUANT_MODE_A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR_SOFTMAX_FP16 &&
        !pagedInt8) {
        CheckInt8Window(mask_mode, win_left, win_right);
        const int64_t rank = layout_q == "TND" ? 3 : 4;
        const int64_t kvHeadAxis = layout_kv == "BSND" ? 2 : 1;
        TORCH_CHECK(q.dim() == rank && k.dim() == rank && v.dim() == rank,
                    "Full INT8 GQA requires layout-matching Q/K/V ranks");
        TORCH_CHECK(q.size(-1) >= 1 && q.size(-1) <= 256, "Full INT8 GQA requires D in [1,256]");
        TORCH_CHECK(q_descale.defined() && k_descale.defined() && v_descale.defined() && p_scale.has_value() &&
                        p_scale->defined(),
                    "Full INT8 GQA requires Q/K/V descales and p_scale");
        TORCH_CHECK(q_descale.scalar_type() == at::kFloat && k_descale.scalar_type() == at::kFloat &&
                        v_descale.scalar_type() == at::kFloat && p_scale->scalar_type() == at::kFloat,
                    "Full INT8 GQA requires FP32 scales");
        TORCH_CHECK(q_descale.sizes() == q.sizes().slice(0, rank - 1) &&
                        k_descale.sizes() == k.sizes().slice(0, rank - 1) && v_descale.dim() == 1 &&
                        v_descale.size(0) == k.size(kvHeadAxis) && p_scale->dim() == 1 && p_scale->numel() == 1,
                    "Full INT8 GQA requires q_descale [B,Hq,Sq], k_descale [B,Hkv,Sk], v_descale [Hkv], p_scale [1]");
        TORCH_CHECK(q_descale.is_contiguous() && k_descale.is_contiguous() && v_descale.is_contiguous() &&
                        p_scale->is_contiguous(),
                    "Full INT8 GQA requires contiguous scales");
        TORCH_CHECK(q_descale.device() == q.device() && k_descale.device() == q.device() &&
                        v_descale.device() == q.device() && p_scale->device() == q.device(),
                    "Full INT8 GQA scales must be on the Q device");
        const auto bounds = ValidateInt8Inputs(q, k, cu_seqlens_q, cu_seqlens_kv, metadata, max_seqlen_q, max_seqlen_kv,
                                               layout_q, layout_kv, layout_out, mask_mode, win_left, win_right);
        max_seqlen_q = bounds[1];
        max_seqlen_kv = bounds[2];
    }
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
        dSize = pagedInt8 ? q.size(q.dim() - 1) : v.size(2) * v.size(4);
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
    at::Tensor softmaxLse = at::empty(softmaxOutSize, q.options().dtype(at::kFloat));
    if (output_dim != 0) {
        dSize = output_dim;
    }
    int64_t qDtypeRatio = GetQkvDtypeRatio(quant_mode);
    if (layout_out == "TND") {
        attentionOutSize = {tSize, nSize, qDtypeRatio * dSize};
    } else if (layout_out == "BNSD") {
        attentionOutSize = {bSize, nSize, sSize, qDtypeRatio * dSize};
    } else {
        attentionOutSize = {bSize, sSize, nSize, qDtypeRatio * dSize};
    }
    const auto outputDtype = at::kBFloat16;
    at::Tensor attentionOutput = at::empty(attentionOutSize, q.options().dtype(outputDtype));

    char* layout_q_ptr = const_cast<char*>(layout_q.c_str());
    char* layout_q_descale_ptr = const_cast<char*>(layout_q_descale.c_str());
    char* layout_kv_ptr = const_cast<char*>(layout_kv.c_str());
    char* layout_out_ptr = const_cast<char*>(layout_out.c_str());
    if (quant_mode == QUANT_MODE_MXFP4) {
        TensorWrapper q_wrapper = make_wrapper(q, FLOAT4_E2M1);
        TensorWrapper k_wrapper = make_wrapper(k, FLOAT4_E2M1);
        TensorWrapper v_wrapper = make_wrapper(v, FLOAT4_E2M1);
        TensorWrapper q_descale_wrapper = make_wrapper(q_descale, FLOAT8_E8M0);
        TensorWrapper k_descale_wrapper = make_wrapper(k_descale, FLOAT8_E8M0);
        TensorWrapper v_descale_wrapper = make_wrapper(v_descale, FLOAT8_E8M0);
        ACLNN_CMD(aclnnQuantFlashAttn, q_wrapper, k_wrapper, v_wrapper, q_descale_wrapper, k_descale_wrapper,
                  v_descale_wrapper, block_table, p_scale, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks,
                  attn_mask, metadata, v_tail, block_table_tail, seqused_v_tail, quant_mode, softmax_scale, mask_mode,
                  win_left, win_right, max_seqlen_q, max_seqlen_kv, layout_q_ptr, layout_q_descale_ptr, layout_kv_ptr,
                  layout_out_ptr, return_softmax_lse, attentionOutput, softmaxLse);
    } else if (quant_mode == QUANT_MODE_HIF8) {
        TensorWrapper q_wrapper = make_wrapper(q, HIFLOAT8);
        TensorWrapper k_wrapper = make_wrapper(k, HIFLOAT8);
        TensorWrapper v_wrapper = make_wrapper(v, HIFLOAT8);
        ACLNN_CMD(aclnnQuantFlashAttn, q_wrapper, k_wrapper, v_wrapper, q_descale, k_descale, v_descale, block_table,
                  p_scale, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask, metadata, v_tail,
                  block_table_tail, seqused_v_tail, quant_mode, softmax_scale, mask_mode, win_left, win_right,
                  max_seqlen_q, max_seqlen_kv, layout_q_ptr, layout_q_descale_ptr, layout_kv_ptr, layout_out_ptr,
                  return_softmax_lse, attentionOutput, softmaxLse);
    } else if (quant_mode == QUANT_MODE_HIF4 || quant_mode == QUANT_MODE_HIF4_P_HIF4_LEVEL1) {
        TensorWrapper q_wrapper = make_wrapper(q, HIFLOAT4);
        TensorWrapper k_wrapper = make_wrapper(k, HIFLOAT4);
        TensorWrapper v_wrapper = make_wrapper(v, HIFLOAT4);
        TensorWrapper q_descale_wrapper = make_wrapper(q_descale, HIFLOAT4_SCALE);
        TensorWrapper k_descale_wrapper = make_wrapper(k_descale, HIFLOAT4_SCALE);
        TensorWrapper v_descale_wrapper = make_wrapper(v_descale, HIFLOAT4_SCALE);
        ACLNN_CMD(aclnnQuantFlashAttn, q_wrapper, k_wrapper, v_wrapper, q_descale_wrapper, k_descale_wrapper,
                  v_descale_wrapper, block_table, p_scale, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks,
                  attn_mask, metadata, v_tail, block_table_tail, seqused_v_tail, quant_mode, softmax_scale, mask_mode,
                  win_left, win_right, max_seqlen_q, max_seqlen_kv, layout_q_ptr, layout_q_descale_ptr, layout_kv_ptr,
                  layout_out_ptr, return_softmax_lse, attentionOutput, softmaxLse);
    } else {
        ACLNN_CMD(aclnnQuantFlashAttn, q, k, v, q_descale, k_descale, v_descale, block_table, p_scale, cu_seqlens_q,
                  cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask, metadata, v_tail, block_table_tail,
                  seqused_v_tail, quant_mode, softmax_scale, mask_mode, win_left, win_right, max_seqlen_q,
                  max_seqlen_kv, layout_q_ptr, layout_q_descale_ptr, layout_kv_ptr, layout_out_ptr, return_softmax_lse,
                  attentionOutput, softmaxLse);
    }

    return std::tuple<at::Tensor, at::Tensor>(attentionOutput, softmaxLse);
}

// Preserve the existing pybind signature. Public torch.library and GE shape
// inference continue to return the untrimmed V dimension.
std::tuple<at::Tensor, at::Tensor> quant_flash_attn(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& q_descale,
    const at::Tensor& k_descale, const at::Tensor& v_descale, int64_t quant_mode,
    const c10::optional<at::Tensor>& block_table, const c10::optional<at::Tensor>& p_scale,
    const c10::optional<at::Tensor>& cu_seqlens_q, const c10::optional<at::Tensor>& cu_seqlens_kv,
    const c10::optional<at::Tensor>& seqused_q, const c10::optional<at::Tensor>& seqused_kv,
    const c10::optional<at::Tensor>& sinks, const c10::optional<at::Tensor>& attn_mask,
    const c10::optional<at::Tensor>& metadata, const c10::optional<at::Tensor>& v_tail,
    const c10::optional<at::Tensor>& block_table_tail, const c10::optional<at::Tensor>& seqused_v_tail,
    double softmax_scale, int64_t mask_mode, int64_t win_left, int64_t win_right, int64_t max_seqlen_q,
    int64_t max_seqlen_kv, std::string layout_q, std::string layout_q_descale, std::string layout_kv,
    std::string layout_out, bool return_softmax_lse)
{
    return quant_flash_attn_impl(q, k, v, q_descale, k_descale, v_descale, quant_mode, block_table, p_scale,
                                 cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask, metadata, v_tail,
                                 block_table_tail, seqused_v_tail, softmax_scale, mask_mode, win_left, win_right,
                                 max_seqlen_q, max_seqlen_kv, layout_q, layout_q_descale, layout_kv, layout_out,
                                 return_softmax_lse, 0);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("validate_int8_inputs", &ValidateInt8Inputs);
    m.def("quant_flash_attn_metadata", &quant_flash_attn_metadata, "quant_flash_attn_metadata");
    m.def("quant_flash_attn", &quant_flash_attn, "quant_flash_attn");
    // Internal eager-only entry. The padding helper supplies every argument,
    // including the trailing logical output dimension.
    m.def("quant_flash_attn_padded_out", &quant_flash_attn_impl, "Internal aligned INT8 compact output");
}
} // namespace op_api
