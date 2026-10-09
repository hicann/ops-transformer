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
 * \file mixed_quant_flash_attn.cpp
 * \brief mixed_quant_flash_attn 与 mixed_quant_flash_attn_metadata 的 torch extension 绑定
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t DIM_FOUR = 4;
const int64_t DIM_FIVE = 5;
const int64_t MAX_DIM_SIZE = 8;

std::tuple<at::Tensor, at::Tensor> mixed_quant_flash_attn(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& k_descale,
    const at::Tensor& v_descale, const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& cu_seqlens_q, const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv, const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& attn_mask, const c10::optional<at::Tensor>& metadata, int64_t quant_compute_mode,
    double softmax_scale, int64_t mask_mode, int64_t win_left, int64_t win_right, int64_t max_seqlen_q,
    int64_t max_seqlen_kv, string layout_q, string layout_kv, string layout_attn_out, bool return_softmax_lse)
{
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
        dSize = q.size(2);
    } else if (layout_q == "BSND") {
        bSize = q.size(0);
        sSize = q.size(1);
        nSize = q.size(2);
        dSize = q.size(3);
    } else {
        bSize = q.size(0);
        nSize = q.size(1);
        sSize = q.size(2);
        dSize = q.size(3);
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

    if (layout_attn_out == "TND") {
        attentionOutSize = {tSize, nSize, dSize};
    } else if (layout_attn_out == "BNSD") {
        attentionOutSize = {bSize, nSize, sSize, dSize};
    } else {
        attentionOutSize = {bSize, sSize, nSize, dSize};
    }
    at::Tensor attentionOutput = at::empty(attentionOutSize, q.options().dtype(q.dtype()));

    char* layout_q_ptr = const_cast<char*>(layout_q.c_str());
    char* layout_kv_ptr = const_cast<char*>(layout_kv.c_str());
    char* layout_attn_out_ptr = const_cast<char*>(layout_attn_out.c_str());

    TensorWrapper wrapperK = {k, ACL_DT_UNDEFINED};
    int64_t tempK = static_cast<int64_t>(k.scalar_type());
    if (kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(k.scalar_type())] == ACL_UINT8) {
        if (quant_compute_mode == 1) {
            wrapperK.dtype = ACL_FLOAT4_E2M1;
        } else if (quant_compute_mode == 2) {
            wrapperK.dtype = ACL_HIFLOAT4;
        }
    } else {
        wrapperK.dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(k.scalar_type())];
    }

    TensorWrapper wrapperV = {v, ACL_DT_UNDEFINED};
    int64_t tempV = static_cast<int64_t>(v.scalar_type());
    if (kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(v.scalar_type())] == ACL_UINT8) {
        if (quant_compute_mode == 1) {
            wrapperV.dtype = ACL_FLOAT4_E2M1;
        } else if (quant_compute_mode == 2) {
            wrapperV.dtype = ACL_HIFLOAT4;
        }
    } else {
        wrapperV.dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(v.scalar_type())];
    }

    TensorWrapper wrapperKDescale = {k_descale, ACL_DT_UNDEFINED};
    int64_t tempKDescale = static_cast<int64_t>(k_descale.scalar_type());
    if (quant_compute_mode == 1 && kATenScalarTypeToAclDataTypeTable[tempKDescale] == ACL_UINT8) {
        wrapperKDescale.dtype = ACL_FLOAT8_E8M0;
    } else if (quant_compute_mode == 2 && kATenScalarTypeToAclDataTypeTable[tempKDescale] == ACL_FLOAT) {
        wrapperKDescale.dtype = ACL_HIFLOAT4_SCALE;
    } else {
        wrapperKDescale.dtype = kATenScalarTypeToAclDataTypeTable[tempKDescale];
    }

    TensorWrapper wrapperVDescale = {v_descale, ACL_DT_UNDEFINED};
    int64_t tempVDescale = static_cast<int64_t>(v_descale.scalar_type());
    if (quant_compute_mode == 1 && kATenScalarTypeToAclDataTypeTable[tempVDescale] == ACL_UINT8) {
        wrapperVDescale.dtype = ACL_FLOAT8_E8M0;
    } else if (quant_compute_mode == 2 && kATenScalarTypeToAclDataTypeTable[tempVDescale] == ACL_FLOAT) {
        wrapperVDescale.dtype = ACL_HIFLOAT4_SCALE;
    } else {
        wrapperVDescale.dtype = kATenScalarTypeToAclDataTypeTable[tempVDescale];
    }

    ACLNN_CMD(aclnnMixedQuantFlashAttn, q, wrapperK, wrapperV, wrapperKDescale, wrapperVDescale, block_table,
              cu_seqlens_q, seqused_q, seqused_kv, sinks, attn_mask, metadata, quant_compute_mode, softmax_scale,
              mask_mode, win_left, win_right, max_seqlen_q, max_seqlen_kv, layout_q_ptr, layout_kv_ptr,
              layout_attn_out_ptr, return_softmax_lse, attentionOutput, softmaxLse);

    return std::tuple<at::Tensor, at::Tensor>(attentionOutput, softmaxLse);
}

at::Tensor mixed_quant_flash_attn_metadata(const c10::optional<at::Tensor>& cu_seqlens_q,
                                           const c10::optional<at::Tensor>& seqused_q,
                                           const c10::optional<at::Tensor>& seqused_kv, int64_t num_heads_q,
                                           int64_t num_heads_kv, int64_t head_dim, int64_t quant_compute_mode,
                                           int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_kv,
                                           int64_t mask_mode, int64_t win_left, int64_t win_right, std::string layout_q,
                                           std::string layout_kv, std::string layout_attn_out, const at::Tensor& output)
{
    ACLNN_CMD(aclnnMixedQuantFlashAttnMetadata, cu_seqlens_q, seqused_q, seqused_kv, batch_size, max_seqlen_q,
              max_seqlen_kv, num_heads_q, num_heads_kv, head_dim, quant_compute_mode, mask_mode, win_left, win_right,
              layout_q, layout_kv, layout_attn_out, output);
    return output;
}

// Bind the C++ functions to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mixed_quant_flash_attn", &mixed_quant_flash_attn, "mixed_quant_flash_attn");
    m.def("mixed_quant_flash_attn_metadata", &mixed_quant_flash_attn_metadata, "mixed_quant_flash_attn_metadata");
}
} // namespace op_api
