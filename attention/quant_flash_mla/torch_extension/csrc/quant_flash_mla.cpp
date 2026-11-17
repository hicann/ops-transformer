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
 * \file quant_flash_mla.cpp
 * \brief QuantFlashMla/QuantFlashMlaMetadata torch extension: 调用aclnn接口
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_THREE = 3;
const int64_t DIM_FOUR = 4;
const int64_t MAX_DIM_SIZE = 8;
const int64_t QUANT_MODE_HIF8 = 0;
const int64_t HIFLOAT8 = 34;

// HIF8场景: torch侧以uint8承载hif8编码, 传给aclnn前需映射为ACL_HIFLOAT8
inline TensorWrapper make_wrapper(const at::Tensor& tensor, const int64_t real_dtype)
{
    TensorWrapper wrapper = {tensor, ACL_DT_UNDEFINED};
    if (kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())] == ACL_UINT8) {
        if (real_dtype == HIFLOAT8) {
            wrapper.dtype = ACL_HIFLOAT8;
        }
    } else {
        wrapper.dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(tensor.scalar_type())];
    }
    return wrapper;
}

at::Tensor quant_flash_mla_metadata(const at::Tensor& cache_seqlens, int64_t num_heads_q, int64_t num_heads_kv,
                                    int64_t quant_mode, const c10::optional<at::Tensor>& cu_seqlens_q,
                                    const c10::optional<at::Tensor>& seqused_q, int64_t max_seqlen_q,
                                    int64_t max_seqlen_kv, int64_t head_dim_qk, int64_t head_dim_v, int64_t mask_mode,
                                    std::string layout_q, const at::Tensor& output)
{
    ACLNN_CMD(aclnnQuantFlashMlaMetadata, cache_seqlens, num_heads_q, num_heads_kv, quant_mode, cu_seqlens_q, seqused_q,
              max_seqlen_q, max_seqlen_kv, head_dim_qk, head_dim_v, mask_mode, layout_q, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> quant_flash_mla(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& q_descale, const at::Tensor& k_descale,
    const at::Tensor& block_table, const at::Tensor& cache_seqlens, int64_t quant_mode,
    const c10::optional<at::Tensor>& cu_seqlens_q, const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& attn_mask, const c10::optional<at::Tensor>& metadata,
    const c10::optional<at::Tensor>& q_rope, const c10::optional<at::Tensor>& k_rope, int64_t head_dim_v,
    double softmax_scale, int64_t mask_mode, int64_t max_seqlen_q, int64_t max_seqlen_kv, std::string layout_q,
    std::string layout_kv, std::string layout_out, bool return_softmax_lse)
{
    const c10::string_view device = "npu";
    at::Device outputDevice = at::Device(std::string(device));
    int64_t tSize = 0;
    int64_t nSize = 0;
    int64_t bSize = 1;
    int64_t sSize = 0;
    at::SmallVector<int64_t, MAX_DIM_SIZE> attentionOutSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> softmaxOutSize;

    // q: TND(T,N,D) / BSND(B,S,N,D) / BNSD(B,N,S,D), D固定576(nope512+rope64);
    // BSH(arch22 int8全量化, D=512 nope, rope分离): (B,S,N*512)
    if (layout_q == "TND") {
        tSize = q.size(0);
        nSize = q.size(1);
        // TND时bSize从cu_seqlens_q(B+1个元素)或cache_seqlens(B个元素)推导
        if (cu_seqlens_q.has_value() && cu_seqlens_q->defined()) {
            bSize = cu_seqlens_q->size(0) - 1;
        } else if (cache_seqlens.defined()) {
            bSize = cache_seqlens.size(0);
        }
    } else if (layout_q == "BSND") {
        bSize = q.size(0);
        sSize = q.size(1);
        nSize = q.size(2);
    } else if (layout_q == "BSH") {
        bSize = q.size(0);
        sSize = q.size(1);
        nSize = q.size(2) / 512; // nope D=512
    } else {                     // BNSD
        bSize = q.size(0);
        nSize = q.size(1);
        sSize = q.size(2);
    }
    if (return_softmax_lse) {
        if (layout_q == "TND") {
            softmaxOutSize = {nSize, tSize};
        } else {
            softmaxOutSize = {bSize, nSize, sSize};
        }
    } else {
        softmaxOutSize = {0};
    }
    at::Tensor softmaxLse = at::empty(softmaxOutSize, torch::dtype(at::kFloat).device(outputDevice));

    // attn_out: BF16, head_dim_v默认512由python侧保证
    if (layout_out == "BSND") {
        attentionOutSize = {bSize, sSize, nSize, head_dim_v};
    } else if (layout_out == "BNSD") {
        attentionOutSize = {bSize, nSize, sSize, head_dim_v};
    } else if (layout_out == "BSH") {
        // arch22 int8全量化: 输出跟随q layout, 3D (B, S, N*512)
        attentionOutSize = {bSize, sSize, nSize * head_dim_v};
    } else if (layout_out == "TND") {
        attentionOutSize = {tSize, nSize, head_dim_v};
    } else { // NTD
        attentionOutSize = {nSize, tSize, head_dim_v};
    }
    at::Tensor attentionOutput = at::empty(attentionOutSize, torch::dtype(at::kBFloat16).device(outputDevice));

    if (quant_mode == QUANT_MODE_HIF8) {
        // HIF8: q/k_cache为uint8承载的hif8编码, 映射为ACL_HIFLOAT8
        TensorWrapper q_wrapper = make_wrapper(q, HIFLOAT8);
        TensorWrapper k_cache_wrapper = make_wrapper(k_cache, HIFLOAT8);
        ACLNN_CMD(aclnnQuantFlashMla, q_wrapper, k_cache_wrapper, q_descale, k_descale, block_table, cache_seqlens,
                  cu_seqlens_q, seqused_q, attn_mask, metadata, q_rope, k_rope, quant_mode, softmax_scale, mask_mode,
                  max_seqlen_q, max_seqlen_kv, head_dim_v, layout_q, layout_kv, layout_out, return_softmax_lse,
                  attentionOutput, softmaxLse);
        return std::tuple<at::Tensor, at::Tensor>(attentionOutput, softmaxLse);
    }

    ACLNN_CMD(aclnnQuantFlashMla, q, k_cache, q_descale, k_descale, block_table, cache_seqlens, cu_seqlens_q, seqused_q,
              attn_mask, metadata, q_rope, k_rope, quant_mode, softmax_scale, mask_mode, max_seqlen_q, max_seqlen_kv,
              head_dim_v, layout_q, layout_kv, layout_out, return_softmax_lse, attentionOutput, softmaxLse);

    return std::tuple<at::Tensor, at::Tensor>(attentionOutput, softmaxLse);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_flash_mla_metadata", &quant_flash_mla_metadata, "quant_flash_mla_metadata");
    m.def("quant_flash_mla", &quant_flash_mla, "quant_flash_mla");
}
} // namespace op_api
