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
#include <string>
#include <tuple>
#include "aclnn_common.h"

namespace op_api {
constexpr int64_t DLI_KL_LOSS_GRAD_METADATA_SIZE = 64;

at::Device get_dense_lightning_kl_loss_grad_metadata_device(const c10::optional<at::Tensor> &cu_seqlens_q,
                                                            const c10::optional<at::Tensor> &cu_seqlens_k,
                                                            const c10::optional<at::Tensor> &seqused_q,
                                                            const c10::optional<at::Tensor> &seqused_k,
                                                            const c10::optional<at::Tensor> &cmp_residual_k)
{
    if (cu_seqlens_q.has_value() && cu_seqlens_q.value().defined()) {
        return cu_seqlens_q.value().device();
    }
    if (cu_seqlens_k.has_value() && cu_seqlens_k.value().defined()) {
        return cu_seqlens_k.value().device();
    }
    if (seqused_q.has_value() && seqused_q.value().defined()) {
        return seqused_q.value().device();
    }
    if (seqused_k.has_value() && seqused_k.value().defined()) {
        return seqused_k.value().device();
    }
    if (cmp_residual_k.has_value() && cmp_residual_k.value().defined()) {
        return cmp_residual_k.value().device();
    }
    return at::Device(torch_npu::utils::get_npu_device_type());
}

at::Tensor dense_lightning_indexer_kl_loss_grad_metadata(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k, const c10::optional<at::Tensor> &cmp_residual_k, int64_t batch_size,
    int64_t max_seqlen_q, int64_t max_seqlen_k, std::string layout_q, std::string layout_k, int64_t mask_mode,
    int64_t cmp_ratio)
{
    at::Tensor output =
        at::empty({DLI_KL_LOSS_GRAD_METADATA_SIZE},
                  at::TensorOptions().dtype(at::kInt).device(get_dense_lightning_kl_loss_grad_metadata_device(
                      cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k)));

    char *layout_q_ptr = const_cast<char *>(layout_q.c_str());
    char *layout_k_ptr = const_cast<char *>(layout_k.c_str());

    ACLNN_CMD(aclnnDenseLightningIndexerKLLossGradMetadata, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
              cmp_residual_k, batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_k, head_dim, layout_q_ptr,
              layout_k_ptr, mask_mode, cmp_ratio, output);

    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> dense_lightning_indexer_kl_loss_grad(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &w, const at::Tensor &attn_softmax_l1_norm,
    const at::Tensor &softmax_lse, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k, const c10::optional<at::Tensor> &cmp_residual_k,
    const c10::optional<at::Tensor> &metadata, std::string layout_q, std::string layout_k, int64_t mask_mode,
    int64_t cmp_ratio)
{
    at::Tensor dq = at::empty_like(q);
    at::Tensor dk = at::empty_like(k);
    at::Tensor dw = at::empty_like(w);
    at::Tensor softmax_out = at::empty_like(attn_softmax_l1_norm);

    char *layout_q_ptr = const_cast<char *>(layout_q.c_str());
    char *layout_k_ptr = const_cast<char *>(layout_k.c_str());

    ACLNN_CMD(aclnnDenseLightningIndexerKLLossGrad, q, k, w, attn_softmax_l1_norm, softmax_lse, cu_seqlens_q,
              cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, metadata, layout_q_ptr, layout_k_ptr, mask_mode,
              cmp_ratio, dq, dk, dw, softmax_out);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(dq, dk, dw, softmax_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dense_lightning_indexer_kl_loss_grad_metadata", &dense_lightning_indexer_kl_loss_grad_metadata,
          "dense_lightning_indexer_kl_loss_grad_metadata");
    m.def("dense_lightning_indexer_kl_loss_grad", &dense_lightning_indexer_kl_loss_grad,
          "dense_lightning_indexer_kl_loss_grad");
}

} // namespace op_api
