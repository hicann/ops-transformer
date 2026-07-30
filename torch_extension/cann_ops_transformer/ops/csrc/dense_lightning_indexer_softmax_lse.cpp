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
#include "aclnn_common.h"

namespace op_api {
constexpr int64_t DLI_METADATA_SIZE = 64;

at::Device get_dense_lightning_indexer_softmax_lse_v2_metadata_device(const c10::optional<at::Tensor> &cu_seqlens_q,
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

at::Tensor dense_lightning_indexer_softmax_lse_v2_metadata(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k, const c10::optional<at::Tensor> &cmp_residual_k, int64_t batch_size,
    int64_t max_seqlen_q, int64_t max_seqlen_k, std::string layout_q, std::string layout_k, int64_t mask_mode,
    int64_t cmp_ratio)
{
    TORCH_CHECK((num_heads_q > 0), "The num_heads_q should be greater than 0, current is: ", num_heads_q);
    TORCH_CHECK((num_heads_k > 0), "The num_heads_k should be greater than 0, current is: ", num_heads_k);
    TORCH_CHECK((head_dim == 128), "head_dim only supports 128, current is: ", head_dim);
    TORCH_CHECK((num_heads_q >= 1 && num_heads_q <= 128),
                "num_heads_q should be in [1, 128], current is: ", num_heads_q);
    TORCH_CHECK((num_heads_k == 1), "num_heads_k only supports 1, current is: ", num_heads_k);
    TORCH_CHECK((num_heads_q % num_heads_k == 0),
                "num_heads_q should be divisible by num_heads_k, but got nq=", num_heads_q, " nk=", num_heads_k);
    TORCH_CHECK((cmp_ratio >= 1 && cmp_ratio <= 128), "cmp_ratio should be in [1, 128], current is: ", cmp_ratio);
    TORCH_CHECK((mask_mode == 0 || mask_mode == 3), "mask_mode only supports 0 or 3, current is: ", mask_mode);
    TORCH_CHECK((layout_q == layout_k), "layout_q and layout_k must be the same, but got layout_q=", layout_q,
                " layout_k=", layout_k);
    TORCH_CHECK(
        !((mask_mode == 3 && cmp_ratio > 1) && (!cmp_residual_k.has_value() || !cmp_residual_k.value().defined())),
        "cmp_residual_k must be provided when mask_mode=3 and cmp_ratio>1.");

    at::Tensor output{nullptr};
    {
        auto local_device = get_dense_lightning_indexer_softmax_lse_v2_metadata_device(
            cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k);
        const c10::OptionalDeviceGuard device_guard(local_device);
        output = at::empty({DLI_METADATA_SIZE}, at::TensorOptions().dtype(at::kInt).device(local_device));
    }

    char *layout_q_ptr = const_cast<char *>(layout_q.c_str());
    char *layout_k_ptr = const_cast<char *>(layout_k.c_str());

    ACLNN_CMD(aclnnDenseLightningIndexerSoftmaxLseV2Metadata, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
              cmp_residual_k, batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_k, head_dim, layout_q_ptr,
              layout_k_ptr, mask_mode, cmp_ratio, output);

    return output;
}

at::Tensor dense_lightning_indexer_softmax_lse_v2(
    const at::Tensor &query_index, const at::Tensor &key_index, const at::Tensor &weight,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmp_residual_k, const c10::optional<at::Tensor> &metadata, std::string layout_q,
    std::string layout_k, int64_t mask_mode, int64_t cmp_ratio)
{
    TORCH_CHECK((mask_mode == 0 || mask_mode == 3), "mask_mode only supports 0 or 3, current is: ", mask_mode);
    TORCH_CHECK((cmp_ratio >= 1 && cmp_ratio <= 128), "cmp_ratio should be in [1, 128], current is: ", cmp_ratio);

    at::Tensor softmax_lse_out{nullptr};
    {
        const c10::OptionalDeviceGuard device_guard(query_index.device());
        if (layout_q == "BSND") {
            int64_t bSize = query_index.size(0);
            int64_t s1Size = query_index.size(1);
            int64_t n2Size = key_index.size(2);
            softmax_lse_out = at::empty({bSize, n2Size, s1Size}, query_index.options().dtype(at::kFloat));
        } else {
            int64_t t1Size = query_index.size(0);
            int64_t n2Size = key_index.size(1);
            softmax_lse_out = at::empty({n2Size, t1Size}, query_index.options().dtype(at::kFloat));
        }
    }

    char *layout_q_ptr = const_cast<char *>(layout_q.c_str());
    char *layout_k_ptr = const_cast<char *>(layout_k.c_str());

    ACLNN_CMD(aclnnDenseLightningIndexerSoftmaxLseV2, query_index, key_index, weight, cu_seqlens_q, cu_seqlens_k,
              seqused_q, seqused_k, cmp_residual_k, metadata, layout_q_ptr, layout_k_ptr, mask_mode, cmp_ratio,
              softmax_lse_out);

    return softmax_lse_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dense_lightning_indexer_softmax_lse_v2_metadata", &dense_lightning_indexer_softmax_lse_v2_metadata,
          "dense_lightning_indexer_softmax_lse_v2_metadata");
    m.def("dense_lightning_indexer_softmax_lse_v2", &dense_lightning_indexer_softmax_lse_v2,
          "dense_lightning_indexer_softmax_lse_v2");
}

} // namespace op_api
