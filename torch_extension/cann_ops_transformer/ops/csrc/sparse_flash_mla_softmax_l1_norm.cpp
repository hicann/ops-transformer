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
constexpr int64_t SMLA_METADATA_SIZE = 64;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;

at::Device get_sparse_flash_mla_softmax_l1_norm_metadata_device(const c10::optional<at::Tensor> &cu_seqlens_q,
                                                                const c10::optional<at::Tensor> &cu_seqlens_k,
                                                                const c10::optional<at::Tensor> &seqused_q,
                                                                const c10::optional<at::Tensor> &seqused_k,
                                                                const c10::optional<at::Tensor> &cmp_residual_k,
                                                                const c10::optional<at::Tensor> &topk_length)
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
    if (topk_length.has_value() && topk_length.value().defined()) {
        return topk_length.value().device();
    }
    return at::Device(torch_npu::utils::get_npu_device_type());
}

at::Tensor sparse_flash_mla_softmax_l1_norm_metadata(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k, const c10::optional<at::Tensor> &cmp_residual_k,
    const c10::optional<at::Tensor> &topk_length, int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t topk, std::string layout_q, std::string layout_k, int64_t mask_mode, int64_t cmp_ratio)
{
    TORCH_CHECK((num_heads_q > 0), "The num_heads_q should be greater than 0, current is: ", num_heads_q);
    TORCH_CHECK((num_heads_k > 0), "The num_heads_k should be greater than 0, current is: ", num_heads_k);
    TORCH_CHECK((head_dim > 0), "The head_dim should be greater than 0, current is: ", head_dim);
    TORCH_CHECK((num_heads_q % num_heads_k == 0),
                "num_heads_q should be divisible by num_heads_k, but got nq=", num_heads_q, " nk=", num_heads_k);
    TORCH_CHECK((cmp_ratio >= 1 && cmp_ratio <= 128), "cmp_ratio should be in [1, 128], current is: ", cmp_ratio);

    at::Tensor output{nullptr};
    {
        auto local_device = get_sparse_flash_mla_softmax_l1_norm_metadata_device(
            cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, topk_length);
        const c10::OptionalDeviceGuard device_guard(local_device);
        output = at::empty({SMLA_METADATA_SIZE}, at::TensorOptions().dtype(at::kInt).device(local_device));
    }

    char *layout_q_ptr = const_cast<char *>(layout_q.c_str());
    char *layout_k_ptr = const_cast<char *>(layout_k.c_str());

    ACLNN_CMD(aclnnSparseFlashMlaSoftmaxL1NormMetadata, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
              cmp_residual_k, topk_length, batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_k, head_dim,
              topk, cmp_ratio, mask_mode, layout_q_ptr, layout_k_ptr, output);

    return output;
}

at::Tensor sparse_flash_mla_softmax_l1_norm(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &lse,
    const c10::optional<at::Tensor> &sparse_indices, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k, const c10::optional<at::Tensor> &cmp_residual_k,
    const c10::optional<at::Tensor> &topk_length, const c10::optional<at::Tensor> &metadata, double scale_value,
    int64_t max_seqlen_k, int64_t cmp_ratio, int64_t mask_mode, std::string layout_q, std::string layout_k)
{
    TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query and key should have the same dtype.");
    TORCH_CHECK((query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16),
                "query dtype only support float16 or bfloat16, but got ", query.scalar_type());
    TORCH_CHECK(lse.scalar_type() == at::kFloat, "lse dtype only support float32, but got ", lse.scalar_type());
    TORCH_CHECK((cmp_ratio >= 1 && cmp_ratio <= 128), "cmp_ratio should be in [1, 128], current is: ", cmp_ratio);
    TORCH_CHECK((mask_mode == 0 || mask_mode == 3), "mask_mode only supports 0 or 3, current is: ", mask_mode);

    std::string layout_q_str = std::string(layout_q);
    std::string layout_k_str = std::string(layout_k);
    TORCH_CHECK(layout_q_str == "BSND" || layout_q_str == "TND", "layout_q only support BSND or TND, but got ",
                layout_q_str);
    TORCH_CHECK(layout_k_str == "BSND" || layout_k_str == "TND", "layout_k only support BSND or TND, but got ",
                layout_k_str);

    int64_t kv_head_num;
    int64_t s1;
    int64_t s2;
    if (layout_q_str == "BSND") {
        TORCH_CHECK(query.dim() == 4, "BSND layout, query dim must be 4, but got ", query.dim());
        TORCH_CHECK(key.dim() == 4, "BSND layout, key dim must be 4, but got ", key.dim());
        kv_head_num = key.size(DIM_2);
        s1 = query.size(DIM_1);
        s2 = key.size(DIM_1);
    } else {
        TORCH_CHECK(query.dim() == 3, "TND layout, query dim must be 3, but got ", query.dim());
        TORCH_CHECK(key.dim() == 3, "TND layout, key dim must be 3, but got ", key.dim());
        kv_head_num = key.size(DIM_1);
        s1 = query.size(DIM_0);
        s2 = key.size(DIM_0);
    }

    at::Tensor softmax_l1_norm{nullptr};
    {
        auto local_device = c10::Device(query.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        const at::Tensor &sparse_indices_value = sparse_indices.value_or(at::Tensor());
        if (sparse_indices_value.defined()) {
            softmax_l1_norm = at::empty(sparse_indices_value.sizes(), query.options().dtype(at::kFloat));
        } else {
            if (layout_q_str == "BSND") {
                softmax_l1_norm = at::empty({query.size(DIM_0), s1, 1, s2}, query.options().dtype(at::kFloat));
            } else {
                softmax_l1_norm = at::empty({s1, 1, max_seqlen_k}, query.options().dtype(at::kFloat));
            }
        }
    }

    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_k_ptr = const_cast<char *>(layout_k_str.c_str());

    ACLNN_CMD(aclnnSparseFlashMlaSoftmaxL1Norm, query, key, lse, sparse_indices, cu_seqlens_q, cu_seqlens_k, seqused_q,
              seqused_k, cmp_residual_k, topk_length, metadata, scale_value, max_seqlen_k, cmp_ratio, mask_mode,
              layout_q_ptr, layout_k_ptr, softmax_l1_norm);

    return softmax_l1_norm;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_flash_mla_softmax_l1_norm_metadata", &sparse_flash_mla_softmax_l1_norm_metadata,
          "sparse_flash_mla_softmax_l1_norm_metadata");
    m.def("sparse_flash_mla_softmax_l1_norm", &sparse_flash_mla_softmax_l1_norm, "sparse_flash_mla_softmax_l1_norm");
}

} // namespace op_api
