/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int64_t DIM_THREE = 3;
const int64_t DIM_FOUR = 4;
const int64_t QFLAT_B_DIM = 0;
const int64_t QFLAT_N1_DIM = 1;
const int64_t QFLAT_QB_DIM = 2;
const int64_t KFLAT_KB_DIM = 2;
const int64_t TOPK_SCORE_PRECISION_UINT32 = 1;
const int64_t TOPK_SCORE_PRECISION_UINT16 = 2;

// 工具函数，推导输出 sparse_indices / sparse_seq_len 的 shape 与 dtype
//   - sparse_indices: INT32 [B, q_heads, Qb, Kb]
//   - sparse_seq_len: INT32 [B, q_heads, Qb]
std::tuple<at::Tensor, at::Tensor> construct_stem_output_tensors(const at::Tensor &qflat, const at::Tensor &kflat)
{
    for (auto i = 0; i < qflat.sizes().size(); i++) {
        TORCH_CHECK(qflat.size(i) > 0, "All values within qflat's shape should be greater than 0, but shape[", i,
                    "] is ", qflat.size(i));
    }

    at::SmallVector<int64_t, SIZE> sparse_indices_size;
    at::SmallVector<int64_t, SIZE> sparse_seq_len_size;
    int64_t b_size = qflat.size(QFLAT_B_DIM);
    int64_t n1_size = qflat.size(QFLAT_N1_DIM);
    int64_t qb_size = qflat.size(QFLAT_QB_DIM);
    int64_t kb_size = kflat.size(KFLAT_KB_DIM);
    sparse_indices_size = {b_size, n1_size, qb_size, kb_size};
    sparse_seq_len_size = {b_size, n1_size, qb_size};

    at::Tensor sparse_indices = at::empty(sparse_indices_size, qflat.options().dtype(at::kInt));
    at::Tensor sparse_seq_len = at::empty(sparse_seq_len_size, qflat.options().dtype(at::kInt));
    return std::tuple<at::Tensor, at::Tensor>(sparse_indices, sparse_seq_len);
}

// step2, 为NPU设备实现前向接口（函数形参顺序 = schema 顺序）
std::tuple<at::Tensor, at::Tensor>
npu_stem_indexer_npu(const at::Tensor &qflat, const at::Tensor &kflat, const at::Tensor &vbias,
                     const at::Tensor &q_seq_lens, const at::Tensor &kv_seq_lens,
                     const c10::optional<at::Tensor> &num_prompt_tokens, const c10::optional<at::Tensor> &metadata,
                     bool causal, int64_t stem_block_size, int64_t stem_stride, double alpha, int64_t initial_blocks,
                     int64_t window_size, double k_block_num_rate_medium, int64_t k_block_num_bias_medium,
                     double k_block_num_rate_large, int64_t k_block_num_bias_large, int64_t topk_score_precision)
{
    TORCH_CHECK(qflat.numel() > 0, "Tensor qflat is empty.");
    TORCH_CHECK(qflat.dim() == DIM_FOUR, "qflat must be a 4D tensor with shape (B, N1, Qb, stem_stride * D), but got ",
                qflat.dim(), "D.");
    TORCH_CHECK(kflat.dim() == DIM_FOUR, "kflat must be a 4D tensor with shape (B, N2, Kb, stem_stride * D), but got ",
                kflat.dim(), "D.");
    TORCH_CHECK(vbias.dim() == DIM_THREE, "vbias must be a 3D tensor with shape (B, N2, Kb), but got ", vbias.dim(),
                "D.");
    TORCH_CHECK(qflat.scalar_type() == at::ScalarType::BFloat16, "qflat dtype must be bfloat16, but got ",
                qflat.scalar_type(), ".");
    TORCH_CHECK(kflat.scalar_type() == at::ScalarType::BFloat16, "kflat dtype must be bfloat16, but got ",
                kflat.scalar_type(), ".");
    TORCH_CHECK(vbias.scalar_type() == at::ScalarType::Float, "vbias dtype must be float32, but got ",
                vbias.scalar_type(), ".");
    TORCH_CHECK(kflat.size(KFLAT_KB_DIM) == vbias.size(KFLAT_KB_DIM),
                "The Kb dimensions of kflat and vbias must be equal.");
    TORCH_CHECK(qflat.size(qflat.dim() - 1) == kflat.size(kflat.dim() - 1),
                "The last dimensions of qflat and kflat must be equal.");
    TORCH_CHECK(topk_score_precision == TOPK_SCORE_PRECISION_UINT32 ||
                    topk_score_precision == TOPK_SCORE_PRECISION_UINT16,
                "topk_score_precision must be ", TOPK_SCORE_PRECISION_UINT32, "(uint32) or ",
                TOPK_SCORE_PRECISION_UINT16, "(uint16), but got ", topk_score_precision, ".");

    // construct the output tensors
    std::tuple<at::Tensor, at::Tensor> outputs = construct_stem_output_tensors(qflat, kflat);
    at::Tensor sparse_indices = std::get<0>(outputs);
    at::Tensor sparse_seq_len = std::get<1>(outputs);

    // EXEC_NPU_CMD_V1 实参顺序 = 算子 IR 声明顺序（输入 -> 属性 -> 输出），与 schema 形参顺序不同
    EXEC_NPU_CMD_V1(aclnnStemIndexer, qflat, kflat, vbias, q_seq_lens, kv_seq_lens, num_prompt_tokens, metadata,
                    causal, stem_block_size, stem_stride, alpha, initial_blocks, window_size,
                    k_block_num_rate_medium, k_block_num_bias_medium, k_block_num_rate_large, k_block_num_bias_large,
                    topk_score_precision, sparse_indices, sparse_seq_len);

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices, sparse_seq_len);
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor>
npu_stem_indexer_meta(const at::Tensor &qflat, const at::Tensor &kflat, const at::Tensor &vbias,
                      const at::Tensor &q_seq_lens, const at::Tensor &kv_seq_lens,
                      const c10::optional<at::Tensor> &num_prompt_tokens, const c10::optional<at::Tensor> &metadata,
                      bool causal, int64_t stem_block_size, int64_t stem_stride, double alpha, int64_t initial_blocks,
                      int64_t window_size, double k_block_num_rate_medium, int64_t k_block_num_bias_medium,
                      double k_block_num_rate_large, int64_t k_block_num_bias_large, int64_t topk_score_precision)
{
    TORCH_CHECK(topk_score_precision == TOPK_SCORE_PRECISION_UINT32 ||
                    topk_score_precision == TOPK_SCORE_PRECISION_UINT16,
                "topk_score_precision must be ", TOPK_SCORE_PRECISION_UINT32, "(uint32) or ",
                TOPK_SCORE_PRECISION_UINT16, "(uint16), but got ", topk_score_precision, ".");
    return construct_stem_output_tensors(qflat, kflat);
}
} // namespace custom

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_stem_indexer", &custom::npu_stem_indexer_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_stem_indexer", &custom::npu_stem_indexer_meta);
}
