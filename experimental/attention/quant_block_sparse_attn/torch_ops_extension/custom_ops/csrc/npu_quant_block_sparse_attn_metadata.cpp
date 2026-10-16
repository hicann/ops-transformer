/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/library.h>
#include "ops_common.h"
#include "../../../../quant_block_sparse_attn_metadata/op_kernel_aicpu/quant_block_sparse_attn_metadata.h"

namespace custom {
using namespace at_npu::native;

uint64_t GetQbsaMetadataOutputSize(int64_t batchSize, int64_t numHeadsQ)
{
    // Keep eager and Meta output shapes identical on A5/A6; host isA5 still
    // selects the actual metadata layout within this capacity.
    return optiling::GetQbsaMetadataOutputCapacity(static_cast<uint64_t>(batchSize), static_cast<uint64_t>(numHeadsQ));
}

void CheckQbsaMetadataLayout(const at::Tensor& sparseSeqLen, const c10::optional<at::Tensor>& cuSeqlensQ,
                             const c10::optional<at::Tensor>& sequsedQ, int64_t batchSize, int64_t numHeadsQ,
                             int64_t quantMode, c10::string_view layoutQ, c10::string_view layoutKv)
{
    TORCH_CHECK(optiling::detail::IsSupportedQueryLayout(std::string(layoutQ).c_str(), quantMode),
                "metadata requires FP8 TND/NTD or MXFP8 TND/BSND/BNSD");
    TORCH_CHECK(layoutKv == "PA_BNBD", "metadata layout_kv must be PA_BNBD");
    TORCH_CHECK(batchSize > 0 && numHeadsQ > 0, "metadata batch_size and num_heads_q must be positive");
    TORCH_CHECK(sparseSeqLen.scalar_type() == at::kInt && (sparseSeqLen.dim() == 2 || sparseSeqLen.dim() == 3),
                "metadata sparse_seq_len must be 2D or 3D INT32");
    TORCH_CHECK(sparseSeqLen.size(0) == batchSize && (sparseSeqLen.dim() != 3 || sparseSeqLen.size(1) == numHeadsQ),
                "metadata sparse_seq_len batch/head dimensions must match attributes");
    if (layoutQ == "BSND" || layoutQ == "BNSD") {
        TORCH_CHECK(!cuSeqlensQ.has_value() && !sequsedQ.has_value(),
                    "BSND/BNSD metadata requires cu_seqlens_q=None and seqused_q=None");
        TORCH_CHECK(sparseSeqLen.dim() == 3 && sparseSeqLen.size(2) > 0,
                    "BSND/BNSD metadata requires sparse_seq_len [B,Nq,Qb] with Qb > 0");
    }
}

at::Tensor npu_quant_block_sparse_attn_metadata_npu(
    const at::Tensor& sparseSeqLen, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    const c10::optional<at::Tensor>& cuSeqlensQ, const c10::optional<at::Tensor>& cuSeqlensKv,
    const c10::optional<at::Tensor>& sequsedQ, const c10::optional<at::Tensor>& sequsedKv, int64_t batchSize,
    int64_t sparseBlockSizeQ, int64_t sparseBlockSizeK, int64_t quantMode, int64_t maskMode, c10::string_view layoutQ,
    c10::string_view layoutKv, c10::string_view layoutSparseIndices)
{
    CheckQbsaMetadataLayout(sparseSeqLen, cuSeqlensQ, sequsedQ, batchSize, numHeadsQ, quantMode, layoutQ, layoutKv);
    at::Device outputDevice = sparseSeqLen.device();
    uint64_t metadata_size = GetQbsaMetadataOutputSize(batchSize, numHeadsQ);
    at::Tensor output = torch::empty({metadata_size}, torch::dtype(torch::kInt32).device(outputDevice));

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    std::string layoutSparseIndicesStr = std::string(layoutSparseIndices);
    char* layoutQPtr = const_cast<char*>(layoutQStr.c_str());
    char* layoutKvPtr = const_cast<char*>(layoutKvStr.c_str());
    char* layoutSparseIndicesPtr = const_cast<char*>(layoutSparseIndicesStr.c_str());

    EXEC_NPU_CMD_V1(aclnnQuantBlockSparseAttnMetadata, sparseSeqLen, cuSeqlensQ, cuSeqlensKv, sequsedQ, sequsedKv,
                    batchSize, numHeadsQ, numHeadsKv, headDim, sparseBlockSizeQ, sparseBlockSizeK, quantMode, maskMode,
                    layoutQPtr, layoutKvPtr, layoutSparseIndicesPtr, output);
    return output;
}

at::Tensor npu_quant_block_sparse_attn_metadata_meta(
    const at::Tensor& sparseSeqLen, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    const c10::optional<at::Tensor>& cuSeqlensQ, const c10::optional<at::Tensor>& cuSeqlensKv,
    const c10::optional<at::Tensor>& sequsedQ, const c10::optional<at::Tensor>& sequsedKv, int64_t batchSize,
    int64_t sparseBlockSizeQ, int64_t sparseBlockSizeK, int64_t quantMode, int64_t maskMode, c10::string_view layoutQ,
    c10::string_view layoutKv, c10::string_view layoutSparseIndices)
{
    CheckQbsaMetadataLayout(sparseSeqLen, cuSeqlensQ, sequsedQ, batchSize, numHeadsQ, quantMode, layoutQ, layoutKv);
    uint64_t metadata_size = GetQbsaMetadataOutputSize(batchSize, numHeadsQ);
    return torch::empty({metadata_size}, sparseSeqLen.options().dtype(torch::kInt32));
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_quant_block_sparse_attn_metadata", &custom::npu_quant_block_sparse_attn_metadata_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_quant_block_sparse_attn_metadata", &custom::npu_quant_block_sparse_attn_metadata_meta);
}
