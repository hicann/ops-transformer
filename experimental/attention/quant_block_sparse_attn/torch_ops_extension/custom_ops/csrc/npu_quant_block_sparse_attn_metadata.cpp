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

namespace custom {
using namespace at_npu::native;

constexpr uint64_t QBSA_HEAD_METADATA_SIZE = 8;
constexpr uint64_t QBSA_AIC_CORE_NUM = 36;
constexpr uint64_t QBSA_AIV_CORE_NUM = 72;
constexpr uint64_t QBSA_METADATA_SIZE = 8;
constexpr uint64_t QBSA_FD_METADATA_SIZE = 8;

uint64_t GetQbsaMetadataOutputSize(int64_t batchSize, int64_t numHeadsQ)
{
    return QBSA_HEAD_METADATA_SIZE +
        static_cast<uint64_t>(batchSize) * static_cast<uint64_t>(numHeadsQ) *
        QBSA_AIC_CORE_NUM * QBSA_METADATA_SIZE +
        QBSA_AIV_CORE_NUM * QBSA_FD_METADATA_SIZE;
}

c10::optional<at::Tensor> get_qbsa_valid_tensor(
    const c10::optional<at::Tensor> &tensorOpt, const at::Device &device)
{
    if (tensorOpt.has_value() && tensorOpt.value().defined()) {
        return tensorOpt;
    }
    return c10::optional<at::Tensor>(torch::empty({0}, torch::dtype(torch::kInt32).device(device)));
}

at::Tensor npu_quant_block_sparse_attn_metadata_npu(
    const at::Tensor &sparseSeqLen, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensKv,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedKv,
    int64_t batchSize, int64_t sparseBlockSizeQ, int64_t sparseBlockSizeK, int64_t quantMode, int64_t maskMode,
    c10::string_view layoutQ, c10::string_view layoutKv, c10::string_view layoutSparseIndices)
{
    at::Device outputDevice = sparseSeqLen.device();
    uint64_t metadata_size = GetQbsaMetadataOutputSize(batchSize, numHeadsQ);
    at::Tensor output = torch::empty({metadata_size}, torch::dtype(torch::kInt32).device(outputDevice));

    auto cuSeqlensQValue = get_qbsa_valid_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensKvValue = get_qbsa_valid_tensor(cuSeqlensKv, outputDevice);
    auto sequsedQValue = get_qbsa_valid_tensor(sequsedQ, outputDevice);
    auto sequsedKvValue = get_qbsa_valid_tensor(sequsedKv, outputDevice);

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    std::string layoutSparseIndicesStr = std::string(layoutSparseIndices);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());
    char *layoutSparseIndicesPtr = const_cast<char *>(layoutSparseIndicesStr.c_str());

    EXEC_NPU_CMD_V1(aclnnQuantBlockSparseAttnMetadata,
        sparseSeqLen, cuSeqlensQValue, cuSeqlensKvValue, sequsedQValue, sequsedKvValue,
        batchSize, numHeadsQ, numHeadsKv, headDim, sparseBlockSizeQ, sparseBlockSizeK, quantMode, maskMode,
        layoutQPtr, layoutKvPtr, layoutSparseIndicesPtr, output);
    return output;
}

at::Tensor npu_quant_block_sparse_attn_metadata_meta(
    const at::Tensor &sparseSeqLen, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensKv,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedKv,
    int64_t batchSize, int64_t sparseBlockSizeQ, int64_t sparseBlockSizeK, int64_t quantMode, int64_t maskMode,
    c10::string_view layoutQ, c10::string_view layoutKv, c10::string_view layoutSparseIndices)
{
    uint64_t metadata_size = GetQbsaMetadataOutputSize(batchSize, numHeadsQ);
    return torch::empty({metadata_size}, sparseSeqLen.options().dtype(torch::kInt32));
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_quant_block_sparse_attn_metadata", &custom::npu_quant_block_sparse_attn_metadata_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_quant_block_sparse_attn_metadata", &custom::npu_quant_block_sparse_attn_metadata_meta);
}
