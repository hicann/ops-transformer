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
 * \file npu_minimax_sparse_attention_split_kv.cpp
 * \brief MiniMaxSparseAttentionSplitKv PyTorch NPU extension
 */

#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <string>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;

void CheckInt32Tensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " dtype must be int32, got ", tensor.scalar_type());
}

static void CheckInputs(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                        const c10::optional<at::Tensor> &blockTable, const at::Tensor &k2qRowPtr,
                        const at::Tensor &k2qQIndices, const at::Tensor &k2qSlotIndices,
                        const at::Tensor &actualSeqLengths, const at::Tensor &actualSeqLengthsKv,
                        int64_t numKeyValueHeads, int64_t blockSize, int64_t topK, int64_t innerPrecise,
                        const std::string &inputLayout)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(value.numel() > 0, "Tensor value is empty.");
    TORCH_CHECK(innerPrecise == 0 || innerPrecise == 1 || innerPrecise == 4, "inner_precise must be 0, 1 or 4, got ",
                innerPrecise);
    TORCH_CHECK(inputLayout == "TND" || inputLayout == "BNSD" || inputLayout == "BSND",
                "input_layout must be TND, BNSD or BSND, got ", inputLayout);
    TORCH_CHECK(numKeyValueHeads > 0, "num_key_value_heads must be > 0, got ", numKeyValueHeads);
    TORCH_CHECK(blockSize > 0, "block_size must be > 0, got ", blockSize);
    TORCH_CHECK(topK > 0, "top_k must be > 0, got ", topK);
    if (blockTable.has_value() && blockTable.value().defined()) {
        CheckInt32Tensor(blockTable.value(), "block_table");
    }
    CheckInt32Tensor(k2qRowPtr, "k2q_row_ptr");
    CheckInt32Tensor(k2qQIndices, "k2q_q_indices");
    CheckInt32Tensor(k2qSlotIndices, "k2q_slot_indices");
    CheckInt32Tensor(actualSeqLengths, "actual_seq_lengths");
    CheckInt32Tensor(actualSeqLengthsKv, "actual_seq_lengths_kv");
    if (inputLayout == "TND") {
        TORCH_CHECK(query.dim() == DIM_THREE, "TND query must be rank 3 [T, N, D], got ", query.dim());
    } else {
        TORCH_CHECK(query.dim() == DIM_FOUR, inputLayout, " query must be rank 4, got ", query.dim());
    }
}

static c10::TensorOptions AttentionOutOptions(const at::Tensor &query)
{
    auto outOpts = query.options();
    // FP8 Q/K/V still writes BF16 attentionOut (see infershape / aclnn).
    if (query.scalar_type() == at::kFloat8_e4m3fn) {
        outOpts = outOpts.dtype(at::kBFloat16);
    }
    return outOpts;
}

static std::tuple<at::Tensor, at::Tensor> ConstructOutputs(const at::Tensor &query, bool softmaxLseFlag,
                                                           const std::string &inputLayout)
{
    at::Tensor attentionOut = at::empty(query.sizes(), AttentionOutOptions(query));
    at::Tensor softmaxLse;
    if (softmaxLseFlag) {
        if (inputLayout == "TND") {
            softmaxLse = at::empty({query.size(0), query.size(1), 1}, query.options().dtype(at::kFloat));
        } else {
            softmaxLse = at::empty({query.size(0), query.size(1), query.size(2), 1}, query.options().dtype(at::kFloat));
        }
    } else {
        softmaxLse = at::empty({0}, query.options().dtype(at::kFloat));
    }
    return std::tuple<at::Tensor, at::Tensor>(attentionOut, softmaxLse);
}

std::tuple<at::Tensor, at::Tensor> npu_minimax_sparse_attention_split_kv_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &blockTable, const at::Tensor &k2qRowPtr, const at::Tensor &k2qQIndices,
    const at::Tensor &k2qSlotIndices, const at::Tensor &actualSeqLengths, const at::Tensor &actualSeqLengthsKv,
    int64_t numKeyValueHeads, double scaleValue, int64_t blockSize, int64_t topK, int64_t innerPrecise,
    bool softmaxLseFlag, c10::string_view inputLayout)
{
    std::string inputLayoutStr(inputLayout);
    CheckInputs(query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualSeqLengths,
                actualSeqLengthsKv, numKeyValueHeads, blockSize, topK, innerPrecise, inputLayoutStr);

    at::Tensor attentionOut;
    at::Tensor softmaxLse;
    {
        const c10::OptionalDeviceGuard deviceGuard(c10::Device(query.device()));
        // Zero-init so BNSD/BSND padding tokens (kernel leaves them unwritten) stay 0.
        attentionOut = at::zeros(query.sizes(), AttentionOutOptions(query));
        if (softmaxLseFlag) {
            if (inputLayoutStr == "TND") {
                softmaxLse = at::zeros({query.size(0), query.size(1), 1}, query.options().dtype(at::kFloat));
            } else {
                softmaxLse =
                    at::zeros({query.size(0), query.size(1), query.size(2), 1}, query.options().dtype(at::kFloat));
            }
        } else {
            softmaxLse = at::empty({0}, query.options().dtype(at::kFloat));
        }
    }

    char *layoutPtr = const_cast<char *>(inputLayoutStr.c_str());
    EXEC_NPU_CMD_V1(aclnnMinimaxSparseAttentionSplitKv, query, key, value, blockTable, k2qRowPtr, k2qQIndices,
                    k2qSlotIndices, actualSeqLengths, actualSeqLengthsKv, numKeyValueHeads, scaleValue, blockSize, topK,
                    innerPrecise, softmaxLseFlag, layoutPtr, attentionOut, softmaxLse);
    return std::tuple<at::Tensor, at::Tensor>(attentionOut, softmaxLse);
}

std::tuple<at::Tensor, at::Tensor> npu_minimax_sparse_attention_split_kv_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &blockTable, const at::Tensor &k2qRowPtr, const at::Tensor &k2qQIndices,
    const at::Tensor &k2qSlotIndices, const at::Tensor &actualSeqLengths, const at::Tensor &actualSeqLengthsKv,
    int64_t numKeyValueHeads, double, int64_t blockSize, int64_t topK, int64_t innerPrecise, bool softmaxLseFlag,
    c10::string_view inputLayout)
{
    std::string inputLayoutStr(inputLayout);
    CheckInputs(query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualSeqLengths,
                actualSeqLengthsKv, numKeyValueHeads, blockSize, topK, innerPrecise, inputLayoutStr);
    return ConstructOutputs(query, softmaxLseFlag, inputLayoutStr);
}

} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_minimax_sparse_attention_split_kv", &custom::npu_minimax_sparse_attention_split_kv_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_minimax_sparse_attention_split_kv", &custom::npu_minimax_sparse_attention_split_kv_meta);
}
