/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer.cpp
 * \brief sparse_lightning_indexer torch 扩展（candidate consumer，设计 §5.2）
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

// 工具函数，推导输出 shape（与 LIV2 ConstructLightningIndexerOutputTensor 同构，returnValue 恒 false）
std::tuple<at::Tensor, at::Tensor> ConstructSparseLightningIndexerOutputTensor(const at::Tensor &query,
                                                                               const at::Tensor &key, int64_t topk,
                                                                               std::string queryLayoutStr,
                                                                               std::string keyLayoutStr)
{
    at::SmallVector<int64_t, SIZE> outputSize;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0,
                    "All values within key's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", key.size(i));
    }
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);
    int64_t keyHeadNum = (keyLayoutStr == "TND") ? key.size(DIM_1) : key.size(DIM_2);
    if (queryLayoutStr == "BSND") {
        outputSize = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, topk};
    } else {
        int nDimIndex = 0;
        nDimIndex = (keyLayoutStr == "TND") ? DIM_1 : DIM_2;
        outputSize = {query.size(DIM_0), key.size(nDimIndex), topk};
    }
    at::Tensor sparseIndicesOut = at::empty(outputSize, query.options().dtype(at::kInt));
    // return_value 固定 0（C6：leak NEG_HUGE 降级会污染泄漏槽 value），sparse_values 恒空占位
    at::Tensor sparseValuesOut = at::empty({0}, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

// candidate (two-level topk) consumer：消费 source 算子输出的候选块索引，输出 leak 降级 topk。
// 输入签名按 aclnn 原型序：inputs 0-12（candidate_topk_indices 必传、candidate_block_length 预留仅空）、
// outputs 0-1、attrs 0-7（return_value 固定 0，不暴露）。
std::tuple<at::Tensor, at::Tensor> SparseLightningIndexer(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &w, int64_t topk, const at::Tensor &candidateTopkIndices,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensK,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedK,
    const c10::optional<at::Tensor> &cmpResidualK, const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &outputIdxOffset, const c10::optional<at::Tensor> &metadata, int64_t maxSeqlenQ,
    c10::string_view layoutQ, c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio,
    const c10::optional<at::Tensor> &candidateBlockLength, int64_t candidateBlockSize)
{
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.")
    TORCH_CHECK(k.numel() > 0, "Tensor k is empty.")
    TORCH_CHECK(candidateTopkIndices.numel() > 0, "Tensor candidate_topk_indices is empty.")
    TORCH_CHECK(candidateTopkIndices.scalar_type() == at::kInt, "candidate_topk_indices must be int32, but now is ",
                candidateTopkIndices.scalar_type())
    // candBlocks 由输入 shape 末维推导（无属性源，N2）
    int64_t candBlocks = candidateTopkIndices.size(candidateTopkIndices.dim() - 1);
    TORCH_CHECK(candBlocks > 0 && candBlocks <= 2048 && candBlocks % 64 == 0,
                "The last dim of candidate_topk_indices must be in (0, 2048] and a multiple of 64, but now is ",
                candBlocks)
    TORCH_CHECK(
        candidateBlockSize >= 2 && candidateBlockSize <= 64 && (candidateBlockSize & (candidateBlockSize - 1)) == 0,
        "candidate_block_size must be a power of 2 in [2, 64], but now is ", candidateBlockSize)
    // C7：candidate_block_length 预留，仅接受 None/空 tensor
    if (candidateBlockLength.has_value() && candidateBlockLength.value().numel() != 0) {
        TORCH_CHECK(false, "candidate_block_length is reserved and only empty tensor is supported yet");
    }
    // topk ≤ 2048（C5）
    TORCH_CHECK(topk > 0 && topk <= 2048, "topk must be in (0, 2048], but now is ", topk)

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> sparseOutput =
        ConstructSparseLightningIndexerOutputTensor(q, k, topk, queryLayoutStr, keyLayoutStr);
    at::Tensor sparseIndicesOut = std::get<0>(sparseOutput);
    at::Tensor sparseValuesOut = std::get<1>(sparseOutput);
    // convert str
    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());
    int64_t returnValueOff = 0; // C6：固定 0（左值供 ACLNN_CMD 转发）

    ACLNN_CMD(aclnnSparseLightningIndexer, q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK,
              blockTable, outputIdxOffset, metadata, candidateTopkIndices, candidateBlockLength, topk, maxSeqlenQ,
              queryLayoutPtr, keyLayoutPtr, maskMode, cmpRatio, returnValueOff, candidateBlockSize, sparseIndicesOut,
              sparseValuesOut);
    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_lightning_indexer", &SparseLightningIndexer, "sparse_lightning_indexer");
}
} // namespace op_api
