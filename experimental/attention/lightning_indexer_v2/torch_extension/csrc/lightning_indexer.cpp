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
 * \file lightning_indexer.cpp
 * \brief
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

constexpr int64_t LI_V2_METADATA_SIZE = 1024;

at::Tensor LightningIndexerMetadata(int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
                                    const c10::optional<at::Tensor> &cuSeqlensQ,
                                    const c10::optional<at::Tensor> &cuSeqlensK,
                                    const c10::optional<at::Tensor> &sequsedQ,
                                    const c10::optional<at::Tensor> &sequsedK,
                                    const c10::optional<at::Tensor> &cmpResidualK, int64_t batchSize,
                                    int64_t maxSeqlenQ, int64_t maxSeqlenK, c10::string_view layoutQ,
                                    c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio)
{
    at::Device outputDevice = at::Device(std::string("npu"));
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else if (cuSeqlensK.has_value()) {
        outputDevice = cuSeqlensK.value().device();
    } else if (sequsedQ.has_value()) {
        outputDevice = sequsedQ.value().device();
    } else if (sequsedK.has_value()) {
        outputDevice = sequsedK.value().device();
    } else if (cmpResidualK.has_value()) {
        outputDevice = cmpResidualK.value().device();
    }

    at::Tensor output = torch::empty({LI_V2_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = get_valid_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensKVal = get_valid_tensor(cuSeqlensK, outputDevice);
    auto sequsedQVal = get_valid_tensor(sequsedQ, outputDevice);
    auto sequsedKVal = get_valid_tensor(sequsedK, outputDevice);
    auto cmpResidualKVal = get_valid_tensor(cmpResidualK, outputDevice);

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKStr = std::string(layoutK);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKPtr = const_cast<char *>(layoutKStr.c_str());

    ACLNN_CMD(aclnnLightningIndexerV2Metadata, cuSeqlensQVal, cuSeqlensKVal, sequsedQVal, sequsedKVal, cmpResidualKVal,
              numHeadsQ, numHeadsK, headDim, topk, batchSize, maxSeqlenQ, maxSeqlenK, layoutQPtr, layoutKPtr, maskMode,
              cmpRatio, output);
    return output;
}

// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor> ConstructLightningIndexerOutputTensor(const at::Tensor &query, const at::Tensor &key,
                                                                         int64_t topk, std::string queryLayoutStr,
                                                                         std::string keyLayoutStr, bool returnValue)
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
                    "All values within query's shape should be greater "
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
    at::Tensor sparseValuesOut;
    if (returnValue) {
        sparseValuesOut = at::empty(outputSize, query.options().dtype(at::kFloat));
    } else {
        sparseValuesOut = at::empty({0}, query.options().dtype(at::kFloat));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

// candidate (two-level topk) 输出构造：source on 时 candidate_topk_indices 为
// [B,S1,N2,candBlocks]/[T,N2,candBlocks] int32；off 时 (0,) 占位；candidate_block_length 恒空 (0,)（预留）
std::tuple<at::Tensor, at::Tensor> ConstructCandidateOutputTensor(const at::Tensor &query, const at::Tensor &key,
                                                                  int64_t candTopkBlocks, std::string queryLayoutStr,
                                                                  std::string keyLayoutStr)
{
    at::Tensor candidateTopkIndicesOut;
    if (candTopkBlocks > 0) {
        at::SmallVector<int64_t, SIZE> candSize;
        int64_t keyHeadNum = (keyLayoutStr == "TND") ? key.size(DIM_1) : key.size(DIM_2);
        if (queryLayoutStr == "BSND") {
            candSize = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, candTopkBlocks};
        } else {
            int nDimIndex = (keyLayoutStr == "TND") ? DIM_1 : DIM_2;
            candSize = {query.size(DIM_0), key.size(nDimIndex), candTopkBlocks};
        }
        candidateTopkIndicesOut = at::empty(candSize, query.options().dtype(at::kInt));
    } else {
        candidateTopkIndicesOut = at::empty({0}, query.options().dtype(at::kInt));
    }
    at::Tensor candidateBlockLengthOut = at::empty({0}, query.options().dtype(at::kInt)); // 预留，恒空
    return std::tuple<at::Tensor, at::Tensor>(candidateTopkIndicesOut, candidateBlockLengthOut);
}

std::tuple<at::Tensor, at::Tensor> LightningIndexer(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &w, int64_t topk,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensK,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedK,
    const c10::optional<at::Tensor> &cmpResidualK, const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &outputIdxOffset, const c10::optional<at::Tensor> &metadata, int64_t maxSeqlenQ,
    c10::string_view layoutQ, c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio, int64_t returnValue)
{
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.")
    TORCH_CHECK(k.numel() > 0, "Tensor k is empty.")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> lightningIndexerOutput =
        ConstructLightningIndexerOutputTensor(q, k, topk, queryLayoutStr, keyLayoutStr, returnValue);
    at::Tensor sparseIndicesOut = std::get<0>(lightningIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(lightningIndexerOutput);
    // candidate (two-level topk)：旧入口固定 off（candidate_topk_blocks=-1），行为与现网一致；
    // 两个 candidate 输出为 (0,) 空 tensor 占位（预留接口）
    std::tuple<at::Tensor, at::Tensor> candidateOutput =
        ConstructCandidateOutputTensor(q, k, -1, queryLayoutStr, keyLayoutStr);
    at::Tensor candidateTopkIndicesOut = std::get<0>(candidateOutput);
    at::Tensor candidateBlockLengthOut = std::get<1>(candidateOutput);
    // convert str
    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());
    // candidate (two-level topk)：旧入口固定 off（-1/8 需为左值，ACLNN_CMD 的 DecodeDevice 按左值引用转发）
    int64_t candidateTopkBlocksOff = -1;
    int64_t candidateBlockSizeDefault = 8;

    ACLNN_CMD(aclnnLightningIndexerV2, q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable,
              outputIdxOffset, metadata, topk, maxSeqlenQ, queryLayoutPtr, keyLayoutPtr, maskMode, cmpRatio,
              returnValue, candidateTopkBlocksOff, candidateBlockSizeDefault, sparseIndicesOut, sparseValuesOut,
              candidateTopkIndicesOut, candidateBlockLengthOut);
    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

// candidate (two-level topk) source 模式入口：candidate_topk_blocks != -1 时生成候选块索引。
// 返回 (sparse_indices, sparse_values, candidate_topk_indices, candidate_block_length)；
// candidate_block_length 为预留接口恒空 tensor；return_value 固定 0（与 candidate 互斥）。
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> LightningIndexerCandidate(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &w, int64_t topk,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensK,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedK,
    const c10::optional<at::Tensor> &cmpResidualK, const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &outputIdxOffset, const c10::optional<at::Tensor> &metadata, int64_t maxSeqlenQ,
    c10::string_view layoutQ, c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio, int64_t candidateTopkBlocks,
    int64_t candidateBlockSize)
{
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.")
    TORCH_CHECK(k.numel() > 0, "Tensor k is empty.")
    TORCH_CHECK(candidateTopkBlocks == -1 ||
                    (candidateTopkBlocks > 0 && candidateTopkBlocks <= 2048 && candidateTopkBlocks % 64 == 0),
                "candidate_topk_blocks must be -1(off) or in (0, 2048] and a multiple of 64, but now is ",
                candidateTopkBlocks)
    TORCH_CHECK(
        candidateBlockSize >= 2 && candidateBlockSize <= 64 && (candidateBlockSize & (candidateBlockSize - 1)) == 0,
        "candidate_block_size must be a power of 2 in [2, 64], but now is ", candidateBlockSize)

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    std::tuple<at::Tensor, at::Tensor> lightningIndexerOutput =
        ConstructLightningIndexerOutputTensor(q, k, topk, queryLayoutStr, keyLayoutStr, false);
    at::Tensor sparseIndicesOut = std::get<0>(lightningIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(lightningIndexerOutput);
    std::tuple<at::Tensor, at::Tensor> candidateOutput =
        ConstructCandidateOutputTensor(q, k, candidateTopkBlocks, queryLayoutStr, keyLayoutStr);
    at::Tensor candidateTopkIndicesOut = std::get<0>(candidateOutput);
    at::Tensor candidateBlockLengthOut = std::get<1>(candidateOutput);
    // convert str
    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());
    int64_t returnValueOff = 0; // candidate 与 return_value 互斥，本入口固定 0（左值供 ACLNN_CMD 转发）

    ACLNN_CMD(aclnnLightningIndexerV2, q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable,
              outputIdxOffset, metadata, topk, maxSeqlenQ, queryLayoutPtr, keyLayoutPtr, maskMode, cmpRatio,
              returnValueOff, candidateTopkBlocks, candidateBlockSize, sparseIndicesOut, sparseValuesOut,
              candidateTopkIndicesOut, candidateBlockLengthOut);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut,
                                                                      candidateTopkIndicesOut, candidateBlockLengthOut);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lightning_indexer_metadata", &LightningIndexerMetadata, "lightning_indexer_metadata");
    m.def("lightning_indexer", &LightningIndexer, "lightning_indexer");
    m.def("lightning_indexer_candidate", &LightningIndexerCandidate, "lightning_indexer_candidate");
}
} // namespace op_api
