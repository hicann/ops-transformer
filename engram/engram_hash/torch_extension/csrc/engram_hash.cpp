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
 * \file engram_hash.cpp
 * \brief ACLNN Wrapper for aclnnEngramHash
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

namespace {
constexpr int64_t IDS_DIM_NUM = 2;
constexpr int64_t VOCAB_DIM_NUM = 3;
} // namespace

at::Tensor EngramHash(const at::Tensor& ngramTokenIds, const at::Tensor& multipliers, const at::Tensor& vocabSizes,
                      const at::Tensor& offsets)
{
    // 仅校验维度 (推导输出 shape 所需); dtype/shape 链/1024 线程约束等由 tiling 侧统一校验,
    // 连续性由框架 AutoContiguous 处理。
    TORCH_CHECK(ngramTokenIds.dim() == IDS_DIM_NUM,
                "ngram_token_ids must have shape (num_tokens, max_ngram_size), but got ", ngramTokenIds.dim(), "-D.");
    TORCH_CHECK(vocabSizes.dim() == VOCAB_DIM_NUM,
                "vocab_sizes must have shape (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), "
                "but got ",
                vocabSizes.dim(), "-D.");

    const int64_t numTokens = ngramTokenIds.size(0);
    const int64_t numNgramLayers = vocabSizes.size(0);
    const int64_t numOutCols = (ngramTokenIds.size(1) - 1) * vocabSizes.size(2);

    auto out = at::empty({numNgramLayers, numTokens, numOutCols}, ngramTokenIds.options().dtype(at::kInt));
    ACLNN_CMD(aclnnEngramHash, ngramTokenIds, multipliers, vocabSizes, offsets, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("engram_hash", &EngramHash, "engram_hash");
}
} // namespace op_api
