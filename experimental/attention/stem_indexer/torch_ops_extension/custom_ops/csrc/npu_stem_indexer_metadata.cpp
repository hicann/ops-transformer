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

constexpr int64_t STEM_INDEXER_METADATA_HEADER_SIZE = 16;
constexpr int64_t STEM_INDEXER_METADATA_CORE_STRIDE = 16;
constexpr int64_t STEM_INDEXER_METADATA_AIC_CORE_NUM = 36;
constexpr int64_t STEM_INDEXER_METADATA_AIV_CORE_NUM = 72;
constexpr int64_t STEM_INDEXER_METADATA_ALIGN_SIZE = 4096;

static int64_t CalcStemIndexerMetadataOutputSize(int64_t batchSize, int64_t kvHeads)
{
    TORCH_CHECK(batchSize > 0, "q_seq_lens must contain at least one element.");
    TORCH_CHECK(kvHeads > 0, "kv_heads must be greater than 0, but got ", kvHeads, ".");
    const int64_t maxSectionNum = batchSize * kvHeads;
    const int64_t metadataSize =
        STEM_INDEXER_METADATA_HEADER_SIZE +
        maxSectionNum * (STEM_INDEXER_METADATA_AIC_CORE_NUM + STEM_INDEXER_METADATA_AIV_CORE_NUM) *
            STEM_INDEXER_METADATA_CORE_STRIDE;
    return (metadataSize + STEM_INDEXER_METADATA_ALIGN_SIZE - 1) / STEM_INDEXER_METADATA_ALIGN_SIZE *
           STEM_INDEXER_METADATA_ALIGN_SIZE;
}

at::Tensor npu_stem_indexer_metadata_npu(const at::Tensor &qSeqLens, const at::Tensor &kvSeqLens, int64_t qHeads,
                                         int64_t kvHeads, bool causal, int64_t stemBlockSize, int64_t dimQkflat,
                                         int64_t windowSize)
{
    at::Device outputDevice = qSeqLens.device();
    int64_t metadataOutputSize = CalcStemIndexerMetadataOutputSize(qSeqLens.size(0), kvHeads);
    at::Tensor output = torch::empty({metadataOutputSize}, torch::dtype(torch::kInt32).device(outputDevice));

    // EXEC_NPU_CMD_V1 实参顺序 = 算子 IR 声明顺序（输入 -> 属性 -> 输出），与 schema 形参顺序不同
    EXEC_NPU_CMD_V1(aclnnStemIndexerMetadata, qSeqLens, kvSeqLens, qHeads, kvHeads, causal, stemBlockSize, dimQkflat,
                    windowSize, output);
    return output;
}

at::Tensor npu_stem_indexer_metadata_meta(const at::Tensor &qSeqLens, const at::Tensor &kvSeqLens, int64_t qHeads,
                                          int64_t kvHeads, bool causal, int64_t stemBlockSize, int64_t dimQkflat,
                                          int64_t windowSize)
{
    int64_t metadataOutputSize = CalcStemIndexerMetadataOutputSize(qSeqLens.size(0), kvHeads);
    return torch::empty({metadataOutputSize}, qSeqLens.options().dtype(torch::kInt32));
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_stem_indexer_metadata", &custom::npu_stem_indexer_metadata_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_stem_indexer_metadata", &custom::npu_stem_indexer_metadata_meta);
}
