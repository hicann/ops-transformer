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
 * \file npu_turbo_quant_sparse_flash_attention.cpp
 * \brief TurboQuantSparseFlashAttention operator implementation for PyTorch NPU extension
 */

#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;
const int64_t DIM_0 = 0;
const int64_t DIM_1 = 1;
const int64_t DIM_2 = 2;
const int64_t KEY_DIM = 4; // PA_BSND: [blockNum, blockSize, N2, D]
const int64_t QUERY_DIM = 3;

// query is TND: [T, N, D] where D = nope(512) + rope(64).
// attention_out drops the rope tail: [T, N, D - rope_head_dim].
static at::Tensor ConstructAttentionOut(const at::Tensor &query, int64_t ropeHeadDim)
{
    auto queryShape = query.sizes();
    return at::empty({queryShape[DIM_0], queryShape[DIM_1], queryShape[DIM_2] - ropeHeadDim}, query.options());
}

// softmax_max/softmax_sum mirror kv_quant's TND shape so a DCP decode merge can consume them.
// Empty {0} when the caller does not ask for them.
static at::SmallVector<int64_t, 8> ConstructSoftmaxSize(const at::Tensor &query, const at::Tensor &key,
                                                        const std::string &layoutKv, bool returnSoftmaxLse)
{
    at::SmallVector<int64_t, 8> softmaxSize;
    if (returnSoftmaxLse) {
        // layout 与 key rank 已由 CheckInputsAndLayout 校验，此处可安全取 dim2。
        const int64_t kvHeadDim = key.size(DIM_2);
        TORCH_CHECK(kvHeadDim > 0, "kv head dim should be greater than 0");
        softmaxSize = {kvHeadDim, query.size(DIM_0), query.size(DIM_1) / kvHeadDim};
    } else {
        softmaxSize = {0};
    }
    return softmaxSize;
}

// Runtime and Meta implementations share these checks to keep validation and output inference aligned.
static void CheckInputsAndLayout(const at::Tensor &query, const at::Tensor &key, const std::string &layoutQuery,
                                 const std::string &layoutKv, int64_t ropeHeadDim)
{
    TORCH_CHECK(query.defined(), "Check query != nullptr failed");
    TORCH_CHECK(key.defined(), "Check key != nullptr failed");
    TORCH_CHECK(layoutQuery == "TND", "layout_query[", layoutQuery, "] is not supported, only TND");
    TORCH_CHECK(layoutKv == "PA_BSND", "layout_kv[", layoutKv, "] is not supported, only PA_BSND");
    TORCH_CHECK(query.dim() == QUERY_DIM, "query dim num[", query.dim(), "] should be 3 (TND)");
    TORCH_CHECK(key.dim() == KEY_DIM, "key dim num[", key.dim(), "] should be 4 (PA_BSND)");
    TORCH_CHECK(ropeHeadDim >= 0 && ropeHeadDim < query.size(DIM_2), "rope_head_dim[", ropeHeadDim,
                "] should be in [0, query last dim)");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_turbo_quant_sparse_flash_attention_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparseIndices,
    const c10::optional<at::Tensor> &keyDequantScale, const c10::optional<at::Tensor> &valueDequantScale,
    const at::Tensor &blockTable, const at::Tensor &actualSeqLengthsQuery, const at::Tensor &actualSeqLengthsKv,
    double scaleValue, int64_t keyQuantMode, int64_t valueQuantMode, int64_t sparseBlockSize,
    c10::string_view layoutQuery, c10::string_view layoutKv, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t attentionMode, int64_t quantScaleRepoMode, int64_t tileSize, int64_t ropeHeadDim, bool returnSoftmaxLse)
{
    std::string layoutQueryStr(layoutQuery);
    std::string layoutKvStr(layoutKv);
    CheckInputsAndLayout(query, key, layoutQueryStr, layoutKvStr, ropeHeadDim);
    char *layoutQueryPtr = const_cast<char *>(layoutQueryStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    at::Tensor attentionOut = ConstructAttentionOut(query, ropeHeadDim);
    auto softmaxSize = ConstructSoftmaxSize(query, key, layoutKvStr, returnSoftmaxLse);
    at::Tensor softmaxMax = at::empty(softmaxSize, query.options().dtype(at::kFloat));
    at::Tensor softmaxSum = at::empty(softmaxSize, query.options().dtype(at::kFloat));

    EXEC_NPU_CMD_V1(aclnnTurboQuantSparseFlashAttention, query, key, value, sparseIndices, keyDequantScale,
                    valueDequantScale, blockTable, actualSeqLengthsQuery, actualSeqLengthsKv, scaleValue, keyQuantMode,
                    valueQuantMode, sparseBlockSize, layoutQueryPtr, layoutKvPtr, sparseMode, preTokens, nextTokens,
                    attentionMode, quantScaleRepoMode, tileSize, ropeHeadDim, returnSoftmaxLse, attentionOut,
                    softmaxMax, softmaxSum);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(attentionOut, softmaxMax, softmaxSum);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_turbo_quant_sparse_flash_attention_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &, const at::Tensor &,
    const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, double, int64_t, int64_t, int64_t, c10::string_view layoutQuery, c10::string_view layoutKv,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t ropeHeadDim, bool returnSoftmaxLse)
{
    std::string layoutQueryStr(layoutQuery);
    std::string layoutKvStr(layoutKv);
    CheckInputsAndLayout(query, key, layoutQueryStr, layoutKvStr, ropeHeadDim);
    TORCH_CHECK(query.scalar_type() == at::kBFloat16, "query dtype is not supported, only bfloat16");
    at::Tensor attentionOut = ConstructAttentionOut(query, ropeHeadDim);
    auto softmaxSize = ConstructSoftmaxSize(query, key, layoutKvStr, returnSoftmaxLse);
    at::Tensor softmaxMax = at::empty(softmaxSize, query.options().dtype(at::kFloat));
    at::Tensor softmaxSum = at::empty(softmaxSize, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(attentionOut, softmaxMax, softmaxSum);
}

} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_turbo_quant_sparse_flash_attention", &custom::npu_turbo_quant_sparse_flash_attention_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_turbo_quant_sparse_flash_attention", &custom::npu_turbo_quant_sparse_flash_attention_meta);
}
