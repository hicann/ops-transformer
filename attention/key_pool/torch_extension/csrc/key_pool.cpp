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
 * \file key_pool.cpp
 * \brief KeyPool operator implementation for PyTorch NPU extension
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;
const int64_t VALUE_0 = 0;
const int64_t VALUE_1 = 1;

at::Tensor ConstructKeyPoolOutputTensor(const at::Tensor &hidden_states, const at::Tensor &wk,
                                        const at::Tensor &stateCache, const at::Tensor &cacheBlockTable,
                                        int64_t cmpRatio)
{
    auto hiddenStatesDim = hidden_states.dim();
    at::SmallVector<int64_t, MAX_DIM_SIZE> pooledKeySize;
    at::Tensor pooledKey;

    TORCH_CHECK(wk.defined(), "Check hidden_states != nullptr failed");
    auto wkDim = wk.dim();
    TORCH_CHECK(wkDim == DIM_TWO, "wk dim num[", wkDim, "] should be 2");

    TORCH_CHECK(hiddenStatesDim == DIM_TWO || hiddenStatesDim == DIM_THREE, "hidden_states dim num[", hiddenStatesDim,
                "] should be 2 or 3");
    TORCH_CHECK(stateCache.dim() == DIM_THREE, "state_cache dim num[", stateCache.dim(), "] should be 3");
    TORCH_CHECK(cacheBlockTable.dim() == DIM_TWO, "cache_block_table dim num[", cacheBlockTable.dim(), "] should be 2");
    int64_t pcap = (cacheBlockTable.size(1) * stateCache.size(1) + cmpRatio - 1) / cmpRatio;
    pooledKeySize = {cacheBlockTable.size(0), pcap, wk.size(0)};

    pooledKey = at::zeros(pooledKeySize, hidden_states.options().dtype(hidden_states.dtype()));
    return pooledKey;
}

at::Tensor KeyPool(const at::Tensor &hidden_states, const at::Tensor &wk, const at::Tensor &gate_weight,
                   const at::Tensor &ape, at::Tensor &stateCache, const at::Tensor &cacheBlockTable,
                   const at::Tensor &startPos, const c10::optional<at::Tensor> &normWeight,
                   const c10::optional<at::Tensor> &normBias, const c10::optional<at::Tensor> &cos,
                   const c10::optional<at::Tensor> &sin, const c10::optional<at::Tensor> &cuSeqlens,
                   const c10::optional<at::Tensor> &seqused, int64_t cmpRatio, double normEps, int64_t rotaryMode)
{
    TORCH_CHECK(hidden_states.defined(), "Check hidden_states != nullptr failed");
    auto hiddenStatesDim = hidden_states.dim();
    TORCH_CHECK(hiddenStatesDim == DIM_TWO || hiddenStatesDim == DIM_THREE, "hidden_states dim num[", hiddenStatesDim,
                "] should be 2 or 3");

    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");

    TORCH_CHECK(normWeight.has_value() == normBias.has_value(), "norm_weight and norm_bias must be provided as a pair");
    if (normWeight.has_value()) {
        TORCH_CHECK(normWeight->scalar_type() == at::kFloat && normBias->scalar_type() == at::kFloat,
                    "norm_weight and norm_bias must be FP32");
        TORCH_CHECK(normWeight->dim() == DIM_ONE && normBias->dim() == DIM_ONE,
                    "norm_weight and norm_bias must be rank-1");
        TORCH_CHECK(normWeight->size(0) == wk.size(0) && normBias->size(0) == wk.size(0),
                    "norm_weight and norm_bias size must match wk.size(0)");
    }
    TORCH_CHECK(normEps > 0.0, "norm_eps should be greater than 0");
    TORCH_CHECK(cos.has_value() == sin.has_value(), "cos and sin must be provided as a pair");
    TORCH_CHECK(!cos.has_value(), "KeyPool RoPE is not implemented in this stage");
    TORCH_CHECK(hiddenStatesDim != DIM_TWO || cuSeqlens.has_value(), "cu_seqlens is required for TH layout");
    TORCH_CHECK(hiddenStatesDim != DIM_THREE || !cuSeqlens.has_value(), "cu_seqlens must be absent for BSH layout");
    TORCH_CHECK(!seqused.has_value(), "seqused is reserved and must be None in this stage");
    at::Tensor pooledKey = ConstructKeyPoolOutputTensor(hidden_states, wk, stateCache, cacheBlockTable, cmpRatio);

    int64_t stateCacheStrideDim0 = stateCache.stride(0);

    if (cuSeqlens.has_value()) {
        TORCH_CHECK(cuSeqlens->scalar_type() == at::kInt, "cu_seqlens must be INT32");
        TORCH_CHECK(cuSeqlens->dim() == DIM_ONE, "cu_seqlens dim num[", cuSeqlens->dim(), "] should be 1");
        TORCH_CHECK(cuSeqlens->size(0) == cacheBlockTable.size(0) + VALUE_1, "cu_seqlens shape must be [B+1]");
    }
    ACLNN_CMD(aclnnKeyPool, hidden_states, wk, gate_weight, ape, stateCache, cacheBlockTable, startPos, normWeight,
              normBias, cos, sin, cuSeqlens, seqused, cmpRatio, normEps, rotaryMode, stateCacheStrideDim0, pooledKey);

    return pooledKey;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("key_pool", &KeyPool, "key_pool");
}
} // namespace op_api
