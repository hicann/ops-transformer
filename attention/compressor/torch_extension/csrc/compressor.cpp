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
 * \file compressor.cpp
 * \brief Compressor operator implementation for PyTorch NPU extension
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
const int64_t VALUE_2 = 2;

std::vector<bool> IsContiguousAxes(const at::Tensor &tensor)
{
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t ndim = sizes.size();
    if (ndim == 0) {
        return {};
    }
    std::vector<bool> result(ndim, false);

    std::vector<int64_t> contiguousStride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        contiguousStride[i] = contiguousStride[i + 1] * sizes[i + 1];
    }

    for (int64_t i = 0; i < ndim; i++) {
        result[i] = (strides[i] == contiguousStride[i]);
    }
    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ConstructCompressorOutputTensor(
    const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &ape, const c10::optional<at::Tensor> &cuSeqlens,
    int64_t cmpRatio, int64_t coff)
{
    auto xDim = x.dim();
    at::SmallVector<int64_t, MAX_DIM_SIZE> cmpKvSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> softmaxScoreSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> kvSize;
    at::Tensor cmpKv;
    at::Tensor softmaxScore;
    at::Tensor kv;
    int64_t cmpS = 0;
    int64_t bSize = 0;

    TORCH_CHECK(wkv.defined(), "Check x != nullptr failed");
    auto wkvDim = wkv.dim();
    TORCH_CHECK(wkvDim == DIM_TWO, "wkv dim num[", wkvDim, "] should be 2");

    TORCH_CHECK(coff == VALUE_1 || coff == VALUE_2, "coff value[", coff, "] should be 1 or 2");

    if (xDim == DIM_THREE) {
        cmpS = (x.size(1) + cmpRatio - 1) / cmpRatio;
        cmpKvSize = {x.size(0), cmpS, wkv.size(0) / coff};
        softmaxScoreSize = {x.size(0), cmpS, coff * cmpRatio, wkv.size(0) / coff};
        kvSize = {x.size(0), cmpS, coff * cmpRatio, wkv.size(0) / coff};
    } else {
        TORCH_CHECK(cuSeqlens.has_value(), "Check cu_seqlens != nullptr failed");
        bSize = cuSeqlens->size(0) - 1;
        cmpS = std::min(x.size(0), x.size(0) / cmpRatio + bSize);
        cmpKvSize = {cmpS, wkv.size(0) / coff};
        softmaxScoreSize = {cmpS, coff * cmpRatio, wkv.size(0) / coff};
        kvSize = {cmpS, coff * cmpRatio, wkv.size(0) / coff};
    }

    cmpKv = at::empty(cmpKvSize, x.options().dtype(x.dtype()));
    softmaxScore = at::empty(softmaxScoreSize, ape.options().dtype(ape.dtype()));
    kv = at::empty(kvSize, ape.options().dtype(ape.dtype()));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(cmpKv, softmaxScore, kv);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> Compressor(
    const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate, at::Tensor &stateCache, const at::Tensor &ape,
    int64_t cmpRatio, const c10::optional<at::Tensor> &stateBlockTable, const c10::optional<at::Tensor> &cuSeqlens,
    const c10::optional<at::Tensor> &seqused, const c10::optional<at::Tensor> &startPos, int64_t coff,
    int64_t cacheMode, bool gradEnabled)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto xDim = x.dim();
    TORCH_CHECK(xDim == DIM_TWO || xDim == DIM_THREE, "x dim num[", xDim, "] should be 2 or 3");

    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");
    auto [cmpKv, softmaxScore, kv] = ConstructCompressorOutputTensor(x, wkv, ape, cuSeqlens, cmpRatio, coff);
    auto stateCacheDim = stateCache.dim();
    TORCH_CHECK(stateCacheDim == DIM_THREE, "state_cache dim num[", stateCacheDim, "] should be 3");

    auto contiguousAxesResult = IsContiguousAxes(stateCache);
    int64_t stateCacheStrideDim0 = stateCache.stride(0);

    ACLNN_CMD(aclnnCompressor, x, wkv, wgate, stateCache, ape, stateBlockTable, cuSeqlens, seqused, startPos, cmpRatio,
              coff, cacheMode, stateCacheStrideDim0, gradEnabled, cmpKv, softmaxScore, kv);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(cmpKv, softmaxScore, kv);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> CompressorBackward(
    at::Tensor &dCmpKv, const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
    const at::Tensor &softmaxScore, const at::Tensor &kv, const c10::optional<at::Tensor> &cuSeqlens,
    const c10::optional<at::Tensor> &seqused, const c10::optional<at::Tensor> &startPos, int64_t cmpRatio, int64_t coff)
{
    auto xDim = x.dim();
    at::SmallVector<int64_t, MAX_DIM_SIZE> dxSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> dWkvSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> dWgateSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> dApeSize;
    at::Tensor dx;
    at::Tensor dWkv;
    at::Tensor dWgate;
    at::Tensor dApe;

    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    TORCH_CHECK(xDim == DIM_TWO || xDim == DIM_THREE, "x dim num[", xDim, "] should be 2 or 3");

    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");

    dxSize = x.sizes().vec();
    dx = at::empty(dxSize, x.options().dtype(x.dtype()));
    dWkvSize = wkv.sizes().vec();
    dWkv = at::empty(dWkvSize, wkv.options().dtype(wkv.dtype()));
    dWgateSize = wgate.sizes().vec();
    dWgate = at::empty(dWgateSize, wgate.options().dtype(wgate.dtype()));
    dApeSize = {cmpRatio, wkv.size(0)};
    dApe = at::empty(dApeSize, kv.options().dtype(kv.dtype()));

    ACLNN_CMD(aclnnCompressorGrad, x, wkv, wgate, dCmpKv, softmaxScore, kv, cuSeqlens, seqused, startPos, cmpRatio,
              coff, dx, dWkv, dWgate, dApe);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(dx, dWkv, dWgate, dApe);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> CompressorMeta(
    const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate, at::Tensor &stateCache, const at::Tensor &ape,
    int64_t cmpRatio, const c10::optional<at::Tensor> &stateBlockTable, const c10::optional<at::Tensor> &cuSeqlens,
    const c10::optional<at::Tensor> &seqused, const c10::optional<at::Tensor> &startPos, int64_t coff,
    int64_t cacheMode)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto xDim = x.dim();
    TORCH_CHECK(xDim == DIM_TWO || xDim == DIM_THREE, "x dim num[", xDim, "] should be 2 or 3");

    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");

    auto [cmpKv, softmaxScore, kv] = ConstructCompressorOutputTensor(x, wkv, ape, cuSeqlens, cmpRatio, coff);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(cmpKv, softmaxScore, kv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compressor", &Compressor, "compressor");
    m.def("compressor_backward", &CompressorBackward, "compressor_backward");
}
} // namespace op_api
