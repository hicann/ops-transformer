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
 * \file quant_compressor.cpp
 * \brief QuantCompressor operator implementation for PyTorch NPU extension
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

inline TensorWrapper make_wrapper(const at::Tensor &tensor, aclDataType tensorAcltype)
{
    return {tensor, tensorAcltype};
}

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

at::Tensor ConstructQuantCompressorOutputTensor(const at::Tensor &x, const at::Tensor &wkv,
                                                const c10::optional<at::Tensor> &cuSeqlens, int64_t cmpRatio,
                                                int64_t coff)
{
    auto xDim = x.dim();
    at::SmallVector<int64_t, MAX_DIM_SIZE> cmpKvSize;
    at::Tensor cmpKv;
    int64_t cmpS = 0;

    TORCH_CHECK(wkv.defined(), "Check wkv != nullptr failed");
    auto wkvDim = wkv.dim();
    TORCH_CHECK(wkvDim == DIM_TWO, "wkv dim num[", wkvDim, "] should be 2");

    auto D = wkv.size(0) / coff;
    if (xDim == DIM_THREE) {
        cmpS = (x.size(1) + cmpRatio - 1) / cmpRatio;
        cmpKvSize = {x.size(0), cmpS, D};
    } else {
        TORCH_CHECK(cuSeqlens.has_value(), "Check cu_seqlens != nullptr failed");
        int64_t bSize = cuSeqlens->size(0) - 1;
        cmpS = std::min(x.size(0), x.size(0) / cmpRatio + bSize);
        cmpKvSize = {cmpS, D};
    }

    cmpKv = at::empty(cmpKvSize, x.options().dtype(at::kBFloat16));
    return cmpKv;
}

at::Tensor QuantCompressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate, at::Tensor &stateCache,
                           const at::Tensor &ape, int64_t quantMode, int64_t cmpRatio,
                           const c10::optional<at::Tensor> &xDescale, const c10::optional<at::Tensor> &wkvDescale,
                           const c10::optional<at::Tensor> &wgateDescale,
                           const c10::optional<at::Tensor> &stateBlockTable, const c10::optional<at::Tensor> &cuSeqlens,
                           const c10::optional<at::Tensor> &seqused, const c10::optional<at::Tensor> &startPos,
                           int64_t coff, int64_t cacheMode)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto xDim = x.dim();
    TORCH_CHECK(xDim == DIM_TWO || xDim == DIM_THREE, "x dim num[", xDim, "] should be 2 or 3");

    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");
    TORCH_CHECK(coff > VALUE_0, "coff should be greater than 0");

    at::Tensor cmpKv = ConstructQuantCompressorOutputTensor(x, wkv, cuSeqlens, cmpRatio, coff);

    auto stateCacheDim = stateCache.dim();
    TORCH_CHECK(stateCacheDim == DIM_THREE, "state_cache dim num[", stateCacheDim, "] should be 3");

    auto contiguousAxesResult = IsContiguousAxes(stateCache);
    TORCH_CHECK(contiguousAxesResult[DIM_ONE] && contiguousAxesResult[DIM_TWO],
                "state_cache must be contiguous on all axes except axis 0");

    int64_t stateCacheStrideDim0 = stateCache.stride(0);

    TensorWrapper xWrapper = make_wrapper(x, ACL_HIFLOAT8);
    TensorWrapper wkvWrapper = make_wrapper(wkv, ACL_HIFLOAT8);
    TensorWrapper wgateWrapper = make_wrapper(wgate, ACL_HIFLOAT8);

    ACLNN_CMD(aclnnQuantCompressor, xWrapper, wkvWrapper, wgateWrapper, stateCache, ape, xDescale, wkvDescale,
              wgateDescale, stateBlockTable, cuSeqlens, seqused, startPos, quantMode, cmpRatio, coff, cacheMode,
              stateCacheStrideDim0, cmpKv);

    return cmpKv;
}

at::Tensor QuantCompressorMeta(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                               at::Tensor &stateCache, const at::Tensor &ape, int64_t quantMode, int64_t cmpRatio,
                               const c10::optional<at::Tensor> &xDescale, const c10::optional<at::Tensor> &wkvDescale,
                               const c10::optional<at::Tensor> &wgateDescale,
                               const c10::optional<at::Tensor> &stateBlockTable,
                               const c10::optional<at::Tensor> &cuSeqlens, const c10::optional<at::Tensor> &seqused,
                               const c10::optional<at::Tensor> &startPos, int64_t coff, int64_t cacheMode)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto xDim = x.dim();
    TORCH_CHECK(xDim == DIM_TWO || xDim == DIM_THREE, "x dim num[", xDim, "] should be 2 or 3");

    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");
    TORCH_CHECK(coff > VALUE_0, "coff should be greater than 0");

    return ConstructQuantCompressorOutputTensor(x, wkv, cuSeqlens, cmpRatio, coff);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("quant_compressor", &QuantCompressor, "quant_compressor"); }
} // namespace op_api
