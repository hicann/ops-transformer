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
 * \file compressor_v2.cpp
 * \brief Compressor V2 operator implementation for PyTorch NPU extension
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;
const int64_t VALUE_0 = 0;

at::Tensor ConstructCompressorOutputTensor(const at::Tensor &x, const at::Tensor &wkv,
                                           const c10::optional<at::Tensor> &cuSeqlens, int64_t cmpRatio)
{
    auto xDim = x.dim();
    at::SmallVector<int64_t, MAX_DIM_SIZE> cmpKvSize;
    at::Tensor cmpKv;
    int64_t cmpS = 0;

    TORCH_CHECK(wkv.defined(), "Check wkv != nullptr failed");
    auto wkvDim = wkv.dim();
    TORCH_CHECK(wkvDim == DIM_TWO, "wkv dim num[", wkvDim, "] should be 2");

    if (xDim == DIM_THREE) {
        cmpS = (x.size(1) + cmpRatio - 1) / cmpRatio;
        cmpKvSize = {x.size(0), cmpS, wkv.size(0)};
    } else {
        TORCH_CHECK(cuSeqlens.has_value(), "Check cu_seqlens != nullptr failed");
        cmpS = std::min(x.size(0), x.size(0) / cmpRatio + cuSeqlens->size(0) - 1);
        cmpKvSize = {cmpS, wkv.size(0)};
    }

    cmpKv = at::empty(cmpKvSize, x.options().dtype(x.dtype()));
    return cmpKv;
}

at::Tensor Compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate, at::Tensor &stateCache,
                      int64_t cmpRatio, const c10::optional<at::Tensor> &stateBlockTable,
                      const c10::optional<at::Tensor> &cuSeqlens, const c10::optional<at::Tensor> &seqused,
                      const c10::optional<at::Tensor> &startPos)
{
    TORCH_CHECK(x.defined(), "Check x != nullptr failed");
    auto xDim = x.dim();
    TORCH_CHECK(xDim == DIM_TWO || xDim == DIM_THREE, "x dim num[", xDim, "] should be 2 or 3");
    TORCH_CHECK(cmpRatio > VALUE_0, "cmp_ratio should be greater than 0");

    auto cmpKv = ConstructCompressorOutputTensor(x, wkv, cuSeqlens, cmpRatio);
    auto stateCacheDim = stateCache.dim();
    TORCH_CHECK(stateCacheDim == DIM_THREE, "state_cache dim num[", stateCacheDim, "] should be 3");

    int64_t stateCacheStrideDim0 = stateCache.stride(0);
    ACLNN_CMD(aclnnCompressorV2, x, wkv, wgate, stateCache, stateBlockTable, cuSeqlens, seqused, startPos, cmpRatio,
              stateCacheStrideDim0, cmpKv);

    return cmpKv;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compressor", &Compressor, "compressor");
}
} // namespace op_api
