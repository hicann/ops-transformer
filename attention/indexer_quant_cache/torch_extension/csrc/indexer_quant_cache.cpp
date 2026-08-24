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
 * \file indexer_quant_cache.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

namespace {
constexpr int64_t TAIL_DIM_ALIGN = 32; // x 尾轴 d 必须 32 对齐
constexpr int64_t QUANT_MODE_MIN = 0;
constexpr int64_t QUANT_MODE_MAX = 3; // 0:MX-FP8 1:Normal 2:HiFloat8 3:MX-FP4
} // namespace

void IndexerQuantCache(at::Tensor &cache, at::Tensor &cacheScale, const at::Tensor &x, const at::Tensor &slotMapping,
                       int64_t quantMode, bool roundScale, double xScale)
{
    // 入参校验
    TORCH_CHECK(cache.numel() > 0, "Tensor cache is empty.");
    TORCH_CHECK(cacheScale.numel() > 0, "Tensor cache_scale is empty.");
    TORCH_CHECK(x.dim() >= 2, "x should be at least 2-dim, but got ", x.dim());
    TORCH_CHECK(slotMapping.dim() == x.dim() - 1, "slot_mapping dim should equal x dim - 1, but got slot_mapping dim ",
                slotMapping.dim(), " and x dim ", x.dim());
    TORCH_CHECK(quantMode >= QUANT_MODE_MIN && quantMode <= QUANT_MODE_MAX,
                "quant_mode should be 0, 1, 2 or 3, but got ", quantMode);
    int64_t tailDim = x.size(x.dim() - 1);
    TORCH_CHECK(tailDim > 0 && tailDim % TAIL_DIM_ALIGN == 0,
                "The last dim (d) of x should be positive and 32-aligned, but got ", tailDim);

    auto localDevice = c10::Device(cache.device());
    const c10::OptionalDeviceGuard deviceGuard(localDevice);

    float xScaleF = static_cast<float>(xScale);
    StorageShapeTensor xWrapped{x};
    StorageShapeTensor slotMappingWrapped{slotMapping};
    ACLNN_CMD(aclnnIndexerQuantCache, cache, cacheScale, xWrapped, slotMappingWrapped, quantMode, roundScale, xScaleF);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("indexer_quant_cache", &IndexerQuantCache, "indexer_quant_cache"); }
} // namespace op_api
