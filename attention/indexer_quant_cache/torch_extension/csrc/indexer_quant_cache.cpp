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
constexpr int64_t MXFP8_QUANT_MODE = 0;
constexpr int64_t FP8_QUANT_MODE = 1;
constexpr int64_t HIFLOAT8_QUANT_MODE = 2;
constexpr int64_t MXFP4_QUANT_MODE = 3;

enum class TensorRole {
    CACHE,
    CACHE_SCALE
};

TensorWrapper MakeTensorWrapper(const at::Tensor &tensor, int64_t quantMode, TensorRole role)
{
    const auto tensorType = tensor.scalar_type();
    switch (quantMode) {
        case MXFP8_QUANT_MODE:
            if (role == TensorRole::CACHE) {
                TORCH_CHECK(tensorType == at::kByte || tensorType == at::kFloat8_e4m3fn,
                            "When quant_mode is mxfp8, cache must be torch.uint8 or torch.float8_e4m3fn, but got ",
                            tensorType);
                return {tensor, ACL_FLOAT8_E4M3FN};
            }
            TORCH_CHECK(tensorType == at::kByte || tensorType == at::kFloat8_e8m0fnu,
                        "When quant_mode is mxfp8, cache_scale must be torch.uint8 or torch.float8_e8m0fnu, but got ",
                        tensorType);
            return {tensor, ACL_FLOAT8_E8M0};
        case FP8_QUANT_MODE:
            if (role == TensorRole::CACHE) {
                TORCH_CHECK(
                    tensorType == at::kByte || tensorType == at::kFloat8_e4m3fn || tensorType == at::kFloat8_e5m2,
                    "When quant_mode is fp8, cache must be torch.uint8, torch.float8_e4m3fn or "
                    "torch.float8_e5m2, but got ",
                    tensorType);
                return {tensor, tensorType == at::kFloat8_e5m2 ? ACL_FLOAT8_E5M2 : ACL_FLOAT8_E4M3FN};
            }
            TORCH_CHECK(tensorType == at::kFloat, "When quant_mode is fp8, cache_scale must be torch.float32, but got ",
                        tensorType);
            return {tensor, ACL_FLOAT};
        case HIFLOAT8_QUANT_MODE:
            if (role == TensorRole::CACHE) {
                TORCH_CHECK(tensorType == at::kByte,
                            "When quant_mode is hifloat8, cache must be torch.uint8 carrying HiFloat8 data, but got ",
                            tensorType);
                return {tensor, ACL_UINT8};
            }
            TORCH_CHECK(tensorType == at::kFloat,
                        "When quant_mode is hifloat8, cache_scale must be torch.float32, but got ", tensorType);
            return {tensor, ACL_FLOAT};
        case MXFP4_QUANT_MODE:
            if (role == TensorRole::CACHE) {
                TORCH_CHECK(tensorType == at::kByte || tensorType == at::kFloat4_e2m1fn_x2,
                            "When quant_mode is mxfp4, cache must be packed torch.uint8 or "
                            "torch.float4_e2m1fn_x2, but got ",
                            tensorType);
                return {tensor, ACL_FLOAT4_E2M1};
            }
            TORCH_CHECK(tensorType == at::kByte || tensorType == at::kFloat8_e8m0fnu,
                        "When quant_mode is mxfp4, cache_scale must be torch.uint8 or torch.float8_e8m0fnu, but got ",
                        tensorType);
            return {tensor, ACL_FLOAT8_E8M0};
        default:
            TORCH_CHECK(false, "quant_mode should be one of mxfp8, fp8, hifloat8 or mxfp4");
    }
    return {tensor, ACL_DT_UNDEFINED};
}
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
                "quant_mode should be one of mxfp8, fp8, hifloat8 or mxfp4");
    int64_t tailDim = x.size(x.dim() - 1);
    TORCH_CHECK(tailDim > 0 && tailDim % TAIL_DIM_ALIGN == 0,
                "The last dim (d) of x should be positive and 32-aligned, but got ", tailDim);

    auto localDevice = c10::Device(cache.device());
    const c10::OptionalDeviceGuard deviceGuard(localDevice);

    float xScaleF = static_cast<float>(xScale);
    auto cacheWrapped = MakeTensorWrapper(cache, quantMode, TensorRole::CACHE);
    auto cacheScaleWrapped = MakeTensorWrapper(cacheScale, quantMode, TensorRole::CACHE_SCALE);
    StorageShapeTensor xWrapped{x};
    StorageShapeTensor slotMappingWrapped{slotMapping};
    ACLNN_CMD(aclnnIndexerQuantCache, cacheWrapped, cacheScaleWrapped, xWrapped, slotMappingWrapped, quantMode,
              roundScale, xScaleF);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("indexer_quant_cache", &IndexerQuantCache, "indexer_quant_cache");
}
} // namespace op_api
