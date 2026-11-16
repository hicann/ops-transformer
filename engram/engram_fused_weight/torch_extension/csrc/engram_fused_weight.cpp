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
 * \file engram_fused_weight.cpp
 * \brief ACLNN Wrapper for aclnnEngramFusedWeight
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

namespace {
constexpr int64_t INPUT_DIM_NUM = 2;
} // namespace

at::Tensor EngramFusedWeight(const at::Tensor& weightHidden, const at::Tensor& weightEmbed)
{
    // 仅校验维度; dtype/shape 一致性/256 对齐等由 tiling 侧统一校验,
    // 连续性由框架 AutoContiguous 处理。
    TORCH_CHECK(weightHidden.dim() == INPUT_DIM_NUM, "weight_hidden must be 2D [hcMult, hiddenSize], but got ",
                weightHidden.dim(), "-D.");
    TORCH_CHECK(weightEmbed.dim() == INPUT_DIM_NUM, "weight_embed must be 2D [hcMult, hiddenSize], but got ",
                weightEmbed.dim(), "-D.");

    auto out = at::empty(weightHidden.sizes(), weightHidden.options().dtype(at::kFloat));
    ACLNN_CMD(aclnnEngramFusedWeight, weightHidden, weightEmbed, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("engram_fused_weight", &EngramFusedWeight, "engram_fused_weight");
}
} // namespace op_api
