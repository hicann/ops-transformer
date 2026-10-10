/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file grouped_matmul_quant_torch.cpp
 *  \brief PyTorch registration for GroupedMatmulQuant.
 */

#include <torch/all.h>
#include <torch/library.h>

// aclnn_common.h is shared official code and intentionally remains unchanged.
// Its public declarations require the full Torch C++ API and std::string to be
// visible before inclusion when used by this standalone extension.
using std::string;
#include "torch_extension/cann_ops_transformer/common/aclnn_common.h"
#include "grouped_matmul_quant_torch_adpt.h"

namespace npu_ops_transformer_ext {
namespace {

at::Tensor GroupedMatmulQuantMeta(const at::Tensor& x, const at::Tensor& quantizedWeight, const at::Tensor& weightScale,
                                  const at::Tensor& weightOffset, const c10::optional<at::Tensor>& groupList,
                                  int64_t scaleGroupSize)
{
    static_cast<void>(weightScale);
    static_cast<void>(weightOffset);
    static_cast<void>(groupList);
    static_cast<void>(scaleGroupSize);
    constexpr int64_t FRACTAL_SIZE = 16;
    const int64_t m = x.size(0);
    const int64_t n = quantizedWeight.size(2) * FRACTAL_SIZE;
    return at::empty({m, n}, x.options());
}

} // namespace

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("grouped_matmul_quant", vllm_ascend::grouped_matmul_quant);
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("grouped_matmul_quant", TORCH_FN(GroupedMatmulQuantMeta));
}

} // namespace npu_ops_transformer_ext
