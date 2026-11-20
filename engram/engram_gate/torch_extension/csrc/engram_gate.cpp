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
 * \file engram_gate.cpp
 * \brief ACLNN Wrapper for aclnnEngramGate (前向)
 */

#include <tuple>
#include <vector>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

namespace {
constexpr int64_t HIDDEN_STATES_DIM_NUM = 3;
constexpr int64_t KV_DIM_NUM = 2;
constexpr int64_t WEIGHT_DIM_NUM = 2;
constexpr int64_t SUPPORTED_HEADS = 4;
constexpr int64_t VECTOR_LANES = 64;

} // namespace

std::tuple<at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
           c10::optional<at::Tensor>>
EngramGate(const at::Tensor& hiddenStates, const at::Tensor& kv, const at::Tensor& weightFused, double eps,
           double clampValue, bool saveForBackward)
{
    // 仅校验维度 (推导输出 shape 所需); dtype/shape 链 (kv/weight 与 hidden_states 的一致性)
    // 由 tiling 侧统一校验, 连续性由当前封装校验。
    TORCH_CHECK(hiddenStates.dim() == HIDDEN_STATES_DIM_NUM,
                "hidden_states must have shape (num_tokens, num_heads, head_dim), but got ", hiddenStates.dim(), "-D.");
    TORCH_CHECK(kv.dim() == KV_DIM_NUM, "kv must have shape (num_tokens, (num_heads + 1) * head_dim), but got ",
                kv.dim(), "-D.");
    TORCH_CHECK(weightFused.dim() == WEIGHT_DIM_NUM, "weight_fused must have shape (num_heads, head_dim), but got ",
                weightFused.dim(), "-D.");

    const int64_t numTokens = hiddenStates.size(0);
    const int64_t hcMult = hiddenStates.size(1);
    const int64_t headDim = hiddenStates.size(2);
    TORCH_CHECK(hcMult == SUPPORTED_HEADS, "EngramGate requires H == ", SUPPORTED_HEADS, ", but got ", hcMult, ".");
    TORCH_CHECK(headDim > 0 && headDim % VECTOR_LANES == 0, "EngramGate requires D to be a positive multiple of ",
                VECTOR_LANES, ", but got ", headDim, ".");
    TORCH_CHECK(hiddenStates.is_contiguous() && kv.is_contiguous() && weightFused.is_contiguous(),
                "EngramGate requires contiguous tensors");
    auto output = at::empty(hiddenStates.sizes(), hiddenStates.options());
    // save_for_backward 映射到同名属性：false 时中间结果输出为空 Tensor，kernel 跳过写出。
    auto statsOptions = hiddenStates.options().dtype(at::kFloat);
    const std::vector<int64_t> statsShape =
        saveForBackward ? std::vector<int64_t>{numTokens, hcMult} : std::vector<int64_t>{0};
    auto dot = at::empty(statsShape, statsOptions);
    auto gateScore = at::empty(statsShape, statsOptions);
    auto rstdX = at::empty(statsShape, statsOptions);
    auto rstdK = at::empty(statsShape, statsOptions);
    ACLNN_CMD(aclnnEngramGate, hiddenStates, kv, weightFused, eps, clampValue, saveForBackward, output, dot, gateScore,
              rstdX, rstdK);
    if (!saveForBackward) {
        return {output, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt};
    }
    return {output, dot, gateScore, rstdX, rstdK};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("engram_gate", &EngramGate, "engram_gate");
}
} // namespace op_api
