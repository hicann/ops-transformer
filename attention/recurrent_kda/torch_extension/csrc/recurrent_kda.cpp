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
 * \file recurrent_kda.cpp
 * \brief Torch bridge for aclnnRecurrentKda.
 */

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor> RecurrentKda(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g, const at::Tensor &beta,
    at::Tensor &initialState, const c10::optional<at::Tensor> &cuSeqlens,
    const c10::optional<at::Tensor> &ssmStateIndices, const c10::optional<at::Tensor> &aLog,
    const c10::optional<at::Tensor> &dtBias, const c10::optional<at::Tensor> &numAcceptedTokens,
    const std::string &layout, c10::optional<double> scale, c10::optional<bool> outputFinalState,
    c10::optional<bool> inplaceFinalState, c10::optional<bool> useQkL2normInKernel, c10::optional<bool> useGateInKernel,
    c10::optional<bool> useBetaSigmoidInKernel, c10::optional<bool> allowNegEigval, c10::optional<bool> safeGate,
    c10::optional<double> lowerBound, c10::optional<bool> stateVFirst)
{
    TORCH_CHECK(layout == "BSND" || layout == "TND", "recurrent_kda: layout must be BSND or TND.");
    const bool isTnd = layout == "TND";
    TORCH_CHECK((isTnd && q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && g.dim() == 3 && beta.dim() == 2) ||
                    (!isTnd && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && g.dim() == 4 && beta.dim() == 3),
                "recurrent_kda: input ranks do not match layout.");
    TORCH_CHECK(q.sizes() == k.sizes(), "recurrent_kda: q and k must have identical shape.");
    TORCH_CHECK(
        q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
        "recurrent_kda: q/k/v currently support bfloat16 only.");
    TORCH_CHECK((g.scalar_type() == at::kFloat || g.scalar_type() == at::kBFloat16 || g.scalar_type() == at::kHalf) &&
                    (beta.scalar_type() == at::kFloat || beta.scalar_type() == at::kBFloat16 ||
                     beta.scalar_type() == at::kHalf),
                "recurrent_kda: g and beta must be float32, bfloat16 or float16.");

    const int64_t batch = isTnd ? 1 : q.size(0);
    const int64_t totalTokens = isTnd ? q.size(0) : q.size(0) * q.size(1);
    const int64_t denseSeqLen = isTnd ? q.size(0) : q.size(1);
    const int64_t h = isTnd ? q.size(1) : q.size(2);
    const int64_t kDim = isTnd ? q.size(2) : q.size(3);
    const int64_t hv = isTnd ? v.size(1) : v.size(2);
    const int64_t vDim = isTnd ? v.size(2) : v.size(3);
    TORCH_CHECK(h > 0 && hv > 0 && hv % h == 0 && kDim == 128 && (vDim == 128 || vDim == 256),
                "recurrent_kda: invalid H/HV/K/V dimensions.");
    TORCH_CHECK((isTnd && v.size(0) == totalTokens && g.size(0) == totalTokens && beta.size(0) == totalTokens &&
                 g.size(1) == hv && beta.size(1) == hv && g.size(2) == kDim) ||
                    (!isTnd && v.size(0) == batch && v.size(1) == denseSeqLen && g.size(0) == batch &&
                     g.size(1) == denseSeqLen && g.size(2) == hv && g.size(3) == kDim && beta.size(0) == batch &&
                     beta.size(1) == denseSeqLen && beta.size(2) == hv),
                "recurrent_kda: v/g/beta shape mismatch.");

    const bool hasCuSeqlens = cuSeqlens.has_value() && cuSeqlens->defined();
    if (hasCuSeqlens) {
        TORCH_CHECK(cuSeqlens->dim() == 1 && cuSeqlens->size(0) >= 2 &&
                        (cuSeqlens->scalar_type() == at::kInt || cuSeqlens->scalar_type() == at::kLong),
                    "recurrent_kda: cu_seqlens must be a 1D int32 or int64 tensor.");
    }
    const int64_t seqNum = hasCuSeqlens ? cuSeqlens->size(0) - 1 : batch;
    const bool stateVFirstValue = stateVFirst.value_or(false);
    const bool inplaceFinalStateValue = inplaceFinalState.value_or(true);
    const bool outputFinalStateValue = outputFinalState.value_or(false);
    const std::vector<int64_t> stateShape =
        stateVFirstValue ? std::vector<int64_t>{seqNum, hv, vDim, kDim} : std::vector<int64_t>{seqNum, hv, kDim, vDim};
    TORCH_CHECK(initialState.scalar_type() == at::kFloat || initialState.scalar_type() == at::kBFloat16,
                "recurrent_kda: initial_state must be float32 or bfloat16.");
    TORCH_CHECK(initialState.dim() == 4 && initialState.size(1) == hv && initialState.size(2) == stateShape[2] &&
                    initialState.size(3) == stateShape[3],
                "recurrent_kda: initial_state shape does not match state_v_first.");

    const bool useGateValue = useGateInKernel.value_or(false);
    const bool safeGateValue = safeGate.value_or(false);
    const bool useQkL2normValue = useQkL2normInKernel.value_or(false);
    const bool useBetaSigmoidValue = useBetaSigmoidInKernel.value_or(false);
    const bool allowNegEigvalValue = allowNegEigval.value_or(false);
    const double lowerBoundValue = lowerBound.value_or(-5.0);
    TORCH_CHECK(!safeGateValue || (useGateValue && lowerBoundValue >= -5.0 && lowerBoundValue < 0.0),
                "recurrent_kda: safe_gate/lower_bound attributes are invalid.");
    const double scaleValue = scale.value_or(std::pow(static_cast<double>(kDim), -0.5));
    const char *layoutValue = layout.c_str();

    at::Tensor out;
    at::Tensor finalState;
    {
        const auto localDevice = c10::Device(q.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        out = at::empty_like(v);
        finalState = at::empty_like(initialState);
    }

    ACLNN_CMD(aclnnRecurrentKda, q, k, v, g, beta, initialState, cuSeqlens, ssmStateIndices, aLog, dtBias,
              numAcceptedTokens, layoutValue, scaleValue, outputFinalStateValue, inplaceFinalStateValue,
              useQkL2normValue, useGateValue, useBetaSigmoidValue, allowNegEigvalValue, safeGateValue, lowerBoundValue,
              stateVFirstValue, out, finalState);

    at::Tensor returnedState;
    if (outputFinalStateValue) {
        returnedState = inplaceFinalStateValue ? initialState : finalState;
    }
    return std::make_tuple(out, returnedState);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("recurrent_kda", &RecurrentKda, "recurrent KDA");
}

} // namespace op_api
