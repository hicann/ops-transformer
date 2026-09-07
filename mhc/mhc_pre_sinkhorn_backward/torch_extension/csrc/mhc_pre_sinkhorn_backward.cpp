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
 * \file mhc_pre_sinkhorn_backward.cpp
 * \brief ACLNN Wrapper composed from aclnnMhcSinkhorn and aclnnMhcPreBackward
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> MhcPreSinkhornBackward(
    const at::Tensor &gradHin, const at::Tensor &gradHPost, const at::Tensor &gradHRes, const at::Tensor &x,
    const at::Tensor &phi, const at::Tensor &alpha, const at::Tensor &bias, const at::Tensor &hPre,
    const at::Tensor &hcBeforeNorm, const at::Tensor &invRms, const at::Tensor &sumOut, const at::Tensor &normOut,
    double hcEps)
{
    at::Tensor gradX = at::empty_like(x);
    at::Tensor gradPhi = at::empty_like(phi);
    at::Tensor gradAlpha = at::empty_like(alpha);
    at::Tensor gradBias = at::empty_like(bias);

    // Keep the empty-tensor behavior of aclnnMhcPreSinkhornBackward: no computation is launched.
    if (gradHin.numel() == 0 || gradHPost.numel() == 0 || gradHRes.numel() == 0 || x.numel() == 0 ||
        hPre.numel() == 0 || hcBeforeNorm.numel() == 0 || invRms.numel() == 0 || sumOut.numel() == 0 ||
        normOut.numel() == 0) {
        return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(gradX, gradPhi, gradAlpha, gradBias);
    }

    TORCH_CHECK(x.dim() == 3 || x.dim() == 4, "x dim must be 3 (t, n, d) or 4 (b, s, n, d), but got ", x.dim());
    TORCH_CHECK(sumOut.dim() >= 1 && sumOut.size(0) > 0 && sumOut.size(0) % 2 == 0,
                "sumOut first dimension must be a positive even number, but got shape ", sumOut.sizes());
    TORCH_CHECK(normOut.dim() == sumOut.dim() + 1 && normOut.size(0) == sumOut.size(0),
                "normOut shape is incompatible with sumOut: ", normOut.sizes(), " vs ", sumOut.sizes());

    const int64_t n = x.size(-2);
    const int64_t numIters = sumOut.size(0) / 2;
    const bool isFlattenedGradHRes = gradHRes.dim() == x.dim() - 1 && gradHRes.size(-1) == n * n;
    const bool isMatrixGradHRes = gradHRes.dim() == x.dim() && gradHRes.size(-2) == n && gradHRes.size(-1) == n;
    TORCH_CHECK(isFlattenedGradHRes || isMatrixGradHRes, "gradHRes shape must be [..., N * N] or [..., N, N], but got ",
                gradHRes.sizes(), " and N = ", n);
    for (int64_t i = 0; i < x.dim() - 2; ++i) {
        TORCH_CHECK(gradHRes.size(i) == x.size(i), "gradHRes prefix dimensions must match x, but got ",
                    gradHRes.sizes(), " and ", x.sizes());
    }
    const int64_t totalLength = gradHRes.numel() / (n * n);
    TORCH_CHECK(hcBeforeNorm.size(-1) == n * n + 2 * n,
                "hcBeforeNorm last dimension must equal N * N + 2 * N, but got ", hcBeforeNorm.size(-1),
                " and N = ", n);

    at::Tensor normalizedMix = hcBeforeNorm * invRms;
    at::Tensor hPostLogits = normalizedMix.slice(-1, n, 2 * n) * alpha.select(0, 1) + bias.slice(0, n, 2 * n);
    at::Tensor hPost = at::sigmoid(hPostLogits) * 2.0;
    at::Tensor hResLogits = normalizedMix.slice(-1, 2 * n, 2 * n + n * n) * alpha.select(0, 2) + bias.slice(0, 2 * n);

    c10::SmallVector<int64_t, 4> hResShape(x.sizes().begin(), x.sizes().end() - 2);
    hResShape.emplace_back(n);
    hResShape.emplace_back(n);
    at::Tensor hResLogitsMatrix = hResLogits.reshape(hResShape);
    at::Tensor gradHResMatrix = gradHRes.reshape(hResShape);

    constexpr int64_t sinkhornAlign = 8;
    at::Tensor hRes = at::empty_like(hResLogitsMatrix);
    at::Tensor sinkhornNormOut = at::empty({2 * numIters * totalLength * n * sinkhornAlign}, hResLogits.options());
    at::Tensor sinkhornSumOut = at::empty({2 * numIters * totalLength * sinkhornAlign}, hResLogits.options());
    ACLNN_CMD(aclnnMhcSinkhorn, hResLogitsMatrix, hcEps, numIters, hRes, sinkhornNormOut, sinkhornSumOut);

    at::Tensor gradHResBeforeSinkhorn = at::empty_like(gradHResMatrix);
    ACLNN_CMD(aclnnMhcSinkhornBackward, gradHResMatrix, sinkhornNormOut, sinkhornSumOut, gradHResBeforeSinkhorn);

    TORCH_CHECK(invRms.size(-1) == 1, "invRms last dimension must be 1, but got ", invRms.size(-1));
    at::Tensor invRmsSqueezed = invRms.squeeze(-1);
    const c10::optional<at::Tensor> gamma = c10::nullopt;
    const c10::optional<at::Tensor> gradXPost = c10::nullopt;
    const c10::optional<at::Tensor> gradGamma = c10::nullopt;

    ACLNN_CMD(aclnnMhcPreBackward, x, phi, alpha, gradHin, gradHPost, gradHResBeforeSinkhorn, invRmsSqueezed,
              hcBeforeNorm, hPre, hPost, gamma, gradXPost, hcEps, gradX, gradPhi, gradAlpha, gradBias, gradGamma);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(gradX, gradPhi, gradAlpha, gradBias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_pre_sinkhorn_backward", &MhcPreSinkhornBackward, "mhc_pre_sinkhorn_backward");
}

} // namespace op_api
