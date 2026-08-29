/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include <c10/core/DeviceGuard.h>

#include <cmath>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t MATRIX_DIM_NUM = 2L;
constexpr int64_t VECTOR_DIM_NUM = 1L;
constexpr int64_t TOKEN_DIM_INDEX = 0L;
constexpr int64_t HIDDEN_DIM_INDEX = 1L;
constexpr int64_t MAX_HIDDEN_SIZE = 8192L;

void CheckTensorDevice(const at::Tensor &tensor, const at::Device &expectedDevice, const char *name)
{
    TORCH_CHECK(tensor.device() == expectedDevice, name, " must be on device ", expectedDevice, ", but got ",
                tensor.device(), ".");
}

void CheckTensorContiguous(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous, but got size ", tensor.sizes(), " and stride ",
                tensor.strides(), ".");
}

void CheckEps(double eps)
{
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and greater than 0, but got ", eps, ".");
    const float aclnnEps = static_cast<float>(eps);
    TORCH_CHECK(std::isfinite(aclnnEps) && aclnnEps > 0.0F,
                "eps must be representable as a finite positive float32 value, but got ", eps, ".");
}

void CheckInputs(const at::Tensor &partialBlock, const at::Tensor &delta, const at::Tensor &pseudoQuery,
                 const at::Tensor &numerator, const at::Tensor &logitMax, const at::Tensor &expSum, double eps)
{
    TORCH_CHECK(partialBlock.scalar_type() == at::kFloat, "partial_block must be float32, but got ",
                partialBlock.scalar_type(), ".");
    TORCH_CHECK(delta.scalar_type() == at::kBFloat16, "delta must be bfloat16, but got ", delta.scalar_type(), ".");
    TORCH_CHECK(pseudoQuery.scalar_type() == at::kFloat, "pseudo_query must be float32, but got ",
                pseudoQuery.scalar_type(), ".");
    TORCH_CHECK(numerator.scalar_type() == at::kFloat, "numerator must be float32, but got ", numerator.scalar_type(),
                ".");
    TORCH_CHECK(logitMax.scalar_type() == at::kFloat, "logit_max must be float32, but got ", logitMax.scalar_type(),
                ".");
    TORCH_CHECK(expSum.scalar_type() == at::kFloat, "exp_sum must be float32, but got ", expSum.scalar_type(), ".");
    CheckTensorContiguous(partialBlock, "partial_block");
    CheckTensorContiguous(delta, "delta");
    CheckTensorContiguous(pseudoQuery, "pseudo_query");
    CheckTensorContiguous(numerator, "numerator");
    CheckTensorContiguous(logitMax, "logit_max");
    CheckTensorContiguous(expSum, "exp_sum");

    const auto expectedDevice = partialBlock.device();
    CheckTensorDevice(delta, expectedDevice, "delta");
    CheckTensorDevice(pseudoQuery, expectedDevice, "pseudo_query");
    CheckTensorDevice(numerator, expectedDevice, "numerator");
    CheckTensorDevice(logitMax, expectedDevice, "logit_max");
    CheckTensorDevice(expSum, expectedDevice, "exp_sum");

    CheckEps(eps);

    TORCH_CHECK(partialBlock.dim() == MATRIX_DIM_NUM, "partial_block must be ", MATRIX_DIM_NUM, "D, but got ",
                partialBlock.dim(), "D.");
    TORCH_CHECK(delta.dim() == MATRIX_DIM_NUM, "delta must be ", MATRIX_DIM_NUM, "D, but got ", delta.dim(), "D.");
    TORCH_CHECK(pseudoQuery.dim() == VECTOR_DIM_NUM, "pseudo_query must be ", VECTOR_DIM_NUM, "D, but got ",
                pseudoQuery.dim(), "D.");
    TORCH_CHECK(numerator.dim() == MATRIX_DIM_NUM, "numerator must be ", MATRIX_DIM_NUM, "D, but got ", numerator.dim(),
                "D.");
    TORCH_CHECK(logitMax.dim() == VECTOR_DIM_NUM, "logit_max must be ", VECTOR_DIM_NUM, "D, but got ", logitMax.dim(),
                "D.");
    TORCH_CHECK(expSum.dim() == VECTOR_DIM_NUM, "exp_sum must be ", VECTOR_DIM_NUM, "D, but got ", expSum.dim(), "D.");

    TORCH_CHECK(delta.sizes() == partialBlock.sizes(), "delta shape must match partial_block, but got ", delta.sizes(),
                " and ", partialBlock.sizes(), ".");
    TORCH_CHECK(numerator.sizes() == partialBlock.sizes(), "numerator shape must match partial_block, but got ",
                numerator.sizes(), " and ", partialBlock.sizes(), ".");
    TORCH_CHECK(pseudoQuery.size(TOKEN_DIM_INDEX) == partialBlock.size(HIDDEN_DIM_INDEX),
                "pseudo_query length must match the hidden dimension of partial_block, but got ", pseudoQuery.sizes(),
                " and ", partialBlock.sizes(), ".");
    TORCH_CHECK(logitMax.size(TOKEN_DIM_INDEX) == partialBlock.size(TOKEN_DIM_INDEX),
                "logit_max length must match the token dimension of partial_block, but got ", logitMax.sizes(), " and ",
                partialBlock.sizes(), ".");
    TORCH_CHECK(expSum.size(TOKEN_DIM_INDEX) == partialBlock.size(TOKEN_DIM_INDEX),
                "exp_sum length must match the token dimension of partial_block, but got ", expSum.sizes(), " and ",
                partialBlock.sizes(), ".");
    TORCH_CHECK(partialBlock.size(TOKEN_DIM_INDEX) >= 0,
                "the token dimension T must be greater than or equal to 0, but got ",
                partialBlock.size(TOKEN_DIM_INDEX), ".");
    TORCH_CHECK(partialBlock.size(HIDDEN_DIM_INDEX) >= 0,
                "the hidden dimension D must be greater than or equal to 0, but got ",
                partialBlock.size(HIDDEN_DIM_INDEX), ".");
    TORCH_CHECK(partialBlock.size(HIDDEN_DIM_INDEX) <= MAX_HIDDEN_SIZE, "the hidden dimension D must not exceed ",
                MAX_HIDDEN_SIZE, ", but got ", partialBlock.size(HIDDEN_DIM_INDEX), ".");
}
} // namespace

at::Tensor block_attn_res_update(at::Tensor &partialBlock, const at::Tensor &delta, const at::Tensor &pseudoQuery,
                                 const at::Tensor &numerator, const at::Tensor &logitMax, const at::Tensor &expSum,
                                 double eps)
{
    CheckInputs(partialBlock, delta, pseudoQuery, numerator, logitMax, expSum, eps);
    const float aclnnEps = static_cast<float>(eps);

    at::Tensor h{nullptr};
    {
        const auto localDevice = c10::Device(partialBlock.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        h = at::empty(delta.sizes(), delta.options());
    }

    // aclnnBlockAttnResUpdate updates partial_block in place; the Torch API only returns h.
    ACLNN_CMD(aclnnBlockAttnResUpdate, partialBlock, delta, pseudoQuery, numerator, logitMax, expSum, aclnnEps, h);

    return h;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("block_attn_res_update", &block_attn_res_update, "block_attn_res_update torch wrapper");
}
} // namespace op_api
