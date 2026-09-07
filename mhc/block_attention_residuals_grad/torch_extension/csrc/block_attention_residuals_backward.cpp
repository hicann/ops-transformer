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
 * \file block_attention_residuals_backward.cpp
 * \brief ACLNN wrapper for aclnnBlockAttentionResidualsGrad.
 */

#include <tuple>

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t PARTIAL_BLOCK_DIM_NUM = 2;
constexpr int64_t BLOCK_RES_DIM_NUM = 3;
constexpr int64_t PROJ_WEIGHT_DIM_NUM = 2;
constexpr int64_t NORM_WEIGHT_DIM_NUM = 1;
constexpr int64_t MIN_TOKEN_NUM = 1;
constexpr int64_t MIN_BLOCK_NUM = 0;
constexpr int64_t MAX_BLOCK_NUM = 128;
constexpr int64_t MIN_HIDDEN_SIZE = 1;

bool IsSupportedMainDtype(at::ScalarType dtype)
{
    return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> block_attention_residuals_backward(
    const at::Tensor &partial_block, const at::Tensor &block_res, const at::Tensor &proj_weight,
    const at::Tensor &norm_weight, const at::Tensor &grad_hidden_states, const at::Tensor &inv_norm,
    const at::Tensor &probs, int64_t valid_block_num)
{
    TORCH_CHECK(partial_block.dim() == PARTIAL_BLOCK_DIM_NUM, "partial_block must be 2D, but got ", partial_block.dim(),
                "D");
    TORCH_CHECK(block_res.dim() == BLOCK_RES_DIM_NUM, "block_res must be 3D, but got ", block_res.dim(), "D");
    TORCH_CHECK(proj_weight.dim() == PROJ_WEIGHT_DIM_NUM, "proj_weight must be 2D, but got ", proj_weight.dim(), "D");
    TORCH_CHECK(norm_weight.dim() == NORM_WEIGHT_DIM_NUM, "norm_weight must be 1D, but got ", norm_weight.dim(), "D");
    TORCH_CHECK(grad_hidden_states.dim() == PARTIAL_BLOCK_DIM_NUM, "grad_hidden_states must be 2D, but got ",
                grad_hidden_states.dim(), "D");
    TORCH_CHECK(inv_norm.dim() == PARTIAL_BLOCK_DIM_NUM, "inv_norm must be 2D, but got ", inv_norm.dim(), "D");
    TORCH_CHECK(probs.dim() == PARTIAL_BLOCK_DIM_NUM, "probs must be 2D, but got ", probs.dim(), "D");

    TORCH_CHECK(IsSupportedMainDtype(partial_block.scalar_type()),
                "partial_block dtype must be float16/bfloat16/float32, but got ", partial_block.scalar_type());
    TORCH_CHECK(IsSupportedMainDtype(block_res.scalar_type()),
                "block_res dtype must be float16/bfloat16/float32, but got ", block_res.scalar_type());
    TORCH_CHECK(IsSupportedMainDtype(proj_weight.scalar_type()),
                "proj_weight dtype must be float16/bfloat16/float32, but got ", proj_weight.scalar_type());
    TORCH_CHECK(IsSupportedMainDtype(norm_weight.scalar_type()),
                "norm_weight dtype must be float16/bfloat16/float32, but got ", norm_weight.scalar_type());
    TORCH_CHECK(IsSupportedMainDtype(grad_hidden_states.scalar_type()),
                "grad_hidden_states dtype must be float16/bfloat16/float32, but got ",
                grad_hidden_states.scalar_type());
    TORCH_CHECK(block_res.scalar_type() == partial_block.scalar_type() &&
                    proj_weight.scalar_type() == partial_block.scalar_type() &&
                    norm_weight.scalar_type() == partial_block.scalar_type() &&
                    grad_hidden_states.scalar_type() == partial_block.scalar_type(),
                "block_res/proj_weight/norm_weight/grad_hidden_states dtype must be same as partial_block");
    TORCH_CHECK(inv_norm.scalar_type() == at::kFloat, "inv_norm dtype must be float32, but got ",
                inv_norm.scalar_type());
    TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs dtype must be float32, but got ", probs.scalar_type());

    TORCH_CHECK(partial_block.device() == block_res.device() && partial_block.device() == proj_weight.device() &&
                    partial_block.device() == norm_weight.device() &&
                    partial_block.device() == grad_hidden_states.device() &&
                    partial_block.device() == inv_norm.device() && partial_block.device() == probs.device(),
                "all inputs must be on the same device, but partial_block is on ", partial_block.device());

    const int64_t tokenNum = partial_block.size(DIM_0);
    const int64_t hiddenSize = partial_block.size(DIM_1);
    const int64_t blockNum = block_res.size(DIM_1);
    TORCH_CHECK(tokenNum >= MIN_TOKEN_NUM, "partial_block.size(0) must be >= 1, but got ", tokenNum);
    TORCH_CHECK(hiddenSize >= MIN_HIDDEN_SIZE, "partial_block.size(1) must be >= 1, but got ", hiddenSize);
    TORCH_CHECK(blockNum >= MIN_BLOCK_NUM && blockNum <= MAX_BLOCK_NUM,
                "block_res.size(1) must be in [0, 128], but got ", blockNum);
    TORCH_CHECK(block_res.size(DIM_0) == tokenNum, "block_res.size(0) must equal partial_block.size(0), but got ",
                block_res.size(DIM_0), " and ", tokenNum);
    TORCH_CHECK(block_res.size(DIM_2) == hiddenSize, "block_res.size(2) must equal partial_block.size(1), but got ",
                block_res.size(DIM_2), " and ", hiddenSize);
    TORCH_CHECK(proj_weight.size(DIM_0) == 1 && proj_weight.size(DIM_1) == hiddenSize,
                "proj_weight must have shape [1, ", hiddenSize, "], but got ", proj_weight.sizes());
    TORCH_CHECK(norm_weight.size(DIM_0) == hiddenSize, "norm_weight must have shape [", hiddenSize, "], but got ",
                norm_weight.sizes());
    TORCH_CHECK(grad_hidden_states.size(DIM_0) == tokenNum && grad_hidden_states.size(DIM_1) == hiddenSize,
                "grad_hidden_states must have shape [", tokenNum, ", ", hiddenSize, "], but got ",
                grad_hidden_states.sizes());
    TORCH_CHECK(inv_norm.size(DIM_0) == tokenNum && inv_norm.size(DIM_1) == blockNum + 1, "inv_norm must have shape [",
                tokenNum, ", ", blockNum + 1, "], but got ", inv_norm.sizes());
    TORCH_CHECK(probs.size(DIM_0) == tokenNum && probs.size(DIM_1) == blockNum + 1, "probs must have shape [", tokenNum,
                ", ", blockNum + 1, "], but got ", probs.sizes());

    at::Tensor gradPartialBlock;
    at::Tensor gradBlockRes;
    at::Tensor gradProjWeight;
    at::Tensor gradNormWeight;
    {
        const c10::OptionalDeviceGuard deviceGuard(partial_block.device());
        gradPartialBlock = at::empty(partial_block.sizes(), partial_block.options());
        gradBlockRes = at::empty(block_res.sizes(), block_res.options());
        gradProjWeight = at::empty(proj_weight.sizes(), proj_weight.options());
        gradNormWeight = at::empty(norm_weight.sizes(), norm_weight.options());
    }

    ACLNN_CMD(aclnnBlockAttentionResidualsGrad, partial_block, block_res, proj_weight, norm_weight, grad_hidden_states,
              inv_norm, probs, valid_block_num, gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight);

    return std::make_tuple(gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("block_attention_residuals_backward", &block_attention_residuals_backward,
          "block_attention_residuals_backward");
}

} // namespace op_api
