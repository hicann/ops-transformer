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
 * \file block_attention_residuals.cpp
 * \brief PyTorch binding for BlockAttentionResiduals.
 */

#include <cmath>
#include <tuple>

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {

constexpr int64_t PARTIAL_BLOCK_RANK = 2;
constexpr int64_t BLOCK_RES_RANK = 3;
constexpr int64_t NORM_WEIGHT_RANK = 1;
constexpr int64_t PROJ_WEIGHT_RANK_1D = 1;
constexpr int64_t PROJ_WEIGHT_RANK_2D = 2;
constexpr int64_t PROJ_WEIGHT_LEADING_ONES = 1;
constexpr int64_t T_DIM_INDEX = 0;
constexpr int64_t N_DIM_INDEX = 1;
constexpr int64_t H_DIM_INDEX = 1;
constexpr int64_t BLOCK_RES_H_DIM_INDEX = 2;
constexpr int64_t MIN_HIDDEN_SIZE = 1;
constexpr int64_t MIN_BLOCK_NUM = 1;
constexpr int64_t MAX_BLOCK_NUM = 100;

bool IsSupportedComputeDtype(at::ScalarType dtype)
{
    return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> block_attention_residuals(
    const at::Tensor &partialBlock, const at::Tensor &blockRes, const at::Tensor &projWeight,
    const at::Tensor &normWeight, int64_t validBlockNum, double normEps, bool needBackward)
{
    TORCH_CHECK(partialBlock.dim() == PARTIAL_BLOCK_RANK, "partial_block must be a 2D tensor, but got ",
                partialBlock.dim(), "D");
    TORCH_CHECK(blockRes.dim() == BLOCK_RES_RANK, "block_res must be a 3D tensor, but got ", blockRes.dim(), "D");
    TORCH_CHECK(normWeight.dim() == NORM_WEIGHT_RANK, "norm_weight must be a 1D tensor, but got ", normWeight.dim(),
                "D");
    TORCH_CHECK(projWeight.dim() == PROJ_WEIGHT_RANK_1D || (projWeight.dim() == PROJ_WEIGHT_RANK_2D &&
                                                            projWeight.size(T_DIM_INDEX) == PROJ_WEIGHT_LEADING_ONES),
                "proj_weight must be [H] or [1,H], but got ", projWeight.sizes());

    const auto computeDtype = partialBlock.scalar_type();
    TORCH_CHECK(IsSupportedComputeDtype(computeDtype),
                "partial_block dtype must be float16, bfloat16 or float32, but got ", computeDtype);
    TORCH_CHECK(blockRes.scalar_type() == computeDtype, "block_res dtype must match partial_block, but got ",
                blockRes.scalar_type());
    TORCH_CHECK(projWeight.scalar_type() == computeDtype, "proj_weight dtype must match partial_block, but got ",
                projWeight.scalar_type());
    TORCH_CHECK(normWeight.scalar_type() == computeDtype, "norm_weight dtype must match partial_block, but got ",
                normWeight.scalar_type());

    TORCH_CHECK(partialBlock.device() == blockRes.device() && partialBlock.device() == projWeight.device() &&
                    partialBlock.device() == normWeight.device(),
                "all inputs must be on the same device, but got partial_block=", partialBlock.device(),
                ", block_res=", blockRes.device(), ", proj_weight=", projWeight.device(),
                ", norm_weight=", normWeight.device());

    const int64_t numTokens = partialBlock.size(T_DIM_INDEX);
    const int64_t hiddenSize = partialBlock.size(H_DIM_INDEX);
    const int64_t numBlocks = blockRes.size(N_DIM_INDEX);
    const int64_t blockCount = numBlocks + 1;
    TORCH_CHECK(
        numTokens >= 0 && hiddenSize >= MIN_HIDDEN_SIZE && numBlocks >= MIN_BLOCK_NUM && numBlocks <= MAX_BLOCK_NUM,
        "shape range requires T>=0, H>=1 and 1<=N<=", MAX_BLOCK_NUM, ", but got T=", numTokens, ", H=", hiddenSize,
        ", N=", numBlocks);
    TORCH_CHECK(blockRes.size(T_DIM_INDEX) == numTokens && blockRes.size(BLOCK_RES_H_DIM_INDEX) == hiddenSize,
                "block_res shape must be [T,N,H], but got ", blockRes.sizes());
    TORCH_CHECK(projWeight.size(-1) == hiddenSize, "proj_weight last dim must equal H=", hiddenSize, ", but got ",
                projWeight.sizes());
    TORCH_CHECK(normWeight.size(0) == hiddenSize, "norm_weight must be [H], but got ", normWeight.sizes());
    TORCH_CHECK(validBlockNum == -1 || validBlockNum == numBlocks,
                "valid_block_num must be -1 or block_res.shape[1], got ", validBlockNum);
    TORCH_CHECK(std::isfinite(normEps) && normEps > 0.0, "norm_eps must be finite and greater than zero, but got ",
                normEps);

    at::Tensor hiddenStates{nullptr};
    at::Tensor invNorm{nullptr};
    at::Tensor probs{nullptr};
    {
        const c10::OptionalDeviceGuard deviceGuard(partialBlock.device());
        hiddenStates = at::empty({numTokens, hiddenSize}, partialBlock.options());
        if (needBackward) {
            invNorm = at::empty({numTokens, blockCount}, partialBlock.options().dtype(at::kFloat));
            probs = at::empty({numTokens, blockCount}, partialBlock.options().dtype(at::kFloat));
        } else {
            invNorm = at::empty({0}, partialBlock.options().dtype(at::kFloat));
            probs = at::empty({0}, partialBlock.options().dtype(at::kFloat));
        }
    }

    c10::optional<at::Tensor> invNormOpt = c10::nullopt;
    c10::optional<at::Tensor> probsOpt = c10::nullopt;
    if (needBackward) {
        invNormOpt = invNorm;
        probsOpt = probs;
    }

    ACLNN_CMD(aclnnBlockAttentionResiduals, partialBlock, blockRes, projWeight, normWeight, validBlockNum, normEps,
              needBackward, hiddenStates, invNormOpt, probsOpt);
    return {hiddenStates, invNorm, probs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("block_attention_residuals", &block_attention_residuals, "block_attention_residuals");
}

} // namespace op_api
