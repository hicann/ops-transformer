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
 * \file block_attn_res_prepare.cpp
 * \brief PyTorch binding for BlockAttnResPrepare.
 */

#include <limits>
#include <tuple>

#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>

#include "aclnn_common.h"

namespace op_api {
namespace {

constexpr int64_t BLOCK_RES_RANK = 3;
constexpr int64_t VALID_BLOCKS_RANK = 1;
constexpr int64_t PSEUDO_QUERY_RANK = 2;
constexpr int64_t T_DIM_INDEX = 0;
constexpr int64_t N_DIM_INDEX = 1;
constexpr int64_t D_DIM_INDEX = 2;
constexpr int64_t S_DIM_INDEX = 0;
constexpr int64_t PSEUDO_QUERY_D_DIM_INDEX = 1;
constexpr int64_t VALID_BLOCKS_VALUE_DIM_INDEX = 0;
constexpr int64_t MIN_BLOCK_NUM = 1;
constexpr int64_t MAX_BLOCK_NUM = 64;
constexpr int64_t MIN_HEAD_DIM = 1;
constexpr int64_t MAX_HEAD_DIM = 8192;

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> BlockAttnResPrepare(const at::Tensor& blockRes,
                                                                   const at::Tensor& validBlocks,
                                                                   const at::Tensor& pseudoQuery, double eps)
{
    TORCH_CHECK(blockRes.dim() == BLOCK_RES_RANK, "block_res must be a 3D tensor, but got ", blockRes.dim(), "D");
    TORCH_CHECK(validBlocks.dim() == VALID_BLOCKS_RANK && validBlocks.size(VALID_BLOCKS_VALUE_DIM_INDEX) == 1,
                "valid_blocks must have shape [1], but got ", validBlocks.sizes());
    TORCH_CHECK(pseudoQuery.dim() == PSEUDO_QUERY_RANK, "pseudo_query must be a 2D tensor, but got ", pseudoQuery.dim(),
                "D");
    TORCH_CHECK(blockRes.scalar_type() == at::kFloat || blockRes.scalar_type() == at::kBFloat16,
                "block_res must have dtype float32 or bfloat16, but got ", blockRes.scalar_type());
    TORCH_CHECK(validBlocks.scalar_type() == at::ScalarType::UInt64, "valid_blocks must have dtype uint64, but got ",
                validBlocks.scalar_type());
    TORCH_CHECK(pseudoQuery.scalar_type() == at::kFloat, "pseudo_query must have dtype float32, but got ",
                pseudoQuery.scalar_type());
    TORCH_CHECK(blockRes.device() == validBlocks.device() && blockRes.device() == pseudoQuery.device(),
                "all inputs must be on the same device, but got block_res=", blockRes.device(),
                ", valid_blocks=", validBlocks.device(), ", pseudo_query=", pseudoQuery.device());
    TORCH_CHECK(blockRes.is_contiguous(), "block_res must be contiguous, but got shape ", blockRes.sizes(),
                " and strides ", blockRes.strides());
    TORCH_CHECK(validBlocks.is_contiguous(), "valid_blocks must be contiguous, but got shape ", validBlocks.sizes(),
                " and strides ", validBlocks.strides());
    TORCH_CHECK(pseudoQuery.is_contiguous(), "pseudo_query must be contiguous, but got shape ", pseudoQuery.sizes(),
                " and strides ", pseudoQuery.strides());
    TORCH_CHECK(blockRes.size(N_DIM_INDEX) >= MIN_BLOCK_NUM && blockRes.size(N_DIM_INDEX) <= MAX_BLOCK_NUM,
                "block_res.size(1) must be in [1, 64], but got ", blockRes.size(N_DIM_INDEX));
    TORCH_CHECK(blockRes.size(D_DIM_INDEX) >= MIN_HEAD_DIM && blockRes.size(D_DIM_INDEX) <= MAX_HEAD_DIM,
                "block_res.size(2) must be in [1, 8192], but got ", blockRes.size(D_DIM_INDEX));
    TORCH_CHECK(blockRes.size(D_DIM_INDEX) == pseudoQuery.size(PSEUDO_QUERY_D_DIM_INDEX),
                "block_res.size(2) must equal pseudo_query.size(1), but got block_res.size(2)=",
                blockRes.size(D_DIM_INDEX), " and pseudo_query.size(1)=", pseudoQuery.size(PSEUDO_QUERY_D_DIM_INDEX));

    TORCH_CHECK(eps >= std::numeric_limits<float>::min() && eps <= std::numeric_limits<float>::max(),
                "eps must be in [FLT_MIN, FLT_MAX] before conversion to float32, but got ", eps);

    const int64_t totalT = blockRes.size(T_DIM_INDEX);
    const int64_t totalS = pseudoQuery.size(S_DIM_INDEX);
    const int64_t totalD = blockRes.size(D_DIM_INDEX);

    const auto outputOptions = blockRes.options().dtype(at::kFloat);
    at::Tensor numerator{nullptr};
    at::Tensor logitMax{nullptr};
    at::Tensor expSum{nullptr};
    {
        const c10::OptionalDeviceGuard deviceGuard(blockRes.device());
        numerator = at::empty({totalS, totalT, totalD}, outputOptions);
        logitMax = at::empty({totalS, totalT}, outputOptions);
        expSum = at::empty({totalS, totalT}, outputOptions);
    }

    ACLNN_CMD(aclnnBlockAttnResPrepare, blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum, eps);
    return {numerator, logitMax, expSum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("block_attn_res_prepare", &BlockAttnResPrepare, "block_attn_res_prepare", py::arg("block_res"),
               py::arg("valid_blocks"), py::arg("pseudo_query"), py::arg("eps"));
}

} // namespace op_api
