// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_ONE = 1;
const int DIM_TWO = 2;

/**
 * @brief Wrapper for ffn_to_attention. The op has no host-visible output tensors;
 *        data is sent to peer ranks via the HCCL window. We therefore return an empty
 *        tensor so PyTorch's dispatcher contract (at least one output) is satisfied.
 */
void NpuFFNToAttention(const at::Tensor& context, const at::Tensor& x, const at::Tensor& sessionIds,
                       const at::Tensor& microBatchIds, const at::Tensor& tokenIds, const at::Tensor& expertOffsets,
                       const at::Tensor& actualTokenNum, const c10::optional<at::Tensor>& attnRankTable,
                       std::string group, int64_t worldSize, const std::vector<int64_t>& tokenInfoTableShape,
                       const std::vector<int64_t>& tokenDataShape, int64_t cclBufferSize)
{
    TORCH_CHECK((x.dim() == DIM_TWO), "The x should be 2D, current dim is: ", x.dim());
    TORCH_CHECK((sessionIds.dim() == DIM_ONE) && (microBatchIds.dim() == DIM_ONE) && (tokenIds.dim() == DIM_ONE) &&
                    (expertOffsets.dim() == DIM_ONE) && (actualTokenNum.dim() == DIM_ONE),
                "sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum should be 1D.");
    TORCH_CHECK((worldSize > 0), "The world_size should be greater than 0, current is: ", worldSize);
    TORCH_CHECK(!group.empty(), "group should not be empty.");
    TORCH_CHECK((tokenInfoTableShape.size() == 3),
                "token_info_table_shape should be 3D, current size is: ", tokenInfoTableShape.size());
    TORCH_CHECK((tokenDataShape.size() == 4),
                "token_data_shape should be 4D, current size is: ", tokenDataShape.size());
    TORCH_CHECK(((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)),
                "dtype of x should be bfloat16 or float16, but got ", x.scalar_type());
    TORCH_CHECK((sessionIds.scalar_type() == at::kInt), "dtype of sessionIds should be int32.");
    TORCH_CHECK((microBatchIds.scalar_type() == at::kInt), "dtype of microBatchIds should be int32.");
    TORCH_CHECK((tokenIds.scalar_type() == at::kInt), "dtype of tokenIds should be int32.");
    TORCH_CHECK((expertOffsets.scalar_type() == at::kInt), "dtype of expertOffsets should be int32.");
    TORCH_CHECK((actualTokenNum.scalar_type() == at::kLong), "dtype of actualTokenNum should be int64.");

    // Set DeviceGuard on x (the first input tensor that lives on NPU)
    auto localDevice = c10::Device(x.device());
    const c10::OptionalDeviceGuard deviceGuard(localDevice);

    std::string groupStr = std::string(group);
    char* groupPtr = const_cast<char*>(groupStr.c_str());

    at::IntArrayRef tokenInfoTableShapeRef(tokenInfoTableShape);
    at::IntArrayRef tokenDataShapeRef(tokenDataShape);

    ACLNN_CMD(aclnnFFNToAttentionV2, context, x, sessionIds, microBatchIds, tokenIds, expertOffsets, actualTokenNum,
              attnRankTable, groupPtr, worldSize, tokenInfoTableShapeRef, tokenDataShapeRef, cclBufferSize);
}

namespace {
constexpr int64_t MB_ALIGN = 2LL * 1024LL * 1024LL; // 2MB
constexpr int64_t BYTES_PER_FP16 = 2;
constexpr int64_t BYTES_PER_INT32 = 4;
constexpr int64_t WIN_REGION_ALIGN = 512;          // window region start alignment (matches kernel WIN_ALIGN)
constexpr int64_t FLAG_SLOT_BYTES = 32;            // URMA flag staging slot (matches kernel URMA_FLAG_SLOT_SIZE)
constexpr int64_t REMOTE_ADDRESS_ENTRY_BYTES = 16; // address table entry: slot id + source token index
constexpr int64_t WORLD_SIZE_UPPER_BOUND = 1024;
// Reserve for up to 1024 AIVs; host tiling uses the actual core count.
constexpr int64_t AIV_RESERVE_COUNT = 1024LL;
constexpr int64_t FLAG_SOURCE_RESERVE_BYTES = AIV_RESERVE_COUNT * FLAG_SLOT_BYTES;

int64_t CeilAlign(int64_t val, int64_t align)
{
    TORCH_CHECK(val >= 0 && val <= std::numeric_limits<int64_t>::max() - (align - 1), "Buffer alignment overflow");
    return (val + align - 1) / align * align;
}

int64_t CheckedMul(int64_t lhs, int64_t rhs)
{
    TORCH_CHECK(lhs >= 0 && rhs >= 0 && (rhs == 0 || lhs <= std::numeric_limits<int64_t>::max() / rhs),
                "Buffer size multiplication overflow");
    return lhs * rhs;
}

int64_t CheckedAdd(int64_t lhs, int64_t rhs)
{
    TORCH_CHECK(lhs >= 0 && rhs >= 0 && lhs <= std::numeric_limits<int64_t>::max() - rhs,
                "Buffer size addition overflow");
    return lhs + rhs;
}
} // namespace

int64_t GetFFNToAttentionCclBufferSize(const std::vector<int64_t>& tokenInfoTableShape,
                                       const std::vector<int64_t>& tokenDataShape, int64_t worldSize)
{
    TORCH_CHECK(tokenInfoTableShape.size() == 3, "token_info_table_shape should have 3 elements, but got ",
                tokenInfoTableShape.size());
    TORCH_CHECK(tokenDataShape.size() == 4, "token_data_shape should have 4 elements, but got ", tokenDataShape.size());

    int64_t microBatchNum = tokenInfoTableShape[0];
    int64_t bs = tokenInfoTableShape[1];
    int64_t expertNumPerToken = tokenInfoTableShape[2];
    int64_t hs = tokenDataShape[3];

    TORCH_CHECK(microBatchNum > 0 && bs > 0 && expertNumPerToken > 0 && hs > 0, "Token dimensions must be positive");
    TORCH_CHECK(std::equal(tokenInfoTableShape.begin(), tokenInfoTableShape.end(), tokenDataShape.begin()),
                "Token info/data dimensions must match");
    TORCH_CHECK(worldSize > 0 && worldSize <= WORLD_SIZE_UPPER_BOUND,
                "Complete window sizing requires world_size in [1, ", WORLD_SIZE_UPPER_BOUND, "]");
    int64_t tokenSlots = CheckedMul(CheckedMul(microBatchNum, bs), expertNumPerToken);
    int64_t maxWorkerNum = worldSize - 1;
    int64_t infoTableLastDim = CheckedAdd(CheckedMul(bs, expertNumPerToken), 2);
    int64_t recvInfoSize =
        CeilAlign(CheckedMul(CheckedMul(CheckedMul(maxWorkerNum, microBatchNum), infoTableLastDim), BYTES_PER_INT32),
                  WIN_REGION_ALIGN);
    int64_t recvDataSize =
        CeilAlign(CheckedMul(CheckedMul(CheckedMul(maxWorkerNum, tokenSlots), hs), BYTES_PER_FP16), WIN_REGION_ALIGN);
    int64_t rawSize =
        CeilAlign(CheckedAdd(CheckedAdd(recvInfoSize, recvDataSize), FLAG_SOURCE_RESERVE_BYTES), WIN_REGION_ALIGN);
    const int64_t tokenCapacity = CheckedMul(worldSize, tokenSlots);
    const int64_t rankTableStride = CeilAlign(CheckedMul(tokenCapacity, REMOTE_ADDRESS_ENTRY_BYTES), WIN_REGION_ALIGN);
    const int64_t countRowStride = CeilAlign(CheckedMul(worldSize, BYTES_PER_INT32), WIN_REGION_ALIGN);
    const int64_t tableBytes =
        CheckedAdd(CheckedMul(worldSize, rankTableStride), CheckedMul(AIV_RESERVE_COUNT + 1, countRowStride));
    rawSize = CheckedAdd(rawSize, tableBytes);
    return CeilAlign(rawSize, MB_ALIGN);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_ffn_to_attention", &NpuFFNToAttention, "npu_ffn_to_attention");
    m.def("get_ffn_to_attention_ccl_buffer_size", &GetFFNToAttentionCclBufferSize,
          pybind11::arg("token_info_table_shape"), pybind11::arg("token_data_shape"), pybind11::arg("world_size"),
          "URMA window size hosting the attention_to_ffn receiving area, flag sources, rank address tables and "
          "count rows");
}
} // namespace op_api
