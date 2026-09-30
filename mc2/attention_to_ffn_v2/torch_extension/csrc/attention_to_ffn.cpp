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
#include <vector>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_ONE = 1;
const int DIM_TWO = 2;
const int DIM_THREE = 3;

/**
 * @brief Wrapper for attention_to_ffn. The op has no host-visible output tensors;
 *        data is sent to peer ranks via the HCCL window. We therefore return an empty
 *        tensor so PyTorch's dispatcher contract (at least one output) is satisfied.
 */
void NpuAttentionToFfn(const at::Tensor& context, const at::Tensor& x, const at::Tensor& sessionId,
                       const at::Tensor& microBatchId, const at::Tensor& layerId, const at::Tensor& expertIds,
                       const at::Tensor& expertRankTable, const c10::optional<at::Tensor>& scalesOptional,
                       const c10::optional<at::Tensor>& activeMaskOptional, std::string group, int64_t worldSize,
                       const std::vector<int64_t>& ffnTokenInfoTableShape,
                       const std::vector<int64_t>& ffnTokenDataShape,
                       const std::vector<int64_t>& attnTokenInfoTableShape, int64_t moeExpertNum, int64_t quantMode,
                       int64_t syncFlag, int64_t ffnStartRankId, int64_t cclBufferSize)
{
    TORCH_CHECK((x.dim() == DIM_THREE), "The x should be 3D, current dim is: ", x.dim());
    TORCH_CHECK((sessionId.dim() == DIM_ONE) && (microBatchId.dim() == DIM_ONE) && (layerId.dim() == DIM_ONE),
                "sessionId, microBatchId, layerId should be 1D.");
    TORCH_CHECK((expertIds.dim() == DIM_THREE), "The expertIds should be 3D, current dim is: ", expertIds.dim());
    TORCH_CHECK((expertRankTable.dim() == DIM_THREE),
                "The expertRankTable should be 3D, current dim is: ", expertRankTable.dim());
    TORCH_CHECK((worldSize > 0), "The world_size should be greater than 0, current is: ", worldSize);
    TORCH_CHECK(!group.empty(), "group should not be empty.");
    TORCH_CHECK((ffnTokenInfoTableShape.size() == 3),
                "ffn_token_info_table_shape should be 3D, current size is: ", ffnTokenInfoTableShape.size());
    TORCH_CHECK((ffnTokenDataShape.size() == 5),
                "ffn_token_data_shape should be 5D, current size is: ", ffnTokenDataShape.size());
    TORCH_CHECK((attnTokenInfoTableShape.size() == 3),
                "attn_token_info_table_shape should be 3D, current size is: ", attnTokenInfoTableShape.size());
    TORCH_CHECK((moeExpertNum > 0), "The moe_expert_num should be greater than 0, current is: ", moeExpertNum);
    TORCH_CHECK(((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)),
                "dtype of x should be bfloat16 or float16, but got ", x.scalar_type());
    TORCH_CHECK((sessionId.scalar_type() == at::kInt), "dtype of sessionId should be int32.");
    TORCH_CHECK((microBatchId.scalar_type() == at::kInt), "dtype of microBatchId should be int32.");
    TORCH_CHECK((layerId.scalar_type() == at::kInt), "dtype of layerId should be int32.");
    TORCH_CHECK((expertIds.scalar_type() == at::kInt), "dtype of expertIds should be int32.");
    TORCH_CHECK((expertRankTable.scalar_type() == at::kInt), "dtype of expertRankTable should be int32.");

    // Set DeviceGuard on x (the first input tensor that lives on NPU)
    auto localDevice = c10::Device(x.device());
    const c10::OptionalDeviceGuard deviceGuard(localDevice);

    std::string groupStr = std::string(group);
    char* groupPtr = const_cast<char*>(groupStr.c_str());

    at::IntArrayRef ffnTokenInfoTableShapeRef(ffnTokenInfoTableShape);
    at::IntArrayRef ffnTokenDataShapeRef(ffnTokenDataShape);
    at::IntArrayRef attnTokenInfoTableShapeRef(attnTokenInfoTableShape);

    ACLNN_CMD(aclnnAttentionToFfnV2, context, x, sessionId, microBatchId, layerId, expertIds, expertRankTable,
              scalesOptional, activeMaskOptional, groupPtr, worldSize, ffnTokenInfoTableShapeRef, ffnTokenDataShapeRef,
              attnTokenInfoTableShapeRef, moeExpertNum, quantMode, syncFlag, ffnStartRankId, cclBufferSize);
}

namespace {
constexpr int64_t MB_ALIGN = 2LL * 1024LL * 1024LL; // 2MB
constexpr int64_t NO_QUANT_MODE = 0;
constexpr int64_t BYTES_PER_FP16 = 2;
constexpr int64_t BYTES_PER_INT32 = 4;
constexpr int64_t BYTES_PER_QUANT_ELEM = 1; // INT8/FP8 = 1 byte, FP4-packed HS already accounts for packing
constexpr int64_t WIN_REGION_ALIGN = 512;   // window region start alignment (matches kernel WIN_ALIGN)
constexpr int64_t FLAG_SLOT_BYTES = 32;     // URMA flag staging slot (matches kernel URMA_FLAG_SLOT_SIZE)

int64_t CeilAlign(int64_t val, int64_t align)
{
    return (val + align - 1) / align * align;
}
} // namespace

int64_t GetAttentionToFfnCclBufferSize(const std::vector<int64_t>& ffnTokenInfoTableShape,
                                       const std::vector<int64_t>& ffnTokenDataShape, int64_t quantMode)
{
    TORCH_CHECK(ffnTokenInfoTableShape.size() == 3, "ffn_token_info_table_shape should have 3 elements, but got ",
                ffnTokenInfoTableShape.size());
    TORCH_CHECK(ffnTokenDataShape.size() == 5, "ffn_token_data_shape should have 5 elements, but got ",
                ffnTokenDataShape.size());

    int64_t attentionWorkerNum = ffnTokenDataShape[0];
    int64_t microBatchNum = ffnTokenDataShape[1];
    int64_t bs = ffnTokenDataShape[2];
    int64_t kPlusShared = ffnTokenDataShape[3];
    int64_t hs = ffnTokenDataShape[4];
    int64_t infoTableLastDim = ffnTokenInfoTableShape[2];

    int64_t tokenInfoSize = attentionWorkerNum * microBatchNum * infoTableLastDim * BYTES_PER_INT32;
    int64_t bytesPerElem = (quantMode == NO_QUANT_MODE) ? BYTES_PER_FP16 : BYTES_PER_QUANT_ELEM;
    int64_t tokenDataSize = attentionWorkerNum * microBatchNum * bs * kPlusShared * hs * bytesPerElem;
    // Compact contiguous staging area appended after the token-data region: one slot per token
    // (stride = ceil(hs * elemSize, 32B)), physically separated from the session-slot receiving
    // area to avoid HBM bank conflicts between local URMA reads and remote URMA writes.
    int64_t stagingStride = CeilAlign(hs * bytesPerElem, FLAG_SLOT_BYTES);
    int64_t maxTotalSendNum = bs * kPlusShared; // X is always 1
    int64_t stagingSize = maxTotalSendNum * stagingStride;
    // Persistent URMA flag staging area: one 32B flag slot per token plus one layer-flag slot.
    int64_t flagStagingSize = (bs * kPlusShared + 1) * FLAG_SLOT_BYTES;
    int64_t needed = CeilAlign(tokenInfoSize, WIN_REGION_ALIGN) + CeilAlign(tokenDataSize, WIN_REGION_ALIGN) +
                     CeilAlign(stagingSize, WIN_REGION_ALIGN) + flagStagingSize;
    return CeilAlign(needed, MB_ALIGN);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_attention_to_ffn", &NpuAttentionToFfn, "npu_attention_to_ffn");
    m.def("get_attention_to_ffn_ccl_buffer_size", &GetAttentionToFfnCclBufferSize,
          "get_attention_to_ffn_ccl_buffer_size");
}
} // namespace op_api
