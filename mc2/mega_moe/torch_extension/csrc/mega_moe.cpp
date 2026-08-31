// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <torch/extension.h>
#include <cstring>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;

std::tuple<at::Tensor, at::Tensor> NpuMegaMoe(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &topkIds, const at::Tensor &topkWeights,
    const std::vector<at::Tensor> &weight1, const std::vector<at::Tensor> &weight2, int64_t moeExpertNum,
    int64_t epWorldSize, int64_t cclBufferSize, const c10::optional<std::vector<at::Tensor>> &weightScales1,
    const c10::optional<std::vector<at::Tensor>> &weightScales2, const c10::optional<std::vector<at::Tensor>> &bias1,
    const c10::optional<std::vector<at::Tensor>> &bias2, const c10::optional<at::Tensor> &xActiveMask,
    const c10::optional<std::vector<at::Tensor>> &sharedWeight1,
    const c10::optional<std::vector<at::Tensor>> &sharedWeight2,
    const c10::optional<std::vector<at::Tensor>> &sharedWeightScales1,
    const c10::optional<std::vector<at::Tensor>> &sharedWeightScales2,
    const c10::optional<std::vector<at::Tensor>> &sharedBias1,
    const c10::optional<std::vector<at::Tensor>> &sharedBias2, const c10::optional<at::Tensor> &maskBuffer,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t combineQuantMode, std::string commAlg,
    int64_t numMaxTokensPerRank, std::string activation, std::vector<float> activationParams,
    c10::optional<int64_t> dispatchQuantOutDtype, c10::optional<int64_t> weight1Type,
    c10::optional<int64_t> weight2Type, c10::optional<int64_t> topoType, c10::optional<int64_t> rankNumPerServer,
    int64_t topkWeightsType)
{
    TORCH_CHECK((epWorldSize > 0), "The ep_world_sizes should be greater than 0, current is: ", epWorldSize);
    TORCH_CHECK((x.dim() == DIM_TWO) && (topkIds.dim() == DIM_TWO), "The x and topk_ids should be 2D");
    TORCH_CHECK(
        ((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)) && (topkIds.scalar_type() == at::kInt),
        "dtype of x should be bfloat16, float16, dtype of topk_ids should be int.");
    if (maskBuffer.has_value()) {
        const at::Tensor &mask = maskBuffer.value();
        TORCH_CHECK(mask.scalar_type() == at::kInt, "mask_buffer dtype must be int32.");
        TORCH_CHECK(mask.dim() == 1 && mask.numel() == epWorldSize, "mask_buffer shape must be [ep_world_size].");
        TORCH_CHECK(mask.device() == x.device(), "mask_buffer must be on the same device as x.");
        TORCH_CHECK(mask.is_contiguous(), "mask_buffer must be contiguous.");
    }

    at::TensorList weight1Ref = weight1;
    at::TensorList weight2Ref = weight2;

    auto toTensorList = [](const c10::optional<std::vector<at::Tensor>> &opt) -> at::TensorList {
        return opt.has_value() ? at::TensorList(opt.value()) : at::TensorList();
    };
    at::TensorList weightScales1Ref = toTensorList(weightScales1);
    at::TensorList weightScales2Ref = toTensorList(weightScales2);
    at::TensorList bias1Ref = toTensorList(bias1);
    at::TensorList bias2Ref = toTensorList(bias2);

    aclDataType weight1RefDtype = weight1Type.has_value() ? GetAclDataType(weight1Type.value()) :
                                                            ConvertToAclDataType(weight1Ref[0].scalar_type());
    aclDataType weightScales1Dtype;
    if (weight1RefDtype == aclDataType::ACL_FLOAT8_E5M2 || weight1RefDtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight1RefDtype == aclDataType::ACL_FLOAT4_E2M1) {
        weightScales1Dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weightScales1Dtype = aclDataType::ACL_UINT64;
    }

    aclDataType weight2RefDtype = weight2Type.has_value() ? GetAclDataType(weight2Type.value()) :
                                                            ConvertToAclDataType(weight2Ref[0].scalar_type());
    aclDataType weightScales2Dtype;
    if (weight2RefDtype == aclDataType::ACL_FLOAT8_E5M2 || weight2RefDtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight2RefDtype == aclDataType::ACL_FLOAT4_E2M1) {
        weightScales2Dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weightScales2Dtype = aclDataType::ACL_UINT64;
    }

    auto xSize = x.sizes();
    auto topkIdsSize = topkIds.sizes();
    int64_t bs = xSize[0];
    int64_t h = xSize[1];
    int64_t k = topkIdsSize[1];

    if ((dispatchQuantOutDtype.has_value()) &&
        (dispatchQuantOutDtype.value() == static_cast<int64_t>(DType::FLOAT4_E2M1))) {
        TORCH_CHECK(h % 2 == 0, "The last dim input shape must be divisible by 2 if "
                                "dispatch quant output type is torch_npu.float4_e2m1");
    }

    int64_t localMoeExpertNum = 1;
    localMoeExpertNum = moeExpertNum / epWorldSize;
    at::Tensor expertTokenNums;
    expertTokenNums = at::empty({localMoeExpertNum}, x.options().dtype(at::kInt));

    std::string commAlgStr = std::string(commAlg);
    char *commAlgPtr = const_cast<char *>(commAlg.c_str());

    std::string activationStr = std::string(activation);
    char *activationPtr = const_cast<char *>(activationStr.c_str());

    // The eager and graph Python paths validate and serialize activationParams before calling C++.
    int64_t topoTypeValue = topoType.value_or(0);
    int64_t rankNumPerServerValue = rankNumPerServer.value_or(2);

    int64_t dispatchQuantResultType =
        dispatchQuantOutDtype.has_value() ? static_cast<int64_t>(GetAclDataType(dispatchQuantOutDtype.value())) : 28;

    at::Tensor y;
    y = at::empty({bs, h}, topkIds.options().dtype(x.scalar_type()));

    TensorListWrapper weight1Wrapper = {weight1Ref, weight1RefDtype};
    TensorListWrapper weight2Wrapper = {weight2Ref, weight2RefDtype};
    TensorListWrapper weightScales1Wrapper = {weightScales1Ref, weightScales1Dtype};
    TensorListWrapper weightScales2Wrapper = {weightScales2Ref, weightScales2Dtype};
    TensorListWrapper bias1Wrapper = {bias1Ref, aclDataType::ACL_FLOAT};
    TensorListWrapper bias2Wrapper = {bias2Ref, aclDataType::ACL_FLOAT};

    at::TensorList sharedWeight1Ref = toTensorList(sharedWeight1);
    at::TensorList sharedWeight2Ref = toTensorList(sharedWeight2);
    at::TensorList sharedWeightScales1Ref = toTensorList(sharedWeightScales1);
    at::TensorList sharedWeightScales2Ref = toTensorList(sharedWeightScales2);
    at::TensorList sharedBias1Ref = toTensorList(sharedBias1);
    at::TensorList sharedBias2Ref = toTensorList(sharedBias2);

    TensorListWrapper sharedWeight1Wrapper = {sharedWeight1Ref, weight1RefDtype};
    TensorListWrapper sharedWeight2Wrapper = {sharedWeight2Ref, weight2RefDtype};
    TensorListWrapper sharedWeightScales1Wrapper = {sharedWeightScales1Ref, weightScales1Dtype};
    TensorListWrapper sharedWeightScales2Wrapper = {sharedWeightScales2Ref, weightScales2Dtype};
    TensorListWrapper sharedBias1Wrapper = {sharedBias1Ref, aclDataType::ACL_FLOAT};
    TensorListWrapper sharedBias2Wrapper = {sharedBias2Ref, aclDataType::ACL_FLOAT};

    ACLNN_CMD(aclnnMegaMoe, context, x, topkIds, topkWeights, weight1Wrapper, weight2Wrapper, weightScales1Wrapper,
              weightScales2Wrapper, bias1Wrapper, bias2Wrapper, xActiveMask, sharedWeight1Wrapper, sharedWeight2Wrapper,
              sharedWeightScales1Wrapper, sharedWeightScales2Wrapper, sharedBias1Wrapper, sharedBias2Wrapper,
              maskBuffer, moeExpertNum, epWorldSize, cclBufferSize, maxRecvTokenNum, dispatchQuantMode,
              dispatchQuantResultType, combineQuantMode, commAlgPtr, numMaxTokensPerRank, activationPtr,
              activationParams, topoTypeValue, rankNumPerServerValue, topkWeightsType, y, expertTokenNums);

    return std::tie(y, expertTokenNums);
}

namespace {
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int64_t MB_SIZE = 1024LL * 1024LL;
constexpr int64_t RESERVED_SPACE_SIZE = 10LL * 1024 * 1024;
constexpr int64_t MAX_EXPERTS_PER_RANK_A2A3 = 128LL;
constexpr int64_t SYNC_STATE_RESERVED_SIZE = 512LL * 1024;
// 异常 Dump 区
constexpr int64_t EXCEPTION_DUMP_REGION_SIZE = 60LL * 1024LL;
// rankSyncInWorld 同步区
constexpr int64_t PEERMEM_DATA_OFFSET = 60LL * 1024LL;

int64_t CeilAlign(int64_t val, int64_t align)
{
    return (val + align - 1) / align * align;
}

// A2 minimum buffer size (MB).
// Matches tiling_arch22.cpp CalcLeastCclBufferSize with isA3=false.
int64_t CalcLeastCclBufferSizeA2(int64_t maxRecvTokenNum, int64_t h, int64_t epWorldSize, bool isQuantRouting,
                                 int64_t bs, int64_t topK)
{
    // Data block 1: TokenPerExpert
    // EP × CeilAlign(EP × MAX_EXPERTS_PER_RANK_A2A3 + 1, 128) × 4B
    int64_t offsetTokenPerExpert = epWorldSize * CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
                                   static_cast<int64_t>(sizeof(int32_t));

    // Data block 2: tensors
    // ===== winIn =====
    int64_t offsetAAfterDispatch =
        maxRecvTokenNum * (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    int64_t offsetD = bs * topK * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winInTensorSize = offsetAAfterDispatch + offsetD;

    // ===== winOut =====
    int64_t offsetA = bs * topK * (!isQuantRouting ? h * static_cast<int64_t>(sizeof(int16_t)) : (h + ALIGN_512));
    int64_t offsetC = maxRecvTokenNum * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winOutTensorSize = offsetA + offsetC;
    int64_t offsetTensor = std::max(winInTensorSize, winOutTensorSize);
    if (isQuantRouting) {
        offsetTensor += maxRecvTokenNum * static_cast<int64_t>(sizeof(float));
    }

    // Data block 3: sync flags
    int64_t offsetFlag = epWorldSize * ALIGN_512;                 // CrossRankSync
    offsetFlag += epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 * 64LL; // DispatchFlag
    offsetFlag += epWorldSize * 64LL;                             // AllGatherFlag

    return (offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE + MB_SIZE) / MB_SIZE;
}

// A3 minimum buffer size (MB).
// Matches tiling_arch22.cpp CalcLeastCclBufferSize with isA3=true.
int64_t CalcLeastCclBufferSizeA3(int64_t h, int64_t epWorldSize, bool isQuantRouting, int64_t bs, int64_t topK)
{
    // Data block 1: TokenPerExpert
    // EP × CeilAlign(EP × MAX_EXPERTS_PER_RANK_A2A3 + 1, 128) × 4B
    int64_t offsetTokenPerExpert = epWorldSize * CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
                                   static_cast<int64_t>(sizeof(int32_t));

    // Data block 2: tensors (winIn only, no winOut)
    int64_t offsetAAfterDispatch =
        bs * topK * (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    int64_t offsetD = bs * topK * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t offsetTensor = offsetAAfterDispatch + offsetD;
    if (isQuantRouting) {
        offsetTensor += bs * topK * static_cast<int64_t>(sizeof(float));
    }

    // Data block 3: sync flags
    int64_t offsetFlag = std::max(epWorldSize * ALIGN_512, SYNC_STATE_RESERVED_SIZE);

    return (offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE + MB_SIZE) / MB_SIZE;
}

// A5 half-buffer minimum size (MB). Ported 1:1 from the original Python implementation.
int64_t CalcHalfBufferSizeMBA5(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                               int64_t hidden, int64_t topkWeightsType)
{
    int64_t expertPerRank = moeExpertNum / epWorldSize;

    // mask_recv_size
    int64_t compareCount = CeilAlign(numMaxTokensPerRank * numTopk * 4, 256) / 4;
    int64_t maskAlignSize = CeilAlign(compareCount / 8, 32);
    int64_t maskSlotSize = maskAlignSize + 32;
    int64_t maskRecvSize = CeilAlign(expertPerRank * epWorldSize * maskSlotSize, 512);

    // Expert-major raw count table: [localExpert][sourceRank].
    int64_t expertCountRecvSize = CeilAlign(expertPerRank * epWorldSize * static_cast<int64_t>(sizeof(int32_t)), 512);

    // quant_token_scale_size
    int64_t mxScaleNum = (hidden + 31) / 32;
    int64_t dataBytes = CeilAlign(hidden, 256);
    int64_t tokenBytes = CeilAlign(dataBytes + mxScaleNum, 32);
    if (topkWeightsType == 1) {
        int64_t weightBytes = CeilAlign(numTopk * static_cast<int64_t>(sizeof(float)), 32);
        tokenBytes = CeilAlign(tokenBytes + weightBytes, 32);
    }
    int64_t quantTokenScaleSize = CeilAlign(numMaxTokensPerRank * tokenBytes, 512);

    // combine_send_size
    int64_t combineOut = CeilAlign(numMaxTokensPerRank * hidden * numTopk * 2, 512);

    int64_t totalBytes = EXCEPTION_DUMP_REGION_SIZE + PEERMEM_DATA_OFFSET + maskRecvSize + expertCountRecvSize +
                         quantTokenScaleSize + combineOut;

    return totalBytes;
}
} // namespace

int64_t GetMegaMoeCclBufferSize(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                                int64_t hidden, int64_t maxRecvTokenNum, int64_t dispatchQuantMode,
                                c10::optional<int64_t> dispatchQuantOutDtype, int64_t combineQuantMode,
                                std::string commAlg, int64_t topkWeightsType)
{
    const char *socName = aclrtGetSocName();
    bool isA2 = (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr);
    bool isA3 = (socName != nullptr && std::strstr(socName, "Ascend910_93") != nullptr);
    if (isA2 || isA3) {
        TORCH_CHECK(epWorldSize == 2 || epWorldSize == 4 || epWorldSize == 8 || epWorldSize == 16 ||
                        epWorldSize == 32 || epWorldSize == 64 || epWorldSize == 128,
                    "ep_world_size only support {2, 4, 8, 16, 32, 64, 128} on A2/A3, but got ", epWorldSize);
        TORCH_CHECK(hidden >= 1024 && hidden <= 8192 && hidden % 512 == 0,
                    "hidden only support [1024, 8192] and hidden % 512 == 0 on A2/A3, but got ", hidden);
        TORCH_CHECK(numMaxTokensPerRank >= 1 && numMaxTokensPerRank <= 4096,
                    "num_max_tokens_per_rank only support [1, 4096] on A2/A3, but got ", numMaxTokensPerRank);
        TORCH_CHECK(moeExpertNum >= 1 && moeExpertNum <= 2048,
                    "moe_expert_num only support [1, 2048] on A2/A3, but got ", moeExpertNum);
        TORCH_CHECK(numTopk >= 1 && numTopk <= 16, "num_topk only support [1, 16] on A2/A3, but got ", numTopk);
        TORCH_CHECK(dispatchQuantMode == 0 || dispatchQuantMode == 2 || dispatchQuantMode == 4,
                    "dispatch_quant_mode only support {0, 2, 4} on A2/A3, but got ", dispatchQuantMode);

        bool isQuantRouting = (dispatchQuantMode == 4);
        // max_recv_token_num 为 0 时自动计算为 bs * epWorldSize * min(topK, expertPerRank)，
        if (maxRecvTokenNum == 0) {
            int64_t expertPerRank = moeExpertNum / epWorldSize;
            maxRecvTokenNum = numMaxTokensPerRank * epWorldSize * std::min(numTopk, expertPerRank);
        }
        if (isA3) {
            return CalcLeastCclBufferSizeA3(hidden, epWorldSize, isQuantRouting, numMaxTokensPerRank, numTopk);
        }
        return CalcLeastCclBufferSizeA2(maxRecvTokenNum, hidden, epWorldSize, isQuantRouting, numMaxTokensPerRank,
                                        numTopk);
    }

    // A5 / 950 — 校验与原 Python get_mega_moe_ccl_buffer_size 对齐
    TORCH_CHECK(epWorldSize >= 2 && epWorldSize <= 1024, "ep_world_size only support in [2, 1024], but got ",
                epWorldSize);
    TORCH_CHECK(hidden >= 1024 && hidden <= 8192, "hidden only support in [1024, 8192], but got ", hidden);
    TORCH_CHECK(numMaxTokensPerRank >= 1, "num_max_tokens_per_rank should be >= 1, but got ", numMaxTokensPerRank);
    TORCH_CHECK(moeExpertNum >= 1 && moeExpertNum <= 2048, "moe_expert_num only support in [1, 2048], but got ",
                moeExpertNum);
    TORCH_CHECK(numTopk >= 1 && numTopk <= 32, "num_topk only support in [1, 32], but got ", numTopk);

    return CalcHalfBufferSizeMBA5(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden, topkWeightsType);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_mega_moe", &NpuMegaMoe, "npu_mega_moe");
    m.def("get_mega_moe_ccl_buffer_size", &GetMegaMoeCclBufferSize, "get_mega_moe_ccl_buffer_size");
}

} // namespace op_api
