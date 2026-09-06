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
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
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
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int64_t MB_SIZE = 1024LL * 1024LL;
constexpr int64_t RESERVED_SPACE_SIZE = 10LL * 1024 * 1024;
constexpr int64_t MAX_EXPERTS_PER_RANK_A2A3 = 128LL;
constexpr int64_t SYNC_STATE_RESERVED_SIZE = 512LL * 1024;
constexpr int64_t PEERMEM_MIN_RANK_SYNC_SIZE = 48LL * 1024LL;
constexpr int64_t PEERMEM_SYNC_COUNT_REGION_SIZE = 12LL * 1024LL;
constexpr int64_t PEERMEM_SYNC_SLOT_SIZE = 64LL;
constexpr int64_t MXFP_SCALE_GROUP_NUM = 32LL;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2LL;
constexpr int64_t Y_DTYPE_SIZE = 2LL;
constexpr int64_t URMA_H_ALIGN = 1024LL;
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

// The Torch JIT wheel does not package op_kernel headers. Keep this pure sizing mirror synchronized with
// mc2/mega_moe/op_kernel/arch35/common/mega_moe_peermem.h, which remains the host/device layout source of truth.
int64_t CalcTokenScaleBytesA5(int64_t hidden, int64_t numTopk, int64_t topkWeightsType)
{
    int64_t mxScaleNum = (hidden + ALIGN_32 - 1) / ALIGN_32;
    int64_t dataBytes = CeilAlign(hidden, ALIGN_256);
    int64_t tokenBytes = CeilAlign(dataBytes + mxScaleNum, ALIGN_32);
    if (topkWeightsType == 1) {
        int64_t weightBytes = CeilAlign(numTopk * static_cast<int64_t>(sizeof(float)), ALIGN_32);
        tokenBytes = CeilAlign(tokenBytes + weightBytes, ALIGN_32);
    }
    return tokenBytes;
}

int64_t CalcCombineTokenBytesA5(int64_t hidden, int64_t combineQuantMode)
{
    if (combineQuantMode == 0) {
        return hidden * Y_DTYPE_SIZE;
    }
    int64_t tokenStorageBytes = CeilAlign(hidden, ALIGN_256);
    int64_t scaleCount = (hidden + MXFP_SCALE_GROUP_NUM - 1) / MXFP_SCALE_GROUP_NUM;
    int64_t storedScaleBytes = CeilAlign(scaleCount, MXFP_MULTI_BASE_SIZE);
    return CeilAlign(tokenStorageBytes + storedScaleBytes, ALIGN_32);
}

// Preserve the A5 MTE sizing baseline, including its non-quant combine width and exception-dump prefix.
int64_t CalcMteCclBufferSizeA5(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                               int64_t hidden, int64_t topkWeightsType)
{
    int64_t expertPerRank = moeExpertNum / epWorldSize;

    // Compact route-index receive area.
    int64_t routeIndexAlignSize = CeilAlign(numMaxTokensPerRank * static_cast<int64_t>(sizeof(int32_t)), ALIGN_32);
    int64_t routeRecvSize = CeilAlign(expertPerRank * epWorldSize * routeIndexAlignSize, ALIGN_512);

    // Expert-major raw count table: [localExpert][sourceRank].
    int64_t expertCountRecvSize =
        CeilAlign(expertPerRank * epWorldSize * static_cast<int64_t>(sizeof(int32_t)), ALIGN_512);

    int64_t tokenBytes = CalcTokenScaleBytesA5(hidden, numTopk, topkWeightsType);
    int64_t dispatchRecordAreaSize = CeilAlign(numMaxTokensPerRank * tokenBytes, ALIGN_512);

    int64_t combineSendSize = CeilAlign(numMaxTokensPerRank * numTopk * hidden * Y_DTYPE_SIZE, ALIGN_512);

    int64_t totalBytes = EXCEPTION_DUMP_REGION_SIZE + PEERMEM_DATA_OFFSET + routeRecvSize + expertCountRecvSize +
                         dispatchRecordAreaSize + combineSendSize;

    return totalBytes;
}

// A5 URMA peermem minimum size in bytes. The symmetric layout uses capacity, never the current-rank BS.
int64_t CalcUrmaCclBufferSizeA5(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                                int64_t hidden, int64_t combineQuantMode, int64_t topkWeightsType, int64_t serverNum)
{
    int64_t expertPerRank = moeExpertNum / epWorldSize;
    int64_t rankSyncSize = epWorldSize * PEERMEM_SYNC_SLOT_SIZE;
    int64_t dataOffset =
        CeilAlign(std::max(rankSyncSize, PEERMEM_MIN_RANK_SYNC_SIZE) + PEERMEM_SYNC_COUNT_REGION_SIZE, ALIGN_512);

    int64_t routeCapacity = numMaxTokensPerRank * numTopk;
    int64_t alignedRouteCount = CeilAlign(routeCapacity * static_cast<int64_t>(sizeof(int32_t)), ALIGN_256) /
                                static_cast<int64_t>(sizeof(int32_t));
    int64_t maskAlignSize = CeilAlign(alignedRouteCount / 8, ALIGN_32);
    int64_t maskRecvSize = CeilAlign(expertPerRank * epWorldSize * (maskAlignSize + ALIGN_32), ALIGN_512);
    int64_t expertCountRecvSize =
        CeilAlign(expertPerRank * epWorldSize * static_cast<int64_t>(sizeof(int32_t)), ALIGN_512);

    int64_t tokenBytes = CalcTokenScaleBytesA5(hidden, numTopk, topkWeightsType);
    int64_t relayRecordBytes = CeilAlign(tokenBytes, ALIGN_512);
    int64_t relayDataSize = CeilAlign(numMaxTokensPerRank * relayRecordBytes * serverNum, ALIGN_512);
    int64_t relayFlagSize =
        CeilAlign(serverNum * numMaxTokensPerRank * static_cast<int64_t>(sizeof(uint64_t)), ALIGN_512);

    int64_t combineTokenBytes = CalcCombineTokenBytesA5(hidden, combineQuantMode);
    int64_t combineSendSize = CeilAlign(numMaxTokensPerRank * numTopk * combineTokenBytes, ALIGN_512);

    return dataOffset + maskRecvSize + expertCountRecvSize + relayDataSize + relayFlagSize + combineSendSize;
}
} // namespace

int64_t GetMegaMoeCclBufferSize(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                                int64_t hidden, int64_t maxRecvTokenNum, int64_t dispatchQuantMode,
                                c10::optional<int64_t> dispatchQuantOutDtype, int64_t combineQuantMode,
                                std::string commAlg, int64_t topkWeightsType, int64_t serverNum)
{
    // Zero selects the initial MTE layout; a confirmed cross-server context supplies the actual count (> 1).
    TORCH_CHECK(serverNum >= 0, "server_num must be non-negative, but got ", serverNum);
    const char *socName = aclrtGetSocName();
    bool isA2 = (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr);
    bool isA3 = (socName != nullptr && std::strstr(socName, "Ascend910_93") != nullptr);
    if (isA2 || isA3) {
        TORCH_CHECK(serverNum == 0, "server_num is only supported by the Ascend950 channel backend");
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

    // A5 / 950 checks are aligned with the arch35 host tiling contract.
    TORCH_CHECK(epWorldSize >= 2 && epWorldSize <= 1024, "ep_world_size only support in [2, 1024], but got ",
                epWorldSize);
    TORCH_CHECK(hidden >= 1024 && hidden <= 8192, "hidden only support in [1024, 8192], but got ", hidden);
    int64_t hiddenAlignment = serverNum > 0 ? URMA_H_ALIGN : ALIGN_32;
    TORCH_CHECK(hidden % hiddenAlignment == 0, "hidden must be a multiple of ", hiddenAlignment,
                " for the selected communication topology, but got ", hidden);
    TORCH_CHECK(
        numMaxTokensPerRank >= 1 && static_cast<uint64_t>(numMaxTokensPerRank) <= std::numeric_limits<uint32_t>::max(),
        "num_max_tokens_per_rank should be in [1, UINT32_MAX], but got ", numMaxTokensPerRank);
    TORCH_CHECK(maxRecvTokenNum >= 0, "max_recv_token_num should be non-negative, but got ", maxRecvTokenNum);
    TORCH_CHECK(moeExpertNum >= epWorldSize && moeExpertNum <= 2048 && moeExpertNum % epWorldSize == 0,
                "moe_expert_num should be in [ep_world_size, 2048] and divisible by ep_world_size, but got ",
                moeExpertNum, " and ep_world_size ", epWorldSize);
    TORCH_CHECK(numTopk >= 1 && numTopk <= 32, "num_topk only support in [1, 32], but got ", numTopk);
    TORCH_CHECK(topkWeightsType == 0 || topkWeightsType == 1, "topk_weights_type only support 0 or 1, but got ",
                topkWeightsType);
    TORCH_CHECK(combineQuantMode == 0 || combineQuantMode == 3 || combineQuantMode == 4,
                "combine_quant_mode only support 0, 3 or 4 on Ascend950, but got ", combineQuantMode);

    if (serverNum > 0) {
        TORCH_CHECK(serverNum > 1 && serverNum <= epWorldSize && epWorldSize % serverNum == 0,
                    "server_num should be in [2, ep_world_size] and divide ep_world_size, but got ", serverNum,
                    " and ep_world_size ", epWorldSize);
        int64_t routeCapacity = numMaxTokensPerRank * numTopk;
        TORCH_CHECK(routeCapacity <= std::numeric_limits<int32_t>::max(),
                    "num_max_tokens_per_rank * num_topk should be <= INT32_MAX for URMA, but got ", routeCapacity);
        int64_t expertPerRank = moeExpertNum / epWorldSize;
        int64_t maxOutputCapacity = numMaxTokensPerRank * epWorldSize * std::min(numTopk, expertPerRank);
        TORCH_CHECK(maxOutputCapacity <= std::numeric_limits<int32_t>::max(),
                    "maximum receive token capacity should be <= INT32_MAX for URMA, but got ", maxOutputCapacity);
        TORCH_CHECK(maxRecvTokenNum <= maxOutputCapacity,
                    "max_recv_token_num should not exceed the URMA maximum receive token capacity ", maxOutputCapacity,
                    ", but got ", maxRecvTokenNum);
        return CalcUrmaCclBufferSizeA5(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden,
                                       combineQuantMode, topkWeightsType, serverNum);
    }
    return CalcMteCclBufferSizeA5(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden, topkWeightsType);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_mega_moe", &NpuMegaMoe, "npu_mega_moe");
    m.def("get_mega_moe_ccl_buffer_size", &GetMegaMoeCclBufferSize, "get_mega_moe_ccl_buffer_size",
          py::arg("ep_world_size"), py::arg("moe_expert_num"), py::arg("num_max_tokens_per_rank"), py::arg("num_topk"),
          py::arg("hidden"), py::arg("max_recv_token_num"), py::arg("dispatch_quant_mode"),
          py::arg("dispatch_quant_out_dtype"), py::arg("combine_quant_mode"), py::arg("comm_alg"),
          py::arg("topk_weights_type"), py::arg("server_num") = 0);
}

} // namespace op_api
