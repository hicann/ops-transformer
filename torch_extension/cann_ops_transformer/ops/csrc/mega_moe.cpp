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

std::tuple<at::Tensor, at::Tensor> npu_mega_moe(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &topk_ids,
    const at::Tensor &topk_weights, const std::vector<at::Tensor> &weight1,
    const std::vector<at::Tensor> &weight2, int64_t moe_expert_num, int64_t ep_world_size,
    int64_t ccl_buffer_size,
    const c10::optional<std::vector<at::Tensor>> &weight_scales1,
    const c10::optional<std::vector<at::Tensor>> &weight_scales2,
    const c10::optional<std::vector<at::Tensor>> &bias1,
    const c10::optional<std::vector<at::Tensor>> &bias2,
    const c10::optional<at::Tensor> &x_active_mask,
    int64_t max_recv_token_num,
    int64_t dispatch_quant_mode,
    int64_t combine_quant_mode,
    std::string comm_alg,
    int64_t num_max_tokens_per_rank,
    std::string activation,
    c10::optional<float> activation_clamp,
    c10::optional<int64_t> dispatch_quant_out_dtype,
    c10::optional<int64_t> weight1_type,
    c10::optional<int64_t> weight2_type)
{
    TORCH_CHECK((ep_world_size > 0), "The ep_world_sizes should be greater than 0, current is: ", ep_world_size);
    TORCH_CHECK((x.dim() == DIM_TWO) && (topk_ids.dim() == DIM_TWO), "The x and topk_ids should be 2D");
    TORCH_CHECK(((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)) &&
                (topk_ids.scalar_type() == at::kInt),
                "dtype of x should be bfloat16, float16, dtype of topk_ids should be int.");

    at::TensorList weight1_ref = weight1;
    at::TensorList weight2_ref = weight2;

    at::TensorList weight_scales1_ref;
    if (weight_scales1.has_value()) {
        weight_scales1_ref = at::TensorList(weight_scales1.value());
    } else {
        weight_scales1_ref = at::TensorList();
    }
    at::TensorList weight_scales2_ref;
    if (weight_scales2.has_value()) {
        weight_scales2_ref = at::TensorList(weight_scales2.value());
    } else {
        weight_scales2_ref = at::TensorList();
    }
    at::TensorList bias1_ref;
    if (bias1.has_value()) {
        bias1_ref = at::TensorList(bias1.value());
    } else {
        bias1_ref = at::TensorList();
    }
    at::TensorList bias2_ref;
    if (bias2.has_value()) {
        bias2_ref = at::TensorList(bias2.value());
    } else {
        bias2_ref = at::TensorList();
    }

    aclDataType weight1_ref_dtype = weight1_type.has_value() ? GetAclDataType(weight1_type.value())
        : ConvertToAclDataType(weight1_ref[0].scalar_type());
    aclDataType weight_scales1_dtype;
    if (weight1_ref_dtype == aclDataType::ACL_FLOAT8_E5M2 ||
        weight1_ref_dtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight1_ref_dtype == aclDataType::ACL_FLOAT4_E2M1) {
        weight_scales1_dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weight_scales1_dtype = aclDataType::ACL_UINT64;
    }

    aclDataType weight2_ref_dtype = weight2_type.has_value() ? GetAclDataType(weight2_type.value())
        : ConvertToAclDataType(weight2_ref[0].scalar_type());
    aclDataType weight_scales2_dtype;
    if (weight2_ref_dtype == aclDataType::ACL_FLOAT8_E5M2 ||
        weight2_ref_dtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight2_ref_dtype == aclDataType::ACL_FLOAT4_E2M1) {
        weight_scales2_dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weight_scales2_dtype = aclDataType::ACL_UINT64;
    }
    
    auto x_size = x.sizes();
    auto topk_ids_size = topk_ids.sizes();
    int64_t bs = x_size[0];
    int64_t h = x_size[1];
    int64_t k = topk_ids_size[1];

    if ((dispatch_quant_out_dtype.has_value()) &&
        (dispatch_quant_out_dtype.value() == static_cast<int64_t>(DType::FLOAT4_E2M1))) {
        TORCH_CHECK(h % 2 == 0,
                    "The last dim input shape must be divisible by 2 if "
                    "dispatch quant output type is torch_npu.float4_e2m1");
    }

    int64_t local_moe_expert_num = 1;
    local_moe_expert_num = moe_expert_num / ep_world_size;
    at::Tensor expert_token_nums;
    expert_token_nums = at::empty({local_moe_expert_num}, x.options().dtype(at::kInt));

    std::string comm_alg_str = std::string(comm_alg);
    char *comm_alg_ptr = const_cast<char *>(comm_alg.c_str());

    std::string activation_str = std::string(activation);
    char *activation_ptr = const_cast<char *>(activation_str.c_str());

    float activation_clamp_value = activation_clamp.value_or(std::numeric_limits<float>::max());

    int64_t dispatch_quant_result_type = dispatch_quant_out_dtype.has_value()
         ? static_cast<int64_t>(GetAclDataType(dispatch_quant_out_dtype.value()))
         : 28;

    at::Tensor y;
    y = at::empty({bs, h}, topk_ids.options().dtype(x.scalar_type()));

    TensorListWrapper weight1_wrapper = {weight1_ref, weight1_ref_dtype};
    TensorListWrapper weight2_wrapper = {weight2_ref, weight2_ref_dtype};
    TensorListWrapper weight_scales1_wrapper = {weight_scales1_ref, weight_scales1_dtype};
    TensorListWrapper weight_scales2_wrapper = {weight_scales2_ref, weight_scales2_dtype};
    TensorListWrapper bias1_wrapper = {bias1_ref, aclDataType::ACL_FLOAT};
    TensorListWrapper bias2_wrapper = {bias2_ref, aclDataType::ACL_FLOAT};

    ACLNN_CMD(aclnnMegaMoe, context, x, topk_ids, topk_weights, weight1_wrapper,
        weight2_wrapper, weight_scales1_wrapper, weight_scales2_wrapper, bias1_wrapper, bias2_wrapper,
        x_active_mask, moe_expert_num, ep_world_size, ccl_buffer_size, max_recv_token_num,
        dispatch_quant_mode, dispatch_quant_result_type, combine_quant_mode,
        comm_alg_ptr, num_max_tokens_per_rank, activation_ptr, activation_clamp_value,
        y, expert_token_nums);

    return std::tie(y, expert_token_nums);
}

namespace {
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int64_t MB_SIZE = 1024LL * 1024LL;
constexpr int64_t RESERVED_SPACE_SIZE = 10LL * 1024 * 1024;
constexpr int64_t MAX_EXPERTS_PER_RANK_A2A3 = 128LL;
constexpr int64_t SYNC_STATE_RESERVED_SIZE = 512LL * 1024;

int64_t CeilAlign(int64_t val, int64_t align)
{
    return (val + align - 1) / align * align;
}

// A2 minimum buffer size (MB).
// Matches tiling_arch22.cpp CalcLeastCclBufferSize with isA3=false.
int64_t CalcLeastCclBufferSizeA2(int64_t maxRecvTokenNum, int64_t h,
    int64_t epWorldSize, bool isQuantRouting, int64_t bs, int64_t topK)
{
    // Data block 1: TokenPerExpert
    // EP × CeilAlign(EP × MAX_EXPERTS_PER_RANK_A2A3 + 1, 128) × 4B
    int64_t offsetTokenPerExpert = epWorldSize *
        CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
        static_cast<int64_t>(sizeof(int32_t));

    // Data block 2: tensors
    // ===== winIn =====
    int64_t offsetAAfterDispatch = maxRecvTokenNum *
        (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    int64_t offsetD = bs * topK * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winInTensorSize = offsetAAfterDispatch + offsetD;

    // ===== winOut =====
    int64_t offsetA = bs * topK *
        (!isQuantRouting ? h * static_cast<int64_t>(sizeof(int16_t)) : (h + ALIGN_512));
    int64_t offsetC = maxRecvTokenNum * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winOutTensorSize = offsetA + offsetC;
    int64_t offsetTensor = std::max(winInTensorSize, winOutTensorSize);
    if (isQuantRouting) {
        offsetTensor += maxRecvTokenNum * static_cast<int64_t>(sizeof(float));
    }

    // Data block 3: sync flags
    int64_t offsetFlag = epWorldSize * ALIGN_512;  // CrossRankSync
    offsetFlag += epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 * 64LL;  // DispatchFlag
    offsetFlag += epWorldSize * 64LL;  // AllGatherFlag

    return (offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE + MB_SIZE) / MB_SIZE;
}

// A3 minimum buffer size (MB).
// Matches tiling_arch22.cpp CalcLeastCclBufferSize with isA3=true.
int64_t CalcLeastCclBufferSizeA3(int64_t h,
    int64_t epWorldSize, bool isQuantRouting, int64_t bs, int64_t topK)
{
    // Data block 1: TokenPerExpert
    // EP × CeilAlign(EP × MAX_EXPERTS_PER_RANK_A2A3 + 1, 128) × 4B
    int64_t offsetTokenPerExpert = epWorldSize *
        CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
        static_cast<int64_t>(sizeof(int32_t));

    // Data block 2: tensors (winIn only, no winOut)
    int64_t offsetAAfterDispatch = bs * topK *
        (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
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
int64_t CalcHalfBufferSizeMBA5(int64_t epWorldSize, int64_t moeExpertNum,
    int64_t numMaxTokensPerRank, int64_t numTopk, int64_t hidden)
{
    int64_t expertPerRank = moeExpertNum / epWorldSize;

    // 全卡软同步使用 60KB
    int64_t peermemDataOffset = 60LL * 1024LL;

    // mask_recv_size
    int64_t compareCount = CeilAlign(numMaxTokensPerRank * numTopk * 4, 256) / 4;
    int64_t maskAlignSize = CeilAlign(compareCount / 8, 32);
    int64_t maskSlotSize = maskAlignSize + 32;
    int64_t maskRecvSize = CeilAlign(expertPerRank * epWorldSize * maskSlotSize, 512);

    // quant_token_scale_size
    int64_t mxScaleNum = (hidden + 31) / 32;
    int64_t dataBytes = CeilAlign(hidden, 256);
    int64_t tokenBytes = CeilAlign(dataBytes + mxScaleNum, 32);
    int64_t quantTokenScaleSize = CeilAlign(numMaxTokensPerRank * tokenBytes, 512);

    // combine_send_size
    int64_t combineOut = CeilAlign(numMaxTokensPerRank * hidden * numTopk * 2, 512);

    int64_t totalBytes = peermemDataOffset + maskRecvSize + quantTokenScaleSize + combineOut;

    return totalBytes;
}
} // namespace

int64_t GetMegaMoeCclBufferSize(int64_t epWorldSize, int64_t moeExpertNum,
    int64_t numMaxTokensPerRank, int64_t numTopk, int64_t hidden,
    int64_t maxRecvTokenNum,
    int64_t dispatchQuantMode, c10::optional<int64_t> dispatchQuantOutDtype,
    int64_t combineQuantMode, std::string commAlg)
{
    const char *socName = aclrtGetSocName();
    bool isA2 = (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr);
    bool isA3 = (socName != nullptr && std::strstr(socName, "Ascend910_93") != nullptr);
    if (isA2 || isA3) {
        TORCH_CHECK(epWorldSize == 2 || epWorldSize == 4 || epWorldSize == 8 ||
                    epWorldSize == 16 || epWorldSize == 32 || epWorldSize == 64,
            "ep_world_size only support {2, 4, 8, 16, 32, 64} on A2/A3, but got ", epWorldSize);
        TORCH_CHECK(hidden >= 1024 && hidden <= 8192 && hidden % 512 == 0,
            "hidden only support [1024, 8192] and hidden % 512 == 0 on A2/A3, but got ", hidden);
        TORCH_CHECK(numMaxTokensPerRank >= 1 && numMaxTokensPerRank <= 4096,
            "num_max_tokens_per_rank only support [1, 4096] on A2/A3, but got ", numMaxTokensPerRank);
        TORCH_CHECK(moeExpertNum >= 1 && moeExpertNum <= 2048,
            "moe_expert_num only support [1, 2048] on A2/A3, but got ", moeExpertNum);
        TORCH_CHECK(numTopk >= 1 && numTopk <= 16,
            "num_topk only support [1, 16] on A2/A3, but got ", numTopk);
        TORCH_CHECK(dispatchQuantMode == 0 || dispatchQuantMode == 2 || dispatchQuantMode == 4,
            "dispatch_quant_mode only support {0, 2, 4} on A2/A3, but got ", dispatchQuantMode);

        bool isQuantRouting = (dispatchQuantMode == 4);
        // max_recv_token_num 为 0 时自动计算为 bs * epWorldSize * min(topK, expertPerRank)，
        if (maxRecvTokenNum == 0) {
            int64_t expertPerRank = moeExpertNum / epWorldSize;
            maxRecvTokenNum = numMaxTokensPerRank * epWorldSize * std::min(numTopk, expertPerRank);
        }
        if (isA3) {
            return CalcLeastCclBufferSizeA3(
                hidden, epWorldSize, isQuantRouting,
                numMaxTokensPerRank, numTopk);
        }
        return CalcLeastCclBufferSizeA2(
            maxRecvTokenNum, hidden, epWorldSize, isQuantRouting,
            numMaxTokensPerRank, numTopk);
    }

    // A5 / 950
    TORCH_CHECK(epWorldSize >= 2 && epWorldSize <= 768,
        "ep_world_size only support in [2, 768], but got ", epWorldSize);
    TORCH_CHECK(hidden >= 1024 && hidden <= 8192,
        "hidden only support in [1024, 8192], but got ", hidden);
    TORCH_CHECK(numMaxTokensPerRank >= 1 && numMaxTokensPerRank <= 512,
        "num_max_tokens_per_rank only support in [1, 512], but got ", numMaxTokensPerRank);
    TORCH_CHECK(moeExpertNum >= 1 && moeExpertNum <= 1024,
        "moe_expert_num only support in [1, 1024], but got ", moeExpertNum);
    TORCH_CHECK(numTopk >= 1 && numTopk <= 16,
        "num_topk only support in [1, 16], but got ", numTopk);

    return CalcHalfBufferSizeMBA5(epWorldSize, moeExpertNum,
        numMaxTokensPerRank, numTopk, hidden);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_mega_moe", &npu_mega_moe, "npu_mega_moe");
    m.def("get_mega_moe_ccl_buffer_size", &GetMegaMoeCclBufferSize, "get_mega_moe_ccl_buffer_size");
}

} // namespace op_api
