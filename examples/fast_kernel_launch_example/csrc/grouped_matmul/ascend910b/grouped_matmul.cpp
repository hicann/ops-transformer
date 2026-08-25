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
 * \file grouped_matmul.cpp
 * \brief
 */
#include "grouped_matmul_tiling_common.h"
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "platform/platform_ascendc.h"
#include <type_traits>

#include "op_kernel/grouped_matmul_kernel.h"

namespace ascend_ops {
namespace GroupedMatmul {

namespace {
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_X = {at::kBFloat16};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_WEIGHT = {at::kBFloat16};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_BIAS = {at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_SCALE = {at::kLong};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OFFSET = {at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_ANTIQUANTSCALE = {at::kHalf};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_ANTIQUANTOFFSET = {at::kHalf};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_GROUPLIST = {at::kLong};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_PERTOKENSCALE = {at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OUTPUT = {at::kBFloat16};

const std::vector<TypeCombo> &getSupportedCombos()
{
    static const auto combos = TypeComboManager::createCombosFromLists(
        SUPPORTED_DTYPE_X, SUPPORTED_DTYPE_BIAS, SUPPORTED_DTYPE_SCALE, SUPPORTED_DTYPE_OFFSET,
        SUPPORTED_DTYPE_ANTIQUANTSCALE, SUPPORTED_DTYPE_ANTIQUANTOFFSET, SUPPORTED_DTYPE_GROUPLIST,
        SUPPORTED_DTYPE_PERTOKENSCALE, SUPPORTED_DTYPE_WEIGHT, SUPPORTED_DTYPE_OUTPUT);
    return combos;
}
} // namespace

// Register the operator's schema
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("grouped_matmul(Tensor[] x, Tensor[] weight, Tensor[]? bias, Tensor[]? scale, Tensor[]? offset, Tensor[]? "
          "antiquantScale, Tensor[]? antiquantOffset, Tensor? groupList, Tensor[]? perTokenScale, int splitItem, int "
          "groupType, int groupListType, int actType,int[]? tuningConfigOptional) -> Tensor");
}

__global__ __aicore__ void groupedmatmul_kernel(__gm__ uint8_t *x, __gm__ uint8_t *weight, __gm__ uint8_t *bias,
                                                __gm__ uint8_t *scale, __gm__ uint8_t *offset,
                                                __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                                __gm__ uint8_t *groupList, __gm__ uint8_t *perTokenScale,
                                                __gm__ uint8_t *y, __gm__ uint8_t *workspace,
                                                GroupedMatmulTilingData tilingData)
{
    GroupedMatmulKernelImpl<GMM_TPL_BF16, GMM_TPL_BF16, GMM_TPL_BF16, 0, 0, 1, 0, 0, 0, 0, 0>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, y, workspace,
        &tilingData);
}

void groupedmatmul_api(const TypeCombo &matchedCombo, int comboIndex, aclrtStream stream, const at::TensorList &x,
                       const at::TensorList &weight, const c10::optional<at::TensorList> &bias,
                       const c10::optional<at::TensorList> &scale, const c10::optional<at::TensorList> &offset,
                       const c10::optional<at::TensorList> &antiquantScale,
                       const c10::optional<at::TensorList> &antiquantOffset,
                       const c10::optional<torch::Tensor> &groupList,
                       const c10::optional<at::TensorList> &perTokenScale, const at::TensorList &y,
                       const int64_t splitItem, const int64_t groupType, const int64_t groupListType,
                       const int64_t actType, const vector<int64_t> *tuningConfigOptional)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatFrom;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatFrom);
    GroupedMatmulTilingData tilingData;
    GroupedMatmulNs::GroupedMatmulTiling::GroupedMatmulCommonTiling<at::TensorList, c10::optional<at::TensorList>,
                                                                    c10::optional<torch::Tensor>>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, tilingData,
        ascendcPlatform->GetCoreNumAic(), ubSizePlatFrom);
    uint32_t blockDim = tilingData.gmmBaseParams.get_coreNum();
    auto x_ptr = get_first_tensor_address<at::TensorList>(matchedCombo.x, x, false);
    auto weight_ptr = get_first_tensor_address<at::TensorList>(matchedCombo.weight, weight, false);
    auto y_ptr = get_first_tensor_address<at::TensorList>(matchedCombo.output, y, false);
    auto bias_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.bias, bias, true);
    auto scale_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.scale, scale, true);
    auto offset_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.offset, offset, true);
    auto antiquantScale_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.antiquantScale, antiquantScale, true);
    auto antiquantOffset_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.antiquantOffset, antiquantOffset, true);
    auto groupList_ptr =
        get_first_tensor_address<c10::optional<torch::Tensor>>(matchedCombo.groupList, groupList, true);
    auto perTokenScale_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.perTokenScale, perTokenScale, true);
    uint64_t workspaceSize = 16U * 1024U * 1024U;
    void *workspace_ptr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace_ptr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d\n", ret);
    }
    groupedmatmul_kernel<<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t *)x_ptr, (__gm__ uint8_t *)weight_ptr, (__gm__ uint8_t *)bias_ptr, (__gm__ uint8_t *)scale_ptr,
        (__gm__ uint8_t *)offset_ptr, (__gm__ uint8_t *)antiquantScale_ptr, (__gm__ uint8_t *)antiquantOffset_ptr,
        (__gm__ uint8_t *)groupList_ptr, (__gm__ uint8_t *)perTokenScale_ptr, (__gm__ uint8_t *)y_ptr,
        (__gm__ uint8_t *)workspace_ptr, tilingData);

    // Free workspace memory
    if (workspaceSize > 0 && workspace_ptr != nullptr) {
        aclrtFree(workspace_ptr);
        workspace_ptr = nullptr;
    }
}

torch::Tensor grouped_matmul_npu(
    const torch::TensorList &x, const torch::TensorList &weight, const c10::optional<torch::TensorList> &bias,
    const c10::optional<torch::TensorList> &scale, const c10::optional<torch::TensorList> &offset,
    const c10::optional<torch::TensorList> &antiquantScale, const c10::optional<torch::TensorList> &antiquantOffset,
    const c10::optional<torch::Tensor> &groupList, const c10::optional<torch::TensorList> &perTokenScale,
    const int64_t splitItem, const int64_t groupType, const int64_t groupListType, const int64_t actType,
    const c10::optional<c10::IntArrayRef> &tuningConfigOptional)
{
    // OptionalDeviceGuard 确保后续操作在正确的设备上下文执行
    // 它会记录当前设备状态，执行完作用域代码后自动恢复
    const c10::OptionalDeviceGuard guard(x[0].device());
    checkTensorOnNPU(x, "x", false);
    checkTensorOnNPU(weight, "weight", false);
    checkTensorOnNPU(bias, "bias", true);
    checkTensorOnNPU(scale, "scale", true);
    checkTensorOnNPU(offset, "offset", true);
    checkTensorOnNPU(antiquantScale, "antiquantScale", true);
    checkTensorOnNPU(antiquantOffset, "antiquantOffset", true);
    checkTensorOnNPU(perTokenScale, "perTokenScale", true);
    checkTensorOnNPU(groupList, "groupList", true);

    const auto &SUPPORTED_COMBOS = getSupportedCombos();

    int matched_index = TypeComboManager::findMatchingCombo(SUPPORTED_COMBOS, x, weight, bias, scale, offset,
                                                            antiquantScale, antiquantOffset, groupList, perTokenScale);

    if (matched_index == -1) {
        TORCH_CHECK(false, "no match dtype combo");
    }

    const auto &matched_combo = SUPPORTED_COMBOS[matched_index];
    const c10::ScalarType output_type = matched_combo.output;
    auto shapeX = getTensorListFirstShape(x);
    int64_t m = shapeX[0];
    auto shapeWeight = getTensorListFirstShape(weight);
    int64_t n = shapeWeight[1];

    at::Tensor y = at::empty({m, n},                    // 指定目标shape [M, N]
                             at::dtype(output_type)     // 数据类型
                                 .device(x[0].device()) // 对齐x[0]的设备（CPU/GPU/Ascend）
                                 .layout(x[0].layout()) // 对齐x[0]的内存布局
    );

    // RunOpApi enqueues acl_call asynchronously. TensorList and IntArrayRef are
    // non-owning views, so materialize their contents before creating the callback.
    const auto materialize_tensor_list = [](const at::TensorList &tensor_list) {
        return std::vector<at::Tensor>(tensor_list.begin(), tensor_list.end());
    };
    const auto materialize_optional_tensor_list =
        [&materialize_tensor_list](const c10::optional<at::TensorList> &tensor_list) {
            c10::optional<std::vector<at::Tensor>> result = c10::nullopt;
            if (tensor_list.has_value()) {
                result = materialize_tensor_list(tensor_list.value());
            }
            return result;
        };
    const auto x_vec = materialize_tensor_list(x);
    const auto weight_vec = materialize_tensor_list(weight);
    const auto bias_vec = materialize_optional_tensor_list(bias);
    const auto scale_vec = materialize_optional_tensor_list(scale);
    const auto offset_vec = materialize_optional_tensor_list(offset);
    const auto antiquant_scale_vec = materialize_optional_tensor_list(antiquantScale);
    const auto antiquant_offset_vec = materialize_optional_tensor_list(antiquantOffset);
    const auto per_token_scale_vec = materialize_optional_tensor_list(perTokenScale);
    const auto y_vec = std::vector<at::Tensor>{y};

    c10::optional<std::vector<int64_t>> tuning_config_vec = c10::nullopt;
    if (tuningConfigOptional.has_value()) {
        tuning_config_vec = std::vector<int64_t>(tuningConfigOptional->begin(), tuningConfigOptional->end());
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto acl_call = [=, &matched_combo]() -> int {
        const at::TensorList x_ref(x_vec);
        const at::TensorList weight_ref(weight_vec);
        const at::TensorList y_ref(y_vec);
        c10::optional<at::TensorList> bias_ref = c10::nullopt;
        c10::optional<at::TensorList> scale_ref = c10::nullopt;
        c10::optional<at::TensorList> offset_ref = c10::nullopt;
        c10::optional<at::TensorList> antiquant_scale_ref = c10::nullopt;
        c10::optional<at::TensorList> antiquant_offset_ref = c10::nullopt;
        c10::optional<at::TensorList> per_token_scale_ref = c10::nullopt;
        if (bias_vec.has_value()) {
            bias_ref = at::TensorList(*bias_vec);
        }
        if (scale_vec.has_value()) {
            scale_ref = at::TensorList(*scale_vec);
        }
        if (offset_vec.has_value()) {
            offset_ref = at::TensorList(*offset_vec);
        }
        if (antiquant_scale_vec.has_value()) {
            antiquant_scale_ref = at::TensorList(*antiquant_scale_vec);
        }
        if (antiquant_offset_vec.has_value()) {
            antiquant_offset_ref = at::TensorList(*antiquant_offset_vec);
        }
        if (per_token_scale_vec.has_value()) {
            per_token_scale_ref = at::TensorList(*per_token_scale_vec);
        }
        const std::vector<int64_t> *tuning_config_ptr =
            tuning_config_vec.has_value() ? &tuning_config_vec.value() : nullptr;
        groupedmatmul_api(matched_combo, matched_index, stream, x_ref, weight_ref, bias_ref, scale_ref, offset_ref,
                          antiquant_scale_ref, antiquant_offset_ref, groupList, per_token_scale_ref, y_ref, splitItem,
                          groupType, groupListType, actType, tuning_config_ptr);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("GroupedMatmul", acl_call);
    return y;
}

// Register the NPU implementation
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m) { m.impl("grouped_matmul", grouped_matmul_npu); }

} // namespace GroupedMatmul
} // namespace ascend_ops
