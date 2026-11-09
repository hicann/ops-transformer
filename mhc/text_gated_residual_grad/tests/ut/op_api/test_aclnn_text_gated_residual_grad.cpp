/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <array>
#include <vector>
#include "op_api_ut_common/tensor_desc.h"
#include "../../../op_host/op_api/aclnn_text_gated_residual_grad.h"

#include "../../text_gated_residual_grad_test_constants.h"

namespace {
using namespace HcGradTest;
void CheckInvalidPairing(aclDataType type, uint32_t gateMask, uint32_t gammaMask)
{
    std::vector<TensorDesc> descs = {
        {{1, 1, HIDDEN}, type, ACL_FORMAT_ND},         {{1, 1, CHANNELS}, type, ACL_FORMAT_ND},
        {{1, 1, FEATURES}, type, ACL_FORMAT_ND},       {{PROJECTION, FEATURES}, type, ACL_FORMAT_ND},
        {{FEATURES, PROJECTION}, type, ACL_FORMAT_ND}, {{CHANNELS, FEATURES}, type, ACL_FORMAT_ND},
        {{FEATURES}, ACL_FLOAT, ACL_FORMAT_ND},        {{1, 1, CHANNELS}, ACL_FLOAT, ACL_FORMAT_ND},
        {{1, 1, CHANNELS}, type, ACL_FORMAT_ND},       {{1, 1, PROJECTION}, type, ACL_FORMAT_ND},
        {{1, 1, FEATURES}, type, ACL_FORMAT_ND},       {{1, 1, FEATURES}, type, ACL_FORMAT_ND},
        {{PROJECTION, FEATURES}, type, ACL_FORMAT_ND}, {{FEATURES, PROJECTION}, type, ACL_FORMAT_ND},
        {{CHANNELS, FEATURES}, type, ACL_FORMAT_ND},   {{FEATURES}, ACL_FLOAT, ACL_FORMAT_ND}};
    std::vector<decltype(descs.front().ToAclType())> storage;
    std::array<aclTensor*, TENSOR_COUNT> tensors{};
    for (size_t i = 0; i < descs.size(); ++i) {
        storage.push_back(descs[i].ToAclType());
        tensors[i] = storage.back().get();
        ASSERT_NE(tensors[i], nullptr);
    }
    const std::array<size_t, GATE_GROUP_SIZE> gateIndices{GRAD_INJECT, W_GATE, INJECT, DW_GATE};
    for (size_t bit = 0; bit < gateIndices.size(); ++bit) {
        if ((gateMask & (1U << bit)) == 0) {
            tensors[gateIndices[bit]] = nullptr;
        }
    }
    if ((gammaMask & GAMMA_INPUT_MASK) == 0)
        tensors[GAMMA] = nullptr;
    if ((gammaMask & GAMMA_OUTPUT_MASK) == 0)
        tensors[DGAMMA] = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(
        aclnnTextGatedResidualGradGetWorkspaceSize(
            tensors[GRAD_MIXED], tensors[X], tensors[W_DOWN], tensors[W_UP], tensors[INV_RMS], tensors[DOWN_PRE],
            tensors[UP], tensors[GRAD_INJECT], tensors[W_GATE], tensors[GAMMA], tensors[INJECT], CHANNELS, tensors[DX],
            tensors[DW_DOWN], tensors[DW_UP], tensors[DW_GATE], tensors[DGAMMA], &workspaceSize, &executor),
        ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}
} // namespace

TEST(TextGatedResidualGradApi, IncompleteGateGroup_FAILED)
{
    for (auto type : {ACL_BF16, ACL_FLOAT16}) {
        for (uint32_t mask = 1; mask < COMPLETE_GATE_MASK; ++mask) {
            SCOPED_TRACE(mask);
            CheckInvalidPairing(type, mask, COMPLETE_GAMMA_MASK);
        }
    }
}

TEST(TextGatedResidualGradApi, IncompleteGammaPair_FAILED)
{
    for (auto type : {ACL_BF16, ACL_FLOAT16}) {
        for (uint32_t mask : {GAMMA_INPUT_MASK, GAMMA_OUTPUT_MASK}) {
            SCOPED_TRACE(mask);
            CheckInvalidPairing(type, COMPLETE_GATE_MASK, mask);
        }
    }
}
