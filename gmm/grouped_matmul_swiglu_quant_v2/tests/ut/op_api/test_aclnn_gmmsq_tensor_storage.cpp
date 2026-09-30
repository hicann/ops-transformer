/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <limits>
#include <memory>

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "../../../op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"
#include "../../../../common/op_api/gmm_tensor_storage_check.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {
constexpr int64_t TEST_M = 32;
constexpr int64_t TEST_K = 128;
constexpr int64_t TEST_N = 128;
constexpr int64_t TEST_E = 1;
constexpr int64_t TEST_OUTPUT_N = 64;
constexpr int64_t TEST_SCALE_K = 2;
constexpr int64_t SCALE_PAIR_SIZE = 2;
constexpr int64_t NZ_N_BLOCKS = 4;
constexpr int64_t NZ_K_BLOCKS = 8;
constexpr int64_t NZ_K0 = 16;
constexpr int64_t NZ_C0 = 32;
constexpr int64_t TEST_WEIGHT_GROUP_STRIDE = TEST_K * TEST_N;
constexpr int64_t ELEMENT_STRIDE = 1;
constexpr int64_t SHIFTED_VIEW_OFFSET = 1;
constexpr int64_t NEGATIVE_DESCRIPTOR_VALUE = -1;
constexpr int64_t OVERFLOW_STORAGE_MULTIPLIER = 2;
constexpr size_t REQUIRED_TENSOR_COUNT = 7;
constexpr size_t SINGLE_TENSOR_COUNT = 1;
constexpr size_t OPTIONAL_TENSOR_COUNT = 3;
constexpr size_t OPTIONAL_BIAS_INDEX = 1;
constexpr size_t OPTIONAL_SMOOTH_SCALE_INDEX = 2;
constexpr int64_t MX_QUANT_MODE = 2;
constexpr int64_t SPLIT_SWIGLU_MODE = 2;
constexpr double DEFAULT_CLAMP_LIMIT = 7.0;
constexpr double DEFAULT_GLU_ALPHA = 1.702;
constexpr double DEFAULT_GLU_BIAS = 1.0;
constexpr int64_t BOUNDARY_VIEW_ELEMENTS = 2;
constexpr int64_t BOUNDARY_STRIDE = 2;
constexpr int64_t BOUNDARY_STORAGE_ELEMENTS = 4;
constexpr int64_t BOUNDARY_LEGAL_OFFSET = 1;
constexpr int64_t BOUNDARY_INVALID_OFFSET = 2;
using TensorPtr = std::unique_ptr<aclTensor, void (*)(aclTensor*)>;
using TensorListPtr = std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)>;
enum class StorageApi {
    ND_V2,
    NZ_V2,
    NZ_V3
};

struct StorageInputs {
    TensorPtr x = TensorDesc({TEST_M, TEST_K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    TensorPtr weight;
    TensorPtr weightScale =
        TensorDesc({TEST_E, TEST_SCALE_K, TEST_N, SCALE_PAIR_SIZE}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ToAclType();
    TensorPtr xScale = TensorDesc({TEST_M, TEST_SCALE_K, SCALE_PAIR_SIZE}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ToAclType();
    TensorPtr groupList = TensorDesc({TEST_E}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    TensorPtr output = TensorDesc({TEST_M, TEST_OUTPUT_N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    TensorPtr outputScale = TensorDesc({TEST_M, TEST_E, SCALE_PAIR_SIZE}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ToAclType();

    explicit StorageInputs(StorageApi api)
        : weight(api == StorageApi::ND_V2 ?
                     TensorDesc({TEST_E, TEST_K, TEST_N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType() :
                     TensorDesc({TEST_E, TEST_K, TEST_N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ,
                                {TEST_WEIGHT_GROUP_STRIDE, TEST_N, ELEMENT_STRIDE}, 0,
                                {TEST_E, NZ_N_BLOCKS, NZ_K_BLOCKS, NZ_K0, NZ_C0})
                         .ToAclType())
    {}

    std::array<aclTensor*, REQUIRED_TENSOR_COUNT> RequiredTensors()
    {
        return {x.get(),         weight.get(), weightScale.get(), xScale.get(),
                groupList.get(), output.get(), outputScale.get()};
    }
};

void ExpectWorkspaceStatus(StorageApi api, StorageInputs& inputs, aclnnStatus expected,
                           const aclTensorList* assistance = nullptr, const aclTensor* bias = nullptr,
                           const aclTensor* smoothScale = nullptr, bool nullX = false)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    const aclTensor* weightElements[] = {inputs.weight.get()};
    const aclTensor* scaleElements[] = {inputs.weightScale.get()};
    TensorListPtr weights(aclCreateTensorList(weightElements, SINGLE_TENSOR_COUNT), aclDestroyTensorList);
    ASSERT_NE(weights.get(), nullptr);
    (void)inputs.weight.release();
    TensorListPtr scales(aclCreateTensorList(scaleElements, SINGLE_TENSOR_COUNT), aclDestroyTensorList);
    ASSERT_NE(scales.get(), nullptr);
    (void)inputs.weightScale.release();
    constexpr uint64_t WORKSPACE_SENTINEL = 12345;
    uint64_t workspaceSize = WORKSPACE_SENTINEL;
    aclOpExecutor* executor = nullptr;
    const auto* x = nullX ? nullptr : inputs.x.get();
    aclnnStatus status;
    if (api == StorageApi::NZ_V3) {
        status = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
            x, weights.get(), scales.get(), assistance, bias, inputs.xScale.get(), smoothScale, inputs.groupList.get(),
            MX_QUANT_MODE, 0, MX_QUANT_MODE, 0, nullptr, SPLIT_SWIGLU_MODE, DEFAULT_CLAMP_LIMIT, DEFAULT_GLU_ALPHA,
            DEFAULT_GLU_BIAS, "rint", 0, 0.0, inputs.output.get(), inputs.outputScale.get(), &workspaceSize, &executor);
    } else {
        auto function = api == StorageApi::ND_V2 ? aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize :
                                                   aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize;
        status = function(x, weights.get(), scales.get(), assistance, bias, inputs.xScale.get(), smoothScale,
                          inputs.groupList.get(), MX_QUANT_MODE, 0, MX_QUANT_MODE, 0, nullptr, inputs.output.get(),
                          inputs.outputScale.get(), &workspaceSize, &executor);
    }
    EXPECT_EQ(status, expected);
    if (expected != ACLNN_SUCCESS) {
        EXPECT_EQ(workspaceSize, WORKSPACE_SENTINEL);
        EXPECT_EQ(executor, nullptr);
    }
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
}

class GmmsqTensorStorageTest : public testing::TestWithParam<StorageApi> {};

TEST_P(GmmsqTensorStorageTest, RejectsShiftedFullStorageForEveryRequiredTensor)
{
    for (size_t index = 0; index < REQUIRED_TENSOR_COUNT; ++index) {
        SCOPED_TRACE(index);
        StorageInputs inputs(GetParam());
        inputs.RequiredTensors()[index]->SetViewOffset(SHIFTED_VIEW_OFFSET);
        ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_P(GmmsqTensorStorageTest, RejectsNegativeOffset)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetViewOffset(NEGATIVE_DESCRIPTOR_VALUE);
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(GmmsqTensorStorageTest, RejectsOffsetAtStorageEnd)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetViewOffset(inputs.x->GetStorageShape().GetShapeSize());
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(GmmsqTensorStorageTest, RejectsStrideExtentOverflow)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetViewStrides(op::Strides({std::numeric_limits<int64_t>::max(), ELEMENT_STRIDE}));
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(GmmsqTensorStorageTest, RejectsStorageProductOverflow)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetStorageShape(op::Shape({std::numeric_limits<int64_t>::max(), OVERFLOW_STORAGE_MULTIPLIER}));
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
}

TEST_P(GmmsqTensorStorageTest, RejectsNegativeStrideAndRankMismatch)
{
    {
        StorageInputs inputs(GetParam());
        inputs.x->SetViewStrides(op::Strides({TEST_K, NEGATIVE_DESCRIPTOR_VALUE}));
        ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
    }
    {
        StorageInputs inputs(GetParam());
        inputs.x->SetViewStrides(op::Strides({ELEMENT_STRIDE}));
        ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_P(GmmsqTensorStorageTest, ChecksOptionalTensorStorage)
{
    for (size_t index = 0; index < OPTIONAL_TENSOR_COUNT; ++index) {
        SCOPED_TRACE(index);
        StorageInputs inputs(GetParam());
        auto optional = TensorDesc({TEST_E}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
        optional->SetViewOffset(SHIFTED_VIEW_OFFSET);
        if (index == 0) {
            const aclTensor* elements[] = {optional.get()};
            TensorListPtr assistance(aclCreateTensorList(elements, SINGLE_TENSOR_COUNT), aclDestroyTensorList);
            ASSERT_NE(assistance.get(), nullptr);
            (void)optional.release();
            ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID, assistance.get());
        } else {
            ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID, nullptr,
                                  index == OPTIONAL_BIAS_INDEX ? optional.get() : nullptr,
                                  index == OPTIONAL_SMOOTH_SCALE_INDEX ? optional.get() : nullptr);
        }
    }
}

TEST_P(GmmsqTensorStorageTest, RequiredNullPointerHasPriorityOverInvalidStorage)
{
    StorageInputs inputs(GetParam());
    inputs.output->SetViewOffset(SHIFTED_VIEW_OFFSET);
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_NULLPTR, nullptr, nullptr, nullptr, true);
}

TEST_P(GmmsqTensorStorageTest, AcceptsNonzeroOffsetWithinDeclaredStorage)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetStorageShape(op::Shape({TEST_M + ELEMENT_STRIDE, TEST_K}));
    inputs.x->SetViewOffset(SHIFTED_VIEW_OFFSET);
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_SUCCESS);
}

TEST_P(GmmsqTensorStorageTest, RejectsInvalidAuxiliaryStorageBeforeEmptyFastPath)
{
    StorageInputs inputs(GetParam());
    inputs.x->SetViewShape(op::Shape({0, TEST_K}));
    inputs.groupList->SetViewOffset(SHIFTED_VIEW_OFFSET);
    ExpectWorkspaceStatus(GetParam(), inputs, ACLNN_ERR_PARAM_INVALID);
}

INSTANTIATE_TEST_SUITE_P(AllApis, GmmsqTensorStorageTest,
                         testing::Values(StorageApi::ND_V2, StorageApi::NZ_V2, StorageApi::NZ_V3));

TEST(GmmsqStorageBoundsHelper, EmptyViewDoesNotInventStorageAccess)
{
    auto tensor = TensorDesc({0, TEST_K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    EXPECT_TRUE(gmm::CheckTensorStorageBounds(tensor.get(), "GroupedMatmulSwigluQuant", "empty"));
    tensor->SetViewOffset(NEGATIVE_DESCRIPTOR_VALUE);
    EXPECT_FALSE(gmm::CheckTensorStorageBounds(tensor.get(), "GroupedMatmulSwigluQuant", "empty"));
}

TEST(GmmsqStorageBoundsHelper, BoundaryLastElementIsAccepted)
{
    auto tensor = TensorDesc({BOUNDARY_VIEW_ELEMENTS}, ACL_FLOAT, ACL_FORMAT_ND, {BOUNDARY_STRIDE},
                             BOUNDARY_LEGAL_OFFSET, {BOUNDARY_STORAGE_ELEMENTS})
                      .ToAclType();
    EXPECT_TRUE(gmm::CheckTensorStorageBounds(tensor.get(), "GroupedMatmulSwigluQuant", "boundary"));
    tensor->SetViewOffset(BOUNDARY_INVALID_OFFSET);
    EXPECT_FALSE(gmm::CheckTensorStorageBounds(tensor.get(), "GroupedMatmulSwigluQuant", "boundary"));
}
} // namespace
