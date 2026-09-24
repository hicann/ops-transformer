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
#include <memory>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_quant_grouped_matmul_inplace_add.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {

using TensorPtr = std::unique_ptr<aclTensor, void (*)(aclTensor *)>;

struct StorageTestInputs {
    TensorPtr x1 = TensorDesc({3, 2}, ACL_HIFLOAT8, ACL_FORMAT_ND, {1, 3}, 0, {2, 3}).ToAclType();
    TensorPtr x2 = TensorDesc({2, 4}, ACL_HIFLOAT8, ACL_FORMAT_ND).ToAclType();
    TensorPtr scale1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    TensorPtr scale2 = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    TensorPtr groupList = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    TensorPtr y = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();

    std::array<aclTensor *, 6> Tensors()
    {
        return {x1.get(), x2.get(), scale1.get(), scale2.get(), groupList.get(), y.get()};
    }
};

void ExpectWorkspaceError(std::array<aclTensor *, 6> tensors, aclnnStatus expectedStatus)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    uint64_t workspaceSize = 12345;
    aclOpExecutor *executor = nullptr;
    const auto status = aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(
        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], 0, 0, &workspaceSize, &executor);
    EXPECT_EQ(status, expectedStatus);
    EXPECT_EQ(workspaceSize, 12345U);
    EXPECT_EQ(executor, nullptr);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
}

TEST(QuantGroupedMatmulInplaceAddStorage, T92TransposedViewOffsetOneReturnsParameterInvalid)
{
    StorageTestInputs inputs;
    inputs.x1->SetViewOffset(1);
    // maxIndex = 1 + (3 - 1) * 1 + (2 - 1) * 3 = 6, storageSize = 6.
    ExpectWorkspaceError(inputs.Tensors(), ACLNN_ERR_PARAM_INVALID);
}

TEST(QuantGroupedMatmulInplaceAddStorage, T94TransposedViewOffsetThreeReturnsParameterInvalid)
{
    StorageTestInputs inputs;
    inputs.x1->SetViewOffset(3);
    // maxIndex = 3 + (3 - 1) * 1 + (2 - 1) * 3 = 8, storageSize = 6.
    ExpectWorkspaceError(inputs.Tensors(), ACLNN_ERR_PARAM_INVALID);
}

TEST(QuantGroupedMatmulInplaceAddStorage, ChecksInputScaleGroupListAndInplaceOutput)
{
    for (size_t index = 0; index < 6; ++index) {
        SCOPED_TRACE(index);
        StorageTestInputs inputs;
        auto tensors = inputs.Tensors();
        // Every view fills its storage at offset zero, so shifting by one must fail.
        tensors[index]->SetViewOffset(1);
        ExpectWorkspaceError(tensors, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(QuantGroupedMatmulInplaceAddStorage, EmptyFastPathStillRejectsInvalidAuxiliaryStorage)
{
    StorageTestInputs inputs;
    inputs.x1 = TensorDesc({0, 2}, ACL_HIFLOAT8, ACL_FORMAT_ND).ToAclType();
    inputs.y = TensorDesc({1, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    inputs.groupList->SetViewOffset(1);
    ExpectWorkspaceError(inputs.Tensors(), ACLNN_ERR_PARAM_INVALID);
}

TEST(QuantGroupedMatmulInplaceAddStorage, ValidEmptyFastPathIsPreserved)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    StorageTestInputs inputs;
    inputs.x1 = TensorDesc({0, 2}, ACL_HIFLOAT8, ACL_FORMAT_ND).ToAclType();
    inputs.y = TensorDesc({1, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    uint64_t workspaceSize = 12345;
    aclOpExecutor *executor = nullptr;
    EXPECT_EQ(aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(inputs.x1.get(), inputs.x2.get(), inputs.scale1.get(),
                                                                inputs.scale2.get(), inputs.groupList.get(),
                                                                inputs.y.get(), 0, 0, &workspaceSize, &executor),
              ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
}

TEST(QuantGroupedMatmulInplaceAddStorage, RequiredNullptrReturnCodesArePreserved)
{
    for (size_t index = 0; index < 6; ++index) {
        SCOPED_TRACE(index);
        StorageTestInputs inputs;
        auto tensors = inputs.Tensors();
        tensors[index] = nullptr;
        ExpectWorkspaceError(tensors, ACLNN_ERR_PARAM_NULLPTR);
    }
}

} // namespace
