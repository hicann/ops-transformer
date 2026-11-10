/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_generic_block_sparse_attention_metadata.h"
#include "../../../generic_block_sparse_attention_metadata_tiling.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
constexpr int64_t kNumQHeads = 64;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kLayoutSparsePatternKvnTotalQbKb = 4;
constexpr int64_t kMaskType = 1;
constexpr int64_t kSoftmaxPrecisionMixed = 1;

aclIntArray* MakeBlockShape()
{
    const std::vector<int64_t> blockShape = {1, 128};
    return aclCreateIntArray(blockShape.data(), static_cast<uint64_t>(blockShape.size()));
}

aclnnStatus CallGetWorkspaceSize(const aclTensor* sparseBlockIdx, const aclTensor* sparseBlockCount,
                                 const aclTensor* metadata, const aclIntArray* blockShape, uint64_t* workspaceSize,
                                 aclOpExecutor** executor)
{
    return aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
        sparseBlockIdx, sparseBlockCount, nullptr, nullptr, nullptr, nullptr, -1, -1, kNumQHeads, kNumKvHeads, kHeadDim,
        blockShape, "TND", "PA_BBND", kLayoutSparsePatternKvnTotalQbKb, kMaskType, 0, kSoftmaxPrecisionMixed, -1, -1, 0,
        true, metadata, workspaceSize, executor);
}
// Input shapes for the MXFP8 metadata API contract checks.
constexpr int64_t kBatch = 1;
constexpr int64_t kS1 = 4;
constexpr int64_t kS2 = 256;
constexpr int64_t kN1 = 4;
constexpr int64_t kN2 = 1;
constexpr int64_t kD = 128;
constexpr int64_t kTopK = 2;
constexpr int64_t kBlockSize = 128;
constexpr int64_t kT = kBatch * kS1;
constexpr int64_t kLayoutSparse = 4;

// Validate through the public API: metadata checks now live in its translation unit.
aclnnStatus CheckMetadataContract(SocVersion soc, int64_t quantType, int64_t precision, int64_t headDim = kD,
                                  int64_t blockY = kBlockSize, int64_t residualMode = 0)
{
    const auto previousSoc = op::GetCurrentPlatformInfo().GetSocVersion();
    op::SetPlatformSocVersion(soc);
    auto idx = TensorDesc({kN2, kT, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0}).ToAclType();
    auto count = TensorDesc({kN2, kT}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1}).ToAclType();
    auto cuQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kS1}).ToAclType();
    auto usedKv = TensorDesc({kBatch}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{kS2}).ToAclType();
    const int64_t metadataSize = optiling::generic_block_sparse_attention_metadata::METADATA_TOTAL_SIZE;
    auto metadata = TensorDesc({metadataSize}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0}).ToAclType();
    const vector<int64_t> blockShape = {1, blockY};
    auto blocks = aclCreateIntArray(blockShape.data(), blockShape.size());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
        idx.get(), count.get(), cuQ.get(), nullptr, nullptr, usedKv.get(), kS1, kS2, kN1, kN2, headDim, blocks, "TND",
        "PA_BBND", kLayoutSparse, 1, quantType, precision, -1, -1, residualMode, true, metadata.get(), &workspaceSize,
        &executor);
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
    aclDestroyIntArray(blocks);
    op::SetPlatformSocVersion(previousSoc);
    return status;
}
} // namespace

class aclnnGenericBlockSparseAttentionMetadata_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "aclnnGenericBlockSparseAttentionMetadata_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "aclnnGenericBlockSparseAttentionMetadata_test TearDown" << endl;
    }
};

// 公共API边界出参为空是用户参数契约错误：返回 ACLNN_ERR_PARAM_NULLPTR 并经 errno 映射上报正式参数码 EZ1001。
TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, null_workspace_size)
{
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(CallGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, nullptr, &executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, null_executor)
{
    uint64_t workspaceSize = 0;
    EXPECT_EQ(CallGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &workspaceSize, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
}

// 必选输入为空属于参数校验失败：根因层正式上报 EZ1001 并返回 ACLNN_ERR_PARAM_INVALID。
TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, null_sparse_block_idx)
{
    auto sparseBlockCount = TensorDesc({2, 2, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4).ToAclType();
    auto metadata = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0).ToAclType();
    aclIntArray* blockShape = MakeBlockShape();
    ASSERT_NE(blockShape, nullptr);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(
        CallGetWorkspaceSize(nullptr, sparseBlockCount.get(), metadata.get(), blockShape, &workspaceSize, &executor),
        ACLNN_ERR_PARAM_INVALID);
    aclDestroyIntArray(blockShape);
}

// dtype 不满足 INT32 约束：根因层正式上报 EZ1001 并返回 ACLNN_ERR_PARAM_INVALID。
TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, dtype_sparse_block_idx_fp16)
{
    auto sparseBlockIdx = TensorDesc({2, 2, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 4).ToAclType();
    auto sparseBlockCount = TensorDesc({2, 2, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4).ToAclType();
    auto metadata = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0).ToAclType();
    aclIntArray* blockShape = MakeBlockShape();
    ASSERT_NE(blockShape, nullptr);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(CallGetWorkspaceSize(sparseBlockIdx.get(), sparseBlockCount.get(), metadata.get(), blockShape,
                                   &workspaceSize, &executor),
              ACLNN_ERR_PARAM_INVALID);
    aclDestroyIntArray(blockShape);
}

TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, metadata_mxfp8_platform_and_precision)
{
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 6, 1), ACLNN_SUCCESS);
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND910B, 6, 1), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 6, 0), ACLNN_SUCCESS);
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 5, 0), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND910B, 0, 1), ACLNN_SUCCESS);
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 5, 1), ACLNN_SUCCESS);
}

TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, metadata_mxfp8_shape_contract)
{
    for (int64_t headDim : {1, 31, 32, 33, 64, 65, 128, 192, 255, 256}) {
        EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 6, 1, headDim), ACLNN_SUCCESS);
    }
    for (int64_t headDim : {0, 257, 512}) {
        EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 6, 1, headDim), ACLNN_ERR_PARAM_INVALID);
    }
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 5, 1, 256), ACLNN_ERR_PARAM_INVALID);
    for (int64_t blockY : {1, 4, 8, 17, 64, 127, 128}) {
        EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 6, 1, 192, blockY), ACLNN_SUCCESS);
    }
    for (int64_t blockY : {0, 129}) {
        EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 6, 1, 192, blockY), ACLNN_ERR_PARAM_INVALID);
    }
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 6, 1, kD, kBlockSize, 1), ACLNN_ERR_PARAM_INVALID);
}
