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
// Input shapes for metadata API contract checks.
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
                                  int64_t blockY = kBlockSize, int64_t residualMode = 0, int64_t maskType = 1,
                                  int64_t blockX = 1, int64_t numQHeads = kN1, int64_t numKvHeads = kN2)
{
    const auto previousSoc = op::GetCurrentPlatformInfo().GetSocVersion();
    op::SetPlatformSocVersion(soc);
    auto idx = TensorDesc({numKvHeads, kT, kTopK}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0}).ToAclType();
    auto count = TensorDesc({numKvHeads, kT}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1}).ToAclType();
    auto cuQ = TensorDesc({kBatch + 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, kS1}).ToAclType();
    auto usedKv = TensorDesc({kBatch}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{kS2}).ToAclType();
    const int64_t metadataSize = optiling::generic_block_sparse_attention_metadata::METADATA_TOTAL_SIZE;
    auto metadata = TensorDesc({metadataSize}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0}).ToAclType();
    const vector<int64_t> blockShape = {blockX, blockY};
    auto blocks = aclCreateIntArray(blockShape.data(), blockShape.size());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
        idx.get(), count.get(), cuQ.get(), nullptr, nullptr, usedKv.get(), kS1, kS2, numQHeads, numKvHeads, headDim,
        blocks, "TND", "PA_BBND", kLayoutSparse, maskType, quantType, precision, -1, -1, residualMode, true,
        metadata.get(), &workspaceSize, &executor);
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

TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, int8_rejects_group_above_16)
{
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND910B, 7, 0, kD, kBlockSize, 0, kMaskType, 1, 17, 1),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, mxfp8_q32_q64_group_boundary)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto idx = TensorDesc({2, 4, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2).ToAclType();
    auto count = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 3).ToAclType();
    auto cuQ = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 4}).ToAclType();
    auto cuKv = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 257}).ToAclType();
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0).ToAclType();
    const vector<int64_t> shape = {1, 4};
    auto blocks = aclCreateIntArray(shape.data(), shape.size());
    for (int64_t dim : {192, 256}) {
        for (int64_t group : {1, 16, 17, 31, 32, 33, 48, 64, 65}) {
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            EXPECT_EQ(
                aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
                    idx.get(), count.get(), cuQ.get(), cuKv.get(), nullptr, nullptr, 4, 257, group * 2, 2, dim, blocks,
                    "TND", "TND", 4, 0, 6, 0, -1, -1, 0, false, metadata.get(), &workspaceSize, &executor),
                group <= 64 ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID);
            if (executor != nullptr) {
                aclDestroyAclOpExecutor(executor);
            }
        }
    }
    aclDestroyIntArray(blocks);
}

TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, mxfp8_precision_requires_zero)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto idx = TensorDesc({2, 4, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2).ToAclType();
    auto count = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 3).ToAclType();
    auto cuQ = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 4}).ToAclType();
    auto cuKv = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 257}).ToAclType();
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0).ToAclType();
    const vector<int64_t> shape = {1, 4};
    auto blocks = aclCreateIntArray(shape.data(), shape.size());
    for (int64_t dim : {192, 256}) {
        for (int64_t precision : {0, 1, -1, 2}) {
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            EXPECT_EQ(
                aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
                    idx.get(), count.get(), cuQ.get(), cuKv.get(), nullptr, nullptr, 4, 257, 24, 2, dim, blocks, "TND",
                    "TND", 4, 0, 6, precision, -1, -1, 0, false, metadata.get(), &workspaceSize, &executor),
                precision == 0 ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID);
            if (executor != nullptr) {
                aclDestroyAclOpExecutor(executor);
            }
        }
    }
    aclDestroyIntArray(blocks);
}

TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, mxfp8_sparse_block_size_contract)
{
    for (int64_t dim : {192, 256}) {
        for (int64_t residual : {0, 1}) {
            for (int64_t y : {-1, 0, 1, 2, 4, 8, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256}) {
                const bool supported = y == 4 || y == 8 || y == 16 || y == 32 || y == 64 || y == 128;
                EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 6, 0, dim, y, residual),
                          supported ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID)
                    << "D=" << dim << ", Y=" << y << ", residual=" << residual;
            }
        }
    }
}

// 公共API边界出参为空是用户参数契约错误：返回 ACLNN_ERR_PARAM_NULLPTR 并经 errno 映射上报正式参数码 EZ1001。
TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, null_workspace_size)
{
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(CallGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, nullptr, &executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, mxfp8_implicit_residual_mode)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto idx = TensorDesc({2, 4, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2).ToAclType();
    auto count = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3).ToAclType();
    auto cuQ = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 4}).ToAclType();
    auto cuKv = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 257}).ToAclType();
    auto metadata = TensorDesc({1024}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 0).ToAclType();
    const vector<int64_t> shape = {1, 4};
    auto blocks = aclCreateIntArray(shape.data(), shape.size());
    for (int64_t mode : {0, 1, 2}) {
        uint64_t workspace = 0;
        aclOpExecutor* executor = nullptr;
        EXPECT_EQ(aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
                      idx.get(), count.get(), cuQ.get(), cuKv.get(), nullptr, nullptr, 4, 257, 24, 2, 256, blocks,
                      "TND", "TND", 4, 1, 6, 0, -1, -1, mode, false, metadata.get(), &workspace, &executor),
                  mode <= 1 ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID);
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
    }
    aclDestroyIntArray(blocks);
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

TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, metadata_rejects_union_mxfp8)
{
    // Union (maskType=3) only supports quantType=0 or 5; quantType=6 must fail here,
    // not only later in the main operator tiling.
    // Host-only UT cannot register AICPU (561103); keep reject-path only.
    // Allowed (0/5) SUCCESS covered by scheduler tiling UT.
    constexpr int64_t kUnionX = 8;
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, 6, 0, kD, kBlockSize, 0, 3, kUnionX),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(aclnnGenericBlockSparseAttentionMetadata_test, rejects_unsupported_quant_types_and_precision)
{
    for (int64_t quantType : {3, 5, 6}) {
        EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND910B, quantType, 0), ACLNN_ERR_PARAM_INVALID)
            << "910B quantType=" << quantType;
    }
    for (int64_t quantType : {3, 7}) {
        EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND950, quantType, 0), ACLNN_ERR_PARAM_INVALID)
            << "950 quantType=" << quantType;
    }
    EXPECT_EQ(CheckMetadataContract(SocVersion::ASCEND910B, 7, 1), ACLNN_ERR_PARAM_INVALID);
}
