/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include <vector>
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_lightning_indexer_v2.h"
#include "../../../op_host/op_api/aclnn_lightning_indexer_v3.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
using namespace std;
using namespace op;

namespace {
void DestroyAclTensor(aclTensor* tensor)
{
    Release(tensor);
}
} // namespace

class lightning_indexer_v2_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "lightning_indexer_v2_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "lightning_indexer_v2_opapi_ut TearDown" << endl;
    }
};

// 手写 aclnn 接口注册冒烟：libopapi_transformer_ut.so 加载即完成 OpDef 注册与符号解析
// （opapi UT 会整包链接各算子 op_api 手写源文件与 op_host 的 *_def.cpp，见 cmake/custom_build.cmake
// 中 ops_aclnn 归档），用例被执行说明库加载与全部算子定义注册流程无崩溃、无重复注册冲突。
// 注：须用 TEST_F 与同 suite 其余用例统一 fixture（gtest 禁止同 suite 混用 TEST/TEST_F）。
TEST_F(lightning_indexer_v2_opapi_ut, op_def_registration_smoke)
{
    SUCCEED();
}

// V2 兼容接口（无 candidate 参数）：必选输入 q 为空
TEST_F(lightning_indexer_v2_opapi_ut, lightning_indexer_v2_aclnn_0)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseValuesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerV2GetWorkspaceSize(
        nullptr, key.get(), weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 4,
        -1, layoutQuery, layoutKey, 3, 1, 0, sparseIndicesOut.get(), sparseValuesOut.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

// V2 兼容接口：必选输出 sparseValues 为空
TEST_F(lightning_indexer_v2_opapi_ut, lightning_indexer_v2_aclnn_1)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerV2GetWorkspaceSize(
        query.get(), key.get(), weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        4, -1, layoutQuery, layoutKey, 3, 1, 0, sparseIndicesOut.get(), nullptr, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

// V2 兼容接口：必选输入 k 为空
TEST_F(lightning_indexer_v2_opapi_ut, lightning_indexer_v2_aclnn_2)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseValuesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerV2GetWorkspaceSize(
        query.get(), nullptr, weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 4,
        -1, layoutQuery, layoutKey, 3, 1, 0, sparseIndicesOut.get(), sparseValuesOut.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

// V2 兼容接口：第二段接口空执行器直通
TEST_F(lightning_indexer_v2_opapi_ut, lightning_indexer_v2_aclnn_3)
{
    aclnnStatus aclRet = aclnnLightningIndexerV2(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// V2 兼容接口：必选输入 w 为空
TEST_F(lightning_indexer_v2_opapi_ut, lightning_indexer_v2_aclnn_4)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseValuesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerV2GetWorkspaceSize(
        query.get(), key.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 4, -1,
        layoutQuery, layoutKey, 3, 1, 0, sparseIndicesOut.get(), sparseValuesOut.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

// V2 兼容接口：必选输出 sparseIndices 为空
TEST_F(lightning_indexer_v2_opapi_ut, lightning_indexer_v2_aclnn_5)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseValuesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerV2GetWorkspaceSize(
        query.get(), key.get(), weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        4, -1, layoutQuery, layoutKey, 3, 1, 0, nullptr, sparseValuesOut.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

// V3 candidate 接口：必选输入 q 为空
TEST_F(lightning_indexer_v2_opapi_ut, lightning_indexer_v3_aclnn_0)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseValuesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerV3GetWorkspaceSize(
        nullptr, key.get(), weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 4,
        -1, layoutQuery, layoutKey, 3, 1, 0, -1, 8, sparseIndicesOut.get(), sparseValuesOut.get(), nullptr, nullptr,
        &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// V3 candidate 接口：candidate 开启（candidateTopkBlocks != -1）时 candidateBlockIndices 为空，
// 由手写接口的前置一致性校验拒绝
TEST_F(lightning_indexer_v2_opapi_ut, lightning_indexer_v3_aclnn_1)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseValuesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerV3GetWorkspaceSize(
        query.get(), key.get(), weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        4, -1, layoutQuery, layoutKey, 3, 1, 0, 64, 8, sparseIndicesOut.get(), sparseValuesOut.get(), nullptr, nullptr,
        &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// V3 candidate 接口：第二段接口空执行器直通
TEST_F(lightning_indexer_v2_opapi_ut, lightning_indexer_v3_aclnn_2)
{
    aclnnStatus aclRet = aclnnLightningIndexerV3(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
