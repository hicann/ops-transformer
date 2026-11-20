/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <cstdint>
#include <iostream>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_quant_lightning_indexer_v2.h"
#include "../../../op_api/aclnn_quant_lightning_indexer_v3.h"
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

class quant_lightning_indexer_v2_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "quant_lightning_indexer_v2_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "quant_lightning_indexer_v2_opapi_ut TearDown" << endl;
    }
};

// 手写 aclnn 接口注册冒烟：libopapi_transformer_ut.so 加载即完成 OpDef 注册与符号解析
TEST_F(quant_lightning_indexer_v2_opapi_ut, op_def_registration_smoke)
{
    SUCCEED();
}

// V2 兼容接口（签名冻结、无 candidate 参数）：必选输入 q 为空
TEST_F(quant_lightning_indexer_v2_opapi_ut, quant_lightning_indexer_v2_aclnn_null_query)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "PA_BBND";
    auto key = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 16, 1, 128}, ACL_INT8, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 1, 2048}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseValuesOut = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({0}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerV2GetWorkspaceSize(
        nullptr, key.get(), weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, 2048, 2, -1, layoutQuery, layoutKey, 0, 1, 0, sparseIndicesOut.get(), sparseValuesOut.get(),
        &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// V2 兼容接口：第二段接口空执行器直通
TEST_F(quant_lightning_indexer_v2_opapi_ut, quant_lightning_indexer_v2_aclnn_empty_executor)
{
    aclnnStatus aclRet = aclnnQuantLightningIndexerV2(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// V3 candidate 接口：必选输入 q 为空
TEST_F(quant_lightning_indexer_v2_opapi_ut, quant_lightning_indexer_v3_aclnn_null_query)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "PA_BBND";
    auto key = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 16, 1, 128}, ACL_INT8, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 1, 2048}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseValuesOut = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({0}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerV3GetWorkspaceSize(
        nullptr, key.get(), weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, 2048, 2, -1, layoutQuery, layoutKey, 0, 1, 0, -1, -1, sparseIndicesOut.get(),
        sparseValuesOut.get(), nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// V3 candidate 接口：candidate 参数对非法（-1, 8），第一段接口直接拒绝
TEST_F(quant_lightning_indexer_v2_opapi_ut, quant_lightning_indexer_v3_aclnn_bad_candidate_pair_failed)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "PA_BBND";
    auto query = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 64, 128}, ACL_INT8, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 16, 1, 128}, ACL_INT8, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto qScale = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto kScale = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 16, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 1, 2048}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseValuesOut = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({0}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto candidateIndices = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 1, 2048}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto candidateLength = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerV3GetWorkspaceSize(
        query.get(), key.get(), weights.get(), qScale.get(), kScale.get(), nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 2048, 2, -1, layoutQuery, layoutKey, 0, 1, 0, -1, 8, sparseIndicesOut.get(),
        sparseValuesOut.get(), candidateIndices.get(), candidateLength.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// V3 candidate 接口：candidate 开启时 candidateBlockIndices 为空，由前置一致性校验拒绝
TEST_F(quant_lightning_indexer_v2_opapi_ut, quant_lightning_indexer_v3_aclnn_missing_candidate_out)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "PA_BBND";
    auto query = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 64, 128}, ACL_INT8, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 16, 1, 128}, ACL_INT8, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto qScale = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto kScale = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 16, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 39, 1, 2048}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseValuesOut = std::unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({0}, ACL_BF16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerV3GetWorkspaceSize(
        query.get(), key.get(), weights.get(), qScale.get(), kScale.get(), nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 2048, 2, -1, layoutQuery, layoutKey, 0, 1, 0, 2048, 8, sparseIndicesOut.get(),
        sparseValuesOut.get(), nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
}

// V3 candidate 接口：第二段接口空执行器直通
TEST_F(quant_lightning_indexer_v2_opapi_ut, quant_lightning_indexer_v3_aclnn_empty_executor)
{
    aclnnStatus aclRet = aclnnQuantLightningIndexerV3(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
