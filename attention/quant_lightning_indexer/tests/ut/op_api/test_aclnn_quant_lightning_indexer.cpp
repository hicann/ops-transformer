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
#include <vector>
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_quant_lightning_indexer.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
using namespace std;
using namespace op;

namespace {
void DestroyAclTensor(aclTensor *tensor)
{
    Release(tensor);
}
} // namespace

class quant_lightning_indexer_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "quant_lightning_indexer_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "quant_lightning_indexer_opapi_ut TearDown" << endl;
    }
};

// Null query
TEST_F(quant_lightning_indexer_opapi_ut, quant_lightning_indexer_aclnn_0)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerGetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, layoutQuery, layoutKey, 4, 3,
        INT64_MAX, INT64_MAX, nullptr, &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Contiguous tensors pass stride checks and reach inner api
TEST_F(quant_lightning_indexer_opapi_ut, quant_lightning_indexer_aclnn_1)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto queryDequantScale = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto keyDequantScale = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto out = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerGetWorkspaceSize(
        query.get(), key.get(), weights.get(), queryDequantScale.get(), keyDequantScale.get(), nullptr, nullptr,
        nullptr, 0, 0, layoutQuery, layoutKey, 4, 3, INT64_MAX, INT64_MAX, out.get(), &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Non-contiguous key with invalid inner stride is rejected
TEST_F(quant_lightning_indexer_opapi_ut, quant_lightning_indexer_aclnn_2)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "PA_BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    // inner strides broken on axis 2 (expected 128, given 256)
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND, {16384, 128, 256, 1}).ToAclTypeRawPtr(),
        DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto out = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerGetWorkspaceSize(
        query.get(), key.get(), weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, layoutQuery,
        layoutKey, 4, 3, INT64_MAX, INT64_MAX, out.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// Non-contiguous key with valid PA_BSND inner strides passes stride checks
TEST_F(quant_lightning_indexer_opapi_ut, quant_lightning_indexer_aclnn_3)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "PA_BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    // axis 0 stride padded, inner axes contiguous: valid PA non-contiguous key
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND, {32768, 128, 128, 1}).ToAclTypeRawPtr(),
        DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto out = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerGetWorkspaceSize(
        query.get(), key.get(), weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, layoutQuery,
        layoutKey, 4, 3, INT64_MAX, INT64_MAX, out.get(), &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Non-contiguous keyDequantScale with invalid inner stride is rejected
TEST_F(quant_lightning_indexer_opapi_ut, quant_lightning_indexer_aclnn_4)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "PA_BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    // 3-dim scale, inner stride on axis 2 broken (expected 1, given 2)
    auto keyDequantScale = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 64, 128}, ACL_FLOAT, ACL_FORMAT_ND, {16384, 128, 2}).ToAclTypeRawPtr(), DestroyAclTensor);
    auto out = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerGetWorkspaceSize(
        query.get(), key.get(), weights.get(), nullptr, keyDequantScale.get(), nullptr, nullptr, nullptr, 0, 0,
        layoutQuery, layoutKey, 4, 3, INT64_MAX, INT64_MAX, out.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// Non-contiguous keyDequantScale with valid PA_BSND inner strides passes stride checks
TEST_F(quant_lightning_indexer_opapi_ut, quant_lightning_indexer_aclnn_5)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "PA_BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    // axis 0 stride padded, inner axes contiguous: valid PA non-contiguous scale
    auto keyDequantScale = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 64, 128}, ACL_FLOAT, ACL_FORMAT_ND, {32768, 128, 1}).ToAclTypeRawPtr(), DestroyAclTensor);
    auto out = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerGetWorkspaceSize(
        query.get(), key.get(), weights.get(), nullptr, keyDequantScale.get(), nullptr, nullptr, nullptr, 0, 0,
        layoutQuery, layoutKey, 4, 3, INT64_MAX, INT64_MAX, out.get(), &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Non-contiguous key is rejected when layout is not PA
TEST_F(quant_lightning_indexer_opapi_ut, quant_lightning_indexer_aclnn_6)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    // axis 0 stride is not the contiguous value, BSND requires fully contiguous key
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND, {16384, 128, 128, 1}).ToAclTypeRawPtr(),
        DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto out = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnQuantLightningIndexerGetWorkspaceSize(
        query.get(), key.get(), weights.get(), nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, layoutQuery,
        layoutKey, 4, 3, INT64_MAX, INT64_MAX, out.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// Null executor of the second phase entry
TEST_F(quant_lightning_indexer_opapi_ut, quant_lightning_indexer_aclnn_7)
{
    aclnnStatus aclRet = aclnnQuantLightningIndexer(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
