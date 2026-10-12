/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_rope_with_sin_cos_cache.h"
#include "../../../../op_host/op_api/aclnn_rope_with_sin_cos_cache_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_rope_with_sin_cos_cache_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rope_with_sin_cos_cache_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "rope_with_sin_cos_cache_test TearDown" << endl;
    }
};

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_1)
{
    auto tensorPositions =
        TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_2)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64, 50}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64, 51}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64, 51}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_3)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64, 51}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64, 51}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_4)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64, 51}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_5)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rope_with_sin_cos_cache_test, Ascend910B_case_6)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnRopeWithSinCosCache,
        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize, isNeoxStyle),
        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rope_with_sin_cos_cache_test, rope_v2_case_7)
{
    auto tensorPositions =
        TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 255).Value(vector<int64_t>{7, 6, 5, 4, 1, 0, 2, 3});

    auto tensorQueryIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorKeyIn = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    auto tensorCosSinCache = TensorDesc({20, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    vector<int64_t> mropeSectionContent = {16, 24, 24};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionContent.data(), mropeSectionContent.size());

    int64_t headSize = 64;
    bool isNeoxStyle = true;
    int64_t cacheMode = 1;

    auto tensorQueryOutDesc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorKeyOutDesc = TensorDesc({4, 64, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRopeWithSinCosCacheV2,
                        INPUT(tensorPositions, tensorQueryIn, tensorKeyIn, tensorCosSinCache, mropeSection, headSize,
                              isNeoxStyle, cacheMode),
                        OUTPUT(tensorQueryOutDesc, tensorKeyOutDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
