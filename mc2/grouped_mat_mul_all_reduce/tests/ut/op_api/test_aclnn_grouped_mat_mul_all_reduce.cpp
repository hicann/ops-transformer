/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_grouped_mat_mul_all_reduce.cpp
 * \brief
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_grouped_mat_mul_all_reduce.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
namespace {
class L2GroupedMatMulAllReduceTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "gemm_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "gemm_test TearDown" << endl;
    }
};

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case0)
{
    vector<vector<int64_t>> xDims = {{256, 256}, {1024, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{100}, {256}};
    vector<vector<int64_t>> outDims = {{256, 256}, {1024, 256}};
    vector<int64_t> groupList = {};
    int64_t splitItemOptional = 0;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto x = TensorDesc(xDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
        xGroup.emplace_back(x);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        outGroup.emplace_back(out);
    }
    auto xList = TensorListDesc({xGroup[0], xGroup[1]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray *array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case1)
{
    vector<vector<int64_t>> xDims = {{1280, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{100}, {256}};
    vector<vector<int64_t>> outDims = {{256, 256}, {1024, 256}};
    vector<int64_t> groupList = {256, 1280};
    int64_t splitItemOptional = 1;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    auto x = TensorDesc(xDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    xGroup.emplace_back(x);
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        outGroup.emplace_back(out);
    }
    auto xList = TensorListDesc({xGroup[0]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray *array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case2)
{
    vector<vector<int64_t>> xDims = {{256, 256}, {1024, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{100}, {256}};
    vector<vector<int64_t>> outDims = {{1280, 256}};
    vector<int64_t> groupList = {};
    int64_t splitItemOptional = 2;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    auto out = TensorDesc(outDims[0], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    outGroup.emplace_back(out);
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto x = TensorDesc(xDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        xGroup.emplace_back(x);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
    }

    auto xList = TensorListDesc({xGroup[0], xGroup[1]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0]});
    aclIntArray *array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case3)
{
    vector<vector<int64_t>> xDims = {{1280, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{100}, {256}};
    vector<vector<int64_t>> outDims = {{1280, 256}};
    vector<int64_t> groupList = {256, 1280};
    int64_t splitItemOptional = 3;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    auto x = TensorDesc(xDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    xGroup.emplace_back(x);
    auto out = TensorDesc(outDims[0], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    outGroup.emplace_back(out);
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
    }
    auto xList = TensorListDesc({xGroup[0]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0]});
    aclIntArray *array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case4)
{
    vector<vector<int64_t>> xDims = {{256, 256}, {1024, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {100, 256}};
    vector<vector<int64_t>> biasDims = {{256}, {256}};
    vector<vector<int64_t>> outDims = {{256, 256}, {1024, 256}};
    vector<int64_t> groupList = {};
    int64_t splitItemOptional = 0;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    for (size_t i = 0; i < weightDims.size(); i++) {
        auto x = TensorDesc(xDims[i], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto weight = TensorDesc(weightDims[i], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_BF16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
        xGroup.emplace_back(x);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        outGroup.emplace_back(out);
    }
    auto xList = TensorListDesc({xGroup[0], xGroup[1]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray *array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case5)
{
    vector<vector<int64_t>> xDims = {{256, 256}, {1024, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{100}, {256}};
    vector<vector<int64_t>> outDims = {{256, 256}, {1024, 256}};
    vector<int64_t> groupList = {};
    int64_t splitItemOptional = 5;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    for (size_t i = 0; i < weightDims.size(); i++) {
        auto x = TensorDesc(xDims[i], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto weight = TensorDesc(weightDims[i], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_BF16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
        xGroup.emplace_back(x);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        outGroup.emplace_back(out);
    }
    auto xList = TensorListDesc({xGroup[0], xGroup[1]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray *array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case7)
{
    vector<vector<int64_t>> xDims = {{256, 256}, {1024, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{0}};
    vector<vector<int64_t>> outDims = {{256, 100}, {1024, 256}};
    vector<int64_t> groupList = {};
    int64_t splitItemOptional = 0;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    for (size_t i = 0; i < weightDims.size(); i++) {
        auto x = TensorDesc(xDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
        xGroup.emplace_back(x);
        weightGroup.emplace_back(weight);
        outGroup.emplace_back(out);
    }
    auto bias = TensorDesc(biasDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    biasGroup.emplace_back(bias);
    auto xList = TensorListDesc({xGroup[0], xGroup[1]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray *array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case10)
{
    vector<vector<int64_t>> xDims = {{1280, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{256}, {256}};
    vector<vector<int64_t>> scaleDims = {{256}, {256}};
    vector<vector<int64_t>> outDims = {{1280, 256}};
    vector<int64_t> groupList = {256, 1024};
    int64_t splitItemOptional = 3;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> scaleGroup;
    std::vector<TensorDesc> outGroup;

    auto x = TensorDesc(xDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    xGroup.emplace_back(x);
    auto out = TensorDesc(outDims[0], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    outGroup.emplace_back(out);
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto weight = TensorDesc(weightDims[i], ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto scale = TensorDesc(scaleDims[i], ACL_UINT64, ACL_FORMAT_ND).ValueRange(0.01, 0.04);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        scaleGroup.emplace_back(scale);
    }
    auto xList = TensorListDesc({xGroup[0]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto scaleList = TensorListDesc({scaleGroup[0], scaleGroup[1]});
    auto outList = TensorListDesc({outGroup[0]});
    aclIntArray *array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case11)
{
    vector<vector<int64_t>> xDims = {{1280, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{256}, {256}};
    vector<vector<int64_t>> outDims = {{256, 256}, {1024, 256}};
    vector<int64_t> groupList = {1280, 256};
    int64_t splitItemOptional = 1;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    auto x = TensorDesc(xDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    xGroup.emplace_back(x);
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        outGroup.emplace_back(out);
    }
    auto xList = TensorListDesc({xGroup[0]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray *array = aclCreateIntArray(groupList.data(), groupList.size());

    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 8, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // namespace
