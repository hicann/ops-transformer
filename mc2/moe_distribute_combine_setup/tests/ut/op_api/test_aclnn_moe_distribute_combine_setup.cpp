/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_moe_distribute_combine_setup.cpp
 * \brief aclnn ut
 */

#include <float.h>
#include <array>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_moe_distribute_combine_setup.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;

namespace MoeDistributeCombineSetupUT {
static aclTensor *CreateAclTensor(const std::vector<int64_t> shape, aclDataType dataType, aclFormat format)
{
    void *storage_data = nullptr;
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    aclTensor *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                                        shape.size(), storage_data);
    assert(tensor != nullptr);
    return tensor;
}

static aclTensor *CreateAclTensorOrNull(const std::vector<int64_t> shape, aclDataType dataType, aclFormat format)
{
    if (shape.empty()) {
        return nullptr;
    }
    return CreateAclTensor(shape, dataType, format);
}

class TestAclnnMoeDistributeCombineSetup : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        std::cout << "TestAclnnMoeDistributeCombineSetup SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        std::cout << "TestAclnnMoeDistributeCombineSetup TearDown" << std::endl;
    }
};

struct MoeDistributeCombineSetupAclnnTestParam {
    string caseName;

    // shape
    std::vector<int64_t> expandXShape;
    std::vector<int64_t> expertIdsShape;
    std::vector<int64_t> assistInfoForCombineShape;
    std::vector<int64_t> quantExpandXOutShape;
    std::vector<int64_t> commCmdInfoOutShape;

    // 通信域标识
    char *groupEp;
    int64_t epWorldSize;
    int64_t epRankId;
    int64_t moeExpertNum;
    int64_t expertShardType;
    int64_t sharedExpertNum;
    int64_t sharedExpertRankNum;
    int64_t globalBs;
    int64_t commQuantMode;
    int64_t commType;
    char *commAlg;

    // dtype
    aclDataType expandXDtype;
    aclDataType expertIdsDtype;
    aclDataType assistInfoForCombineDtype;
    aclDataType quantExpandXOutDtype;
    aclDataType commCmdInfoOutDtype;

    // format
    aclFormat expandXFormat;
    aclFormat expertIdsFormat;
    aclFormat assistInfoForCombineFormat;
    aclFormat quantExpandXOutFormat;
    aclFormat commCmdInfoOutFormat;

    // 返回状态
    aclnnStatus aclnnStatusUt;
};

// 用例列表集
static MoeDistributeCombineSetupAclnnTestParam g_casesParams[] = {
    // 正常用例
    {"test_aclnn_moe_distribute_combine_setup_normal_1",
     {96, 4096},
     {8, 6},
     {12288},
     {96, 6144},
     {1568},
     "MoeDistributeCombineSetup_test_groupEp",
     2,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_setup_normal_2",
     {4096, 7168},
     {256, 8},
     {524288},
     {4096, 10752},
     {65568},
     "MoeDistributeCombineSetup_test_groupEp",
     2,
     0,
     32,
     0,
     0,
     0,
     512,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_setup_normal_3",
     {512, 7168},
     {16, 8},
     {65536},
     {512, 10752},
     {8320},
     "MoeDistributeCombineSetup_test_groupEp",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_BF16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_setup_normal_4",
     {512, 4096},
     {16, 6},
     {65536},
     {512, 6144},
     {8320},
     "MoeDistributeCombineSetup_test_groupEp",
     8,
     0,
     32,
     0,
     0,
     0,
     128,
     0,
     2,
     "",
     ACL_BF16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    // 异常用例
    {"test_aclnn_moe_distribute_combine_setup_nullptr_expand_x",
     {},
     {16, 6},
     {24576},
     {192, 6144},
     {3104},
     "MoeDistributeCombineSetup_test_groupEp",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_setup_nullptr_expert_ids",
     {192, 4096},
     {},
     {24576},
     {192, 6144},
     {3104},
     "MoeDistributeCombineSetup_test_groupEp",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_setup_nullptr_assist_info_for_combine",
     {192, 4096},
     {16, 6},
     {},
     {192, 6144},
     {3104},
     "MoeDistributeCombineSetup_test_groupEp",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_setup_nullptr_quant_expand_x_out",
     {192, 4096},
     {16, 6},
     {24576},
     {},
     {3104},
     "MoeDistributeCombineSetup_test_groupEp",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_setup_nullptr_comm_cmd_info_out",
     {192, 4096},
     {16, 6},
     {24576},
     {192, 6144},
     {},
     "MoeDistributeCombineSetup_test_groupEp",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_setup_nullptr_group_ep",
     {192, 4096},
     {16, 6},
     {24576},
     {192, 6144},
     {3104},
     nullptr,
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_setup_empty_group_ep",
     {192, 4096},
     {16, 6},
     {24576},
     {192, 6144},
     {3104},
     "",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_setup_group_ep_too_long",
     {192, 4096},
     {16, 6},
     {24576},
     {192, 6144},
     {3104},
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012"
     "34567890123456789",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_setup_expand_x_empty_shape",
     {},
     {16, 6},
     {24576},
     {192, 6144},
     {3104},
     "MoeDistributeCombineSetup_test_groupEp",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_setup_expand_x_zero_dim",
     {0, 4096},
     {16, 6},
     {24576},
     {192, 6144},
     {3104},
     "MoeDistributeCombineSetup_test_groupEp",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_moe_distribute_combine_setup_expand_x_negative_dim",
     {-1, 4096},
     {16, 6},
     {24576},
     {192, 6144},
     {3104},
     "MoeDistributeCombineSetup_test_groupEp",
     8,
     0,
     32,
     0,
     0,
     0,
     0,
     0,
     2,
     "",
     ACL_FLOAT16,
     ACL_INT32,
     ACL_INT32,
     ACL_INT8,
     ACL_INT32,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID},
};

static void TestOneParamCase(const MoeDistributeCombineSetupAclnnTestParam &param)
{
    std::cout << "run case " << param.caseName << std::endl;
    if (param.groupEp == nullptr) {
        std::cerr << "[ERROR]: groupEp is null" << std::endl;
        return;
    }
    std::vector<int64_t> expandXShape = param.expandXShape;
    std::vector<int64_t> expertIdsShape = param.expertIdsShape;
    std::vector<int64_t> assistInfoForCombineShape = param.assistInfoForCombineShape;
    std::vector<int64_t> quantExpandXOutShape = param.quantExpandXOutShape;
    std::vector<int64_t> commCmdInfoOutShape = param.commCmdInfoOutShape;
    char *groupEp = param.groupEp;
    int64_t epWorldSize = param.epWorldSize;
    int64_t epRankId = param.epRankId;
    int64_t moeExpertNum = param.moeExpertNum;
    int64_t expertShardType = param.expertShardType;
    int64_t sharedExpertNum = param.sharedExpertNum;
    int64_t sharedExpertRankNum = param.sharedExpertRankNum;
    int64_t globalBs = param.globalBs;
    int64_t commQuantMode = param.commQuantMode;
    int64_t commType = param.commType;
    char *commAlg = param.commAlg;
    aclDataType expandXDtype = param.expandXDtype;
    aclDataType expertIdsDtype = param.expertIdsDtype;
    aclDataType assistInfoForCombineDtype = param.assistInfoForCombineDtype;
    aclDataType quantExpandXOutDtype = param.quantExpandXOutDtype;
    aclDataType commCmdInfoOutDtype = param.commCmdInfoOutDtype;
    aclFormat expandXFormat = param.expandXFormat;
    aclFormat expertIdsFormat = param.expertIdsFormat;
    aclFormat assistInfoForCombineFormat = param.assistInfoForCombineFormat;
    aclFormat quantExpandXOutFormat = param.quantExpandXOutFormat;
    aclFormat commCmdInfoOutFormat = param.commCmdInfoOutFormat;
    aclnnStatus retStatus = param.aclnnStatusUt;

    // 封装
    aclTensor *expandX = CreateAclTensorOrNull(expandXShape, expandXDtype, expandXFormat);
    aclTensor *expertIds = CreateAclTensorOrNull(expertIdsShape, expertIdsDtype, expertIdsFormat);
    aclTensor *assistInfoForCombine =
        CreateAclTensorOrNull(assistInfoForCombineShape, assistInfoForCombineDtype, assistInfoForCombineFormat);
    aclTensor *quantExpandXOut =
        CreateAclTensorOrNull(quantExpandXOutShape, quantExpandXOutDtype, quantExpandXOutFormat);
    aclTensor *commCmdInfoOut = CreateAclTensorOrNull(commCmdInfoOutShape, commCmdInfoOutDtype, commCmdInfoOutFormat);

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombineSetup,
        INPUT(expandX, expertIds, assistInfoForCombine, groupEp, epWorldSize, epRankId, moeExpertNum, expertShardType,
              sharedExpertNum, sharedExpertRankNum, globalBs, commQuantMode, commType, commAlg),
        OUTPUT(quantExpandXOut, commCmdInfoOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
}

TEST_F(TestAclnnMoeDistributeCombineSetup, CasesParamsTest)
{
    if (std::size(g_casesParams) != 0) {
        uint64_t numCases = sizeof(g_casesParams) / sizeof(g_casesParams[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(g_casesParams[idx]);
        }
    }
}

TEST_F(TestAclnnMoeDistributeCombineSetup, TestTeardownCalcOutputSizeNormal)
{
    std::vector<int64_t> expandXShape = {192, 4096};
    aclTensor *expandX = CreateAclTensor(expandXShape, ACL_FLOAT16, ACL_FORMAT_ND);
    std::vector<int64_t> expertIdsShape = {16, 6};
    aclTensor *expertIds = CreateAclTensor(expertIdsShape, ACL_INT32, ACL_FORMAT_ND);
    std::vector<int64_t> assistInfoShape = {24576};
    aclTensor *assistInfo = CreateAclTensor(assistInfoShape, ACL_INT32, ACL_FORMAT_ND);
    std::string groupEp = "test_group";
    uint64_t tokenMsgSize = 0;
    uint64_t commCmdInfoOutSize = 0;
    aclnnStatus ret =
        aclnnMoeDistributeCombineSetupTeardownCalcOutputSize(expandX, expertIds, assistInfo, groupEp.c_str(), 8, 0, 32,
                                                             0, 0, 0, 0, 0, 2, "", tokenMsgSize, commCmdInfoOutSize);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_GT(tokenMsgSize, 0);
    EXPECT_GT(commCmdInfoOutSize, 0);
}

TEST_F(TestAclnnMoeDistributeCombineSetup, TestTeardownCalcOutputSizeNullptrExpandX)
{
    uint64_t tokenMsgSize = 0;
    uint64_t commCmdInfoOutSize = 0;
    aclnnStatus ret = aclnnMoeDistributeCombineSetupTeardownCalcOutputSize(
        nullptr, nullptr, nullptr, nullptr, 8, 0, 32, 0, 0, 0, 0, 0, 2, "", tokenMsgSize, commCmdInfoOutSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnMoeDistributeCombineSetup, TestTeardownCalcOutputSizeInvalidDimNum)
{
    std::vector<int64_t> expandXShape = {192};
    aclTensor *expandX = CreateAclTensor(expandXShape, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t tokenMsgSize = 0;
    uint64_t commCmdInfoOutSize = 0;
    aclnnStatus ret = aclnnMoeDistributeCombineSetupTeardownCalcOutputSize(
        expandX, nullptr, nullptr, nullptr, 8, 0, 32, 0, 0, 0, 0, 0, 2, "", tokenMsgSize, commCmdInfoOutSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnMoeDistributeCombineSetup, TestTeardownCalcOutputSizeInvalidDimValue)
{
    std::vector<int64_t> expandXShape = {0, 4096};
    aclTensor *expandX = CreateAclTensor(expandXShape, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t tokenMsgSize = 0;
    uint64_t commCmdInfoOutSize = 0;
    aclnnStatus ret = aclnnMoeDistributeCombineSetupTeardownCalcOutputSize(
        expandX, nullptr, nullptr, nullptr, 8, 0, 32, 0, 0, 0, 0, 0, 2, "", tokenMsgSize, commCmdInfoOutSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnMoeDistributeCombineSetup, TestUnsupportedNpuArch)
{
    op::SetPlatformNpuArch(NpuArch::DAV_1001);
    std::vector<int64_t> expandXShape = {192, 4096};
    aclTensor *expandX = CreateAclTensor(expandXShape, ACL_FLOAT16, ACL_FORMAT_ND);
    std::vector<int64_t> expertIdsShape = {16, 6};
    aclTensor *expertIds = CreateAclTensor(expertIdsShape, ACL_INT32, ACL_FORMAT_ND);
    std::vector<int64_t> assistInfoShape = {24576};
    aclTensor *assistInfo = CreateAclTensor(assistInfoShape, ACL_INT32, ACL_FORMAT_ND);
    std::vector<int64_t> quantExpandXShape = {192, 6144};
    aclTensor *quantExpandX = CreateAclTensor(quantExpandXShape, ACL_INT8, ACL_FORMAT_ND);
    std::vector<int64_t> commCmdInfoShape = {3104};
    aclTensor *commCmdInfo = CreateAclTensor(commCmdInfoShape, ACL_INT32, ACL_FORMAT_ND);
    std::string groupEp = "test_group";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnMoeDistributeCombineSetupGetWorkspaceSize(expandX, expertIds, assistInfo, groupEp.c_str(), 8,
                                                                     0, 32, 0, 0, 0, 0, 0, 2, "", quantExpandX,
                                                                     commCmdInfo, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
}

TEST_F(TestAclnnMoeDistributeCombineSetup, ExecuteCall)
{
    aclnnStatus ret = aclnnMoeDistributeCombineSetup(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(TestAclnnMoeDistributeCombineSetup, TestEmptyDimTensor)
{
    aclTensor *expandX = CreateAclTensor({}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor *expertIds = CreateAclTensor({16, 6}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor *assistInfo = CreateAclTensor({24576}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor *quantExpandXOut = CreateAclTensor({192, 6144}, ACL_INT8, ACL_FORMAT_ND);
    aclTensor *commCmdInfoOut = CreateAclTensor({3104}, ACL_INT32, ACL_FORMAT_ND);
    std::string groupEp = "test_group";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnMoeDistributeCombineSetupGetWorkspaceSize(expandX, expertIds, assistInfo, groupEp.c_str(), 8,
                                                                     0, 32, 0, 0, 0, 0, 0, 2, "", quantExpandXOut,
                                                                     commCmdInfoOut, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

} // namespace MoeDistributeCombineSetupUT
