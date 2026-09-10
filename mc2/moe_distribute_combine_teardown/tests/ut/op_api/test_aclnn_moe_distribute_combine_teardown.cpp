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
 * \file test_aclnn_moe_distribute_combine_teardown.cpp
 * \brief aclnn ut
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_moe_distribute_combine_teardown.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

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

class TestAclnnMoeDistributeCombineTeardown : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
        cout << "TestAclnnMoeDistributeCombineTeardown SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "TestAclnnMoeDistributeCombineTeardown TearDown" << endl;
    }
};

struct MoeDistributeCombineTeardownAclnnTestParam {
    string case_name;
    vector<int64_t> expandXShape;
    vector<int64_t> quantExpandXShape;
    vector<int64_t> expertIdsShape;
    vector<int64_t> expandIdxShape;
    vector<int64_t> expertScalesShape;
    vector<int64_t> commCmdInfoShape;
    vector<int64_t> xActiveMaskOptionalShape;
    vector<int64_t> sharedExpertXOptionalShape;
    vector<int64_t> xOutShape;
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
    aclnnStatus aclnnStatusUt;
};

static char g_emptyGroupEp[] = "";
static char g_longGroupEp[] = "hccl_world_group_very_long_name_that_exceeds_the_maximum_allowed_length_of_127_"
                              "characters_for_group_ep_attribute_in_moe_distribute_combine_teardown_operator";

static MoeDistributeCombineTeardownAclnnTestParam g_casesParams[] = {
    {"test_aclnn_moe_distribute_combine_teardown_bs8_h4096_k6_fp16_globalBs0_epWorldSize2_success",
     {48, 4096},
     {48, 4096},
     {8, 6},
     {8, 6},
     {8, 6},
     {128},
     {},
     {},
     {48, 4096},
     "moe_distribute_combine_teardown_test_group",
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_teardown_bs8_h4096_k8_fp16_globalBs64_epWorldSize8_success",
     {64, 4096},
     {64, 4096},
     {8, 8},
     {8, 8},
     {8, 8},
     {128},
     {},
     {},
     {64, 4096},
     "moe_distribute_combine_teardown_test_group",
     8,
     0,
     8,
     0,
     0,
     0,
     64,
     0,
     0,
     "",
     ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_teardown_bs8_h7168_k6_bf16_globalBs0_epWorldSize8_success",
     {48, 7168},
     {48, 7168},
     {8, 6},
     {8, 6},
     {8, 6},
     {128},
     {},
     {},
     {48, 7168},
     "moe_distribute_combine_teardown_test_group",
     8,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_teardown_bs8_h7168_k8_bf16_globalBs16_epWorldSize2_success",
     {64, 7168},
     {64, 7168},
     {8, 8},
     {8, 8},
     {8, 8},
     {128},
     {},
     {},
     {64, 7168},
     "moe_distribute_combine_teardown_test_group",
     2,
     0,
     8,
     0,
     0,
     0,
     16,
     0,
     0,
     "",
     ACLNN_SUCCESS},
    {"test_aclnn_moe_distribute_combine_teardown_expandX_null_error1",
     {},
     {48, 4096},
     {8, 6},
     {8, 6},
     {8, 6},
     {128},
     {},
     {},
     {48, 4096},
     "moe_distribute_combine_teardown_test_group",
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_quantExpandX_null_error2",
     {48, 4096},
     {},
     {8, 6},
     {8, 6},
     {8, 6},
     {128},
     {},
     {},
     {48, 4096},
     "moe_distribute_combine_teardown_test_group",
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_expertIds_null_error3",
     {48, 4096},
     {48, 4096},
     {},
     {8, 6},
     {8, 6},
     {128},
     {},
     {},
     {48, 4096},
     "moe_distribute_combine_teardown_test_group",
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_expandIdx_null_error4",
     {48, 4096},
     {48, 4096},
     {8, 6},
     {},
     {8, 6},
     {128},
     {},
     {},
     {48, 4096},
     "moe_distribute_combine_teardown_test_group",
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_expertScales_null_error5",
     {48, 4096},
     {48, 4096},
     {8, 6},
     {8, 6},
     {},
     {128},
     {},
     {},
     {48, 4096},
     "moe_distribute_combine_teardown_test_group",
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_commCmdInfo_null_error6",
     {48, 4096},
     {48, 4096},
     {8, 6},
     {8, 6},
     {8, 6},
     {},
     {},
     {},
     {48, 4096},
     "moe_distribute_combine_teardown_test_group",
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_xOut_null_error7",
     {48, 4096},
     {48, 4096},
     {8, 6},
     {8, 6},
     {8, 6},
     {128},
     {},
     {},
     {},
     "moe_distribute_combine_teardown_test_group",
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_groupEp_null_error8",
     {48, 4096},
     {48, 4096},
     {8, 6},
     {8, 6},
     {8, 6},
     {128},
     {},
     {},
     {48, 4096},
     nullptr,
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_groupEp_empty_error9",
     {48, 4096},
     {48, 4096},
     {8, 6},
     {8, 6},
     {8, 6},
     {128},
     {},
     {},
     {48, 4096},
     g_emptyGroupEp,
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_ERR_PARAM_NULLPTR},
    {"test_aclnn_moe_distribute_combine_teardown_groupEp_too_long_error10",
     {48, 4096},
     {48, 4096},
     {8, 6},
     {8, 6},
     {8, 6},
     {128},
     {},
     {},
     {48, 4096},
     g_longGroupEp,
     2,
     0,
     8,
     0,
     0,
     0,
     0,
     0,
     0,
     "",
     ACLNN_ERR_PARAM_INVALID},
};

static void TestOneParamCase(const MoeDistributeCombineTeardownAclnnTestParam &param)
{
    std::cout << "run case " << param.case_name << std::endl;
    vector<int64_t> expandXShape = param.expandXShape;
    vector<int64_t> quantExpandXShape = param.quantExpandXShape;
    vector<int64_t> expertIdsShape = param.expertIdsShape;
    vector<int64_t> expandIdxShape = param.expandIdxShape;
    vector<int64_t> expertScalesShape = param.expertScalesShape;
    vector<int64_t> commCmdInfoShape = param.commCmdInfoShape;
    vector<int64_t> xActiveMaskOptionalShape = param.xActiveMaskOptionalShape;
    vector<int64_t> sharedExpertXOptionalShape = param.sharedExpertXOptionalShape;
    vector<int64_t> xOutShape = param.xOutShape;
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
    aclnnStatus retStatus = param.aclnnStatusUt;
    aclTensor *expandX = CreateAclTensorOrNull(expandXShape, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor *quantExpandX = CreateAclTensorOrNull(quantExpandXShape, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor *expertIds = CreateAclTensorOrNull(expertIdsShape, ACL_INT32, ACL_FORMAT_ND);
    aclTensor *expandIdx = CreateAclTensorOrNull(expandIdxShape, ACL_INT32, ACL_FORMAT_ND);
    aclTensor *expertScales = CreateAclTensorOrNull(expertScalesShape, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor *commCmdInfo = CreateAclTensorOrNull(commCmdInfoShape, ACL_INT32, ACL_FORMAT_ND);
    aclTensor *xActiveMaskOptional = CreateAclTensorOrNull(xActiveMaskOptionalShape, ACL_BOOL, ACL_FORMAT_ND);
    aclTensor *sharedExpertXOptional = CreateAclTensorOrNull(sharedExpertXOptionalShape, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor *xOut = CreateAclTensorOrNull(xOutShape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnMoeDistributeCombineTeardown,
                  INPUT(expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo, xActiveMaskOptional,
                        sharedExpertXOptional, groupEp, epWorldSize, epRankId, moeExpertNum, expertShardType,
                        sharedExpertNum, sharedExpertRankNum, globalBs, commQuantMode, commType, commAlg),
                  OUTPUT(xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
}

TEST_F(TestAclnnMoeDistributeCombineTeardown, CasesParamsTest)
{
    if (std::size(g_casesParams) != 0) {
        uint64_t numCases = sizeof(g_casesParams) / sizeof(g_casesParams[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(g_casesParams[idx]);
        }
    }
}

class TestAclnnMoeDistributeCombineTeardownUnsupportedArch : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3113);
        cout << "TestAclnnMoeDistributeCombineTeardownUnsupportedArch SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
        cout << "TestAclnnMoeDistributeCombineTeardownUnsupportedArch TearDown" << endl;
    }
};

TEST_F(TestAclnnMoeDistributeCombineTeardownUnsupportedArch, UnsupportedArchError)
{
    aclTensor *expandX = CreateAclTensor({48, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor *quantExpandX = CreateAclTensor({48, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor *expertIds = CreateAclTensor({8, 6}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor *expandIdx = CreateAclTensor({8, 6}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor *expertScales = CreateAclTensor({8, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor *commCmdInfo = CreateAclTensor({128}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor *xActiveMaskOptional = nullptr;
    aclTensor *sharedExpertXOptional = nullptr;
    aclTensor *xOut = CreateAclTensor({48, 4096}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t epWorldSize = 2;
    int64_t epRankId = 0;
    int64_t moeExpertNum = 8;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t globalBs = 0;
    int64_t commQuantMode = 0;
    int64_t commType = 0;

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombineTeardown,
        INPUT(expandX, quantExpandX, expertIds, expandIdx, expertScales, commCmdInfo, xActiveMaskOptional,
              sharedExpertXOptional, "moe_distribute_combine_teardown_test_group", epWorldSize, epRankId, moeExpertNum,
              expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, commQuantMode, commType, ""),
        OUTPUT(xOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnMoeDistributeCombineTeardown, ExecuteCall)
{
    aclnnStatus ret = aclnnMoeDistributeCombineTeardown(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}
