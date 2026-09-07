/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_api/aclnn_moe_distribute_combine_v3.h"

#include <gmock/gmock.h>
#include "gtest/gtest.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace MoeDistributeCombineV3 {
namespace {
constexpr int64_t EP_WORLD_SIZE = 288;
constexpr int64_t TP_WORLD_SIZE = 1;
constexpr int64_t EP_RANK_ID = 0;
constexpr int64_t TP_RANK_ID = 0;
constexpr int64_t EXPERT_SHARD_TYPE = 0;
constexpr int64_t SHARED_EXPERT_NUM = 1;
constexpr int64_t SHARED_EXPERT_RANK_NUM = 32;
constexpr int64_t MOE_EXPERT_NUM = 256;
constexpr int64_t GLOBAL_BS = 0;
constexpr int64_t OUT_DTYPE = 0;
constexpr int64_t COMM_QUANT_MODE = 0;
constexpr int64_t GROUP_LIST_TYPE = 0;
} // namespace

class L2MoeDistributeCombineV3Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeCombineV3Test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "L2MoeDistributeCombineV3Test TearDown" << endl;
    }
};

TEST_F(L2MoeDistributeCombineV3Test, TestMoeDistributeCombineV3FirstApi)
{
    TensorDesc expandX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc expertIds = TensorDesc({32, 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc assistInfo = TensorDesc({32 * 8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc epSendCounts = TensorDesc({288}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc expertScales = TensorDesc({32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc elasticInfo = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc oriX = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc constExpertAlpha1 = TensorDesc({1, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc constExpertAlpha2 = TensorDesc({1, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc constExpertV = TensorDesc({1, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc({32, 7168}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnMoeDistributeCombineV3,
        INPUT(expandX, expertIds, assistInfo, epSendCounts, expertScales, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
              "test_moe_distribute_combine_ep", EP_WORLD_SIZE, EP_RANK_ID, MOE_EXPERT_NUM,
              "test_moe_distribute_combine_tp", TP_WORLD_SIZE, TP_RANK_ID, EXPERT_SHARD_TYPE, SHARED_EXPERT_NUM,
              SHARED_EXPERT_RANK_NUM, GLOBAL_BS, OUT_DTYPE, COMM_QUANT_MODE, GROUP_LIST_TYPE, "test", 1, 1, 1),
        OUTPUT(x));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2MoeDistributeCombineV3Test, TestMoeDistributeCombineV3ExecuteEntry)
{
    aclnnStatus ret = aclnnMoeDistributeCombineV3(nullptr, 0, nullptr, nullptr);
    EXPECT_THAT(ret, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                    testing::Eq(ACLNN_ERR_PARAM_INVALID), testing::Eq(ACLNN_ERR_INNER)));
}
} // namespace MoeDistributeCombineV3
