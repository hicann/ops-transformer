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
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_norm_rope_concat.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_interleave_rope_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_norm_rope_concat SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_norm_rope_concat TearDown" << endl;
    }
};

// dtype fp32
TEST_F(l2_norm_rope_concat, Ascend910B2_norm_rope_concat_fp32)
{
    //    query, key, value, encoderQuery, encoderKey, encoderValue, normQueryWeight, normQueryBias, normKeyWeight,
    //     normKeyBias, normAddedQueryWeight, normAddedQueryBias, normAddedKeyWeight, normAddedKeyBias, ropeSin,
    //     ropeCos, normType, normAddedType, ropeType, concatOrder, eps, isTraing, queryOutput, keyOutput, valueOutput,
    //     normQueryMean, normQueryRstd, normKeyMean, normKeyRstd, normAddedQueryMean, normAddedQueryRstd,
    //     normAddedKeyMean, normAddedKeyRstd
    auto query = TensorDesc({1, 8, 8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto key = TensorDesc({1, 8, 8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto value = TensorDesc({1, 8, 8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto encoderQuery = TensorDesc({1, 8, 8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto encoderKey = TensorDesc({1, 8, 8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto encoderValue = TensorDesc({1, 8, 8, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto normQueryWeight = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto normQueryBias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto normKeyWeight = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto normKeyBias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto normAddedQueryWeight = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto normAddedQueryBias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto normAddedKeyWeight = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto normAddedKeyBias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ropeSin = TensorDesc({16, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ropeCos = TensorDesc({16, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto queryOutput = TensorDesc({1, 8, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyOutput = TensorDesc({1, 8, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valueOutput = TensorDesc({1, 8, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normQueryMean = TensorDesc({1, 8, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normQueryRstd = TensorDesc({1, 8, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normKeyMean = TensorDesc({1, 8, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normKeyRstd = TensorDesc({1, 8, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normAddedQueryMean = TensorDesc({1, 8, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normAddedQueryRstd = TensorDesc({1, 8, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normAddedKeyMean = TensorDesc({1, 8, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normAddedKeyRstd = TensorDesc({1, 8, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnNormRopeConcat,
                  INPUT(query, key, value, encoderQuery, encoderKey, encoderValue, normQueryWeight, normQueryBias,
                        normKeyWeight, normKeyBias, normAddedQueryWeight, normAddedQueryBias, normAddedKeyWeight,
                        normAddedKeyBias, ropeSin, ropeCos, 1, 1, 1, 0, 1e-5, true),
                  OUTPUT(queryOutput, keyOutput, valueOutput, normQueryMean, normQueryRstd, normKeyMean, normKeyRstd,
                         normAddedQueryMean, normAddedQueryRstd, normAddedKeyMean, normAddedKeyRstd));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
