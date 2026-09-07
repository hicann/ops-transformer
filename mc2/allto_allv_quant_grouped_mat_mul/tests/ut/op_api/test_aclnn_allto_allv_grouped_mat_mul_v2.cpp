/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "allto_allv_quant_grouped_mat_mul_api_ut_param_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "../../../op_api/aclnn_allto_allv_quant_grouped_mat_mul_v2.h"

namespace AlltoAllvQuantGroupedMatMulV2UT {

class AclnnAlltoAllvQuantGroupedMatMulV2Test : public testing::TestWithParam<AlltoAllvQuantGroupedMatMulV2ApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllvQuantGroupedMatMul AclnnAlltoAllvQuantGroupedMatMulV2Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvQuantGroupedMatMul AclnnAlltoAllvQuantGroupedMatMulV2Test TearDown" << std::endl;
    }
};

TEST_P(AclnnAlltoAllvQuantGroupedMatMulV2Test, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut = OP_API_UT(
        aclnnAlltoAllvQuantGroupedMatMulV2,
        INPUT(param.gmmX, param.gmmWeight, param.gmmXScale, param.gmmWeightScale, nullptr, nullptr, param.mmXOptional,
              param.mmWeightOptional, param.mmXScaleOptional, param.mmWeightScaleOptional, param.gmmXQuantMode,
              param.gmmWeightQuantMode, param.mmXQuantMode, param.mmWeightQuantMode, param.group.c_str(),
              param.commMode.c_str(), param.epWorldSize, param.sendCounts, param.recvCounts, param.transGmmWeight,
              param.transMmWeight, param.groupSize, param.permuteOutFlag),
        OUTPUT(param.gmmY, param.mmYOptional, param.permuteOutOptional));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AlltoAllvQuantGroupedMatMul, AclnnAlltoAllvQuantGroupedMatMulV2Test,
    testing::ValuesIn(GetCasesFromCsv<AlltoAllvQuantGroupedMatMulV2ApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AlltoAllvQuantGroupedMatMulV2ApiUtParam>);

} // namespace AlltoAllvQuantGroupedMatMulV2UT
