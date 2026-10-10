/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "allto_allv_quant_grouped_mat_mul_api_ut_param.h"
#include "op_api_ut_common/op_api_ut.h"
#include "../../../op_api/aclnn_allto_allv_quant_grouped_mat_mul.h"

namespace AlltoAllvQuantGroupedMatMulUT {

class AclnnAlltoAllvQuantGroupedMatMulTest : public testing::TestWithParam<AlltoAllvQuantGroupedMatMulApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllvQuantGroupedMatMul AclnnAlltoAllvQuantGroupedMatMulTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvQuantGroupedMatMul AclnnAlltoAllvQuantGroupedMatMulTest TearDown" << std::endl;
    }
};

TEST_P(AclnnAlltoAllvQuantGroupedMatMulTest, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    auto ut =
        OP_API_UT(aclnnAlltoAllvQuantGroupedMatMul,
                  INPUT(param.gmmX, param.gmmWeight, param.gmmXScale, param.gmmWeightScale, nullptr, nullptr,
                        param.mmXOptional, param.mmWeightOptional, param.mmXScaleOptional, param.mmWeightScaleOptional,
                        param.gmmXQuantMode, param.gmmWeightQuantMode, param.mmXQuantMode, param.mmWeightQuantMode,
                        param.group.c_str(), param.epWorldSize, param.sendCounts, param.recvCounts,
                        param.transGmmWeight, param.transMmWeight, param.groupSize, param.permuteOutFlag),
                  OUTPUT(param.gmmY, param.mmYOptional, param.permuteOutOptional));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AlltoAllvQuantGroupedMatMul, AclnnAlltoAllvQuantGroupedMatMulTest,
    testing::ValuesIn(GetCasesFromCsv<AlltoAllvQuantGroupedMatMulApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AlltoAllvQuantGroupedMatMulApiUtParam>);

} // namespace AlltoAllvQuantGroupedMatMulUT
