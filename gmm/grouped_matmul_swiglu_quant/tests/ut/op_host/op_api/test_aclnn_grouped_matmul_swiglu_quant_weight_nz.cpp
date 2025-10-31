/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class aclnnGroupedMatmulSwigluQuantWeightNz_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "aclnnGroupedMatmulSwigluQuantWeightNz_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "aclnnGroupedMatmulSwigluQuantWeightNz_test TearDown" << endl;
    }
};

TEST_F(aclnnGroupedMatmulSwigluQuantWeightNz_test, ascend910B2_test_opapi_w8a8_normal_case)
{
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 2048;
    int64_t e = 4;
    int64_t bs = 24;
    int64_t bsdp = 8;
    int64_t quantGroupSize = 256;

    TensorDesc x = TensorDesc({m, k}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc weight =
        TensorDesc({e, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {e, n / 32, k / 16, 16, 32}).ValueRange(-1, 1);
    TensorDesc weightScale = TensorDesc({e, n}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc xScale = TensorDesc({m}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 3);
    TensorDesc groupList = TensorDesc({e}, ACL_INT64, ACL_FORMAT_ND).ValueRange(bs, bs);
    TensorDesc out1 = TensorDesc({m, n}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc out2 = TensorDesc({m,}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnGroupedMatmulSwigluQuantWeightNZ,
                        INPUT(x, weight, nullptr, nullptr, weightScale, xScale, groupList),
                        OUTPUT(out1, out2, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}