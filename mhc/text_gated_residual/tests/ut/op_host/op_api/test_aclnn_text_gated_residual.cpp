/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../../op_host/op_api/aclnn_text_gated_residual.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

TEST(TextGatedResidualApi, ValidOutputsBothDtypesAndCacheModes)
{
    for (auto dtype : {ACL_FLOAT16, ACL_BF16}) {
        for (bool cache : {false, true}) {
            auto x = TensorDesc({1, 32, 2048}, dtype, ACL_FORMAT_ND);
            auto down = TensorDesc({64, 2048}, dtype, ACL_FORMAT_ND);
            auto up = TensorDesc({2048, 64}, dtype, ACL_FORMAT_ND);
            auto gate = TensorDesc({4, 2048}, dtype, ACL_FORMAT_ND);
            auto gamma = TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND);
            auto mixed = TensorDesc({1, 32, 512}, dtype, ACL_FORMAT_ND);
            auto inject = TensorDesc({1, 32, 4}, dtype, ACL_FORMAT_ND);
            auto inv =
                TensorDesc(cache ? std::vector<int64_t>{1, 32, 4} : std::vector<int64_t>{0}, ACL_FLOAT, ACL_FORMAT_ND);
            auto d =
                TensorDesc(cache ? std::vector<int64_t>{1, 32, 64} : std::vector<int64_t>{0}, dtype, ACL_FORMAT_ND);
            auto u =
                TensorDesc(cache ? std::vector<int64_t>{1, 32, 2048} : std::vector<int64_t>{0}, dtype, ACL_FORMAT_ND);
            auto ut = OP_API_UT(aclnnTextGatedResidual, INPUT(x, down, up, gate, gamma, int64_t{4}, 1e-6, cache),
                                OUTPUT(mixed, inject, inv, d, u));
            uint64_t workspace = 0;
            aclOpExecutor* executor = nullptr;
            EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace, executor), ACLNN_SUCCESS);
        }
    }
}
