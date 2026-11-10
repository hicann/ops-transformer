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
#include "../../../op_api/aclnn_matmul_allto_all.h"
#include "../../../op_api/aclnn_quant_matmul_allto_all.h"
#include "../../../../common/tests/ut/api_pointer_error_assertions.h"

TEST(MatmulAlltoAllPointerErrors, MissingExecutorReportsOnceThroughPublicApi)
{
    Mc2ErrorCodeUT::ExcelErrorReportScope errors;
    EXPECT_EQ(aclnnMatmulAlltoAll(nullptr, 0, nullptr, nullptr), ACLNN_ERR_INNER);
    Mc2ErrorCodeUT::ExpectRequiredApiPointerError("aclnnMatmulAlltoAllBase", "executor");
}

TEST(MatmulAlltoAllPointerErrors, QuantMissingExecutorReportsOnceThroughPublicApi)
{
    Mc2ErrorCodeUT::ExcelErrorReportScope errors;
    EXPECT_EQ(aclnnQuantMatmulAlltoAll(nullptr, 0, nullptr, nullptr), ACLNN_ERR_INNER);
    Mc2ErrorCodeUT::ExpectRequiredApiPointerError("aclnnQuantMatmulAlltoAllBase", "executor");
}
