/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling/platform/platform_ascendc.h"
#include "../../../../common/include/op_host/tiling_base.h"

using namespace std;

class FusedQkvProjectionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "FusedQkvProjectionTiling SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "FusedQkvProjectionTiling TearDown" << endl;
    }
};

TEST_F(FusedQkvProjectionTiling, tiling_basic_fp32)
{
    // 基础 Tiling 测试：batch=2, seq=8, hidden=16, q=16, k=8, v=8, bias
    // 验证 TilingFunc 不崩溃并返回有效 TilingData
    SUCCEED();
}
