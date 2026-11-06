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
#include "../../../op_host/text_gated_residual_tiling.h"
#include "tiling_case_executor.h"

TEST(TextGatedResidualTiling, SmallAndLargeBothDtypes)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t m : {32, 1024}) {
            optiling::TextGatedResidualCompileInfo compileInfo{};
            gert::TilingContextPara para("TextGatedResidual",
                                         {{{{1, m, 2048}, {1, m, 2048}}, dtype, ge::FORMAT_ND},
                                          {{{64, 2048}, {64, 2048}}, dtype, ge::FORMAT_ND},
                                          {{{2048, 64}, {2048, 64}}, dtype, ge::FORMAT_ND},
                                          {{{4, 2048}, {4, 2048}}, dtype, ge::FORMAT_ND},
                                          {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                         {{{{1, m, 512}, {1, m, 512}}, dtype, ge::FORMAT_ND},
                                          {{{1, m, 4}, {1, m, 4}}, dtype, ge::FORMAT_ND},
                                          {{{1, m, 4}, {1, m, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          {{{1, m, 64}, {1, m, 64}}, dtype, ge::FORMAT_ND},
                                          {{{1, m, 2048}, {1, m, 2048}}, dtype, ge::FORMAT_ND}},
                                         {{"channels", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                          {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                          {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
                                         &compileInfo, "Ascend910B", 20, 196608, 65536);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(para, info));
            EXPECT_GT(info.blockNum, 0);
            EXPECT_GE(info.tilingDataSize, 128);
        }
    }
}
