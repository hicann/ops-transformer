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
 * \file test_masked_causal_conv1d_backward_tiling.cpp
 * \brief Unit tests for MaskedCausalConv1dBackward tiling logic
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/masked_causal_conv1d_backward_tiling_arch35.h"

using namespace std;

class MaskedCausalConv1dBackwardTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaskedCausalConv1dBackwardTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaskedCausalConv1dBackwardTilingArch35 TearDown" << std::endl;
    }
};

TEST_F(MaskedCausalConv1dBackwardTilingArch35, MaskedCausalConv1dBackward_950)
{
    optiling::MaskedCausalConv1dBackwardArch35CompileInfo compileInfo = {64, 261888};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {

    };

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1dBackward",
                                              {
                                                  {{{4096, 4, 768}, {4096, 4, 768}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096, 4, 768}, {4096, 4, 768}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{3, 768}, {3, 768}}, ge::DT_BF16, ge::FORMAT_ND},

                                              },
                                              {
                                                  {{{4096, 4, 768}, {4096, 4, 768}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{3, 768}, {3, 768}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "12 0 64 64 1 26 64 64 158 157 14 12 1 26 64 64 158 157 0 4096 4 768 3 ";

    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
