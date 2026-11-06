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
#include "infer_shape_case_executor.h"

TEST(TextGatedResidualInferShape, CacheOutputShapes)
{
    for (bool cache : {false, true}) {
        gert::InfershapeContextPara para("TextGatedResidual",
                                         {{{{2, 3, 2048}, {2, 3, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                          {{{64, 2048}, {64, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                          {{{2048, 64}, {2048, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                          {{{4, 2048}, {4, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                          {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                         {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
                                         {{"channels", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                          {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                          {"need_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(cache)}});
        std::vector<std::vector<int64_t>> expected = {
            {2, 3, 512},
            {2, 3, 4},
            cache ? std::vector<int64_t>{2, 3, 4} : std::vector<int64_t>{0},
            cache ? std::vector<int64_t>{2, 3, 64} : std::vector<int64_t>{0},
            cache ? std::vector<int64_t>{2, 3, 2048} : std::vector<int64_t>{0}};
        ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
    }
}
