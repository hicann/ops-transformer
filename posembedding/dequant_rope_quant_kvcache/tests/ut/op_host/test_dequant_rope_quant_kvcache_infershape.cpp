/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class DequantRopeQuantKvcache : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DequantRopeQuantKvcache SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DequantRopeQuantKvcache TearDown" << std::endl;
    }
};

TEST_F(DequantRopeQuantKvcache, DequantRopeQuantKvcache_infershape_case_0) {
  gert::InfershapeContextPara infershapeContextPara("DequantRopeQuantKvcache",
                                            {
                                              {{{4, 1, 1280}, {4, 1, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}
                                            },
                                            {
                                              {"size_splits",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1024, 128, 128})},
                                              {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
                                              {"layout",Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                              {"kv_output",Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                                            }
                                          );
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 1, 8, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(DequantRopeQuantKvcache, DequantRopeQuantKvcache_infershape_case_1) {
  gert::InfershapeContextPara infershapeContextPara("DequantRopeQuantKvcache",
                                            {
                                              {{{4, 1280}, {4, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{4, 128}, {4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{4, 128}, {4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND}
                                            },
                                            {
                                              {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                              {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}
                                            },
                                            {
                                              {"size_splits",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1024, 128, 128})},
                                              {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
                                              {"layout",Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                              {"kv_output",Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
                                            }
                                          );
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
