/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class ScatterPaKvCache : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScatterPaKvCache SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ScatterPaKvCache TearDown" << std::endl;
    }
};

TEST_F(ScatterPaKvCache, scatter_pa_kv_cache_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ScatterPaKvCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},           // key
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // keyCache
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},                           // slotMapping
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},           // value
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // valueCache
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},

        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // keyCache
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // valueCache
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"scatter_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("None")},
            {"strides", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"offsets", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{306, 128, 1, 128}, {306, 128, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
