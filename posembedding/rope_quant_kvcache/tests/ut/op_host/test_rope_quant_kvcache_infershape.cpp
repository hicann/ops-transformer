/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_rope_quant_kvcache_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RopeQuantKvcache : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RopeQuantKvcache SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RopeQuantKvcache TearDown" << std::endl;
    }
};

TEST_F(RopeQuantKvcache, RopeQuantKvcache_infershape_case_0)
{
    std::vector<int64_t> size_splits = {1024, 128, 128};
    gert::InfershapeContextPara infershapeContextPara(
        "RopeQuantKvcache",
        {
            // input info
            {{{4, 1, 1280}, {4, 1, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 1}, {4, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            // attr
            {"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(size_splits)},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 1, 8, 128}, {4, 1, 1, 128}, {4, 1, 1, 128}, {4, 2048, 1, 128}, {4, 2048, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
