/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/* !
 * \file test_grouped_matmul_swiglu_quant_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

class GroupedMatmulSwigluQuantInferShape_TEST : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantInferShape_TEST SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantInferShape_TEST TearDown" << std::endl;
    }
};

TEST_F(GroupedMatmulSwigluQuantInferShape_TEST, grouped_matmul_swiglu_quant_infershape_00)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmulSwigluQuant",
        {
            {{{1024, 256}, {16, 256}}, ge::DT_INT8, ge::FORMAT_ND},           // x
            {{{16, 256, 4096}, {16, 256, 4096}}, ge::DT_INT8, ge::FORMAT_ND}, // weight
            {{{16, 4096}, {16, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},          // weightScale
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // xscale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // weightAssistMatrix
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // groupList
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // yscale
        },
        {{"is_enable_weight_assistance_matri", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"dequant_mod", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 2048}, {1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GroupedMatmulSwigluQuantInferShape_TEST, grouped_matmul_swiglu_quant_infershape_01)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmulSwigluQuant",
        {
            {{{1024, 256}, {16, 256}}, ge::DT_INT8, ge::FORMAT_ND},           // x
            {{{16, 256, 4096}, {16, 256, 4096}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{16, 4096}, {16, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},            // weightScale
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // xscale
            {{{16, 4096}, {16, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},          // weightAssistMatrix
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // groupList
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // yscale
        },
        {{"is_enable_weight_assistance_matri", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"dequant_mod", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 2048}, {1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GroupedMatmulSwigluQuantInferShape_TEST, grouped_matmul_swiglu_quant_infershape_02)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmulSwigluQuant",
        {
            {{{1024, 256}, {16, 256}}, ge::DT_INT8, ge::FORMAT_ND},           // x
            {{{16, 256, 4096}, {16, 256, 4096}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{16, 4, 4096}, {16, 4, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},    // weightScale
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // xscale
            {{{16, 4096}, {16, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},          // weightAssistMatrix
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // groupList
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // yscale
        },
        {{"is_enable_weight_assistance_matri", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"dequant_mod", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 2048}, {1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
