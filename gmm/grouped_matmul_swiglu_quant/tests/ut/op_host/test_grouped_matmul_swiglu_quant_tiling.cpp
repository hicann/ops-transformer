/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_gating_top_k_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/grouped_matmul_swiglu_quant_tiling.h"

using namespace std;

class GroupedMatmulSwigluQuantTiling_TEST : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantTiling_TEST SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulSwigluQuantTiling_TEST TearDown" << std::endl;
    }
};

struct GMMSwigluCompileInfo_ {
    uint64_t ubSize_ = 192 * 1024;
    uint32_t aicNum_ = 20;
    uint32_t baseM_ = 128;
    uint32_t baseN_ = 256;
};

TEST_F(GroupedMatmulSwigluQuantTiling_TEST, grouped_matmul_swiglu_quant_tiling_succ_01)
{
    GMMSwigluCompileInfo_ compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulSwigluQuant",
        {
            {{{16, 256}, {16, 256}}, ge::DT_INT8, ge::FORMAT_ND},                 // x
            {{{4, 256, 512}, {4, 256, 512}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // weightScale
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // xscale
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // weightAssistMatrix
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // groupList
        },
        {
            {{{16, 256}, {16, 256}}, ge::DT_INT8, ge::FORMAT_ND}, // y
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND}           // yscale
        },
        {{"is_enable_weight_assistance_matri", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"dequant_mod", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "85899345924 2199023255808 549755813904 70368744177920 0 1 17179869225 512 68719476737 "
                              "1099511628288 549755814144 1099511628032 1099511627904 4294967552 4294967297 1 0 "
                              "211106232532992 131072 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16809984};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GroupedMatmulSwigluQuantTiling_TEST, grouped_matmul_swiglu_quant_tiling_succ_02)
{
    GMMSwigluCompileInfo_ compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulSwigluQuant",
        {
            {{{1024, 256}, {16, 256}}, ge::DT_INT8, ge::FORMAT_ND},           // x
            {{{16, 256, 4096}, {16, 256, 4096}}, ge::DT_INT8, ge::FORMAT_ND}, // weight
            {{{16, 512}, {16, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},            // weightScale
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // xscale
            {{{16, 4096}, {16, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},          // weightAssistMatrix
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // groupList
        },
        {
            {{{1024, 2048}, {1024, 2048}}, ge::DT_INT8, ge::FORMAT_ND}, // y
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}             // yscale
        },
        {{"is_enable_weight_assistance_matri", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"dequant_mod", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "85899345936 17592186044672 549755813904 8796093022464 0 1 68719476740 4096 68719476737 "
                              "1099511631872 549755814144 1099511628032 1099511627904 4294967552 4294967297 1 0 "
                              "211106232532992 131072 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {17039360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GroupedMatmulSwigluQuantTiling_TEST, grouped_matmul_swiglu_quant_tiling_succ_03)
{
    GMMSwigluCompileInfo_ compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulSwigluQuant",
        {
            {{{1024, 256}, {16, 256}}, ge::DT_INT8, ge::FORMAT_ND},           // x
            {{{16, 256, 4096}, {16, 256, 4096}}, ge::DT_INT4, ge::FORMAT_ND}, // weight
            {{{16, 512}, {16, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},            // weightScale
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // xscale
            {{{16, 4096}, {16, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},          // weightAssistMatrix
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // groupList
        },
        {
            {{{1024, 2048}, {1024, 2048}}, ge::DT_INT8, ge::FORMAT_ND}, // y
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}             // yscale
        },
        {{"is_enable_weight_assistance_matri", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"dequant_mod", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData =
        "85899345936 17592186044672 549755813904 2190433321216 287104476245130240 1 68719476740 4096 68719476737 "
        "1099511631872 549755814144 1099511628032 1099511627904 34359738624 4294967304 1 0 211106232532992 131072 "
        "4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {83885056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
