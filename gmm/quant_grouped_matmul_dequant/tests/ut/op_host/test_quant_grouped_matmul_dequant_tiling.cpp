/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../../../op_host/quant_grouped_matmul_dequant_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class QuantGroupedMatmulDequantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantGroupedMatmulDequantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantGroupedMatmulDequantTiling TearDown" << std::endl;
    }
};

TEST_F(QuantGroupedMatmulDequantTiling, test_tiling_float16_1)
{
    int64_t G = 16;
    int64_t M = 256;
    int64_t N = 1024;
    int64_t K = 1024;

    optiling::QuantMatmulDequantCompileInfo compileInfo = {};
    compileInfo.coreNum = 8;
    compileInfo.ubSize = 262144;
    compileInfo.l1Size = 1048576;
    compileInfo.l2Size = 16777216;
    compileInfo.l0CSize = 262144;
    compileInfo.l0ASize = 65536;
    compileInfo.l0BSize = 65536;
    compileInfo.socVersionStr = "Ascend310P";

    string socInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", )"
                           R"("Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, )"
                           R"("Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, )"
                           R"("UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576, )"
                           R"("L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, )"
                           R"("CORE_NUM": 8, "socVersion":"Ascend310P"} })";

    gert::TilingContextPara tilingContextPara(
        "QuantGroupedMatmulDequant",
        {{{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{G, K / 32, N / 16, 16, 32}, {G, K / 32, N / 16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
         {{{G, N}, {G, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{G}, {G}}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{{M}, {M}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{K}, {K}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("pertoken")},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend310P", 8, 262144, 4096, socInfoString);

    int64_t expectTilingKey = 10000003;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(QuantGroupedMatmulDequantTiling, test_tiling_float16_2)
{
    int64_t G = 8;
    int64_t M = 128;
    int64_t N = 5120;
    int64_t K = 1344;

    optiling::QuantMatmulDequantCompileInfo compileInfo = {};
    compileInfo.coreNum = 8;
    compileInfo.ubSize = 262144;
    compileInfo.l1Size = 1048576;
    compileInfo.l2Size = 16777216;
    compileInfo.l0CSize = 262144;
    compileInfo.l0ASize = 65536;
    compileInfo.l0BSize = 65536;
    compileInfo.socVersionStr = "Ascend310P";

    string socInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", )"
                           R"("Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, )"
                           R"("Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, )"
                           R"("UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576, )"
                           R"("L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, )"
                           R"("CORE_NUM": 8, "socVersion":"Ascend310P"} })";

    gert::TilingContextPara tilingContextPara(
        "QuantGroupedMatmulDequant",
        {{{{M, K}, {M, K}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{G, K / 32, N / 16, 16, 32}, {G, K / 32, N / 16, 16, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},
         {{{G, N}, {G, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{G}, {G}}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{K}, {K}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{M, N}, {M, N}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("pertensor")},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend310P", 8, 262144, 4096, socInfoString);

    int64_t expectTilingKey = 10000004;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
