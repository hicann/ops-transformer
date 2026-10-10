/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../all_gather_matmul_v2_host_ut_param.h"
#include "mc2_tiling_case_executor.h"

namespace all_gather_matmul_v2_ut {

class AllGatherMatmulV2Arch35TilingTest : public testing::TestWithParam<AllGatherMatmulV2TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulV2 Arch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulV2 Arch35TilingTest TearDown" << std::endl;
    }
};

struct AllGatherMatmulV2CompileInfo {};

gert::TilingContextPara BuildTilingContextParam(const AllGatherMatmulV2TilingUtParam& param,
                                                AllGatherMatmulV2CompileInfo* compileInfo)
{
    return gert::TilingContextPara(
        "AllGatherMatmulV2",
        {
            param.x1,
            param.x2,
            param.bias,
            param.x1_scale,
            param.x2_scale,
            param.quant_scale,
        },
        {
            param.y,
            param.gather_out,
            param.amax_out,
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gather_index)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rank_size)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_gather_out)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_amax_out)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.comm_mode)}},
        param.inputInstance, param.outputInstance, compileInfo, param.soc, param.coreNum, param.ubsize);
}

TEST_P(AllGatherMatmulV2Arch35TilingTest, param)
{
    auto param = GetParam();
    constexpr char kAscendMc2DebugMode[] = "ASCEND_MC2_DEBUG_MODE";
    bool debugModeEnvSet = false;
    constexpr const char kUtMc2Debug4Prefix[] = "ut_mc2_debug4_";
    if (param.case_name.compare(0, std::size(kUtMc2Debug4Prefix) - 1, kUtMc2Debug4Prefix) == 0) {
        (void)setenv(kAscendMc2DebugMode, "4", 1);
        debugModeEnvSet = true;
    }
    AllGatherMatmulV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara = BuildTilingContextParam(param, &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rank_size}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
                       param.expectTilingDataHash, {}, MC2_TILING_DATA_RESERVED_LEN, true);
    if (debugModeEnvSet) {
        (void)unsetenv(kAscendMc2DebugMode);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmulV2, AllGatherMatmulV2Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    GetCaseInfoString<AllGatherMatmulV2TilingUtParam>);

} // namespace all_gather_matmul_v2_ut
