/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../all_gather_matmul_host_ut_param.h"
#include "mc2_tiling_case_executor.h"

namespace AllGatherMatmulUT {

class AllGatherMatmulArch35TilingTest : public testing::TestWithParam<AllGatherMatmulTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmul Arch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmul Arch35TilingTest TearDown" << std::endl;
    }
};

TEST_P(AllGatherMatmulArch35TilingTest, param)
{
    auto param = GetParam();
    struct AllGatherMatmulCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "AllGatherMatmul", {param.x1, param.x2, param.bias}, {param.y, param.gather_out},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gather_index)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rank_size)},
         {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.topo_type)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_gather_out)}},
        param.inputInstance, param.outputInstance, &compileInfo, param.soc, param.coreNum, param.ubsize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rank_size}, {"topoType", param.topo_type}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
                       param.expectTilingDataHash, {}, MC2_TILING_DATA_RESERVED_LEN, true);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmul, AllGatherMatmulArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AllGatherMatmulTilingUtParam>);

} // namespace AllGatherMatmulUT
