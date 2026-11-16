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
#include "base/registry/op_impl_space_registry_v2.h"

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
#if !ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED
    // devkit < 9.3 无 Ascend960DT(9201) tiling 支持, soc=9201 用例在 tiling 分发处必然失败, 按soc维度跳过
    if (param.soc == "9201") {
        GTEST_SKIP() << "Ascend960DT(9201) requires devkit >= 9.3, skip soc=9201 cases";
    }
#endif
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

TEST(AllGatherMatmulV2A5FallbackTest, NonQuantWorkspaceIncludesBiasLen)
{
    // A5(3510) 非量化应回退原 V2 路径: workspace = libApi + storageA + biasLen。
    // 该用例 gatherOut 非空 => storageA = 0; bf16 bias 需 cast 空间, biasLen = AlignUp(1394, 256) * sizeof(float)。
    // 若 A5 误走 apace 路径, biasLen 不再计入 workspace, 断言将失败。
    constexpr size_t kLibApiWorkspaceSizeDav3510 = 16UL * 1024UL * 1024UL;
    constexpr size_t kBiasLen = 1536UL * sizeof(float); // AlignUp(1394, 256) = 1536
    csv_map csvMap = {
        {"case_name", "ut_a5_non_quant_fallback_legacy_workspace_DAV_3510"},
        {"expectResult", "SUCCESS"},
        {"x1_shape", "172 4814"},
        {"x1_dtype", "BF16"},
        {"x1_format", "ND"},
        {"x2_shape", "4814 1394"},
        {"x2_dtype", "BF16"},
        {"x2_format", "ND"},
        {"bias_shape", "1394"},
        {"bias_dtype", "BF16"},
        {"bias_format", "ND"},
        {"output_y_shape", "1376 1394"},
        {"output_y_dtype", "BF16"},
        {"output_y_format", "ND"},
        {"gather_out_shape", "1376 4814"},
        {"gather_out_dtype", "BF16"},
        {"gather_out_format", "ND"},
        {"group", "hcclCom"},
        {"is_trans_a", "0"},
        {"is_trans_b", "0"},
        {"gather_index", "0"},
        {"comm_turn", "0"},
        {"rank_size", "8"},
        {"block_size", "0"},
        {"group_size", "0"},
        {"is_gather_out", "0"},
        {"is_amax_out", "0"},
        {"y_dtype", "FLOAT"},
        {"comm_mode", "ai_cpu"},
        {"soc", "3510"},
        {"core_num", "32"},
        {"ubsize", "262144"},
        {"expectTilingKey", "257"},
    };
    AllGatherMatmulV2TilingUtParam param(csvMap);
    AllGatherMatmulV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara = BuildTilingContextParam(param, &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rank_size}};

    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
                       param.expectTilingDataHash, {kLibApiWorkspaceSizeDav3510 + kBiasLen},
                       MC2_TILING_DATA_RESERVED_LEN, true);
}

TEST(AllGatherMatmulV2Arch35TilingParseTest, TestTilingParse)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("AllGatherMatmulV2");
    ASSERT_NE(opImpl, nullptr);
    auto tilingParseFunc = opImpl->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(nullptr), ge::GRAPH_SUCCESS);
}

} // namespace all_gather_matmul_v2_ut
