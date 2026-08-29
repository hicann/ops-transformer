/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_all_gather_matmul_v3_tiling.cpp
 * \brief host侧tiling ut
 */

#include <gtest/gtest.h>
#include "../all_gather_matmul_v3_host_ut_param.h"
#include "mc2_tiling_case_executor.h"

namespace AllGatherMatmulV3UT {

static const std::string OP_NAME = "AllGatherMatmulV3";

static struct AllGatherMatmulV3CompileInfo {
} compileInfo;

class AllGatherMatmulV3Arch35TilingTest : public testing::TestWithParam<AllGatherMatmulV3TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        setenv("HCCL_BUFFSIZE", "300", 1);
        std::cout << "AllGatherMatmulV3Arch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        unsetenv("HCCL_BUFFSIZE");
        std::cout << "AllGatherMatmulV3Arch35TilingTest TearDown." << std::endl;
    }
};

TEST_P(AllGatherMatmulV3Arch35TilingTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {param.context, param.x1, param.x2, param.bias, param.x1Scale, param.x2Scale});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({param.y, param.gatherOut});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hcclBufferSize)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.isTransA)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.isTransB)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rankSize)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.yDtypeAttr)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)}});

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.soc, 64, 262144, 8192);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey, "", {},
                       MC2_TILING_DATA_RESERVED_LEN);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmulV3TilingUT, AllGatherMatmulV3Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulV3TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AllGatherMatmulV3TilingUtParam>);

} // namespace AllGatherMatmulV3UT
