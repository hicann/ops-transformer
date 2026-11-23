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
#include "../../../../op_host/op_tiling/mhc_post_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_common/op_host/util/op_const_def.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"

using namespace std;
using namespace optiling;

constexpr char kSocVersion950[] = "Ascend950";

class MhcPostTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPostTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPostTiling TearDown" << std::endl;
    }
};

TEST_F(MhcPostTiling, test_mhc_post_950_3d_fp16_success)
{
    MhcPostCompileInfo compileInfo = {64, 64, platform_ascendc::SocVersion::ASCEND950, Ops::Base::DAV_3510};
    gert::TilingContextPara tilingContextPara("MhcPost",
                                              {
                                                  {{{1024, 4, 5120}, {1024, 4, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4, 4}, {1024, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4}, {1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 4, 5120}, {1024, 4, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, kSocVersion950);
    uint64_t expectTilingKey = 6;
    string expectTilingDataStr = "4 5120 64 16 64 16 4 1 4 2560 2 2560 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcPostTiling, test_mhc_post_950_regbase_4d_fp16)
{
    MhcPostCompileInfo compileInfo = {64, 64, platform_ascendc::SocVersion::ASCEND950, Ops::Base::DAV_3510};
    gert::TilingContextPara tilingContextPara("MhcPost",
                                              {
                                                  {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 128, 8, 8}, {1, 128, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 128, 128}, {1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 128, 8}, {1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 8, 128}, {1, 128, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, kSocVersion950);
    uint64_t expectTilingKey = 6;
    string expectTilingDataStr = "8 128 64 2 64 2 8 1 8 128 1 128 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcPostTiling, test_mhc_post_950_3d_fp16_invalid_hout_shape)
{
    MhcPostCompileInfo compileInfo = {64, 64, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara("MhcPost",
                                              {
                                                  {{{1024, 4, 5120}, {1024, 4, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4, 4}, {1024, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 5121}, {1024, 5121}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4}, {1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 4, 5120}, {1024, 4, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, kSocVersion950);
    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    errors.GetRawErrorMessages();
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", expectWorkspaces);
    const auto reports = errors.GetRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(reports[0].error_id, "EZ0008");
    ASSERT_EQ(reports[0].args_map.at("param_name"), "h_out");
    ASSERT_EQ(reports[0].args_map.at("op_name"), "MhcPost");
    ASSERT_EQ(reports[0].args_map.at("incorrect_shape"), "[1024, 5121]");
    ASSERT_EQ(reports[0].args_map.at("correct_shape"), "[1024, 5120]");
}
