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

using namespace std;
using namespace optiling;

constexpr char kSocVersion910B[] = "Ascend910B";

class MhcPostTilingArch22 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPostTilingArch22 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPostTilingArch22 TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }
    return result;
}

TEST_F(MhcPostTilingArch22, test_mhc_post_3d_fp16_success)
{
    MhcPostCompileInfo compileInfo = {64, 64, platform_ascendc::SocVersion::ASCEND910B, NpuArch::DAV_3002};
    gert::TilingContextPara tilingContextPara("MhcPost",
                                              {
                                                  // input info: x, h_res, h_out, h_post
                                                  {{{1024, 4, 5120}, {1024, 4, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4, 4}, {1024, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 5120}, {1024, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4}, {1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1024, 4, 5120}, {1024, 4, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                              },
                                              &compileInfo, kSocVersion910B);
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcPostTilingArch22, test_mhc_post_4d_fp16_success)
{
    MhcPostCompileInfo compileInfo = {64, 64, platform_ascendc::SocVersion::ASCEND910B, NpuArch::DAV_3002};
    gert::TilingContextPara tilingContextPara(
        "MhcPost",
        {
            // input info: x, h_res, h_out, h_post
            {{{1, 1024, 4, 5120}, {1, 1024, 4, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 4, 4}, {1, 1024, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1024, 5120}, {1, 1024, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 4}, {1, 1024, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{1, 1024, 4, 5120}, {1, 1024, 4, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
        },
        &compileInfo, kSocVersion910B);
    uint64_t expectTilingKey = 1;
    string expectTilingDataStr = "4 5120 64 16 16 1 1024 1 5120 1 5120 5120 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingDataStr, expectWorkspaces);
}

TEST_F(MhcPostTilingArch22, test_mhc_post_3d_n6_invalid)
{
    MhcPostCompileInfo compileInfo = {64, 64, platform_ascendc::SocVersion::ASCEND910B, NpuArch::DAV_3002};
    gert::TilingContextPara tilingContextPara("MhcPost",
                                              {
                                                  // input info: x, h_res, h_out, h_post
                                                  {{{512, 6, 2048}, {512, 6, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{512, 6, 6}, {512, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{512, 6}, {512, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{512, 6, 2048}, {512, 6, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                              },
                                              &compileInfo, kSocVersion910B);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MhcPostTilingArch22, test_mhc_post_invalid_n)
{
    MhcPostCompileInfo compileInfo = {64, 64, platform_ascendc::SocVersion::ASCEND910B, NpuArch::DAV_3002};
    gert::TilingContextPara tilingContextPara("MhcPost",
                                              {
                                                  // input info: x, h_res, h_out, h_post
                                                  {{{512, 7, 2048}, {512, 7, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{512, 7, 7}, {512, 7, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{512, 7}, {512, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{512, 7, 2048}, {512, 7, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                              },
                                              &compileInfo, kSocVersion910B);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MhcPostTilingArch22, test_mhc_post_invalid_d)
{
    MhcPostCompileInfo compileInfo = {64, 64, platform_ascendc::SocVersion::ASCEND910B, NpuArch::DAV_3002};
    gert::TilingContextPara tilingContextPara("MhcPost",
                                              {
                                                  // input info: x, h_res, h_out, h_post
                                                  {{{512, 4, 30000}, {512, 4, 30000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{512, 4, 4}, {512, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 30000}, {512, 30000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{512, 4}, {512, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{512, 4, 30000}, {512, 4, 30000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                              },
                                              &compileInfo, kSocVersion910B);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
