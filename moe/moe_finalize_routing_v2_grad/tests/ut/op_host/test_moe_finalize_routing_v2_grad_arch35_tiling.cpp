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
#include <vector>
#include "../../../op_host/moe_finalize_routing_v2_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace {
static const std::string kA5SocInfo = "{\n"
                                      "  \"hardware_info\": {\n"
                                      "    \"UB_SIZE\": 262144,\n"
                                      "    \"CORE_NUM\": 32,\n"
                                      "    \"socVersion\": \"Ascend910_95\"\n"
                                      "  }\n"
                                      "}";

std::vector<gert::TilingContextPara::OpAttr> DefaultGradAttrs(int64_t dropPadMode = 0)
{
    return {
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
        {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    };
}
} // namespace

class MoeFinalizeRoutingV2GradArch35Tiling : public testing::Test {};

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_not_split_h_float_no_bias)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20011, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_not_split_h_float_with_bias)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20021, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_not_split_h_bf16_no_bias)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{15, 8}, {15, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 8}, {15, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20011, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_not_split_h_invalid_expanded_row_idx)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{15}, {15}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 20011, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_not_split_h_invalid_scales_dtype)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 20011, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_not_split_h_bias_grad_y_mismatch)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 20021, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_not_split_h_expert_idx_dtype_mismatch)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 20021, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_split_h_float_no_bias)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 262144}, {5, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{15, 262144}, {15, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 262144}, {15, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20012, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_split_h_float_with_bias)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 262144}, {5, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{15, 262144}, {15, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{8, 262144}, {8, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 262144}, {15, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20022, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_split_h_invalid_expanded_x_dtype)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 262144}, {5, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{15, 262144}, {15, 262144}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 262144}, {15, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 20012, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_without_scale_not_cut_h)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 1}, {5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10001, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_without_scale_cut_h)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 262144}, {5, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 262144}, {5, 262144}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 1}, {5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10002, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2GradArch35Tiling, MoeFinalizeRoutingV2Grad_arch35_without_scale_invalid_row_idx)
{
    optiling::MoeFinalizeRoutingV2GradCompileInfo compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2Grad",
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{5, 1}, {5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultGradAttrs(0), &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 10001, "", expectWorkspaces);
}
