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
#include "../../../op_host/moe_finalize_routing_v2_tiling.h"
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

std::vector<gert::TilingContextPara::OpAttr> DefaultExpertRangeAttrs(int64_t dropPadMode = 0, int64_t k = 1)
{
    return {
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dropPadMode)},
        {"zero_expert_range",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
        {"copy_expert_range",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
        {"constant_expert_range",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(k)},
    };
}
} // namespace

class MoeFinalizeRoutingV2Arch35Tiling : public testing::Test {};

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_dropless_float_h_full)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultExpertRangeAttrs(0, 1), &compileInfo, "Ascend950", kA5SocInfo,
                                              4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 30000, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_droppad_col_bf16)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{316, 96, 52}, {316, 96, 52}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{60088}, {60088}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{259, 52}, {259, 52}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{259, 52}, {259, 52}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{316, 52}, {316, 52}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{259, 232}, {259, 232}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{259, 232}, {259, 232}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{259, 52}, {259, 52}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              DefaultExpertRangeAttrs(1, 232), &compileInfo, "Ascend950", kA5SocInfo,
                                              4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_dropless_bf16_split_h)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{248, 1, 6825}, {248, 1, 6825}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{157042}, {157042}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{674, 6825}, {674, 6825}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{674, 6825}, {674, 6825}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{248, 6825}, {248, 6825}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{674, 233}, {674, 233}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{674, 233}, {674, 233}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{674, 6825}, {674, 6825}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              DefaultExpertRangeAttrs(3, 233), &compileInfo, "Ascend950", kA5SocInfo,
                                              4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_dropless_float16_h_full)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              DefaultExpertRangeAttrs(0, 1), &compileInfo, "Ascend950", kA5SocInfo,
                                              4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_dropless_no_scales)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultExpertRangeAttrs(0, 1), &compileInfo, "Ascend950", kA5SocInfo,
                                              4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_dropless_active_expert_range)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"zero_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{0, 2})},
        {"copy_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{2, 4})},
        {"constant_expert_range",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{4, 6})},
        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_constant_expert_inputs)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"zero_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{0, 2})},
        {"copy_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{2, 4})},
        {"constant_expert_range",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{4, 6})},
        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo, "Ascend950", kA5SocInfo, 4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_invalid_expert_range_bounds)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"zero_expert_range",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{20, 22})},
        {"copy_expert_range",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
        {"constant_expert_range",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo, "Ascend950", kA5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_dropless_split_h_large_h)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 1}, {4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 1}, {4, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 8192}, {4, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultExpertRangeAttrs(2, 1), &compileInfo, "Ascend950", kA5SocInfo,
                                              4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20020, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_kh_full_load_small_k)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{64, 512}, {64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{64, 512}, {64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 512}, {64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 512}, {64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 4}, {64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 4}, {64, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{64, 512}, {64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultExpertRangeAttrs(0, 4), &compileInfo, "Ascend950", kA5SocInfo,
                                              4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 40000, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_droppad_col_kh_full_load)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{4, 2, 16}, {4, 2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 16}, {4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultExpertRangeAttrs(1, 4), &compileInfo, "Ascend950", kA5SocInfo,
                                              4096);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 30010, "", expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_droppad_invalid_expert_range)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"zero_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{0, 3})},
        {"copy_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{2, 4})},
        {"constant_expert_range",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
    };
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{2, 4, 16}, {2, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo, "Ascend950", kA5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_invalid_expert_range_overlap)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"zero_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{0, 4})},
        {"copy_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{2, 6})},
        {"constant_expert_range",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
        {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    };
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo, "Ascend950", kA5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeFinalizeRoutingV2Arch35Tiling, MoeFinalizeRoutingV2_arch35_invalid_drop_pad_mode)
{
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {40, 262144};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRoutingV2",
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              DefaultExpertRangeAttrs(99, 1), &compileInfo, "Ascend950", kA5SocInfo,
                                              4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}
