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
#include <array>
#include <limits>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "../../../op_host/moe_finalize_routing_v2_tiling.h"
#include "../../../op_kernel/arch35/moe_finalize_routing_v2_tiling_struct.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "common/include/op_host/op_const_def_transformer.h"

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
    std::vector<size_t> expectWorkspaces = {0};
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
    std::vector<size_t> expectWorkspaces = {0};
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
    std::vector<size_t> expectWorkspaces = {0};
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
    std::vector<size_t> expectWorkspaces = {0};
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
    std::vector<size_t> expectWorkspaces = {0};
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
    std::vector<size_t> expectWorkspaces = {0};
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
    std::vector<size_t> expectWorkspaces = {0};
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
    std::vector<size_t> expectWorkspaces = {0};
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
    std::vector<size_t> expectWorkspaces = {0};
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
    std::vector<size_t> expectWorkspaces = {0};
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

namespace {
enum InputIndex : size_t {
    EXPANDED_X_IDX,
    EXPANDED_ROW_IDX_IDX,
    X1_IDX,
    X2_IDX,
    BIAS_IDX,
    SCALES_IDX,
    EXPERT_IDX,
    X_IDX,
    ALPHA1_IDX,
    ALPHA2_IDX,
    V_IDX,
    EXPANDED_X_SCALE_IDX,
    INPUT_COUNT
};
constexpr int64_t DROP_LESS_COL = 0;
constexpr int64_t DROP_LESS_ROW = 2;
constexpr int64_t DROP_PAD_ROW = 3;
constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t SCALE_PAIR_SIZE = 2;
constexpr int32_t kMx4AivCount = 40;
constexpr uint64_t kMx4UbSize = 256 * 1024;
constexpr uint64_t kMx4TilingDataCapacity = 4096;

template <typename Attr = gert::InfershapeContextPara::OpAttr>
std::vector<Attr> Mx4Attrs(int64_t mode, int64_t k)
{
    return {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode)},
            {"zero_expert_range",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
            {"copy_expert_range",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
            {"constant_expert_range",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(k)}};
}

TEST(MoeFinalizeRoutingMx4, InfersOriginalTokenCountFromIndices)
{
    gert::InfershapeContextPara para("MoeFinalizeRoutingV2",
                                     {{{{5, 66}, {5, 66}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
                                      {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                      {{{3, 66}, {3, 66}}, ge::DT_BF16, ge::FORMAT_ND},
                                      {{{3, 66}, {3, 66}}, ge::DT_BF16, ge::FORMAT_ND},
                                      {{{4, 66}, {4, 66}}, ge::DT_BF16, ge::FORMAT_ND},
                                      {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                      {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                      {{{5, 2, 2}, {5, 2, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}}, Mx4Attrs(2, 2));
    para.inputInstanceNum_.assign(INPUT_COUNT, 1);
    para.outputInstanceNum_ = {1};
    for (size_t i = X_IDX; i <= V_IDX; ++i) {
        para.inputInstanceNum_[i] = 0;
    }
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{3, 66}});
}

TEST(MoeFinalizeRoutingMx4, InfersPartialAndEmptyShapes)
{
    struct Case {
        gert::Shape expandedX;
        int64_t indexCount;
        gert::Shape scale;
        int64_t k;
        ge::graphStatus status;
        std::vector<int64_t> output;
    };
    for (const auto& entry :
         {Case{{5, -1}, 6, {5, 2, 2}, 2, ge::GRAPH_SUCCESS, {3, -1}},
          Case{{5, 66}, 0, {5, 2, 2}, 2, ge::GRAPH_SUCCESS, {0, 66}},
          Case{{-2}, 6, {-2}, 2, ge::GRAPH_SUCCESS, {3, -1}},
          Case{{5, 66}, -1, {5, 2, 2}, 2, ge::GRAPH_SUCCESS, {-1, 66}},
          Case{{5, -3}, 6, {5, 2, 2}, 2, ge::GRAPH_FAILED, {}}, Case{{5, 65}, 6, {5, 2, 2}, 2, ge::GRAPH_FAILED, {}},
          Case{{0, 66}, 6, {0, 2, 2}, 2, ge::GRAPH_FAILED, {}}, Case{{5, 66}, 5, {5, 2, 2}, 2, ge::GRAPH_FAILED, {}},
          Case{{5, 66}, 6, {6, 2, 2}, 2, ge::GRAPH_FAILED, {}}, Case{{5, 66}, 6, {5, 1, 2}, 2, ge::GRAPH_FAILED, {}},
          Case{{5, 66}, 6, {5, 2, 2}, 0, ge::GRAPH_FAILED, {}}}) {
        gert::StorageShape expandedX;
        expandedX.MutableOriginShape() = entry.expandedX;
        expandedX.MutableStorageShape() = entry.expandedX;
        gert::StorageShape scale;
        scale.MutableOriginShape() = entry.scale;
        scale.MutableStorageShape() = entry.scale;
        gert::InfershapeContextPara para("MoeFinalizeRoutingV2",
                                         {{expandedX, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
                                          {{{entry.indexCount}, {entry.indexCount}}, ge::DT_INT32, ge::FORMAT_ND},
                                          {scale, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
                                         {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}}, Mx4Attrs(DROP_LESS_ROW, entry.k));
        para.inputInstanceNum_.assign(INPUT_COUNT, 0);
        para.inputInstanceNum_[EXPANDED_X_IDX] = 1;
        para.inputInstanceNum_[EXPANDED_ROW_IDX_IDX] = 1;
        para.inputInstanceNum_[EXPANDED_X_SCALE_IDX] = 1;
        para.outputInstanceNum_ = {1};
        ExecuteTestCase(para, entry.status, {entry.output});
    }
}

TEST(MoeFinalizeRoutingHif4, UsesLegacyFp32Scales)
{
    using TensorDescription = gert::TilingContextPara::TensorDescription;
    const TensorDescription absent = {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    std::vector<TensorDescription> inputs(SCALES_IDX + 1, absent);
    inputs[EXPANDED_X_IDX] = {{{6, 64}, {6, 64}}, Ops::Transformer::DT_HIFLOAT4, ge::FORMAT_ND};
    inputs[EXPANDED_ROW_IDX_IDX] = {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[SCALES_IDX] = {{{3, 2, 1}, {3, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND};
    optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {kMx4AivCount, kMx4UbSize};
    gert::TilingContextPara para("MoeFinalizeRoutingV2", inputs, {{{{3, 64}, {3, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 DefaultExpertRangeAttrs(DROP_LESS_ROW), &compileInfo, "Ascend960DT", kMx4AivCount,
                                 kMx4UbSize, kMx4TilingDataCapacity);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_GE(info.tilingDataSize, sizeof(MoeFinalizeRoutingV2RegbaseTilingData));
    const auto* data = reinterpret_cast<const MoeFinalizeRoutingV2RegbaseTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->k, 2);

    para.inputTensorDesc_.resize(EXPANDED_X_SCALE_IDX, absent);
    para.inputTensorDesc_.push_back({{{3, 2, 4}, {3, 2, 4}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    TilingInfo withUnusedScale;
    ASSERT_TRUE(ExecuteTiling(para, withUnusedScale));
    EXPECT_EQ(withUnusedScale.tilingKey, info.tilingKey);
    ASSERT_EQ(withUnusedScale.tilingDataSize, info.tilingDataSize);
    EXPECT_EQ(std::memcmp(withUnusedScale.tilingData.get(), info.tilingData.get(), info.tilingDataSize), 0);
    para.inputTensorDesc_.pop_back();
    para.inputTensorDesc_[SCALES_IDX] = {{{3, 2, 1}, {3, 2, 1}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
    EXPECT_FALSE(ExecuteTiling(para, info));
}

} // namespace

TEST(MoeFinalizeRoutingMx4Tiling, SelectsStrategiesAndRoutingModes)
{
    constexpr uint64_t kBatchKey = 130000;
    constexpr uint64_t splitHKey = 140000;
    constexpr uint64_t fullKHKey = 150000;
    constexpr uint64_t fullRowKHKey = 160000;
    constexpr uint64_t dropPadKeyStep = 10;
    constexpr int64_t unalignedSplitWidth = 16450;
    constexpr int64_t tokenCount = 3;
    constexpr int64_t biasExpertCount = 4;
    struct Case {
        int64_t r, h, k;
        uint64_t key;
    };
    for (const auto& c : {Case{7, 66, 3, fullRowKHKey}, Case{1024, 512, 3, fullKHKey}, Case{5000, 512, 1000, kBatchKey},
                          Case{7, 65536, 3, splitHKey}, Case{7, unalignedSplitWidth, 3, splitHKey}}) {
        for (int64_t mode = DROP_LESS_COL; mode <= DROP_PAD_ROW; ++mode) {
            optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {kMx4AivCount, kMx4UbSize};
            gert::TilingContextPara::TensorDescription xDesc =
                (mode == DROP_LESS_COL || mode == DROP_LESS_ROW) ?
                    gert::TilingContextPara::TensorDescription{
                        {{c.r, c.h}, {c.r, c.h}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND} :
                    gert::TilingContextPara::TensorDescription{
                        {{c.r, 1, c.h}, {c.r, 1, c.h}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND};
            gert::TilingContextPara para(
                "MoeFinalizeRoutingV2",
                {xDesc,
                 {{{tokenCount * c.k}, {tokenCount * c.k}}, ge::DT_INT32, ge::FORMAT_ND},
                 {{{tokenCount, c.h}, {tokenCount, c.h}}, ge::DT_BF16, ge::FORMAT_ND},
                 {{{tokenCount, c.h}, {tokenCount, c.h}}, ge::DT_BF16, ge::FORMAT_ND},
                 {{{biasExpertCount, c.h}, {biasExpertCount, c.h}}, ge::DT_BF16, ge::FORMAT_ND},
                 {{{tokenCount, c.k}, {tokenCount, c.k}}, ge::DT_FLOAT, ge::FORMAT_ND},
                 {{{tokenCount, c.k}, {tokenCount, c.k}}, ge::DT_INT32, ge::FORMAT_ND}},
                {{{{tokenCount, c.h}, {tokenCount, c.h}}, ge::DT_BF16, ge::FORMAT_ND}},
                Mx4Attrs<gert::TilingContextPara::OpAttr>(mode, c.k), &compileInfo, "Ascend950", kMx4AivCount,
                kMx4UbSize, kMx4TilingDataCapacity, kA5SocInfo);
            para.inputTensorDesc_.resize(EXPANDED_X_SCALE_IDX, {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND});
            const int64_t scaleBlocks = (c.h + BLOCK_SIZE - 1) / BLOCK_SIZE;
            para.inputTensorDesc_.push_back({{{c.r, scaleBlocks, SCALE_PAIR_SIZE}, {c.r, scaleBlocks, SCALE_PAIR_SIZE}},
                                             ge::DT_FLOAT8_E8M0,
                                             ge::FORMAT_ND});
            if (c.h == unalignedSplitWidth) {
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(para, info));
                ASSERT_EQ(info.tilingKey, c.key + mode * dropPadKeyStep);
                ASSERT_GE(info.tilingDataSize, sizeof(MoeFinalizeRoutingV2RegbaseTilingData));
                const auto* data =
                    reinterpret_cast<const MoeFinalizeRoutingV2RegbaseTilingData*>(info.tilingData.get());
                EXPECT_LT(data->hFactor, c.h);
                EXPECT_GT(data->hFactor, 0);
                EXPECT_EQ(data->hFactor % BLOCK_SIZE, 0);
            } else {
                ExecuteTestCase(para, ge::GRAPH_SUCCESS, c.key + mode * dropPadKeyStep, "", std::vector<size_t>{0});
            }
        }
    }
}

TEST(MoeFinalizeRoutingMx4Tiling, RejectsInvalidInputsThroughCommonValidation)
{
    using TensorDescription = gert::TilingContextPara::TensorDescription;
    struct Case {
        size_t index;
        TensorDescription tensor;
    };
    const std::vector<Case> cases = {
        {EXPANDED_X_IDX, {{{5, 65}, {5, 65}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND}},
        {EXPANDED_X_IDX, {{{0, 66}, {0, 66}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND}},
        {EXPANDED_ROW_IDX_IDX, {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {EXPANDED_ROW_IDX_IDX, {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND}},
        {X1_IDX, {{{4, 66}, {4, 66}}, ge::DT_BF16, ge::FORMAT_ND}},
        {X1_IDX, {{{3, 66}, {3, 66}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {X1_IDX, {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {BIAS_IDX, {{{4, 64}, {4, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {EXPANDED_X_SCALE_IDX, {{{5, 1, 2}, {5, 1, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {EXPANDED_X_SCALE_IDX, {{{5, 2, 2}, {5, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {EXPANDED_X_SCALE_IDX, {{{}, {}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND}},
        {EXPERT_IDX, {{{3, 3}, {3, 3}}, ge::DT_INT32, ge::FORMAT_ND}},
        {EXPERT_IDX, {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}},
        {X_IDX, {{{3, 66}, {3, 66}}, ge::DT_BF16, ge::FORMAT_ND}},
        {ALPHA1_IDX, {{{1, 66}, {1, 66}}, ge::DT_BF16, ge::FORMAT_ND}},
        {ALPHA2_IDX, {{{1, 66}, {1, 66}}, ge::DT_BF16, ge::FORMAT_ND}},
        {V_IDX, {{{1, 66}, {1, 66}}, ge::DT_BF16, ge::FORMAT_ND}},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.index);
        optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {kMx4AivCount, kMx4UbSize};
        gert::TilingContextPara para("MoeFinalizeRoutingV2",
                                     {{{{5, 66}, {5, 66}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
                                      {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                      {{{3, 66}, {3, 66}}, ge::DT_BF16, ge::FORMAT_ND},
                                      {{{3, 66}, {3, 66}}, ge::DT_BF16, ge::FORMAT_ND},
                                      {{{4, 66}, {4, 66}}, ge::DT_BF16, ge::FORMAT_ND},
                                      {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                      {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND}},
                                     {{{{3, 66}, {3, 66}}, ge::DT_BF16, ge::FORMAT_ND}},
                                     Mx4Attrs<gert::TilingContextPara::OpAttr>(DROP_LESS_ROW, 2), &compileInfo,
                                     "Ascend950", kMx4AivCount, kMx4UbSize, kMx4TilingDataCapacity, kA5SocInfo);
        para.inputTensorDesc_.resize(EXPANDED_X_SCALE_IDX, {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND});
        para.inputTensorDesc_.push_back({{{5, 2, 2}, {5, 2, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
        if (c.index < para.inputTensorDesc_.size()) {
            para.inputTensorDesc_[c.index] = c.tensor;
        } else {
            para.outputTensorDesc_[0] = c.tensor;
        }
        ExecuteTestCase(para, ge::GRAPH_FAILED, 0, "", {});
    }
}

TEST(MoeFinalizeRoutingMx4Tiling, RejectsSpecialExpertRanges)
{
    using Attr = gert::TilingContextPara::OpAttr;
    const gert::TilingContextPara::TensorDescription absent = {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND};
    std::vector<gert::TilingContextPara::TensorDescription> inputs(INPUT_COUNT, absent);
    inputs[EXPANDED_X_IDX] = {{{5, 66}, {5, 66}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND};
    inputs[EXPANDED_ROW_IDX_IDX] = {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND};
    inputs[EXPANDED_X_SCALE_IDX] = {{{5, 2, 2}, {5, 2, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
    const char* rangeNames[] = {"zero_expert_range", "copy_expert_range", "constant_expert_range"};
    for (size_t i = 0; i < std::size(rangeNames); ++i) {
        for (const auto& range : {std::vector<int64_t>{0, 1}, std::vector<int64_t>{0}}) {
            auto attrs = Mx4Attrs<Attr>(DROP_LESS_ROW, 2);
            attrs[i + 1] = {rangeNames[i],
                            Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>(range))};
            optiling::MoeFinalizeRoutingCompileInfoV2 compileInfo = {kMx4AivCount, kMx4UbSize};
            gert::TilingContextPara para("MoeFinalizeRoutingV2", inputs,
                                         {{{{3, 66}, {3, 66}}, ge::DT_BF16, ge::FORMAT_ND}}, attrs, &compileInfo,
                                         "Ascend950", kMx4AivCount, kMx4UbSize, kMx4TilingDataCapacity, kA5SocInfo);
            ExecuteTestCase(para, ge::GRAPH_FAILED, 0, "", {});
        }
    }
}
