/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include <initializer_list>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shaperange_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "log/log.h"

class MoeInitRoutingV4 : public testing::Test {
protected:
};

// V4: 6 inputs (x, expert_idx, scale, offset, active_num, topk_weight), 5 outputs (+expanded_topk_weight)
// V4: 8 attrs (no active_num): expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type,
//     expert_tokens_num_flag, quant_mode, active_expert_range, row_idx_type
// V4 optional inputs: scale(2), offset(3), active_num(4), topk_weight(5)
// IrInstanceNum controls which inputs are "present" in the IR for GetOptionalInputShape().

// basic: unquant, scale present, no offset, no active_num input, no topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 30})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 1, 0, 0, 0},   // inputInstanceNum: x, expert_idx, scale present
                                                      {1, 1, 1, 1, 1});      // outputInstanceNum: all 5 outputs present
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {30}, {-1}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// active_num=0 (via attr removed, now default), scale present
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_2)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 1, 0, 0, 0},
                                                      {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// static shape: scale present, no active_num input, no topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_3)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{3, 128}, {3, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 1, 0, 0, 0},
                                                      {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{24, 128}, {24}, {7}, {24}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic quant, scale present, no topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_4)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 1, 0, 0, 0},
                                                      {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic quant with per-expert scale
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_5)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{8 * 512, 1024}, {8 * 512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8 * 512, 512}, {8 * 512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{7, 1024}, {7, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 1, 0, 0, 0},
                                                      {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512 * 512, 1024}, {8 * 512 * 512}, {7}, {8 * 512 * 512}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic quant with 1H scale, scatter
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_6)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{8 * 512, 1024}, {8 * 512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8 * 512, 512}, {8 * 512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                      },
                                                      {1, 1, 1, 0, 0, 0},
                                                      {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512 * 512, 1024}, {8 * 512 * 512}, {7}, {8 * 512 * 512}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// static quant with scale + offset
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_7)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{8 * 512, 1024}, {8 * 512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8 * 512, 512}, {8 * 512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 1, 1, 0, 0},   // scale + offset present for static quant
                                                      {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512 * 512, 1024}, {8 * 512 * 512}, {7}, {}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// key_value mode, scatter
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_8)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{2087, 192}, {2087, 192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{2087, 7242}, {2087, 7242}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                      },
                                                      {1, 1, 1, 0, 0, 0},
                                                      {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {256,2}, {15114054}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// with topk_weight input + expanded_topk_weight output present
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_with_topkweight)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{8, 1024}, {8, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8, 512}, {8, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{8, 512}, {8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 0, 0, 0, 1},   // x, expert_idx, topk_weight present; scale/offset absent for unquant
                                                      {1, 1, 1, 1, 1});
    // expanded_topk_weight: topkWeightOutNum = n*k = 8*512 = 4096, shape = {4096, 1}
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512, 1024}, {8 * 512}, {7}, {8 * 512}, {8 * 512, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// with active_num input (scalar tensor) + topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_with_active_num_and_topkweight)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{8, 1024}, {8, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8, 512}, {8, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{8, 512}, {8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 0, 0, 1, 1},   // x, expert_idx, active_num, topk_weight present
                                                      {1, 1, 1, 1, 1});
    // expanded_topk_weight: topkWeightOutNum = n*k = 8*512 = 4096, shape = {4096, 1}
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512, 1024}, {8 * 512}, {7}, {8 * 512}, {8 * 512, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// empty tensor: n==0
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_empty_n)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{0, 128}, {0, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{0, 8}, {0, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 256})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 0, 0, 0, 0},   // x, expert_idx present; no scale/offset/active_num/topk_weight
                                                      {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 128}, {0}, {256}, {0}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// DropPad + topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_droppad_with_topkweight)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{8, 1024}, {8, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8, 512}, {8, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{8, 512}, {8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 256})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 0, 0, 0, 1},
                                                      {1, 1, 1, 1, 1});
    // DropPad: expandedXOut = [expertNum, expertCapacity, H], expandedTopkWeightOut = [expertNum*expertCapacity, 1]
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 4, 1024}, {8 * 512}, {256}, {256 * 4}, {256 * 4, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// BF16, unquant, no scale, no topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND},
                                                        {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 256})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                      },
                                                      {1, 1, 0, 0, 0, 0},   // x, expert_idx present; no scale/offset/active_num/topk_weight
                                                      {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {256}, {-1}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic quant with topk_weight, expanded_topk_weight present
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_dynamic_with_topkweight)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{8, 1024}, {8, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8, 512}, {8, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{8, 512}, {8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 1, 0, 0, 1},   // x, expert_idx, scale, topk_weight present
                                                      {1, 1, 1, 1, 1});
    // expanded_topk_weight: topkWeightOutNum = n*k = 8*512 = 4096, shape = {4096, 1}
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512, 1024}, {8 * 512}, {7}, {8 * 512}, {8 * 512, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// no expert_tokens_num_flag
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_no_expert_tokens)
{
    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4",
                                                      {
                                                        {{{8 * 512, 1024}, {8 * 512, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{8 * 512, 512}, {8 * 512, 512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
                                                        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"expert_tokens_num_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"expert_tokens_num_flag",Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                                        {"quant_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                        {"active_expert_range",Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
                                                        {"row_idx_type",Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      },
                                                      {1, 1, 1, 1, 0, 0},   // x, expert_idx, scale, offset present for static quant
                                                      {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512 * 512, 1024}, {8 * 512 * 512}, {}, {}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
