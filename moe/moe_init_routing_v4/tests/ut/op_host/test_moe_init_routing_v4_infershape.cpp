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
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 30})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 1, 0, 0, 0}, // inputInstanceNum: x, expert_idx, scale present
        {1, 1, 1, 1, 1});   // outputInstanceNum: all 5 outputs present
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {30}, {-1}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// active_num=0 (via attr removed, now default), scale present
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 1, 0, 0, 0}, {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// static shape: scale present, no active_num input, no topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 1, 0, 0, 0}, {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{24, 128}, {24}, {7}, {24}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic quant, scale present, no topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 1, 0, 0, 0}, {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {7}, {-1}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic quant with per-expert scale
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 1, 0, 0, 0}, {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8 * 512 * 512, 1024}, {8 * 512 * 512}, {7}, {8 * 512 * 512}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic quant with 1H scale, scatter
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        {1, 1, 1, 0, 0, 0}, {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8 * 512 * 512, 1024}, {8 * 512 * 512}, {7}, {8 * 512 * 512}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// static quant with scale + offset
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_7)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 1, 1, 0, 0}, // scale + offset present for static quant
        {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512 * 512, 1024}, {8 * 512 * 512}, {7}, {}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// key_value mode, scatter
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({87, 222})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        {1, 1, 1, 0, 0, 0}, {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{15114054, 192}, {15114054}, {256, 2}, {15114054}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// with topk_weight input + expanded_topk_weight output present
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_with_topkweight)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 0, 0, 0, 1}, // x, expert_idx, topk_weight present; scale/offset absent for unquant
        {1, 1, 1, 1, 1});
    // expanded_topk_weight: topkWeightOutNum = n*k = 8*512 = 4096, shape = {4096, 1}
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512, 1024}, {8 * 512}, {7}, {8 * 512}, {8 * 512, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// with active_num input (scalar tensor) + topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_with_active_num_and_topkweight)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 0, 0, 1, 1}, // x, expert_idx, active_num, topk_weight present
        {1, 1, 1, 1, 1});
    // expanded_topk_weight: topkWeightOutNum = n*k = 8*512 = 4096, shape = {4096, 1}
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512, 1024}, {8 * 512}, {7}, {8 * 512}, {8 * 512, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// empty tensor: n==0
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_empty_n)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 256})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 0, 0, 0, 0}, // x, expert_idx present; no scale/offset/active_num/topk_weight
        {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 128}, {0}, {256}, {0}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// DropPad + topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_droppad_with_topkweight)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 256})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 0, 0, 0, 1}, {1, 1, 1, 1, 1});
    // DropPad: expandedXOut = [expertNum, expertCapacity, H], expandedTopkWeightOut = [expertNum*expertCapacity, 1]
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 4, 1024}, {8 * 512}, {256}, {256 * 4}, {256 * 4, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// BF16, unquant, no scale, no topk_weight
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 256})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        {1, 1, 0, 0, 0, 0}, // x, expert_idx present; no scale/offset/active_num/topk_weight
        {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1}, {256}, {-1}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic quant with topk_weight, expanded_topk_weight present
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_dynamic_with_topkweight)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 1, 0, 0, 1}, // x, expert_idx, scale, topk_weight present
        {1, 1, 1, 1, 1});
    // expanded_topk_weight: topkWeightOutNum = n*k = 8*512 = 4096, shape = {4096, 1}
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512, 1024}, {8 * 512}, {7}, {8 * 512}, {8 * 512, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// no expert_tokens_num_flag
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infer_shape_no_expert_tokens)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeInitRoutingV4",
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
            {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8})},
            {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        {1, 1, 1, 1, 0, 0}, // x, expert_idx, scale, offset present for static quant
        {1, 1, 1, 1, 1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{8 * 512 * 512, 1024}, {8 * 512 * 512}, {}, {}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

namespace {
constexpr int64_t EXPERT_CAPACITY = 0LL;
constexpr int64_t DROP_PAD_MODE = 0LL;
constexpr bool EXPERT_TOKENS_NUM_FLAG = true;
constexpr int64_t EXPERT_TOKENS_TYPE_COUNT = 1LL;
constexpr int64_t EXPERT_TOKENS_TYPE_CUMSUM = 0LL;
constexpr int64_t QUANT_MODE_UNQUANT = -1LL;
constexpr int64_t QUANT_MODE_STATIC = 0LL;
constexpr int64_t QUANT_MODE_DYNAMIC = 1LL;
constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2LL;
constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3LL;
constexpr int64_t QUANT_MODE_FP8_GROUP_E5M2 = 4LL;
constexpr int64_t QUANT_MODE_FP8_GROUP_E4M3FN = 5LL;
constexpr int64_t QUANT_MODE_HIF8_CAST = 6LL;
constexpr int64_t QUANT_MODE_HIF8_PERTENSOR = 7LL;
constexpr int64_t QUANT_MODE_HIF8_PERTOKEN = 8LL;
constexpr int64_t QUANT_MODE_MXFP4_E2M1 = 9LL;
constexpr int64_t QUANT_MODE_FP8_PERBLOCK_E5M2 = 11LL;
constexpr int64_t QUANT_MODE_FP8_PERBLOCK_E4M3FN = 12LL;
constexpr int64_t QUANT_MODE_INT4_DYNAMIC = 13LL;
constexpr int64_t QUANT_MODE_FP8_GROUP_AMAX_E5M2 = 14LL;
constexpr int64_t QUANT_MODE_FP8_GROUP_AMAX_E4M3FN = 15LL;
constexpr int64_t QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 = 16LL;
constexpr int64_t QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN = 17LL;
constexpr int64_t ROW_IDX_TYPE_GATHER = 0LL;
} // namespace

using ShapeList = std::initializer_list<int64_t>;

ge::DataType GetExpandedXDtype(ge::DataType xDtype, int64_t quantMode)
{
    switch (quantMode) {
        case QUANT_MODE_UNQUANT:
            return xDtype;
        case QUANT_MODE_STATIC:
        case QUANT_MODE_DYNAMIC:
            return ge::DT_INT8;
        case QUANT_MODE_INT4_DYNAMIC:
            return ge::DT_INT4;
        case QUANT_MODE_MXFP8_E5M2:
        case QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2:
        case QUANT_MODE_FP8_GROUP_E5M2:
        case QUANT_MODE_FP8_GROUP_AMAX_E5M2:
        case QUANT_MODE_FP8_PERBLOCK_E5M2:
            return ge::DT_FLOAT8_E5M2;
        case QUANT_MODE_MXFP8_E4M3FN:
        case QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN:
        case QUANT_MODE_FP8_GROUP_E4M3FN:
        case QUANT_MODE_FP8_GROUP_AMAX_E4M3FN:
        case QUANT_MODE_FP8_PERBLOCK_E4M3FN:
            return ge::DT_FLOAT8_E4M3FN;
        case QUANT_MODE_HIF8_CAST:
        case QUANT_MODE_HIF8_PERTENSOR:
        case QUANT_MODE_HIF8_PERTOKEN:
            return ge::DT_HIFLOAT8;
        case QUANT_MODE_MXFP4_E2M1:
            return ge::DT_FLOAT4_E2M1;
        default:
            return xDtype;
    }
}

ge::DataType GetExpandedScaleDtype(int64_t quantMode)
{
    if (quantMode == QUANT_MODE_MXFP8_E5M2 || quantMode == QUANT_MODE_MXFP8_E4M3FN ||
        quantMode == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 || quantMode == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN ||
        quantMode == QUANT_MODE_MXFP4_E2M1) {
        return ge::DT_FLOAT8_E8M0;
    }
    return ge::DT_FLOAT;
}

std::vector<int64_t> ComputeExpandedScaleShape(int64_t quantMode, int64_t outNum, int64_t cols)
{
    if (quantMode == QUANT_MODE_MXFP8_E5M2 || quantMode == QUANT_MODE_MXFP8_E4M3FN ||
        quantMode == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 || quantMode == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN) {
        int64_t m = (cols + 31) / 32;
        m = (m + 1) / 2 * 2;
        return {outNum, m};
    }
    if (quantMode == QUANT_MODE_FP8_GROUP_E5M2 || quantMode == QUANT_MODE_FP8_GROUP_E4M3FN ||
        quantMode == QUANT_MODE_FP8_GROUP_AMAX_E5M2 || quantMode == QUANT_MODE_FP8_GROUP_AMAX_E4M3FN) {
        return {outNum, (cols + 127) / 128};
    }
    if (quantMode == QUANT_MODE_HIF8_PERTOKEN) {
        return {outNum};
    }
    if (quantMode == QUANT_MODE_MXFP4_E2M1) {
        return {outNum, (cols + 63) / 64, 2};
    }
    if (quantMode == QUANT_MODE_FP8_PERBLOCK_E5M2 || quantMode == QUANT_MODE_FP8_PERBLOCK_E4M3FN) {
        return {outNum, (cols + 255) / 256, 2};
    }
    if (quantMode == QUANT_MODE_STATIC || quantMode == QUANT_MODE_HIF8_CAST || quantMode == QUANT_MODE_HIF8_PERTENSOR) {
        return {};
    }
    return {outNum};
}

void RunQuantInferShape(ShapeList xShape, ge::DataType xDtype, ShapeList expertIdxShape, ShapeList scaleShape,
                        ShapeList offsetShape, int64_t quantMode, std::vector<int64_t> activeExpertRange,
                        ShapeList expectOutShape0, ShapeList expectOutShape1, ShapeList expectOutShape2,
                        ShapeList expectOutShape3)
{
    int64_t n = xShape.begin()[0];
    int64_t k = expertIdxShape.begin()[1];
    int64_t outNum = n * k;
    int64_t cols = xShape.begin()[1];

    ge::DataType expandedXDtype = GetExpandedXDtype(xDtype, quantMode);
    ge::DataType expandedScaleDtype = GetExpandedScaleDtype(quantMode);

    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {{xShape, xShape}, xDtype, ge::FORMAT_ND},
        {{expertIdxShape, expertIdxShape}, ge::DT_INT32, ge::FORMAT_ND},
    };
    if (scaleShape.size() > 0) {
        inputs.emplace_back(
            gert::InfershapeContextPara::TensorDescription{{scaleShape, scaleShape}, ge::DT_FLOAT, ge::FORMAT_ND});
    } else {
        inputs.emplace_back(gert::InfershapeContextPara::TensorDescription{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND});
    }
    if (offsetShape.size() > 0) {
        inputs.emplace_back(
            gert::InfershapeContextPara::TensorDescription{{offsetShape, offsetShape}, ge::DT_FLOAT, ge::FORMAT_ND});
    } else {
        inputs.emplace_back(gert::InfershapeContextPara::TensorDescription{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND});
    }
    inputs.emplace_back(gert::InfershapeContextPara::TensorDescription{{{}, {}}, ge::DT_INT64, ge::FORMAT_ND});
    inputs.emplace_back(gert::InfershapeContextPara::TensorDescription{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND});

    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {{{}, {}}, expandedXDtype, ge::FORMAT_ND}, {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   {{{}, {}}, expandedScaleDtype, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };

    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_CAPACITY)},
        {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
        {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(DROP_PAD_MODE)},
        {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_TOKENS_TYPE_COUNT)},
        {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(EXPERT_TOKENS_NUM_FLAG)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
        {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(activeExpertRange)},
        {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_GATHER)},
    };

    std::vector<uint32_t> inputInstanceNum = {1, 1, (scaleShape.size() > 0 ? 1 : 0), (offsetShape.size() > 0 ? 1 : 0),
                                              0, 0};
    std::vector<uint32_t> outputInstanceNum = {1, 1, 1, 1, 1};

    gert::InfershapeContextPara infershapeContextPara("MoeInitRoutingV4", inputs, outputs, attrs, inputInstanceNum,
                                                      outputInstanceNum);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        expectOutShape0, expectOutShape1, expectOutShape2, expectOutShape3, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// MXFP8 E5M2 量化 + h为32倍数
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_mxfp8_e5m2)
{
    // x=(32,7168) n=32 k=8 outNum=256 cols=7168
    // expandedScale: [256, CeilAlign(CeilDiv(7168,32),2)] = [256, CeilAlign(224,2)] = [256, 224]
    RunQuantInferShape({32, 7168}, ge::DT_FLOAT16, {32, 8}, {}, {}, QUANT_MODE_MXFP8_E5M2, {0, 256}, {256, 7168}, {256},
                       {256}, {256, 224});
}

// MXFP8 E4M3FN 量化 + h不为32倍数
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_mxfp8_e4m3fn)
{
    // x=(32,1111) cols=1111 CeilDiv(1111,32)=35 CeilAlign(35,2)=36
    RunQuantInferShape({32, 1111}, ge::DT_BF16, {32, 8}, {}, {}, QUANT_MODE_MXFP8_E4M3FN, {0, 256}, {256, 1111}, {256},
                       {256}, {256, 36});
}

// MXFP8 RoundScale+Amax E5M2 量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_mxfp8_rs_amax_e5m2)
{
    // x=(32,65) cols=65 CeilDiv(65,32)=3 CeilAlign(3,2)=4
    RunQuantInferShape({32, 65}, ge::DT_FLOAT16, {32, 8}, {}, {}, QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2, {0, 256},
                       {256, 65}, {256}, {256}, {256, 4});
}

// MXFP8 RoundScale+Amax E4M3FN 量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_mxfp8_rs_amax_e4m3fn)
{
    // x=(32,97) cols=97 CeilDiv(97,32)=4 CeilAlign(4,2)=4
    RunQuantInferShape({32, 97}, ge::DT_BF16, {32, 8}, {}, {}, QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN, {0, 256},
                       {256, 97}, {256}, {256}, {256, 4});
}

// FP8 PerGroup E5M2 量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_fp8_group_e5m2)
{
    // x=(32,1024) cols=1024 CeilDiv(1024,128)=8
    RunQuantInferShape({32, 1024}, ge::DT_FLOAT16, {32, 8}, {}, {}, QUANT_MODE_FP8_GROUP_E5M2, {0, 256}, {256, 1024},
                       {256}, {256}, {256, 8});
}

// FP8 PerGroup E4M3FN 量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_fp8_group_e4m3fn)
{
    // x=(32,513) cols=513 CeilDiv(513,128)=5
    RunQuantInferShape({32, 513}, ge::DT_BF16, {32, 8}, {}, {}, QUANT_MODE_FP8_GROUP_E4M3FN, {0, 256}, {256, 513},
                       {256}, {256}, {256, 5});
}

// FP8 PerGroup E5M2 + Amax
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_fp8_group_amax_e5m2)
{
    RunQuantInferShape({32, 1024}, ge::DT_FLOAT16, {32, 8}, {}, {}, QUANT_MODE_FP8_GROUP_AMAX_E5M2, {0, 256},
                       {256, 1024}, {256}, {256}, {256, 8});
}

// FP8 PerGroup E4M3FN + Amax
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_fp8_group_amax_e4m3fn)
{
    RunQuantInferShape({32, 513}, ge::DT_BF16, {32, 8}, {}, {}, QUANT_MODE_FP8_GROUP_AMAX_E4M3FN, {0, 256}, {256, 513},
                       {256}, {256}, {256, 5});
}

// FP8 PerBlock E5M2 量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_fp8_perblock_e5m2)
{
    // x=(32,1024) cols=1024 CeilDiv(1024,256)=4 expandedScale=[256,4,2]
    RunQuantInferShape({32, 1024}, ge::DT_FLOAT16, {32, 8}, {}, {}, QUANT_MODE_FP8_PERBLOCK_E5M2, {0, 256}, {256, 1024},
                       {256}, {256}, {256, 4, 2});
}

// FP8 PerBlock E4M3FN 量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_fp8_perblock_e4m3fn)
{
    // x=(32,512) cols=512 CeilDiv(512,256)=2 expandedScale=[256,2,2]
    RunQuantInferShape({32, 512}, ge::DT_BF16, {32, 8}, {}, {}, QUANT_MODE_FP8_PERBLOCK_E4M3FN, {0, 256}, {256, 512},
                       {256}, {256}, {256, 2, 2});
}

// MXFP4 E2M1 量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_mxfp4_e2m1)
{
    // x=(32,1024) cols=1024 CeilDiv(1024,64)=16 expandedScale=[256,16,2]
    RunQuantInferShape({32, 1024}, ge::DT_FLOAT16, {32, 8}, {}, {}, QUANT_MODE_MXFP4_E2M1, {0, 256}, {256, 1024}, {256},
                       {256}, {256, 16, 2});
}

// HIF8 直转量化（expandedScale不输出，空shape）
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_hif8_cast)
{
    RunQuantInferShape({32, 1024}, ge::DT_FLOAT16, {32, 8}, {}, {}, QUANT_MODE_HIF8_CAST, {0, 256}, {256, 1024}, {256},
                       {256}, {});
}

// HIF8 PERTENSOR 量化（scale必须输入shape=[1]，expandedScale不输出空shape）
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_hif8_pertensor)
{
    RunQuantInferShape({32, 1024}, ge::DT_BF16, {32, 8}, {1}, {}, QUANT_MODE_HIF8_PERTENSOR, {0, 256}, {256, 1024},
                       {256}, {256}, {});
}

// HIF8 PERTOKEN 量化（expandedScale=[outNum]）
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_hif8_pertoken)
{
    // x=(32,1024) n=32 k=8 outNum=256 expandedScale=[256]
    RunQuantInferShape({32, 1024}, ge::DT_FLOAT16, {32, 8}, {}, {}, QUANT_MODE_HIF8_PERTOKEN, {0, 256}, {256, 1024},
                       {256}, {256}, {256});
}

// INT4 动态量化 + scale为(1,H)
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_int4_dynamic_1)
{
    // x=(32,7168) FP32, scale=(1,7168) expandedScale=[256]
    RunQuantInferShape({32, 7168}, ge::DT_FLOAT, {32, 8}, {1, 7168}, {}, QUANT_MODE_INT4_DYNAMIC, {0, 256}, {256, 7168},
                       {256}, {256}, {256});
}

// INT4 动态量化 + BF16 + 小H
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_infershape_int4_dynamic_2)
{
    // x=(32,2048) BF16, scale=(1,2048) expandedScale=[256]
    RunQuantInferShape({32, 2048}, ge::DT_BF16, {32, 8}, {1, 2048}, {}, QUANT_MODE_INT4_DYNAMIC, {0, 256}, {256, 2048},
                       {256}, {256}, {256});
}

void RunTestcaseInferDataType(ge::DataType xDtype, int64_t quantMode, ge::DataType expectOutDtype0,
                              ge::DataType expectOutDtype3)
{
    static ge::DataType FP32Ref = ge::DT_FLOAT;
    static ge::DataType INT32Ref = ge::DT_INT32;
    static ge::DataType INT64Ref = ge::DT_INT64;
    static ge::DataType FLOATRef = ge::DT_FLOAT;

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeInitRoutingV4")->infer_datatype;
    if (inferDtypeFunc == nullptr) {
        return;
    }

    auto holder =
        gert::InferDataTypeContextFaker()
            .IrInputNum(6)
            .NodeIoNum(6, 5)
            .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"expert_capacity", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_CAPACITY)},
                 {"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(DROP_PAD_MODE)},
                 {"expert_tokens_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(EXPERT_TOKENS_TYPE_COUNT)},
                 {"expert_tokens_num_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(EXPERT_TOKENS_NUM_FLAG)},
                 {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
                 {"active_expert_range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1})},
                 {"row_idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ROW_IDX_TYPE_GATHER)}})
            .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&xDtype, &INT32Ref, &FP32Ref, &FP32Ref, &INT64Ref, &FLOATRef})
            .Build();
    ASSERT_EQ(inferDtypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);

    int idx = 0;
    auto outDtype0 = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(idx++);
    auto outDtype1 = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(idx++);
    auto outDtype2 = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(idx++);
    auto outDtype3 = holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(idx++);
    EXPECT_EQ(outDtype0, expectOutDtype0);
    EXPECT_EQ(outDtype1, INT32Ref);
    EXPECT_EQ(outDtype2, INT64Ref);
    EXPECT_EQ(outDtype3, expectOutDtype3);
}

void RunSuccessTestcaseInferDataType(ge::DataType xDtype, int64_t quantMode)
{
    ge::DataType expandedXDtype = GetExpandedXDtype(xDtype, quantMode);
    ge::DataType expandedScaleDtype = GetExpandedScaleDtype(quantMode);
    RunTestcaseInferDataType(xDtype, quantMode, expandedXDtype, expandedScaleDtype);
}

// ====== inferDataType 成功用例 ======

// FP16 非量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_unquant_fp16)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_UNQUANT);
}

// BF16 非量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_unquant_bf16)
{
    RunSuccessTestcaseInferDataType(ge::DT_BF16, QUANT_MODE_UNQUANT);
}

// FP16 静态量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_static_fp16)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_STATIC);
}

// FP32 静态量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_static_fp32)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT, QUANT_MODE_STATIC);
}

// FP16 动态量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_dynamic_fp16)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_DYNAMIC);
}

// FP32 动态量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_dynamic_fp32)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT, QUANT_MODE_DYNAMIC);
}

// FP16 + MXFP8 E5M2量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_mxfp8_e5m2)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_MXFP8_E5M2);
}

// BF16 + MXFP8 E5M2量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_mxfp8_e5m2_bf16)
{
    RunSuccessTestcaseInferDataType(ge::DT_BF16, QUANT_MODE_MXFP8_E5M2);
}

// FP16 + MXFP8 E4M3FN量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_mxfp8_e4m3fn)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_MXFP8_E4M3FN);
}

// BF16 + MXFP8 E4M3FN量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_mxfp8_e4m3fn_bf16)
{
    RunSuccessTestcaseInferDataType(ge::DT_BF16, QUANT_MODE_MXFP8_E4M3FN);
}

// FP16 + MXFP8 RoundScale+Amax E5M2量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_mxfp8_rs_amax_e5m2)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2);
}

// BF16 + MXFP8 RoundScale+Amax E4M3FN量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_mxfp8_rs_amax_e4m3fn)
{
    RunSuccessTestcaseInferDataType(ge::DT_BF16, QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN);
}

// FP16 + FP8 PerGroup E5M2量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_fp8_group_e5m2)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_FP8_GROUP_E5M2);
}

// BF16 + FP8 PerGroup E4M3FN量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_fp8_group_e4m3fn)
{
    RunSuccessTestcaseInferDataType(ge::DT_BF16, QUANT_MODE_FP8_GROUP_E4M3FN);
}

// FP16 + FP8 PerGroup E5M2 + Amax
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_fp8_group_amax_e5m2)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_FP8_GROUP_AMAX_E5M2);
}

// BF16 + FP8 PerGroup E4M3FN + Amax
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_fp8_group_amax_e4m3fn)
{
    RunSuccessTestcaseInferDataType(ge::DT_BF16, QUANT_MODE_FP8_GROUP_AMAX_E4M3FN);
}

// FP16 + FP8 PerBlock E5M2量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_fp8_perblock_e5m2)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_FP8_PERBLOCK_E5M2);
}

// BF16 + FP8 PerBlock E4M3FN量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_fp8_perblock_e4m3fn)
{
    RunSuccessTestcaseInferDataType(ge::DT_BF16, QUANT_MODE_FP8_PERBLOCK_E4M3FN);
}

// FP16 + MXFP4 E2M1量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_mxfp4_e2m1)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_MXFP4_E2M1);
}

// BF16 + MXFP4 E2M1量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_mxfp4_e2m1_bf16)
{
    RunSuccessTestcaseInferDataType(ge::DT_BF16, QUANT_MODE_MXFP4_E2M1);
}

// FP16 + HIF8 直转量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_hif8_cast)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_HIF8_CAST);
}

// BF16 + HIF8 PERTENSOR量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_hif8_pertensor)
{
    RunSuccessTestcaseInferDataType(ge::DT_BF16, QUANT_MODE_HIF8_PERTENSOR);
}

// FP16 + HIF8 PERTOKEN量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_hif8_pertoken)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT16, QUANT_MODE_HIF8_PERTOKEN);
}

// FP32 + INT4动态量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_int4_dynamic_fp32)
{
    RunSuccessTestcaseInferDataType(ge::DT_FLOAT, QUANT_MODE_INT4_DYNAMIC);
}

// BF16 + INT4动态量化
TEST_F(MoeInitRoutingV4, moe_init_routing_v4_inferdatatype_int4_dynamic_bf16)
{
    RunSuccessTestcaseInferDataType(ge::DT_BF16, QUANT_MODE_INT4_DYNAMIC);
}

// ====== inferDataType 失败用例 ======
