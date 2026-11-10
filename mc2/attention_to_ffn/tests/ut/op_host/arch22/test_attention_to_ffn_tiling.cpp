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
#include "../../../../../common/tests/ut/formal_error_assertions.h"
#include "../../../../op_kernel/attention_to_ffn_tiling.h"
#include "mc2_tiling_case_executor.h"
#include "mc2/attention_to_ffn/op_host/op_tiling/attention_to_ffn_tiling_checks.h"

using Mc2ErrorCodeUT::ExcelErrorReportScope;
using Mc2ErrorCodeUT::ExpectExcelFormalError;

namespace AttentionToFFNUT {

class AttentionToFFNArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AttentionToFFNArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AttentionToFFNArch22TilingTest TearDown" << std::endl;
    }
};

TEST_F(AttentionToFFNArch22TilingTest, NoQuantSyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch22TilingTest, NoQuantAsyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 2UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch22TilingTest, QuantSyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 1UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch22TilingTest, QuantAsyncNoActivateMask)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 3UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch22TilingTest, NoQuantSyncActivateMask)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");
    uint64_t expectTilingKey = 4UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch22TilingTest, NoQuantAsyncActivateMask)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 6UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch22TilingTest, QuantSyncActivateMask)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 5UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch22TilingTest, QuantAsyncActivateMask)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 7UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AttentionToFFNArch22TilingTest, MicroBatchNumInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{2, 16, 7168}, {2, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, BsInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 513, 7168}, {1, 513, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, KInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 16}, {1, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, HInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 8196}, {1, 16, 8196}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, LayerNumInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2, 9, 4}, {2, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, MoeExpertNumInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1026, 4}, {1, 1026, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, FfnTokenInfoTableShapeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, FfnTokenDataShapeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, AttnTokenInfoTableShapeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, XDimInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168, 1}, {1, 16, 7168, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, SessionIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, MicroBatchIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, LayerIdDimInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ExpertIdsDimInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ExpertRankTableDimInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{9, 4}, {9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ScalesDimInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 7168, 1}, {1, 9, 7168, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ActiveMaskDimInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, XDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, SessionIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, MicroBatchIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, LayerIdDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ExpertIdsDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT16, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ExpertRankTableDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT16, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ScalesDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ActiveMaskDtypeInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 16}, {1, 16}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, XFormatInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, SessionIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, MicroBatchIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, LayerIdFormatInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ExpertIdsFormatInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ExpertRankTableDormatInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ScalesFormatInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, ActiveMaskFormatInvalid)
{
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 7168}, {1, 9, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_FRACTAL_NZ}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AttentionToFFNArch22TilingTest, InvalidExpertCountReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "moe_expert_num", "0");
}

TEST_F(AttentionToFFNArch22TilingTest, InvalidQuantModeReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "quant_mode", "9");
}

TEST_F(AttentionToFFNArch22TilingTest, InvalidSyncFlagReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "sync_flag", "2");
}

TEST_F(AttentionToFFNArch22TilingTest, InvalidStartRankRelationReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "ffn_start_rank_id", "16");
}

TEST_F(AttentionToFFNArch22TilingTest, InvalidWorldSizeReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "world_size", "1");
}

TEST_F(AttentionToFFNArch22TilingTest, InvalidFfnRangeReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(12)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0027", "ffn_start_rank_id, ffn_num, world_size",
                           "ffn_start_rank_id=12, ffn_num=5, world_size=16",
                           "ffn_start_rank_id + ffn_num must not exceed world_size");
}
TEST_F(AttentionToFFNArch22TilingTest, NegativeExpertCountPreservesSignedActual)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "moe_expert_num", "-1");
}

TEST_F(AttentionToFFNArch22TilingTest, ZeroWorldSizeReportsBeforeRank)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "world_size", "0");
}

TEST_F(AttentionToFFNArch22TilingTest, NegativeWorldSizeReportsBeforeRank)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "world_size", "-1");
}

TEST_F(AttentionToFFNArch22TilingTest, ShapeListLengthPreservesSingleLeafReport)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    uint64_t expectTilingKey = 0UL;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0011", "ffn_token_data_shape", "4D", nullptr, nullptr, {{"correct_dim", "5D"}});
}

TEST_F(AttentionToFFNArch22TilingTest, MissingWorldSizeUsesPublicAttributeName)
{
    ExcelErrorReportScope reportScope;
    struct AttentionToFFNCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "AttentionToFFN",
        {{{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
         {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
         {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93");

    tilingContextPara.attrs_.erase(tilingContextPara.attrs_.begin() + 1, tilingContextPara.attrs_.end());
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    const auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0037");
    EXPECT_EQ(errors[0].args_map.at("param_name"), "world_size");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Required attribute world_size is missing; provide this attribute");
}

} // namespace AttentionToFFNUT
