/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_kernel/ffn_to_attention_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace FFNToAttentionUT {

class FFNToAttentionArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FFNToAttentionArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FFNToAttentionArch22TilingTest TearDown" << std::endl;
    }
};

TEST_F(FFNToAttentionArch22TilingTest, NoRankTable)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}},
        &compileInfo);

    uint64_t expectTilingKey = 2;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTable)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}},
        &compileInfo);

    uint64_t expectTilingKey = 3;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTableHInvalid)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 800}, {1584, 800}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}},
        &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTableSessionIdInvalid)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1583}, {1583}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}},
        &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTableMircoBatchIdsInvalid)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1583}, {1583}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}},
        &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTableTokenIdsInvalid)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1583}, {1583}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}},
        &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTableExpertNumoffsetIdsInvalid)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1583}, {1583}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}},
        &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTableActualtokenNumInvalid)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}},
        &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTableBsInvalid)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 513, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 513, 9, 7168})}},
        &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTableMircoBatchNumInvalid)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({2, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({2, 16, 9, 7168})}},
        &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTableMircoBatchNumTypeInvalid)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}},
        &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(FFNToAttentionArch22TilingTest, RankTableXformatInvalid)
{
    struct FFNToAttentionCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "FFNToAttention",
        {{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
         {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, 7168})}},
        &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}
} // namespace FFNToAttentionUT
