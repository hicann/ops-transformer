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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/attention_worker_combine_tiling_base.h"

using namespace std;
using namespace optiling;

class AttentionWorkerCombineTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AttentionWorkerCombineTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AttentionWorkerCombineTilingTest TearDown" << std::endl;
    }
};

TEST_F(AttentionWorkerCombineTilingTest, attention_worker_combine_tiling_test01)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{32, 8}, {32, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};

    gert::StorageShape y_shape_out = {{32, 7168}, {32, 7168}};
    gert::StorageShape next_layer_id_shape_out = {{1}, {1}};

    AttentionWorkerCombineCompileInfo compileInfo = {64, 256 * 1024};
    gert::TilingContextPara tilingContextPara("AttentionWorkerCombine",
                                              {// input
                                               {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
                                               {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {// output
                                               {y_shape_out, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {next_layer_id_shape_out, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7168)},
                                               {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}},
                                              &compileInfo);
    int64_t expectTilingKey = 10020;
    std::string expectTilingData = "32 32 8 7168 15 1 32 1 1 7168 0 1 0 1 2 0 4 ";
    std::vector<size_t> expectWorkspaces = {32};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AttentionWorkerCombineTilingTest, attention_worker_combine_tiling_test02)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{32, 8}, {32, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};

    gert::StorageShape y_shape_out = {{32, 20480}, {32, 20480}};
    gert::StorageShape next_layer_id_shape_out = {{1}, {1}};

    AttentionWorkerCombineCompileInfo compileInfo = {64, 256 * 1024};
    gert::TilingContextPara tilingContextPara("AttentionWorkerCombine",
                                              {// input
                                               {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
                                               {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {// output
                                               {y_shape_out, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {next_layer_id_shape_out, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20480)},
                                               {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}},
                                              &compileInfo);
    int64_t expectTilingKey = 10010;
    std::string expectTilingData = "32 32 8 20480 15 1 32 1 1 15872 4608 1 0 2 1 0 8 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AttentionWorkerCombineTilingTest, attention_worker_combine_tiling_test03)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{32, 8}, {32, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};

    gert::StorageShape y_shape_out = {{32, 1024}, {32, 1024}};
    gert::StorageShape next_layer_id_shape_out = {{1}, {1}};

    AttentionWorkerCombineCompileInfo compileInfo = {64, 256 * 1024};
    gert::TilingContextPara tilingContextPara("AttentionWorkerCombine",
                                              {// input
                                               {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
                                               {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {// output
                                               {y_shape_out, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {next_layer_id_shape_out, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
                                               {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}},
                                              &compileInfo);
    int64_t expectTilingKey = 10000;
    std::string expectTilingData = "32 32 8 1024 15 1 32 1 1 1024 0 1 1 1 8 8 1 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AttentionWorkerCombineTilingTest, attention_worker_combine_tiling_test04)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{48, 8}, {48, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};

    gert::StorageShape y_shape_out = {{48, 7168}, {48, 7168}};
    gert::StorageShape next_layer_id_shape_out = {{1}, {1}};

    AttentionWorkerCombineCompileInfo compileInfo = {64, 256 * 1024};
    gert::TilingContextPara tilingContextPara("AttentionWorkerCombine",
                                              {// input
                                               {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
                                               {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {// output
                                               {y_shape_out, ge::DT_BF16, ge::FORMAT_ND},
                                               {next_layer_id_shape_out, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7168)},
                                               {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}},
                                              &compileInfo);
    int64_t expectTilingKey = 10021;
    std::string expectTilingData = "48 48 8 7168 15 1 48 1 1 7168 0 1 0 1 2 0 4 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AttentionWorkerCombineTilingTest, attention_worker_combine_tiling_test05)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{48, 8}, {48, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};

    gert::StorageShape y_shape_out = {{48, 20480}, {48, 20480}};
    gert::StorageShape next_layer_id_shape_out = {{1}, {1}};

    AttentionWorkerCombineCompileInfo compileInfo = {64, 256 * 1024};
    gert::TilingContextPara tilingContextPara("AttentionWorkerCombine",
                                              {// input
                                               {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
                                               {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {// output
                                               {y_shape_out, ge::DT_BF16, ge::FORMAT_ND},
                                               {next_layer_id_shape_out, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(20480)},
                                               {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                               {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}},
                                              &compileInfo);
    int64_t expectTilingKey = 10011;
    std::string expectTilingData = "48 48 8 20480 15 1 48 1 1 15872 4608 1 0 2 1 0 8 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

namespace {
void CheckNonquantOutputType(int64_t tokenDtype, ge::DataType outputType, ge::graphStatus status)
{
    constexpr int64_t r = 3;
    constexpr int64_t k = 2;
    constexpr int64_t h = 64;
    AttentionWorkerCombineCompileInfo compileInfo = {8, 256 * 1024};
    gert::TilingContextPara para(
        "AttentionWorkerCombine",
        {{{{1024}, {1024}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{r, k}, {r, k}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{r, h}, {r, h}}, outputType, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(h)},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenDtype)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        {1, 1, 1}, {1, 1}, &compileInfo, "Ascend950", 8, 256 * 1024, 8192);
    ExecuteTestCase(para, status, 0);
}

void CheckMxfpTiling(int64_t h, int64_t scaleCols, int64_t dtype, ge::DataType scaleType, ge::graphStatus status,
                     int64_t r = 3, int64_t k = 2, const char *soc = "Ascend950", ge::DataType outputType = ge::DT_BF16,
                     int64_t keyBase = 11000, int64_t schedule = 0, const std::string &expected = "")
{
    AttentionWorkerCombineCompileInfo compileInfo = {8, 256 * 1024};
    gert::TilingContextPara para(
        "AttentionWorkerCombine",
        {{{{1024}, {1024}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{r, k, scaleCols}, {r, k, scaleCols}}, scaleType, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{r, h}, {r, h}}, outputType, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(h)},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtype)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(schedule)}},
        {1, 1, 1}, {1, 1}, &compileInfo, soc, 8, 256 * 1024, 8192);
    ExecuteTestCase(para, status, status == ge::GRAPH_SUCCESS ? keyBase + dtype : 0, expected);
}
} // namespace

TEST_F(AttentionWorkerCombineTilingTest, nonquant_reject_output_dtype_mismatch)
{
    CheckNonquantOutputType(0, ge::DT_BF16, ge::GRAPH_FAILED);
    CheckNonquantOutputType(1, ge::DT_FLOAT16, ge::GRAPH_FAILED);
}

TEST_F(AttentionWorkerCombineTilingTest, mxfp_tail_and_large_h)
{
    for (int64_t dtype : {2, 3, 4}) {
        for (int64_t h : {1, 31, 32, 33, 64, 65, 95, 96, 97, 32769}) {
            CheckMxfpTiling(h, ((h + 63) / 64) * 2, dtype, ge::DT_FLOAT8_E8M0, ge::GRAPH_SUCCESS, 3, 2, "Ascend950",
                            ge::DT_BF16, h == 32769 ? 11010 : 11000);
        }
    }
}

TEST_F(AttentionWorkerCombineTilingTest, mxfp_reject_invalid_inputs)
{
    CheckMxfpTiling(65, 3, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_FAILED);
    CheckMxfpTiling(65, 6, 3, ge::DT_FLOAT8_E8M0, ge::GRAPH_FAILED);
    CheckMxfpTiling(65, 4, 2, ge::DT_FLOAT, ge::GRAPH_FAILED);
    CheckMxfpTiling(65, 4, 5, ge::DT_FLOAT8_E8M0, ge::GRAPH_FAILED);
    CheckMxfpTiling(0, 0, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_FAILED);
    CheckMxfpTiling(65, 4, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_FAILED, 0);
    CheckMxfpTiling(65, 4, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_FAILED, 3, 66);
    CheckMxfpTiling(65, 4, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_FAILED, 3, 1);
    CheckMxfpTiling(65, 4, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_SUCCESS, 3, 65);
    CheckMxfpTiling(65, 4, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_FAILED, 3, 2, "Ascend910B");
    CheckMxfpTiling(65, 4, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_FAILED, 3, 2, "Ascend950", ge::DT_FLOAT16);
}

TEST_F(AttentionWorkerCombineTilingTest, mxfp_bs_k_h_selection)
{
    // 256 KiB UB, flags reuse input/output queues. Test both sides of the full-E and single-row limits.
    for (int64_t dtype : {2, 3, 4}) {
        const int64_t bsBoundary = dtype == 4 ? 6144 : 3488;
        const int64_t kBoundary = dtype == 4 ? 24960 : 23808;
        for (const auto &point : std::vector<std::pair<int64_t, int64_t>>{
                 {bsBoundary, 11000}, {bsBoundary + 32, 11020}, {kBoundary, 11020}, {kBoundary + 32, 11010}}) {
            CheckMxfpTiling(point.first, ((point.first + 63) / 64) * 2, dtype, ge::DT_FLOAT8_E8M0, ge::GRAPH_SUCCESS, 3,
                            65, "Ascend950", ge::DT_BF16, point.second);
        }
    }
}

TEST_F(AttentionWorkerCombineTilingTest, mxfp_h_parallel_and_scheduled)
{
    // E5M2 H tile=23808, five H tiles. Without scheduling use five cores; with it keep one.
    CheckMxfpTiling(100001, 3126, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_SUCCESS, 1, 65, "Ascend950", ge::DT_BF16, 11010, 0,
                    "5 1 65 100001 0 1 1 1 1 23808 4769 5 1 1 1 0 65 ");
    CheckMxfpTiling(100001, 3126, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_SUCCESS, 1, 65, "Ascend950", ge::DT_BF16, 11010, 1,
                    "1 1 65 100001 1 1 1 1 1 23808 4769 1 5 5 1 0 65 ");
}

TEST_F(AttentionWorkerCombineTilingTest, mxfp_h_rebalances_bs_groups)
{
    // Eight cores: five tokens become three BS groups, each with two H cores.
    // Five H blocks are distributed as three and two contiguous blocks.
    CheckMxfpTiling(100001, 3126, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_SUCCESS, 5, 65, "Ascend950", ge::DT_BF16, 11010, 0,
                    "6 5 65 100001 0 1 3 2 1 23808 4769 2 3 2 1 0 65 ");
    // Scheduling keeps each token's H blocks on one core.
    CheckMxfpTiling(100001, 3126, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_SUCCESS, 5, 65, "Ascend950", ge::DT_BF16, 11010, 1,
                    "5 5 65 100001 1 1 5 1 1 23808 4769 1 5 5 1 0 65 ");
}

TEST_F(AttentionWorkerCombineTilingTest, mxfp_k_batch_and_tail)
{
    // Same logical H, but FP4's packed UB stride allows more complete expert rows per batch.
    CheckMxfpTiling(8193, 258, 2, ge::DT_FLOAT8_E8M0, ge::GRAPH_SUCCESS, 3, 65, "Ascend950", ge::DT_BF16, 11020, 0,
                    "3 3 65 8193 0 1 3 1 1 8224 8193 1 1 1 21 2 3 ");
    CheckMxfpTiling(8193, 258, 4, ge::DT_FLOAT8_E8M0, ge::GRAPH_SUCCESS, 3, 65, "Ascend950", ge::DT_BF16, 11020, 0,
                    "3 3 65 8193 0 1 3 1 1 8224 8193 1 1 1 43 22 1 ");
}
