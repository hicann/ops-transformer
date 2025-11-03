/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include <iostream>
#include <map>
#include <vector>
#include <string>

#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class AlltoAllAllGatherBmmTiling : public testing::Test
{
protected:
    static void SetUpTestCase() {
        std::cout << "AlltoAllAllGatherBmmTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AlltoAllAllGatherBmmTiling TearDown" << std::endl;
    }
};

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_1)
{
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000001UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 256, 32}, {16, 256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_0_invalid_H) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 256, 65536}, {16, 256, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_0_unequal_H) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 256, 64}, {16, 256, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_1_weight_trans) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 128, 64}, {4, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000011UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_xShard_1_actType_1) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000001UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_xShard_1_actType_4) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000001UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_invalid_E) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{32, 128, 64}, {32, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000001UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_invalid_EOverep_intercept) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{160, 128, 64}, {160, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{40, 128, 64}, {40, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{40, 512, 64}, {40, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000101UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_bf16) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000101UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_0) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 256, 32}, {16, 256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_1) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_1_test1) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test1) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>((true))},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test2) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test3) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test4) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test5) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test6) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{1, 128, 64}, {1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test7) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 65536}, {16, 128, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test8) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test9) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128, 1}, {4, 1, 128, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test10) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 64}, {16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{5, 1, 128}, {5, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test11) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 128, 0}, {16, 128, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test12) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 64, 128}, {4, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 128}, {4, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 512, 64}, {4, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_ep2) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 256, 32}, {16, 256, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 128, 128}, {8, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{8, 512, 128}, {8, 512, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_cut_e) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{16, 2254, 2048}, {16, 2254, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4, 9016, 1024}, {4, 9016, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_cut_c) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{8, 2254, 2048}, {8, 2254, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 4096, 1024}, {2, 4096, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2, 9016, 1024}, {2, 9016, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_tile_short) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{8, 2254, 6144}, {8, 2254, 6144}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 12288, 6144}, {2, 12288, 6144}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2, 9016, 6144}, {2, 9016, 6144}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_multi_e) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{40, 2254, 6144}, {40, 2254, 6144}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{10, 12288, 1024}, {10, 12288, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{10, 9016, 1024}, {10, 9016, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(AlltoAllAllGatherBmmTiling, all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_local_tail_e) {
    struct DistributeBarrierCompileInfo {} compileInfo;
    const std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 4096;
    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        {{{{10, 2254, 1024}, {10, 2254, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{5, 8192, 8192}, {5, 8192, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{5, 4508, 8192}, {5, 4508, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("ep_group")},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>("tp_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"x_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y2_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"output_y3_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 1000000000000000000UL;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}