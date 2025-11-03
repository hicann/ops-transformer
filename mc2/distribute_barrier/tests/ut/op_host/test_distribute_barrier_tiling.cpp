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
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class DistributeBarrierTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "DistributeBarrierTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "DistributeBarrierTiling TearDown" << std::endl;
    }
};

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling)
{
    struct DistributeBarrierCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 10000UL;
    std::string expectTilingData = "8 16 20 196352 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    uint64_t mc2TilingDataReservedLen = 42;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_world_size_1)
{
    struct DistributeBarrierCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_world_size_385)
{
    struct DistributeBarrierCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(385)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_time_out) 
{
    struct DistributeBarrierCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 10000UL;
    std::string expectTilingData = "8 16 20 196352 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    uint64_t mc2TilingDataReservedLen = 42;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_elastic_info) 
{
    struct DistributeBarrierCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 10000UL;
    std::string expectTilingData = "8 16 20 196352 0 256 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    uint64_t mc2TilingDataReservedLen = 42;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(DistributeBarrierTiling, distribute_barrier_test_tiling_time_out_elastic_info) 
{
    struct DistributeBarrierCompileInfo {} compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    uint64_t expectTilingKey = 10000UL;
    std::string expectTilingData = "8 16 20 196352 0 257 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    uint64_t mc2TilingDataReservedLen = 42;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}
