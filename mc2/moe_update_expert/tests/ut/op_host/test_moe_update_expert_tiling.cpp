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

class MoeUpdateExpertTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeUpdateExpertTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MoeUpdateExpertTiling TearDown" << std::endl;
    }
};

TEST_F(MoeUpdateExpertTiling, moe_update_expert_test_tiling_no_tailor)
{
    struct MoeUpdateExpertCompileInfo {} compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    
    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert", // 算子类型
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}  
        },
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}   
        },
        {
            {"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize
    );

    uint64_t expectTilingKey = 0; 
    std::string expectTilingData = "34359738496 21474836736 20 0 8 0 0 "; // 根据实际情况设置
    std::vector<size_t> expectWorkspaces = {4294967295}; // 根据实际情况设置
    uint64_t mc2TilingDataReservedLen = 0; // 根据实际情况设置

    // 执行测试用例
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertTiling, moe_update_expert_test_tiling_expert_tailor)
{
    struct MoeUpdateExpertCompileInfo {} compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    // 构造输入输出张量描述
    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert", // 算子类型
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
            {{{8, }, {8, }}, ge::DT_FLOAT, ge::FORMAT_ND}, 
            {{{128, }, {128, }}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        // 输出张量描述
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}  
        },
        // 算子属性
        {
            {"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize
    );

    uint64_t expectTilingKey =11UL; // 根据实际情况设置
    std::string expectTilingData = "34359738496 21474836736 20 0 8 0 1 "; // 根据实际情况设置
    std::vector<size_t> expectWorkspaces = {4294967295}; // 根据实际情况设置
    uint64_t mc2TilingDataReservedLen = 0; // 根据实际情况设置

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertTiling, moe_update_expert_test_tiling_wrongDim_expertIds)
{
    struct MoeUpdateExpertCompileInfo {} compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert", // 算子类型
        {
            {{{128, }, {128, }}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
            {{{8, }, {8, }}, ge::DT_FLOAT, ge::FORMAT_ND}, 
            {{{128, }, {128, }}, ge::DT_BOOL, ge::FORMAT_ND}
        },
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}  
        },
        {
            {"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize
    );

    uint64_t expectTilingKey = 0; // 根据实际情况设置
    std::string expectTilingData = ""; // 根据实际情况设置
    std::vector<size_t> expectWorkspaces = {0}; // 根据实际情况设置
    uint64_t mc2TilingDataReservedLen = 0; // 根据实际情况设置

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertTiling, moe_update_expert_test_tiling_wrongDim_expert_scales)
{
    struct MoeUpdateExpertCompileInfo {} compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},  
            {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, }, {128, }}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
            {{{8, }, {8, }}, ge::DT_FLOAT, ge::FORMAT_ND}, 
            {{{128, }, {128, }}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        {
            {"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize
    );

    // 期望值占位符
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertTiling, moe_update_expert_test_tiling_wrongDim_pruning_threshold)
{
    struct MoeUpdateExpertCompileInfo {} compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, 
            {{{128, }, {128, }}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        {
            {"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize
    );

    // 期望值占位符
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertTiling, moe_update_expert_test_tiling_wrongDim_active_mask)
{
    struct MoeUpdateExpertCompileInfo {} compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
            {{{8, }, {8, }}, ge::DT_FLOAT, ge::FORMAT_ND}, 
            {{{127, }, {127, }}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        {
            {"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize
    );

    // 期望值占位符
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertTiling, moe_update_expert_test_tiling_wrongDim_balancedExpertIds)
{
    struct MoeUpdateExpertCompileInfo {} compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
            {{{8, }, {8, }}, ge::DT_FLOAT, ge::FORMAT_ND}, 
            {{{128, }, {128, }}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        {
            {{{128, }, {128, }}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        {
            {"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize
    );

    // 期望值占位符
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertTiling, moe_update_expert_test_tiling_wrongDim_balanced_active_mask)
{
    struct MoeUpdateExpertCompileInfo {} compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}, 
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
            {{{8, }, {8, }}, ge::DT_FLOAT, ge::FORMAT_ND}, 
            {{{128, }, {128, }}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        {
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 9}, {128, 9}}, ge::DT_BOOL, ge::FORMAT_ND} 
        },
        {
            {"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize
    );

    // 期望值占位符
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces,
                    mc2TilingDataReservedLen);
}
