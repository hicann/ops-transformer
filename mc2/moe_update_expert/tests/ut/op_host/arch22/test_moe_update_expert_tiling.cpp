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
#include "mc2/common/tests/ut/internal_fallback_assertions.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "../../../../../common/tests/ut/formal_error_assertions.h"
#include "mc2_tiling_case_executor.h"

using Mc2ErrorCodeUT::ExcelErrorReportScope;
using Mc2ErrorCodeUT::ExpectExcelFormalError;

using namespace std;

namespace MoeUpdateExpertUT {
class MoeUpdateExpertArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeUpdateExpertArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeUpdateExpertArch22TilingTest TearDown" << std::endl;
    }
};

TEST_F(MoeUpdateExpertArch22TilingTest, NoTailor)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert", // 算子类型
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "34359738496 21474836736 20 0 8 0 0 "; // 根据实际情况设置
    std::vector<size_t> expectWorkspaces = {4294967295};                  // 根据实际情况设置
    uint64_t mc2TilingDataReservedLen = 0;                                // 根据实际情况设置

    // 执行测试用例
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, ExpertTailor)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    // 构造输入输出张量描述
    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert", // 算子类型
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        // 输出张量描述
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        // 算子属性
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 5UL;                                       // 根据实际情况设置
    std::string expectTilingData = "34359738496 21474836736 20 0 8 0 1 "; // 根据实际情况设置
    std::vector<size_t> expectWorkspaces = {4294967295};                  // 根据实际情况设置
    uint64_t mc2TilingDataReservedLen = 0;                                // 根据实际情况设置

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDimExpertIds)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert", // 算子类型
        {{{{
               128,
           },
           {
               128,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;               // 根据实际情况设置
    std::string expectTilingData = "";          // 根据实际情况设置
    std::vector<size_t> expectWorkspaces = {0}; // 根据实际情况设置
    uint64_t mc2TilingDataReservedLen = 0;      // 根据实际情况设置

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDimExpertScales)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_FLOAT16,
          ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    // 期望值占位符
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDimPruningThreshold)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    // 期望值占位符
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDimActiveMask)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               127,
           },
           {
               127,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    // 期望值占位符
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDimBalancedExpertIds)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara("MoeUpdateExpert",
                                              {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{
                                                     8,
                                                 },
                                                 {
                                                     8,
                                                 }},
                                                ge::DT_FLOAT,
                                                ge::FORMAT_ND},
                                               {{{
                                                     128,
                                                 },
                                                 {
                                                     128,
                                                 }},
                                                ge::DT_BOOL,
                                                ge::FORMAT_ND}},
                                              {{{{
                                                     128,
                                                 },
                                                 {
                                                     128,
                                                 }},
                                                ge::DT_INT32,
                                                ge::FORMAT_ND},
                                               {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
                                              {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                               {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                              &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    // 期望值占位符
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDimBalancedActiveMask)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 9}, {128, 9}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    // 期望值占位符
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

// ========== dtype error tests ==========

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDtypeExpertIds)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDtypeEplbTable)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDtypeExpertScales)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDtypePruningThreshold)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDtypeActiveMask)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDtypeBalancedExpertIds)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT64, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDtypeBalancedActiveMask)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

// ========== eplb_table dim error tests ==========

TEST_F(MoeUpdateExpertArch22TilingTest, WrongDimEplbTable)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

// ========== shape error tests ==========

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapeExpertIdsBsZero)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{0, 8}, {0, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{0, 8}, {0, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{0, 8}, {0, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapeExpertIdsBsExceed)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{513, 8}, {513, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1024, 5}, {1024, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{513, 8}, {513, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{513, 8}, {513, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapeExpertIdsKZero)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 0}, {128, 0}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 0}, {128, 0}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 0}, {128, 0}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapeExpertIdsKExceed)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 17}, {128, 17}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 17}, {128, 17}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 17}, {128, 17}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapeEplbTableDim0Zero)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{0, 5}, {0, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapeEplbTableDim0Exceed)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1025, 5}, {1025, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapeEplbTableDim0LessK)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{4, 5}, {4, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapeEplbTableDim1OutOfRange)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 1}, {256, 1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapeExpertScalesMismatch)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 7}, {128, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapePruningThreshold3D)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 4, 8}, {2, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapePruningThreshold1DMismatch)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               7,
           },
           {
               7,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongShapeActiveMaskDim)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

// ========== optional input combination error tests ==========

TEST_F(MoeUpdateExpertArch22TilingTest, ExpertScalesAlone)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

// "absent" placeholder for optional inputs that should not be provided
using TD = gert::TilingContextPara::TensorDescription;
static const TD ABSENT = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};

TEST_F(MoeUpdateExpertArch22TilingTest, PruningThresholdAlone)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         ABSENT,
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         ABSENT},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, ActiveMaskAlone)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         ABSENT,
         ABSENT,
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, ActiveMaskExpertScales)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         ABSENT,
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, ActiveMaskPruningThreshold)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         ABSENT,
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

// ========== format error tests (CheckFormat rejects FORMAT_FRACTAL_NZ) ==========

TEST_F(MoeUpdateExpertArch22TilingTest, WrongFormatExpertIds)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongFormatEplbTable)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongFormatExpertScales)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongFormatPruningThreshold)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_FRACTAL_NZ},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

TEST_F(MoeUpdateExpertArch22TilingTest, WrongFormatActiveMask)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 8}, {128, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               8,
           },
           {
               8,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND},
         {{{
               128,
           },
           {
               128,
           }},
          ge::DT_BOOL,
          ge::FORMAT_FRACTAL_NZ}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    uint64_t mc2TilingDataReservedLen = 0;

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey, expectTilingData,
                       expectWorkspaces, mc2TilingDataReservedLen);
}

// ========== boundary positive tests ==========

TEST_F(MoeUpdateExpertArch22TilingTest, BoundaryBsMax)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{512, 4}, {512, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{512, 4}, {512, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{512, 4}, {512, 4}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeUpdateExpertArch22TilingTest, BoundaryKMax)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 16}, {128, 16}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 16}, {128, 16}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 16}, {128, 16}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeUpdateExpertArch22TilingTest, BoundaryMoeExpertNumMax)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 4}, {128, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1024, 5}, {1024, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 4}, {128, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 4}, {128, 4}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeUpdateExpertArch22TilingTest, BoundaryMoeExpertNumEqualsK)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{8, 5}, {8, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeUpdateExpertArch22TilingTest, BoundaryFMin)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 4}, {128, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 2}, {256, 2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 4}, {128, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 4}, {128, 4}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeUpdateExpertArch22TilingTest, BoundaryFMax)
{
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert",
        {{{{128, 4}, {128, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 9}, {256, 9}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 4}, {128, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 4}, {128, 4}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeUpdateExpertArch22TilingTest, InvalidBalanceModeReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert", // 算子类型
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "34359738496 21474836736 20 0 8 0 0 "; // 根据实际情况设置
    std::vector<size_t> expectWorkspaces = {4294967295};                  // 根据实际情况设置
    uint64_t mc2TilingDataReservedLen = 0;                                // 根据实际情况设置

    // 执行测试用例
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "balance_mode", "2");
}

TEST_F(MoeUpdateExpertArch22TilingTest, InvalidRankRelationReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert", // 算子类型
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "34359738496 21474836736 20 0 8 0 0 "; // 根据实际情况设置
    std::vector<size_t> expectWorkspaces = {4294967295};                  // 根据实际情况设置
    uint64_t mc2TilingDataReservedLen = 0;                                // 根据实际情况设置

    // 执行测试用例
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0027", "local_rank_id, world_size", "local_rank_id=8, world_size=8",
                           "local_rank_id must be in [0, world_size)");
}

TEST_F(MoeUpdateExpertArch22TilingTest, InvalidWorldSizeReportsSingleFormalError)
{
    ExcelErrorReportScope reportScope;
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert", // 算子类型
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "34359738496 21474836736 20 0 8 0 0 "; // 根据实际情况设置
    std::vector<size_t> expectWorkspaces = {4294967295};                  // 根据实际情况设置
    uint64_t mc2TilingDataReservedLen = 0;                                // 根据实际情况设置

    // 执行测试用例
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    ExpectExcelFormalError("EZ0026", "world_size", "0");
}
TEST_F(MoeUpdateExpertArch22TilingTest, MissingDefaultWorldSizeRetainsInternalFallback)
{
    ExcelErrorReportScope reportScope;
    struct MoeUpdateExpertCompileInfo {
    } compileInfo;
    const std::string socVersion = "";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    gert::TilingContextPara tilingContextPara(
        "MoeUpdateExpert", // 算子类型
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{256, 5}, {256, 5}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{128, 8}, {128, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{"local_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"balance_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    tilingContextPara.attrs_.erase(tilingContextPara.attrs_.begin() + 1, tilingContextPara.attrs_.end());
    std::string expectTilingData = "34359738496 21474836736 20 0 8 0 0 "; // 根据实际情况设置
    std::vector<size_t> expectWorkspaces = {4294967295};                  // 根据实际情况设置
    uint64_t mc2TilingDataReservedLen = 0;                                // 根据实际情况设置

    // 执行测试用例
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
    Mc2ErrorCodeUT::ExpectInternalFallbackReport("CheckAttrs: Schema-defaulted world_size metadata is absent");
}
} // namespace MoeUpdateExpertUT
